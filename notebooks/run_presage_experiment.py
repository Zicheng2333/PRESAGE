#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

sys.path.append("../src/")
from train import set_seed, parse_config, get_predictions

from presage_datamodule import ReploglePRESAGEDataModule
from model_harness import ModelHarness
from presage import PRESAGE

"""
Datasets:
  replogle_k562_gw
  replogle_k562_essential_unfiltered
  replogle_rpe1_essential_unfiltered
  nadig_hepg2
  nadig_jurkat

Pathway Files:
"None"
"../sample_files/prior_files/sample.knowledge_experimental.txt"
"/raid/yangpeng_lab/c12212609/PRESAGE/data/topic_embed/*.txt"
"""


def make_experiment_dir(root: Path, exp_time: str) -> Path:
    run_dir = root / exp_time
    suffix = 1
    while run_dir.exists():
        run_dir = root / f"{exp_time}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, torch.Tensor):
        if x.numel() == 1:
            return float(x.detach().cpu().item())
        return x.detach().cpu().tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def _to_python_scalar(x):
    if isinstance(x, torch.Tensor):
        if x.numel() != 1:
            return None
        return float(x.detach().cpu().item())
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, (int, float)):
        return float(x)
    return None


def _convert_test_output_for_json(test_output):
    converted = []
    for item in test_output:
        row = {}
        for k, v in item.items():
            scalar = _to_python_scalar(v)
            row[k] = scalar if scalar is not None else str(v)
        converted.append(row)
    return converted


def extract_per_perturbation_metrics(single_eval_by_set):
    """
    Convert evaluator per-perturbation metrics into a flat table.
    """
    rows = []
    for test_set, eval_dict in single_eval_by_set.items():
        for perturbation, metric_dict in eval_dict.items():
            if not isinstance(metric_dict, dict):
                continue

            row = {
                "test_set": str(test_set),
                "perturbation": str(perturbation),
            }
            has_metric = False
            for metric_name, metric_value in metric_dict.items():
                scalar = _to_python_scalar(metric_value)
                if scalar is None or not np.isfinite(scalar):
                    continue
                row[metric_name] = scalar
                has_metric = True

            if has_metric:
                rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["test_set", "perturbation"])

    return pd.DataFrame(rows)


def bootstrap_sample_means(values, n_bootstrap, rng, chunk_size=128):
    values = np.asarray(values, dtype=np.float64)
    n = int(values.shape[0])
    if n == 0:
        return np.array([], dtype=np.float64)

    out = np.empty(n_bootstrap, dtype=np.float64)
    done = 0
    while done < n_bootstrap:
        b = min(chunk_size, n_bootstrap - done)
        sample_idx = rng.integers(0, n, size=(b, n), dtype=np.int64)
        out[done : done + b] = values[sample_idx].mean(axis=1)
        done += b
    return out


def bootstrap_metric_summary(per_pert_df, n_bootstrap=1000, seed=42, chunk_size=128):
    metric_cols = [
        c for c in per_pert_df.columns if c not in {"test_set", "perturbation"}
    ]
    rng = np.random.default_rng(seed)

    rows = []
    for test_set, df_one_set in per_pert_df.groupby("test_set", sort=False):
        for metric_name in metric_cols:
            values = df_one_set[metric_name].to_numpy(dtype=np.float64, copy=True)
            values = values[np.isfinite(values)]
            n_perts = int(values.shape[0])
            if n_perts == 0:
                continue

            point_mean = float(values.mean())
            if n_bootstrap > 0:
                bs = bootstrap_sample_means(
                    values,
                    n_bootstrap=n_bootstrap,
                    rng=rng,
                    chunk_size=chunk_size,
                )
                ci_low, ci_high = np.quantile(bs, [0.025, 0.975])
                boot_mean = float(bs.mean())
            else:
                ci_low, ci_high = np.nan, np.nan
                boot_mean = point_mean

            rows.append(
                {
                    "test_set": test_set,
                    "metric": metric_name,
                    "point_mean": point_mean,
                    "bootstrap_mean": boot_mean,
                    "ci95_low": float(ci_low),
                    "ci95_high": float(ci_high),
                    "n_perturbations": n_perts,
                    "bootstrap_iters": int(n_bootstrap),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "test_set",
                "metric",
                "point_mean",
                "bootstrap_mean",
                "ci95_low",
                "ci95_high",
                "n_perturbations",
                "bootstrap_iters",
            ]
        )
    return pd.DataFrame(rows)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="replogle_k562_gw")
parser.add_argument("--n_nmf_embedding", type=int, default=128)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--seed", type=str, default="seed_0")
parser.add_argument("--pathway_files", type=str, default=None)
parser.add_argument("--use_training_gex_embeddings", action="store_true")
parser.add_argument("--eval_test", action="store_true")
parser.add_argument("--use_old", action="store_true")
parser.add_argument(
    "--bootstrap_iters",
    type=int,
    default=1000,
    help="Bootstrap iterations for post-training evaluation metrics.",
)
parser.add_argument(
    "--bootstrap_seed",
    type=int,
    default=42,
    help="Random seed for bootstrap resampling.",
)
parser.add_argument(
    "--bootstrap_chunk_size",
    type=int,
    default=128,
    help="Chunk size for vectorized bootstrap sampling.",
)
parser.add_argument("--tag", type=str, default=None)

parser.add_argument("--batch_size", type=int, default=None)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--output_root", type=str, default="./experiment_runs")

args = parser.parse_args()

dataset = args.dataset
seed = args.seed
run_tag = args.tag or ""
run_name = f"{dataset}{run_tag}" if run_tag else dataset
experiment_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = make_experiment_dir(Path(args.output_root), exp_time=experiment_time)
print("experiment_dir:", run_dir)

default_config_file = "../configs/defaults_config.json"
singles_config_file = "../configs/singles_config.json"
ds_config_file = f"../configs/{dataset}_config.json"

# Load the default config
with open(default_config_file, "r") as f:
    config = json.load(f)
with open(singles_config_file, "r") as f:
    singles_config = json.load(f)
with open(ds_config_file, "r") as f:
    ds_config = json.load(f)

singles_config.update(singles_config)
singles_config.update(ds_config)

new_config = {}
for key, value in singles_config.items():
    if value is not None and key not in {"config", "data_config"}:
        new_config[key.replace("_", ".", 1)] = value
singles_config = new_config
config.update(singles_config)

if args.batch_size is not None:
    modify_config = {
    "training.eval_test": args.eval_test,
    "model.pathway_files": args.pathway_files,
    "model.n_nmf_embedding": args.n_nmf_embedding,
    "model.use_training_gex_embeddings": args.use_training_gex_embeddings,
    "data.data_dir": "../data/",
    "data.batch_size": args.batch_size,
    "model.lr": args.lr 
}
else:
    modify_config = {
    "training.eval_test": args.eval_test,
    "model.pathway_files": args.pathway_files,
    "model.n_nmf_embedding": args.n_nmf_embedding,
    "model.use_training_gex_embeddings": args.use_training_gex_embeddings,
    "data.data_dir": "../data/",
    #"model.lr": args.lr 
}

config.update(modify_config)
config = parse_config(config)

config["data"]["dataset"] = dataset
config["data"]["seed"] = f"../splits/{dataset}_random_splits/{seed}.json"

config_used = deepcopy(config)
config_used["cli_args"] = vars(args)
config_used["experiment"] = {
    "time": experiment_time,
    "run_dir": str(run_dir),
}
with open(run_dir / "config_used.json", "w") as f:
    json.dump(_to_jsonable(config_used), f, indent=2)

set_seed(config["training"].pop("seed", None))

offline = config["training"].pop("offline", False)
do_test_eval = config["training"].pop("eval_test", True)
predictions_file = config["training"].pop("predictions_file", None)
embedding_pref = config["training"].pop("embedding_file", None)
attention_file = config["training"].pop("attention_file", None)

seed = config["data"].pop("seed")
datamodule = ReploglePRESAGEDataModule.from_config(config["data"])
datamodule.do_test_eval = do_test_eval

if hasattr(datamodule, "set_seed"):
    datamodule.set_seed(seed)
config["data"]["seed"] = seed

datamodule.prepare_data()
datamodule.setup("fit")
print("datamodule setup complete.")

# initialize model
model_config = config["model"]
model_config["dataset"] = dataset

# legacy unused parameters
model_config["pca_dim"] = None
model_config["source"] = "temp"
model_config["learnable_gene_embedding"] = False

if args.use_old:
    from presage_old import PRESAGE

    module = PRESAGE(
        model_config,
        datamodule,
        datamodule.pert_covariates.shape[1],
        datamodule.n_genes,
    )
else:
    from presage import PRESAGE

    module = PRESAGE(
        model_config,
        datamodule,
        datamodule.pert_covariates.shape[1],
        datamodule.n_genes,
    )

if hasattr(module, "custom_init"):
    module.custom_init()

lightning_module = ModelHarness(
    module,
    datamodule,
    model_config,
)

print("model initialization complete.")

# run trainer
logger = pl.loggers.CSVLogger(
    save_dir=str(run_dir / "logs"),
    name=run_name,
    version=seed.split('/')[-1].split('.json')[0]
)

print("default prediction file:", predictions_file)
predictions_file = run_dir / f"predictions_all_{dataset}.csv"
print("adjusted prediction file:", predictions_file)

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=1e-6,
    patience=args.patience,
    verbose=True,
    mode="min",
)

now_str = experiment_time

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath=str(run_dir / "saved_models"),
    filename=f"my_model-{dataset}-{seed.split('/')[-1].split('.json')[0]}-{now_str}-{{epoch:02d}}-{{val_loss:.2f}}",
    save_top_k=1,
    mode="min",
)

torch.autograd.set_detect_anomaly(True)
trainer = pl.Trainer(
    logger=logger,
    log_every_n_steps=3,
    num_sanity_val_steps=10,
    callbacks=[early_stop_callback, checkpoint_callback],
    reload_dataloaders_every_n_epochs=1,
    **config["training"],
    gradient_clip_val=0.1,
)

trainer.fit(lightning_module, datamodule=datamodule)
best_model_path = checkpoint_callback.best_model_path

datamodule.setup("test")
datamodule._data_setup = False

checkpoint = torch.load(best_model_path)
lightning_module.load_state_dict(checkpoint["state_dict"])

# log final eval metrics
test_output = trainer.test(lightning_module, datamodule=datamodule)
test_metrics_file = run_dir / f"test_metrics_{dataset}.json"
with open(test_metrics_file, "w") as f:
    json.dump(_convert_test_output_for_json(test_output), f, indent=2)
print("saved test metrics:", test_metrics_file)

# bootstrap over per-perturbation metrics collected during trainer.test
single_eval_by_set = getattr(lightning_module, "test_single_eval_results", None)
if not single_eval_by_set:
    # fallback for older harness behavior
    evaluator = getattr(lightning_module, "evaluator", None)
    if evaluator is not None and hasattr(evaluator, "all_single_evals"):
        single_eval_by_set = {"default": dict(evaluator.all_single_evals)}
    else:
        single_eval_by_set = {}

per_pert_df = extract_per_perturbation_metrics(single_eval_by_set)
per_pert_file = run_dir / f"per_perturbation_metrics_{dataset}.csv"
bootstrap_file = run_dir / f"bootstrap_summary_{dataset}.csv"

if per_pert_df.empty:
    print("[WARN] No per-perturbation scalar metrics found, skip bootstrap summary.")
else:
    per_pert_df.to_csv(per_pert_file, index=False)
    print("saved per-perturbation metrics:", per_pert_file)

    bootstrap_df = bootstrap_metric_summary(
        per_pert_df=per_pert_df,
        n_bootstrap=int(args.bootstrap_iters),
        seed=int(args.bootstrap_seed),
        chunk_size=int(args.bootstrap_chunk_size),
    )
    bootstrap_df.to_csv(bootstrap_file, index=False)
    print("saved bootstrap summary:", bootstrap_file)
    if bootstrap_df.shape[0] > 0:
        print(
            bootstrap_df.sort_values(
                by=["test_set", "metric"], ascending=[True, True]
            ).to_string(index=False)
        )

dataloader = datamodule.test_dataloader()
avg_predictions = get_predictions(
    trainer, lightning_module, dataloader, datamodule.var_names
)
avg_predictions = avg_predictions.loc[
    :, datamodule.train_dataset.adata.var.measured_gene
]
avg_predictions.to_csv(predictions_file)
print("saved averaged predictions:", predictions_file)
