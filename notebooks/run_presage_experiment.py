#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append('../src/')  # Replace with your actual path
from train import set_seed, parse_config, get_predictions

import json

import datetime

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# from datamodule import  ReplogleDataModule
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
"""

dataset = "nadig_jurkat"
seed = "seed_0"

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

modify_config = {
    "training.eval_test": False,
    # "model.pathway_files": "../sample_files/prior_files/sample.knowledge_experimental.txt",
    # "model.pathway_files": "None",
    "model.pathway_files": "/raid/yangpeng_lab/c12212609/PRESAGE/data/topic_embed/all_embed.txt",
    "data.data_dir": "../data/",
}

config.update(modify_config)
config = parse_config(config)

set_seed(config["training"].pop("seed", None))

offline = config["training"].pop("offline", False)
do_test_eval = config["training"].pop("eval_test", True)
predictions_file = config["training"].pop("predictions_file", None)
embedding_pref = config["training"].pop("embedding_file", None)
attention_file = config["training"].pop("attention_file", None)

config['data']['dataset'] = dataset

config['data']['seed'] = f"../splits/{dataset}_random_splits/{seed}.json"

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
model_config['pca_dim'] = None
model_config['source'] = 'temp'
model_config['learnable_gene_embedding'] = False

module = PRESAGE(
    model_config,
    datamodule,
    datamodule.pert_covariates.shape[1],
    datamodule.n_genes,
    # latent_dim or datamodule.n_genes,
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
    save_dir="./logs",
    name=dataset,
    version=seed.split('/')[-1].split('.json')[0]
)

# if predictions_file == "None":
#     predictions_file = f"predictions_{dataset}.csv"

print("default prediction file:", predictions_file)
predictions_file = f"predictions_all_{dataset}.csv"
print("adjusted prediction file:", predictions_file)

early_stop_callback = EarlyStopping(
    monitor="val_loss",
    min_delta=1e-6,
    patience=10,
    verbose=True,
    mode="min",
)

# Get current date and time
now = datetime.datetime.now()

# Format the date and time
now_str = now.strftime("%Y-%m-%d-%H-%M-%S")

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    dirpath="./saved_models",
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
# lightning_module is the pytorch lighting, datamodule from datamodule.py
# Get the best model path
best_model_path = checkpoint_callback.best_model_path

datamodule.setup("test")
datamodule._data_setup = False

checkpoint = torch.load(best_model_path)
lightning_module.load_state_dict(checkpoint["state_dict"])
# os.remove(best_model_path)

# log final eval metrics
trainer.test(lightning_module, datamodule=datamodule)

dataloader = datamodule.test_dataloader()
avg_predictions = get_predictions(
    trainer, lightning_module, dataloader, datamodule.var_names
)
avg_predictions = avg_predictions.loc[
    :, datamodule.train_dataset.adata.var.measured_gene
]
avg_predictions.to_csv(predictions_file)
avg_predictions
