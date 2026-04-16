#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
NOTEBOOK_DIR="$REPO_ROOT/notebooks"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-6}"
DATASET="nadig_hepg2"
SEED_NAME="${SEED_NAME:-seed_0}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiment_runs/hep5_grid_search}"
PATHWAY_FILE="${PATHWAY_FILE:-/raid/yangpeng_lab/c12212609/PRESAGE/data/topic_embed/all_ess_g2.txt}"

# Anchor config: lr=8.2e-4, wd=1e-15, batch=16; keep default hidden sizes from defaults_config
mkdir -p "$OUTPUT_ROOT"
cd "$NOTEBOOK_DIR"

run_idx=0
for lr in 5.5e-4 8.2e-4 1.2e-3; do
  for n_nmf in 64 128; do
for pathway_hidden in 64 128; do
  for patience in 10 30; do
        run_idx=$((run_idx + 1))
        tag="hep5_gs_${run_idx}"
        echo "[hep5_gs ${run_idx}] lr=${lr} n_nmf=${n_nmf} pathway_hidden=${pathway_hidden} patience=${patience}"
        CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" run_presage_experiment.py               --dataset "$DATASET"               --seed "$SEED_NAME"               --pathway_files "$PATHWAY_FILE"               --tag "$tag"               --output_root "$OUTPUT_ROOT"               --bootstrap_iters "$BOOTSTRAP_ITERS"               --patience "$patience"     --enhance_added_embeddings               --optimizer AdamW               --scheduler cosine               --warmup_epochs 5               --min_lr_ratio 0.05               --gradient_clip_val 0.5               --batch_size 16               --lr "$lr"               --weight_decay 1e-6               --n_nmf_embedding "$n_nmf"               --max_epochs 10000               --precision 32               --pathway_weight_type gat               --pool_nlayers 2               --softmax_temperature 0.10               --gat_weight 0.85               --item_hidden_size 512               --item_nlayers 0               --pathway_item_hidden_size "$pathway_hidden"               --pathway_item_nlayers 2               --item_dropout 0.12               --pathway_dropout 0.10               --source_dropout 0.10               --pathway_layer_norm
      done
    done
  done
done

    python - "$OUTPUT_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
all_keys = set()
preferred = [
    "test_loss",
    "test_avg_cossim_top20_de",
    "test_avg_cossim_top20_unionde",
    "test_perturbations_with_effect_avg_cossim_top20_unionde",
    "test_pn_mse_top20_de",
    "test_perturbations_with_effect_pn_mse_top20_unionde",
]
for run_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
    cfg_file = run_dir / "config_used.json"
    summary_file = run_dir / "training_summary.json"
    test_files = list(run_dir.glob("test_metrics_*.json"))
    if not (cfg_file.exists() and summary_file.exists() and test_files):
        continue
    cfg = json.loads(cfg_file.read_text())
    summary = json.loads(summary_file.read_text())
    test_metrics = json.loads(test_files[0].read_text())
    test_metrics = test_metrics[0] if isinstance(test_metrics, list) and test_metrics else {}
    row = {
        "run_dir": str(run_dir),
        "tag": cfg.get("cli_args", {}).get("tag"),
        "best_val_loss": summary.get("best_val_loss"),
"lr": cfg.get("model", {}).get("lr"),
"n_nmf_embedding": cfg.get("model", {}).get("n_nmf_embedding"),
"pathway_item_hidden_size": cfg.get("model", {}).get("pathway_item_hidden_size"),
"patience": cfg.get("cli_args", {}).get("patience"),
"batch_size": cfg.get("data", {}).get("batch_size"),
    }
    for key in preferred:
        if key in test_metrics:
            row[key] = test_metrics[key]
    for key, value in test_metrics.items():
        if isinstance(value, (int, float)):
            row.setdefault(key, value)
    rows.append(row)
    all_keys.update(row)

rows.sort(key=lambda r: float("inf") if r.get("best_val_loss") is None else float(r["best_val_loss"]))
fieldnames = [
    "run_dir",
    "tag",
    "best_val_loss",
    "test_loss",
    "test_avg_cossim_top20_de",
    "test_avg_cossim_top20_unionde",
    "test_perturbations_with_effect_avg_cossim_top20_unionde",
    "test_pn_mse_top20_de",
    "test_perturbations_with_effect_pn_mse_top20_unionde",
"lr",
"n_nmf_embedding",
"pathway_item_hidden_size",
"patience",
"batch_size",
]
fieldnames += sorted(k for k in all_keys if k not in fieldnames)
out_file = root / "grid_summary.csv"
with out_file.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"saved summary: {out_file}")
for row in rows[:5]:
    print(row)
PY
