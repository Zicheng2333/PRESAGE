#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
NOTEBOOK_DIR="$REPO_ROOT/notebooks"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-12}"
DATASET="replogle_k562_gw"
SEED_NAME="${SEED_NAME:-seed_0}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiment_runs/gw3_grid_search}"
PATHWAY_FILE="${PATHWAY_FILE:-/raid/yangpeng_lab/c12212609/PRESAGE/data/topic_embed/ours_gw.txt}"

mkdir -p "$OUTPUT_ROOT"
cd "$NOTEBOOK_DIR"

COMMON_ARGS=(
  --dataset "$DATASET"
  --seed "$SEED_NAME"
  --pathway_files "$PATHWAY_FILE"
  --use_training_gex_embeddings
  --output_root "$OUTPUT_ROOT"
  --bootstrap_iters "$BOOTSTRAP_ITERS"
  --enhance_added_embeddings
  --optimizer AdamW
  --scheduler cosine
  --warmup_epochs 5
  --min_lr_ratio 0.05
  --gradient_clip_val 0.5
  --batch_size 256
  --max_epochs 10000
  --precision 32
  --pool_nlayers 2
  --softmax_temperature 0.10
  --gat_weight 0.80
  --item_hidden_size 1024
  --item_nlayers 0
  --pathway_item_hidden_size 256
  --pathway_item_nlayers 3
  --item_dropout 0.08
  --pathway_dropout 0.06
  --source_dropout 0.05
  --pathway_layer_norm
)

run_idx=0
for lr in 1.0e-3 1.35e-3 1.8e-3; do
  for wd in 1e-15 1e-6; do
    for pool in gat vector; do
      for patience in 10 30; do
        ((run_idx += 1))
        tag="gw3_gs_${run_idx}"
        echo "[gw3_gs ${run_idx}] lr=${lr} wd=${wd} pool=${pool} patience=${patience}"
        CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" run_presage_experiment.py \
          "${COMMON_ARGS[@]}" \
          --tag "$tag" \
          --patience "$patience" \
          --lr "$lr" \
          --weight_decay "$wd" \
          --pathway_weight_type "$pool"
      done
    done
  done
done

"$PYTHON_BIN" "$SCRIPT_DIR/summarize_grid_search.py" "$OUTPUT_ROOT" \
  --field lr=model.lr \
  --field weight_decay=model.weight_decay \
  --field pathway_weight_type=model.pathway_weight_type \
  --field patience=cli_args.patience \
  --field batch_size=data.batch_size \
  --field item_hidden_size=model.item_hidden_size
