#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
NOTEBOOK_DIR="$REPO_ROOT/notebooks"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-15}"
DATASET="replogle_k562_gw"
SEED_NAME="${SEED_NAME:-seed_0}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiment_runs/gw5_grid_search}"
PATHWAY_FILE="${PATHWAY_FILE:-/raid/yangpeng_lab/c12212609/PRESAGE/data/topic_embed/all_gw.txt}"

mkdir -p "$OUTPUT_ROOT"
cd "$NOTEBOOK_DIR"

COMMON_ARGS=(
  --dataset "$DATASET"
  --seed "$SEED_NAME"
  --pathway_files "$PATHWAY_FILE"
  --output_root "$OUTPUT_ROOT"
  --bootstrap_iters "$BOOTSTRAP_ITERS"
  --enhance_added_embeddings
  --optimizer AdamW
  --scheduler cosine
  --warmup_epochs 5
  --min_lr_ratio 0.05
  --gradient_clip_val 0.5
  --batch_size 256
  --weight_decay 1e-6
  --max_epochs 10000
  --precision 32
  --pathway_weight_type gat
  --pool_nlayers 2
  --softmax_temperature 0.10
  --gat_weight 0.85
  --item_hidden_size 1024
  --item_nlayers 0
  --pathway_item_nlayers 3
  --item_dropout 0.10
  --pathway_dropout 0.08
  --source_dropout 0.08
  --pathway_layer_norm
)

run_idx=0
for lr in 1.0e-3 1.35e-3 1.8e-3; do
  for n_nmf in 128 256; do
    for pathway_hidden in 256 384; do
      for patience in 10 30; do
        ((run_idx += 1))
        tag="gw5_gs_${run_idx}"
        echo "[gw5_gs ${run_idx}] lr=${lr} n_nmf=${n_nmf} pathway_hidden=${pathway_hidden} patience=${patience}"
        CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" run_presage_experiment.py \
          "${COMMON_ARGS[@]}" \
          --tag "$tag" \
          --patience "$patience" \
          --lr "$lr" \
          --n_nmf_embedding "$n_nmf" \
          --pathway_item_hidden_size "$pathway_hidden"
      done
    done
  done
done

"$PYTHON_BIN" "$SCRIPT_DIR/summarize_grid_search.py" "$OUTPUT_ROOT" \
  --field lr=model.lr \
  --field n_nmf_embedding=model.n_nmf_embedding \
  --field pathway_item_hidden_size=model.pathway_item_hidden_size \
  --field patience=cli_args.patience \
  --field batch_size=data.batch_size \
  --field item_hidden_size=model.item_hidden_size \
  --field pathway_item_nlayers=model.pathway_item_nlayers
