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
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiment_runs/hep5_grid_search_refined_v3}"
PATHWAY_FILE="${PATHWAY_FILE:-/raid/yangpeng_lab/c12212609/PRESAGE/data/topic_embed/all_ess_g2.txt}"
TAIL_SOURCE_COUNT="${TAIL_SOURCE_COUNT:-1}"

mkdir -p "$OUTPUT_ROOT"
cd "$NOTEBOOK_DIR"

# Hep5 already beats the baseline on union-DE cosine but still lags on standard top20 metrics.
# We therefore monitor a balanced validation score and explicitly damp the added tail source.
COMMON_ARGS=(
  --dataset "$DATASET"
  --seed "$SEED_NAME"
  --pathway_files "$PATHWAY_FILE"
  --use_old
  --separate_embedding_channels
  --learn_source_scaling
  --tail_source_count "$TAIL_SOURCE_COUNT"
  --output_root "$OUTPUT_ROOT"
  --bootstrap_iters "$BOOTSTRAP_ITERS"
  --batch_size 16
  --max_epochs 10000
  --precision 32
  --optimizer Adam
  --scheduler none
  --weight_decay 1e-15
  --gradient_clip_val 0.1
  --batch_norm true
  --pathway_weight_type gat
  --pool_nlayers 2
  --gat_weight 0.85
  --item_hidden_size 512
  --item_nlayers 0
  --pathway_item_nlayers 2
  --eval_val_metrics
  --monitor_metric val_balanced_top20
  --monitor_mode max
)

run_idx=0
for lr in 5.5e-4 6.0e-4 6.5e-4; do
  for n_nmf in 32 48 64; do
    for pathway_hidden in 88 96; do
      for temp in 0.10 0.12; do
        for tail_scale in 0.25 0.50; do
          ((run_idx += 1))
          tag="hep5_ref_v3_${run_idx}"
          echo "[hep5_ref_v3 ${run_idx}] lr=${lr} n_nmf=${n_nmf} pathway_hidden=${pathway_hidden} temp=${temp} tail_scale=${tail_scale}"
          CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" run_presage_experiment.py \
            "${COMMON_ARGS[@]}" \
            --tag "$tag" \
            --lr "$lr" \
            --n_nmf_embedding "$n_nmf" \
            --pathway_item_hidden_size "$pathway_hidden" \
            --softmax_temperature "$temp" \
            --tail_source_scale "$tail_scale" \
            --patience 10
        done
      done
    done
  done
done

"$PYTHON_BIN" "$SCRIPT_DIR/summarize_grid_search.py" --root "$OUTPUT_ROOT" \
  --sort-by bootstrap_cossim_unionde_20 \
  --descending \
  --field use_old=cli_args.use_old \
  --field separate_embedding_channels=model.separate_embedding_channels \
  --field learn_source_scaling=model.learn_source_scaling \
  --field tail_source_count=model.tail_source_count \
  --field tail_source_scale=model.tail_source_scale \
  --field lr=model.lr \
  --field weight_decay=model.weight_decay \
  --field optimizer=model.optimizer \
  --field scheduler=model.scheduler \
  --field softmax_temperature=model.softmax_temperature \
  --field gat_weight=model.gat_weight \
  --field n_nmf_embedding=model.n_nmf_embedding \
  --field pathway_item_hidden_size=model.pathway_item_hidden_size \
  --field patience=cli_args.patience \
  --field batch_size=data.batch_size
