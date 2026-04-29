#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
NOTEBOOK_DIR="$REPO_ROOT/notebooks"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-12}"
DATASET="nadig_hepg2"
SEED_NAME="${SEED_NAME:-seed_0}"
BOOTSTRAP_ITERS="${BOOTSTRAP_ITERS:-1000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiment_runs/hep1_grid_search_refined_v3}"
PATHWAY_FILE="${PATHWAY_FILE:-/raid/yangpeng_lab/c12212609/PRESAGE/data/topic_embed/all_ess_g2.txt}"
TAIL_SOURCE_COUNT="${TAIL_SOURCE_COUNT:-1}"

mkdir -p "$OUTPUT_ROOT"
cd "$NOTEBOOK_DIR"

# Assumption: the added embedding is appended as the last source in all_ess_g2.txt.
# We keep the HepG2-old backbone and baseline-like optimization, but let the model
# start by downweighting the last source and learn whether it should be used.
COMMON_ARGS=(
  --dataset "$DATASET"
  --seed "$SEED_NAME"
  --pathway_files "$PATHWAY_FILE"
  --use_training_gex_embeddings
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
  --item_hidden_size 512
  --item_nlayers 0
  --pathway_item_hidden_size 128
  --pathway_item_nlayers 2
  --n_nmf_embedding 128
  --eval_val_metrics
  --monitor_metric val_balanced_top20
  --monitor_mode max
)

run_idx=0
for lr in 8.2e-4 9.0e-4 1.0e-3; do
  for temp in 0.12 0.15 0.18 0.21; do
    for gat_weight in 0.85 0.90; do
      for tail_scale in 0.25 0.50 0.75; do
        for patience in 10 20; do
          ((run_idx += 1))
          tag="hep1_ref_v3_${run_idx}"
          echo "[hep1_ref_v3 ${run_idx}] lr=${lr} temp=${temp} gat_weight=${gat_weight} tail_scale=${tail_scale} patience=${patience}"
          CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" run_presage_experiment.py \
            "${COMMON_ARGS[@]}" \
            --tag "$tag" \
            --lr "$lr" \
            --softmax_temperature "$temp" \
            --gat_weight "$gat_weight" \
            --tail_source_scale "$tail_scale" \
            --patience "$patience"
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
