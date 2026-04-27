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
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiment_runs/hep1_grid_search_refined_v2}"
PATHWAY_FILE="${PATHWAY_FILE:-/raid/yangpeng_lab/c12212609/PRESAGE/data/topic_embed/all_ess_g2.txt}"

mkdir -p "$OUTPUT_ROOT"
cd "$NOTEBOOK_DIR"

# HepG2 results indicate the best region is very close to the old baseline:
# use_old + Adam + no scheduler + near-zero weight decay.
COMMON_ARGS=(
  --dataset "$DATASET"
  --seed "$SEED_NAME"
  --pathway_files "$PATHWAY_FILE"
  --use_training_gex_embeddings
  --use_old
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
  --monitor_metric val_avg_cossim_top20_unionde
  --monitor_mode max
)

run_idx=0
for lr in 8.2e-4 9.5e-4 1.10e-3; do
  for temp in 0.10 0.12 0.15 0.18; do
    for gat_weight in 0.80 0.85 0.90; do
      for patience in 10 20; do
        ((run_idx += 1))
        tag="hep1_ref_v2_${run_idx}"
        echo "[hep1_ref_v2 ${run_idx}] lr=${lr} temp=${temp} gat_weight=${gat_weight} patience=${patience}"
        CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" run_presage_experiment.py \
          "${COMMON_ARGS[@]}" \
          --tag "$tag" \
          --lr "$lr" \
          --softmax_temperature "$temp" \
          --gat_weight "$gat_weight" \
          --patience "$patience"
      done
    done
  done
done

"$PYTHON_BIN" "$SCRIPT_DIR/summarize_grid_search.py" --root "$OUTPUT_ROOT" \
  --sort-by bootstrap_cossim_unionde_20 \
  --descending \
  --field use_old=cli_args.use_old \
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
