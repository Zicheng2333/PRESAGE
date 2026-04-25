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
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/experiment_runs/hep5_grid_search_refined}"
PATHWAY_FILE="${PATHWAY_FILE:-/raid/yangpeng_lab/c12212609/PRESAGE/data/topic_embed/all_ess_g2.txt}"
PATIENCE="${PATIENCE:-15}"

mkdir -p "$OUTPUT_ROOT"
cd "$NOTEBOOK_DIR"

COMMON_ARGS=(
  --dataset "$DATASET"
  --seed "$SEED_NAME"
  --pathway_files "$PATHWAY_FILE"
  --output_root "$OUTPUT_ROOT"
  --bootstrap_iters "$BOOTSTRAP_ITERS"
  --enhance_added_embeddings
  --batch_size 16
  --max_epochs 10000
  --precision 32
  --pathway_weight_type gat
  --pool_nlayers 2
  --gat_weight 0.85
  --item_hidden_size 512
  --item_nlayers 0
  --pathway_item_nlayers 2
  --softmax_temperature 0.10
  --patience "$PATIENCE"
  --eval_val_metrics
  --monitor_metric val_avg_cossim_top20_unionde
  --monitor_mode max
)

run_idx=0
for backbone in new old; do
  if [[ "$backbone" == "old" ]]; then
    BACKBONE_ARGS=(--use_old)
  else
    BACKBONE_ARGS=()
  fi

  for profile in legacy_like cosine_light; do
    case "$profile" in
      legacy_like)
        PROFILE_ARGS=(
          --optimizer Adam
          --scheduler none
          --weight_decay 0.0
          --gradient_clip_val 0.1
          --item_dropout 0.0
          --pathway_dropout 0.0
          --source_dropout 0.0
          --no_pathway_layer_norm
        )
        ;;
      cosine_light)
        PROFILE_ARGS=(
          --optimizer AdamW
          --scheduler cosine
          --warmup_epochs 3
          --min_lr_ratio 0.10
          --weight_decay 1e-6
          --gradient_clip_val 0.3
          --item_dropout 0.05
          --pathway_dropout 0.05
          --source_dropout 0.0
          --no_pathway_layer_norm
        )
        ;;
      *)
        echo "Unknown profile: $profile" >&2
        exit 1
        ;;
    esac

    for lr in 4.5e-4 5.5e-4 7.0e-4; do
      for n_nmf in 32 64; do
        for pathway_hidden in 96 128; do
          ((run_idx += 1))
          tag="hep5_ref_${backbone}_${profile}_${run_idx}"
          echo "[hep5_ref ${run_idx}] backbone=${backbone} profile=${profile} lr=${lr} n_nmf=${n_nmf} pathway_hidden=${pathway_hidden}"
          CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" run_presage_experiment.py \
            "${COMMON_ARGS[@]}" \
            "${BACKBONE_ARGS[@]}" \
            "${PROFILE_ARGS[@]}" \
            --tag "$tag" \
            --lr "$lr" \
            --n_nmf_embedding "$n_nmf" \
            --pathway_item_hidden_size "$pathway_hidden"
        done
      done
    done
  done
done

"$PYTHON_BIN" "$SCRIPT_DIR/summarize_grid_search.py" "$OUTPUT_ROOT" \
  --sort-by bootstrap_cossim_unionde_20 \
  --descending \
  --field use_old=cli_args.use_old \
  --field lr=model.lr \
  --field n_nmf_embedding=model.n_nmf_embedding \
  --field pathway_item_hidden_size=model.pathway_item_hidden_size \
  --field weight_decay=model.weight_decay \
  --field optimizer=model.optimizer \
  --field scheduler=model.scheduler \
  --field item_dropout=model.item_dropout \
  --field pathway_dropout=model.pathway_dropout \
  --field source_dropout=model.source_dropout \
  --field pathway_layer_norm=model.pathway_layer_norm \
  --field patience=cli_args.patience \
  --field batch_size=data.batch_size
