#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <train|val> <panel-index:0..7> <gpu:0..3> <output-root>" >&2
  exit 2
fi

split=$1
panel_index=$2
gpu=$3
output_root=$4

case "$split" in
  train|val) ;;
  *) echo "split must be train or val" >&2; exit 2 ;;
esac
if [[ ! "$panel_index" =~ ^[0-7]$ ]]; then
  echo "panel-index must be an integer from 0 through 7" >&2
  exit 2
fi
if [[ ! "$gpu" =~ ^[0-3]$ ]]; then
  echo "gpu must be an integer from 0 through 3" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source_root=$(cd -- "$script_dir/../../.." && pwd -P)
panel="$script_dir/panels/${split}8.json"
python_bin=/opt/venvs/gaudp-robofactory-py310/bin/python
evaluator="$source_root/experiments/robofactory/diagnose_place_food_fixed.py"
dataset_root=/mnt/workspace/datasets/robofactory_multi_robot
robofactory_root=/mnt/workspace/fastwam_eval_runtime/RoboFactory-challenge-2d34fb3
gaussian_cache=/oss-chengjuntao/fastwam-gaudp/robofactory_multi_robot/v2/noposplat-c944b498-4a35bc8c/builds/fastwam-8a035024af96-s42-20260801T230944Z/compact-s42-13x28x40-fp16-meanalpha-v2
checkpoint=/oss-chengjuntao/artifacts/fastwam-action-n234-formal-r5-20260812/fastwam-act-n2-placefood-1k-s42-r5-20260812/checkpoints/weights/step_001000.pt
stats=/oss-chengjuntao/artifacts/fastwam-nohash-inputs-20260809/fastwam_multi_robot_n234_train_s42_stats_cpfs_nohash_v1.json
context_file=/oss-chengjuntao/cpfs-user-chengjuntao/datasets/robofactory_multi_robot/text_embeds_cache_n234_named_20260811/PlaceFood-rf.t5_len128.wan22ti2v5b.pt
model_cache_root=/mnt/workspace/checkpoints/FastWAM/model-cache
policy_lightning_repo=/mnt/workspace/Policy-Lightning
noposplat_checkpoint=/mnt/workspace/checkpoints/noposplat/664ba9156f10a6203f0a0fad2f02c069c6894f4f/mixRe10kDl3dv_512x512.ckpt
episode_dir="$output_root/$split/episode-$(printf '%02d' "$panel_index")"
policy_seed=$((10000 + panel_index))

for required in \
  "$panel" \
  "$python_bin" \
  "$evaluator" \
  "$dataset_root" \
  "$robofactory_root" \
  "$gaussian_cache" \
  "$checkpoint" \
  "$stats" \
  "$context_file" \
  "$model_cache_root" \
  "$policy_lightning_repo" \
  "$noposplat_checkpoint"; do
  if [[ ! -e "$required" ]]; then
    echo "required input is absent: $required" >&2
    exit 3
  fi
done
if [[ -e "$episode_dir" || -L "$episode_dir" ]]; then
  echo "refusing to overwrite existing episode output: $episode_dir" >&2
  exit 4
fi
mkdir -p -- "$output_root/$split"

export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES="$gpu"
export PYTHONPATH="$source_root/src:$source_root/experiments/robofactory${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" -B "$evaluator" \
  --mode rollout \
  --formal-contract \
  --task PlaceFood-rf \
  --panel "$panel" \
  --dataset-root "$dataset_root" \
  --robofactory-root "$robofactory_root" \
  --gaussian-cache "$gaussian_cache" \
  --output-dir "$episode_dir" \
  --episode-start "$panel_index" \
  --policy-seed "$policy_seed" \
  --max-steps 300 \
  --initial-state raw \
  --exec-horizon 5 \
  --checkpoint "$checkpoint" \
  --integrity-mode metadata_no_hash \
  --stats "$stats" \
  --context-file "$context_file" \
  --model-cache-root "$model_cache_root" \
  --policy-lightning-repo "$policy_lightning_repo" \
  --noposplat-checkpoint "$noposplat_checkpoint" \
  --device cuda:0 \
  --teacher-device cuda:0 \
  --action-horizon 32 \
  --num-inference-steps 20
