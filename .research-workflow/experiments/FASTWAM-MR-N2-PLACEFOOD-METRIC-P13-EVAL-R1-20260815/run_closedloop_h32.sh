#!/usr/bin/env bash
set -euo pipefail

experiment_id=FASTWAM-MR-N2-PLACEFOOD-METRIC-P13-OFFICIAL-TOPP-H32-VAL8-R1-20260815
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_root=$(git -C "$script_dir" rev-parse --show-toplevel)
model_root=${P13_MODEL_ROOT:-/mnt/workspace/experiments/FastWAM-p13-e5f20bb-20260815}
metric_cache=${P13_METRIC_CACHE_ROOT:?set P13_METRIC_CACHE_ROOT to the completed cache used by P13 training}
train_root=${P13_TRAIN_ROOT:-/oss-chengjuntao/artifacts/fastwam-placefood-metric-gaussian-p13-s42-8g-r1-20260815}
offline_root=${P13_TF_OUTPUT_ROOT:-/oss-chengjuntao/artifacts/fastwam-placefood-metric-gaussian-p13-paired-tf-r1-20260815}
output=${P13_CLOSEDLOOP_OUTPUT_ROOT:-/oss-chengjuntao/artifacts/fastwam-placefood-metric-gaussian-p13-official-topp-h32-val8-r1-20260815}
python=${P13_EVAL_PYTHON:-/opt/venvs/gaudp-robofactory-py310/bin/python}
panel=${P13_EVAL_PANEL:-/mnt/workspace/fastwam_eval_runtime/panels/robofactory_n234_s42_val8_v1.json}
dataset=${P13_DATASET_ROOT:-/oss-chengjuntao/cpfs-user-chengjuntao/datasets/robofactory_multi_robot}
robofactory=${P13_ROBOFACTORY_ROOT:-/mnt/workspace/fastwam_eval_runtime/RoboFactory-challenge-2d34fb3}
stats=${P13_STATS:-/oss-chengjuntao/artifacts/fastwam-nohash-inputs-20260809/fastwam_multi_robot_n234_train_s42_stats_cpfs_nohash_v1.json}
context=${P13_CONTEXT_FILE:-/oss-chengjuntao/cpfs-user-chengjuntao/datasets/robofactory_multi_robot/text_embeds_cache_n234_named_20260811/PlaceFood-rf.t5_len128.wan22ti2v5b.pt}
model_cache=${P13_MODEL_CACHE_ROOT:-/oss-chengjuntao/cpfs-user-chengjuntao/checkpoints/FastWAM/model-cache}
graphics=${P13_GRAPHICS_ROOT:-/cpfs/user/chengjuntao/fastwam-deploy/nvidia-graphics-570.153.02}
checkpoint=$train_root/checkpoints/weights/step_001000.pt
train_commit=e5f20bbf91477b82990e5c571d54305c639705c6
eval_commit=$(git -C "$source_root" rev-parse HEAD)
read -r -a gpus <<< "${P13_EVAL_GPUS:-0 1 2 3}"

mountpoint -q /oss-chengjuntao
test -w /oss-chengjuntao
test ! -e "$output"
test -x "$python"
test "$(git -C "$model_root" rev-parse HEAD)" = "$train_commit"
test -z "$(git -C "$model_root" status --short)"
test -z "$(git -C "$source_root" status --short)"
test -s "$checkpoint"
test -f "$checkpoint.COMPLETE"
test -s "$offline_root/comparison.json"
test "$(tr -d '[:space:]' <"$offline_root/terminal.status")" = SUCCEEDED
test "$(tr -d '[:space:]' <"$metric_cache/COMPLETE")" = complete

exec "$python" "$source_root/experiments/robofactory/run_fixed_policy_closedloop_panel.py" formal \
  --experiment-id "$experiment_id" \
  --candidate p13-step001000 \
  --source-root "$source_root" \
  --output-root "$output" \
  --python "$python" \
  --panel "$panel" \
  --dataset-root "$dataset" \
  --robofactory-root "$robofactory" \
  --gaussian-cache "$metric_cache" \
  --checkpoint "$checkpoint" \
  --training-code-commit "$train_commit" \
  --evaluation-code-commit "$eval_commit" \
  --model-project-root "$model_root" \
  --action-architecture metric_gaussian_v5 \
  --gaussian-source metric_geometry \
  --stats "$stats" \
  --context-file "$context" \
  --model-cache-root "$model_cache" \
  --nvidia-driver-lib-dir "$graphics/driver-lib" \
  --nvidia-vulkan-icd "$graphics/nvidia_icd.json" \
  --nvidia-egl-vendor-json "$graphics/10_nvidia.json" \
  --exec-horizon 32 \
  --control-adapter official_topp \
  --topp-step 0.05 \
  --max-policy-queries 60 \
  --max-simulator-steps 30000 \
  --gpus "${gpus[@]}"
