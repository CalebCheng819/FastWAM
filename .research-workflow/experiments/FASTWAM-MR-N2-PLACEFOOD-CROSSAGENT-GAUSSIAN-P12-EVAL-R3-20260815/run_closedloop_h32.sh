#!/usr/bin/env bash
set -euo pipefail

step=${P12_EVAL_STEP:?set P12_EVAL_STEP to 000500 or 001000}
case "$step" in
  000500|001000) ;;
  *) echo "P12_EVAL_STEP must be 000500 or 001000" >&2; exit 2 ;;
esac

experiment_id=FASTWAM-MR-N2-PLACEFOOD-CROSSAGENT-GAUSSIAN-P12-STEP${step}-OFFICIAL-TOPP-H32-VAL8-R3-20260815
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_root=$(git -C "$script_dir" rev-parse --show-toplevel)
model_root=${P12_MODEL_ROOT:-/mnt/workspace/experiments/FastWAM-p12-render-1181a37-20260814}
train_root=${P12_TRAIN_ROOT:-/oss-chengjuntao/artifacts/fastwam-placefood-crossagent-gaussian-p12-s42-8g-r2-20260814}
offline_root=${P12_TF_OUTPUT_ROOT:-/oss-chengjuntao/artifacts/fastwam-placefood-crossagent-gaussian-p12-paired-tf-20260815-r3}
output=${P12_CLOSEDLOOP_OUTPUT_ROOT:-/oss-chengjuntao/artifacts/fastwam-p12-step${step}-official-topp-h32-val8-20260815-r3}
python=${P12_EVAL_PYTHON:-/mnt/workspace/venvs/fastwam-py310/bin/python}
python_extra=${P12_PYTHON_EXTRA:-/mnt/workspace/venvs/fastwam-gau0-eval-r7-py310-extra-20260813}
panel=${P12_EVAL_PANEL:-/mnt/workspace/fastwam_eval_runtime/panels/robofactory_n234_s42_val8_v1.json}
dataset=${P12_DATASET_ROOT:-/oss-chengjuntao/cpfs-user-chengjuntao/datasets/robofactory_multi_robot}
robofactory=${P12_ROBOFACTORY_ROOT:-/mnt/workspace/fastwam_eval_runtime/RoboFactory-challenge-2d34fb3}
stats=${P12_STATS:-/oss-chengjuntao/artifacts/fastwam-nohash-inputs-20260809/fastwam_multi_robot_n234_train_s42_stats_cpfs_nohash_v1.json}
context=${P12_CONTEXT_FILE:-/oss-chengjuntao/cpfs-user-chengjuntao/datasets/robofactory_multi_robot/text_embeds_cache_n234_named_20260811/PlaceFood-rf.t5_len128.wan22ti2v5b.pt}
model_cache=${P12_MODEL_CACHE_ROOT:-/oss-chengjuntao/cpfs-user-chengjuntao/checkpoints/FastWAM/model-cache}
gaussian_cache=${P12_GAUSSIAN_CACHE:-/oss-chengjuntao/fastwam-gaudp/robofactory_multi_robot/v2/noposplat-c944b498-4a35bc8c/builds/fastwam-8a035024af96-s42-20260801T230944Z/compact-s42-13x28x40-fp16-meanalpha-v2}
policy_lightning=${P12_POLICY_LIGHTNING_ROOT:-/mnt/workspace/Policy-Lightning}
noposplat_checkpoint=${P12_NOPOSPLAT_CHECKPOINT:-/mnt/workspace/checkpoints/noposplat/664ba9156f10a6203f0a0fad2f02c069c6894f4f/mixRe10kDl3dv_512x512.ckpt}
graphics=${P12_GRAPHICS_ROOT:-/cpfs/user/chengjuntao/fastwam-deploy/nvidia-graphics-570.153.02}
checkpoint=$train_root/checkpoints/weights/step_${step}.pt
train_commit=1181a375c880a4a51df2ae78d533e16dde757465
eval_commit=$(git -C "$source_root" rev-parse HEAD)
read -r -a gpus <<< "${P12_EVAL_GPUS:-0 1 2 3}"

mountpoint -q /oss-chengjuntao
test -w /oss-chengjuntao
test ! -e "$output"
test -x "$python"
test -d "$python_extra"
test "$(git -C "$model_root" rev-parse HEAD)" = "$train_commit"
test -z "$(git -C "$model_root" status --short)"
test -z "$(git -C "$source_root" status --short)"
test "$(git -C "$policy_lightning" rev-parse HEAD)" = c944b4989a89c99c69d2572ea870f6a04680f5e7
test -z "$(git -C "$policy_lightning" status --short)"
test -s "$checkpoint"
test -s "$checkpoint.COMPLETE"
test -s "$offline_root/comparison.json"
test "$(tr -d '[:space:]' <"$offline_root/terminal.status")" = SUCCEEDED
for path in \
  "$panel" "$dataset" "$robofactory" "$stats" "$context" "$model_cache" \
  "$gaussian_cache" "$gaussian_cache/COMPLETE" "$noposplat_checkpoint" \
  "$graphics/driver-lib" "$graphics/nvidia_icd.json" "$graphics/10_nvidia.json"; do
  test -e "$path"
done

export PYTHONPATH="$python_extra:$model_root/src:$source_root/experiments/robofactory:$source_root:$policy_lightning:$robofactory:${PYTHONPATH:-}"
exec "$python" -B "$source_root/experiments/robofactory/run_fixed_policy_closedloop_panel.py" formal \
  --experiment-id "$experiment_id" \
  --candidate "p12-step${step}" \
  --source-root "$source_root" \
  --output-root "$output" \
  --python "$python" \
  --panel "$panel" \
  --dataset-root "$dataset" \
  --robofactory-root "$robofactory" \
  --gaussian-cache "$gaussian_cache" \
  --checkpoint "$checkpoint" \
  --training-code-commit "$train_commit" \
  --evaluation-code-commit "$eval_commit" \
  --model-project-root "$model_root" \
  --action-architecture cross_agent_gaussian_v4 \
  --gaussian-source noposplat \
  --stats "$stats" \
  --context-file "$context" \
  --model-cache-root "$model_cache" \
  --policy-lightning-repo "$policy_lightning" \
  --noposplat-checkpoint "$noposplat_checkpoint" \
  --nvidia-driver-lib-dir "$graphics/driver-lib" \
  --nvidia-vulkan-icd "$graphics/nvidia_icd.json" \
  --nvidia-egl-vendor-json "$graphics/10_nvidia.json" \
  --exec-horizon 32 \
  --control-adapter official_topp \
  --topp-step 0.05 \
  --max-policy-queries 60 \
  --max-simulator-steps 30000 \
  --gpus "${gpus[@]}"
