#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 <train|val> <panel-index:0..7> <gpu:0..7> <output-root> <evaluation-code-commit>" >&2
  exit 2
fi

split=$1
panel_index=$2
gpu=$3
output_root=$4
evaluation_code_commit=$5

case "$split" in
  train|val) ;;
  *) echo "split must be train or val" >&2; exit 2 ;;
esac
if [[ ! "$panel_index" =~ ^[0-7]$ ]]; then
  echo "panel-index must be an integer from 0 through 7" >&2
  exit 2
fi
if [[ ! "$gpu" =~ ^[0-7]$ ]]; then
  echo "gpu must be an integer from 0 through 7" >&2
  exit 2
fi
if [[ "$output_root" != /oss-chengjuntao/* ]]; then
  echo "output-root must be under /oss-chengjuntao" >&2
  exit 2
fi
if ! mountpoint -q /oss-chengjuntao || [[ ! -w /oss-chengjuntao ]]; then
  echo "/oss-chengjuntao is not a writable mount" >&2
  exit 3
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source_root=$(cd -- "$script_dir/../../.." && pwd -P)
shared_panel_dir="$source_root/.research-workflow/experiments/FASTWAM-MR-FT-ACT-N2-PLACEFOOD-TRAINLAYOUT-EVAL-R5-20260813/panels"
panel="$shared_panel_dir/${split}8.json"
python_bin=/opt/venvs/gaudp-robofactory-py310/bin/python
evaluator="$source_root/experiments/robofactory/diagnose_place_food_fixed.py"
dataset_root=/mnt/workspace/datasets/robofactory_multi_robot
robofactory_root=/mnt/workspace/fastwam_eval_runtime/RoboFactory-challenge-2d34fb3
graphics_root=${FASTWAM_NVIDIA_GRAPHICS_ROOT:-/cpfs/user/chengjuntao/fastwam-deploy/nvidia-graphics-570.153.02}
vulkan_icd="$graphics_root/nvidia_icd.json"
egl_vendor="$graphics_root/10_nvidia.json"
graphics_driver_lib="$graphics_root/driver-lib"
episode_dir="$output_root/$split/episode-$(printf '%02d' "$panel_index")"
policy_seed=$((10000 + panel_index))

for required in \
  "$panel" \
  "$python_bin" \
  "$evaluator" \
  "$dataset_root" \
  "$robofactory_root" \
  "$vulkan_icd" \
  "$egl_vendor" \
  "$graphics_driver_lib"; do
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
export VK_ICD_FILENAMES="$vulkan_icd"
export VK_DRIVER_FILES="$vulkan_icd"
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __EGL_VENDOR_LIBRARY_FILENAMES="$egl_vendor"
export LD_LIBRARY_PATH="$graphics_driver_lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$python_bin" -B "$evaluator" \
  --mode expert-replay \
  --formal-contract \
  --evaluation-code-commit "$evaluation_code_commit" \
  --task PlaceFood-rf \
  --panel "$panel" \
  --dataset-root "$dataset_root" \
  --robofactory-root "$robofactory_root" \
  --output-dir "$episode_dir" \
  --episode-start "$panel_index" \
  --policy-seed "$policy_seed" \
  --max-steps 300 \
  --initial-state raw
