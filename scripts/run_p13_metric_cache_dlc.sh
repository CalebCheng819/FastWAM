#!/usr/bin/env bash
set -euo pipefail

die() {
  printf 'P13 metric cache error: %s\n' "$*" >&2
  exit 1
}

RUN_ID="${RUN_ID:?RUN_ID is required}"
CODE_REVISION="${FASTWAM_P13_CODE_REVISION:?FASTWAM_P13_CODE_REVISION is required}"
SOURCE_BUNDLE="${FASTWAM_P13_SOURCE_BUNDLE:?FASTWAM_P13_SOURCE_BUNDLE is required}"
RUNTIME_ARCHIVE="${FASTWAM_P13_RUNTIME_ARCHIVE:?FASTWAM_P13_RUNTIME_ARCHIVE is required}"
OUTPUT_ROOT="${FASTWAM_P13_CACHE_OUTPUT_ROOT:?FASTWAM_P13_CACHE_OUTPUT_ROOT is required}"
DATASET_ROOT="${FASTWAM_P13_DATASET_ROOT:?FASTWAM_P13_DATASET_ROOT is required}"
ROBOFACTORY_ROOT="${FASTWAM_P13_ROBOFACTORY_ROOT:?FASTWAM_P13_ROBOFACTORY_ROOT is required}"
PYTHON_BIN="${FASTWAM_P13_PYTHON:?FASTWAM_P13_PYTHON is required}"
DRIVER_ROOT="${FASTWAM_P13_DRIVER_ROOT:?FASTWAM_P13_DRIVER_ROOT is required}"
LOCAL_ROOT="/tmp/fastwam-p13-metric-cache/${RUN_ID}"
LOCAL_REPO="${LOCAL_ROOT}/source"
PARTIAL_REPO="${LOCAL_REPO}.partial.${BASHPID}"
RUNTIME_ROOT="${LOCAL_ROOT}/runtime"
LOCAL_CACHE="${LOCAL_ROOT}/cache-output"

[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || die "unsafe RUN_ID=${RUN_ID}"
[[ "${CODE_REVISION}" =~ ^[0-9a-f]{40}$ ]] || die "invalid Git revision"
[[ -x "${PYTHON_BIN}" ]] || die "Python is missing: ${PYTHON_BIN}"
[[ -f "${SOURCE_BUNDLE}" && ! -L "${SOURCE_BUNDLE}" ]] || die "source bundle is missing"
[[ -f "${RUNTIME_ARCHIVE}" && ! -L "${RUNTIME_ARCHIVE}" ]] || die "runtime archive is missing"
[[ -d "${DATASET_ROOT}" && ! -L "${DATASET_ROOT}" ]] || die "dataset root is missing"
[[ -d "${ROBOFACTORY_ROOT}" && ! -L "${ROBOFACTORY_ROOT}" ]] || die "RoboFactory root is missing"
[[ -f "${DRIVER_ROOT}/nvidia_icd.json" ]] || die "Vulkan ICD is missing"
[[ -f "${DRIVER_ROOT}/10_nvidia.json" ]] || die "EGL vendor file is missing"
[[ ! -e "${LOCAL_ROOT}" && ! -L "${LOCAL_ROOT}" ]] || die "node-local run root already exists: ${LOCAL_ROOT}"

mkdir -p -- "${RUNTIME_ROOT}"
tar -xzf "${RUNTIME_ARCHIVE}" -C "${RUNTIME_ROOT}"
[[ -d "${RUNTIME_ROOT}/site-packages" ]] || die "runtime archive layout is invalid"

git clone --quiet --no-checkout -- "${SOURCE_BUNDLE}" "${PARTIAL_REPO}"
git -C "${PARTIAL_REPO}" checkout --quiet --detach "${CODE_REVISION}"
[[ "$(git -C "${PARTIAL_REPO}" rev-parse HEAD)" == "${CODE_REVISION}" ]] || die "source revision mismatch"
[[ -z "$(git -C "${PARTIAL_REPO}" status --porcelain --untracked-files=all)" ]] || die "restored source is dirty"
mv -T -- "${PARTIAL_REPO}" "${LOCAL_REPO}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${RUNTIME_ROOT}/site-packages:${LOCAL_REPO}/src:${ROBOFACTORY_ROOT%/robofactory}:${ROBOFACTORY_ROOT}"
export LD_LIBRARY_PATH="${DRIVER_ROOT}/driver-lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="${DRIVER_ROOT}/nvidia_icd.json"
export __EGL_VENDOR_LIBRARY_FILENAMES="${DRIVER_ROOT}/10_nvidia.json"
export DISPLAY=

nvidia-smi -L
"${PYTHON_BIN}" - <<'PY'
import importlib
import json

import torch

modules = ["gymnasium", "h5py", "mani_skill", "robofactory", "sapien"]
for name in modules:
    importlib.import_module(name)
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(f"expected exactly one visible CUDA GPU, got {torch.cuda.device_count()}")
print(json.dumps({
    "cache_preflight": "PASS",
    "cuda_device": torch.cuda.get_device_name(0),
    "visible_gpus": torch.cuda.device_count(),
}, sort_keys=True))
PY

"${PYTHON_BIN}" "${LOCAL_REPO}/scripts/build_robofactory_metric_geometry_cache.py" \
  --dataset-root "${DATASET_ROOT}" \
  --robofactory-root "${ROBOFACTORY_ROOT}" \
  --output-root "${LOCAL_CACHE}" \
  --task-name PlaceFood-rf \
  --required-agent-count 2 \
  --num-frames 33 \
  --train-window-stride 16 \
  --val-window-stride 32 \
  --val-set-proportion 0.1 \
  --split-seed 42 \
  --output-size 60 80 \
  --progress-every 25

for required in frames.f16 manifest.json COMPLETE stat-cmp.allowlist; do
  [[ -f "${LOCAL_CACHE}/${required}" && ! -L "${LOCAL_CACHE}/${required}" ]] || die "cache artifact missing: ${required}"
done

"${PYTHON_BIN}" "${LOCAL_REPO}/scripts/publish_metric_geometry_cache.py" \
  --source-root "${LOCAL_CACHE}" \
  --target-root "${OUTPUT_ROOT}" \
  --run-id "${RUN_ID}" \
  --timeout-seconds 1800 \
  --poll-seconds 15

printf 'P13_METRIC_CACHE_COMPLETE run_id=%s revision=%s output=%s\n' \
  "${RUN_ID}" "${CODE_REVISION}" "${OUTPUT_ROOT}"
