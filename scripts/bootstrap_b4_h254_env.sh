#!/usr/bin/env bash
set -Eeuo pipefail

die() {
  printf 'B4 H254 environment bootstrap error: %s\n' "$*" >&2
  exit 1
}

GPFS_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao"
CACHE_ROOT="${GPFS_ROOT}/data-cache-migration/h-eailabagent-20260819/fastwam_env_cache"
BASE_PYTHON="${CACHE_ROOT}/bootstrap-py310/bin/python3.10"
TORCH_WHEEL="${CACHE_ROOT}/direct_test/torch-2.7.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl"
TARGET="${GPFS_ROOT}/envs/fastwam-b4-h254-py310-20260820"
PYPI_INDEX="http://mirrors.i.h.pjlab.org.cn/repository/pypi-proxy/simple/"
PYPI_EXTRA_INDEX="http://pypi.i.h.pjlab.org.cn/brain/dev/+simple"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"

[[ "$TARGET" == "${GPFS_ROOT}/envs/"* ]] || die "target is outside the approved environment root"
[[ ! -e "$TARGET" && ! -L "$TARGET" ]] || die "refusing to overwrite target: ${TARGET}"
[[ -x "$BASE_PYTHON" && ! -L "$BASE_PYTHON" ]] || die "base Python is not a regular executable"
[[ -f "$TORCH_WHEEL" && ! -L "$TORCH_WHEEL" ]] || die "exact Torch wheel is missing"
[[ "$(stat -c %s "$TORCH_WHEEL")" -eq 1039365846 ]] || die "exact Torch wheel byte size changed"

mkdir -p "$(dirname "$TARGET")"
BUILD_DIR="$(mktemp -d "${TARGET}.build.XXXXXX")"
mkdir -p "${CACHE_ROOT}/tmp"
B4_PIP_TMP_DIR="$(mktemp -d "${CACHE_ROOT}/tmp/b4-h254-pip.XXXXXX")"
cleanup() {
  local rc=$?
  if [[ -n "${B4_PIP_TMP_DIR:-}" && "$B4_PIP_TMP_DIR" == "${CACHE_ROOT}/tmp/b4-h254-pip."* ]]; then
    rm -rf -- "$B4_PIP_TMP_DIR"
  fi
  if [[ $rc -ne 0 && -n "${BUILD_DIR:-}" && "$BUILD_DIR" == "${TARGET}.build."* ]]; then
    rm -rf -- "$BUILD_DIR"
  fi
  exit "$rc"
}
trap cleanup EXIT

"$BASE_PYTHON" -m venv --copies "$BUILD_DIR"
ENV_PYTHON="${BUILD_DIR}/bin/python3.10"
[[ -x "$ENV_PYTHON" ]] || die "venv Python was not created"

PIP_ARGS=(
  --disable-pip-version-check
  --cache-dir "${CACHE_ROOT}/pip"
  --index-url "$PYPI_INDEX"
  --extra-index-url "$PYPI_EXTRA_INDEX"
  --extra-index-url "$PYTORCH_INDEX"
  --trusted-host mirrors.i.h.pjlab.org.cn
  --trusted-host pypi.i.h.pjlab.org.cn
)

TMPDIR="$B4_PIP_TMP_DIR" "$ENV_PYTHON" -m pip install "${PIP_ARGS[@]}" "$TORCH_WHEEL"
TMPDIR="$B4_PIP_TMP_DIR" "$ENV_PYTHON" -m pip install "${PIP_ARGS[@]}" \
  accelerate==1.12.0 \
  av==16.0.1 \
  boto3==1.35.99 \
  datasets==3.6.0 \
  deepspeed==0.18.5 \
  einops==0.8.1 \
  gitpython==3.1.45 \
  h5py==3.14.0 \
  huggingface-hub==0.29.2 \
  hydra-core==1.3.2 \
  imageio==2.37.0 \
  imageio-ffmpeg==0.6.0 \
  jsonlines==4.0.0 \
  modelscope==1.34.0 \
  numpy==1.26.4 \
  omegaconf==2.3.0 \
  packaging==25.0 \
  pandas==2.2.3 \
  pillow==12.0.0 \
  pyarrow==23.0.0 \
  regex==2025.11.3 \
  rich==14.2.0 \
  safetensors==0.5.3 \
  termcolor==2.5.0 \
  torch==2.7.1+cu128 \
  torchcodec==0.5+cu128 \
  torchvision==0.22.1+cu128 \
  tqdm==4.66.5 \
  transformers==4.49.0 \
  typing-extensions==4.15.0 \
  wandb==0.23.1

"$ENV_PYTHON" -m pip check
"$ENV_PYTHON" - <<'PY'
import importlib.metadata as md
import sys

expected = {
    "accelerate": "1.12.0",
    "av": "16.0.1",
    "boto3": "1.35.99",
    "datasets": "3.6.0",
    "deepspeed": "0.18.5",
    "einops": "0.8.1",
    "gitpython": "3.1.45",
    "h5py": "3.14.0",
    "huggingface-hub": "0.29.2",
    "hydra-core": "1.3.2",
    "imageio": "2.37.0",
    "imageio-ffmpeg": "0.6.0",
    "jsonlines": "4.0.0",
    "modelscope": "1.34.0",
    "numpy": "1.26.4",
    "omegaconf": "2.3.0",
    "packaging": "25.0",
    "pandas": "2.2.3",
    "pillow": "12.0.0",
    "pyarrow": "23.0.0",
    "regex": "2025.11.3",
    "rich": "14.2.0",
    "safetensors": "0.5.3",
    "termcolor": "2.5.0",
    "torch": "2.7.1+cu128",
    "torchcodec": "0.5+cu128",
    "torchvision": "0.22.1+cu128",
    "tqdm": "4.66.5",
    "transformers": "4.49.0",
    "typing-extensions": "4.15.0",
    "wandb": "0.23.1",
}
if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"expected Python 3.10, got {sys.version.split()[0]}")
bad = {
    name: (md.version(name), want)
    for name, want in expected.items()
    if md.version(name) != want
}
if bad:
    raise SystemExit(f"dependency contract mismatch: {bad}")

import torch
import torchvision

print(
    "B4 H254 environment metadata PASS "
    f"python={sys.version.split()[0]} torch={torch.__version__} "
    f"torchvision={torchvision.__version__}"
)
PY

[[ ! -e "$TARGET" && ! -L "$TARGET" ]] || die "target appeared during build"
mv -- "$BUILD_DIR" "$TARGET"
BUILD_DIR=""
printf 'B4 H254 environment published: %s\n' "$TARGET"
