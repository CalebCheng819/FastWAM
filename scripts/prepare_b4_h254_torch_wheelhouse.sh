#!/usr/bin/env bash
set -Eeuo pipefail

die() {
  printf 'B4 H254 Torch wheelhouse error: %s\n' "$*" >&2
  exit 1
}

GPFS_ROOT="/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/chengjuntao"
CACHE_ROOT="${GPFS_ROOT}/data-cache-migration/h-eailabagent-20260819/fastwam_env_cache"
BASE_PYTHON="${CACHE_ROOT}/bootstrap-py310/bin/python3.10"
TORCH_WHEEL="${CACHE_ROOT}/direct_test/torch-2.7.1+cu128-cp310-cp310-manylinux_2_28_x86_64.whl"
TARGET="${CACHE_ROOT}/wheelhouse/torch-2.7.1-cu128-cp310"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"

[[ "$TARGET" == "${CACHE_ROOT}/wheelhouse/"* ]] || die "target is outside the approved wheelhouse root"
[[ ! -e "$TARGET" && ! -L "$TARGET" ]] || die "refusing to overwrite target: ${TARGET}"
[[ -x "$BASE_PYTHON" && ! -L "$BASE_PYTHON" ]] || die "base Python is not a regular executable"
[[ -f "$TORCH_WHEEL" && ! -L "$TORCH_WHEEL" ]] || die "exact Torch wheel is missing"
[[ "$(stat -c %s "$TORCH_WHEEL")" -eq 1039365846 ]] || die "exact Torch wheel byte size changed"

mkdir -p "$(dirname "$TARGET")"
BUILD_DIR="$(mktemp -d "${TARGET}.build.XXXXXX")"
cleanup() {
  local rc=$?
  if [[ $rc -ne 0 && -n "${BUILD_DIR:-}" && "$BUILD_DIR" == "${TARGET}.build."* ]]; then
    rm -rf -- "$BUILD_DIR"
  fi
  exit "$rc"
}
trap cleanup EXIT

cp -- "$TORCH_WHEEL" "$BUILD_DIR/"

"$BASE_PYTHON" - "$CACHE_ROOT" "$BUILD_DIR" <<'PY'
from pathlib import Path
import shutil
import stat
import sys
import zipfile

cache_root = Path(sys.argv[1])
destination = Path(sys.argv[2])
expected = {
    "nvidia_cublas_cu12-12.8.3.14-py3-none-manylinux_2_27_x86_64.whl": 609620630,
    "nvidia_cuda_cupti_cu12-12.8.57-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl": 10237547,
    "nvidia_cuda_nvrtc_cu12-12.8.61-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl": 88024585,
    "nvidia_cuda_runtime_cu12-12.8.57-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl": 954762,
    "nvidia_cudnn_cu12-9.7.1.26-py3-none-manylinux_2_27_x86_64.whl": 726851421,
    "nvidia_cufft_cu12-11.3.3.41-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl": 193118795,
    "nvidia_cufile_cu12-1.13.0.11-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl": 1197801,
    "nvidia_curand_cu12-10.3.9.55-py3-none-manylinux_2_27_x86_64.whl": 63618038,
    "nvidia_cusolver_cu12-11.7.2.55-py3-none-manylinux_2_27_x86_64.whl": 260373342,
    "nvidia_cusparse_cu12-12.5.7.53-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl": 292064180,
    "nvidia_cusparselt_cu12-0.6.3-py3-none-manylinux2014_x86_64.whl": 156785796,
    "nvidia_nccl_cu12-2.26.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl": 201319755,
    "nvidia_nvjitlink_cu12-12.8.61-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl": 39243473,
    "nvidia_nvtx_cu12-12.8.55-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl": 89896,
}

for filename, expected_size in expected.items():
    candidates = []
    for path in cache_root.rglob(filename):
        info = path.lstat()
        if stat.S_ISREG(info.st_mode) and info.st_size == expected_size:
            candidates.append(path)
    if len(candidates) != 1:
        raise SystemExit(
            f"expected exactly one regular cached wheel for {filename}, found {len(candidates)}"
        )
    with zipfile.ZipFile(candidates[0]) as wheel:
        if not any(name.endswith(".dist-info/METADATA") for name in wheel.namelist()):
            raise SystemExit(f"wheel has no METADATA: {candidates[0]}")
    shutil.copyfile(candidates[0], destination / filename)
PY

TMPDIR="${BUILD_DIR}/tmp"
mkdir -p "$TMPDIR"
for requirement in \
  triton==3.3.1 \
  torchvision==0.22.1+cu128 \
  torchcodec==0.5+cu128
do
  TMPDIR="$TMPDIR" "$BASE_PYTHON" -m pip download \
    --disable-pip-version-check \
    --no-deps \
    --only-binary=:all: \
    --index-url "$PYTORCH_INDEX" \
    --dest "$BUILD_DIR" \
    "$requirement"
done
rm -rf -- "$TMPDIR"

"$BASE_PYTHON" - "$BUILD_DIR" <<'PY'
from email.parser import Parser
from pathlib import Path
import stat
import sys
import zipfile

root = Path(sys.argv[1])
expected = {
    "nvidia-cublas-cu12": "12.8.3.14",
    "nvidia-cuda-cupti-cu12": "12.8.57",
    "nvidia-cuda-nvrtc-cu12": "12.8.61",
    "nvidia-cuda-runtime-cu12": "12.8.57",
    "nvidia-cudnn-cu12": "9.7.1.26",
    "nvidia-cufft-cu12": "11.3.3.41",
    "nvidia-cufile-cu12": "1.13.0.11",
    "nvidia-curand-cu12": "10.3.9.55",
    "nvidia-cusolver-cu12": "11.7.2.55",
    "nvidia-cusparse-cu12": "12.5.7.53",
    "nvidia-cusparselt-cu12": "0.6.3",
    "nvidia-nccl-cu12": "2.26.2",
    "nvidia-nvjitlink-cu12": "12.8.61",
    "nvidia-nvtx-cu12": "12.8.55",
    "torch": "2.7.1+cu128",
    "torchcodec": "0.5+cu128",
    "torchvision": "0.22.1+cu128",
    "triton": "3.3.1",
}
observed = {}
for path in sorted(root.glob("*.whl")):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise SystemExit(f"wheel is not a regular file: {path}")
    with zipfile.ZipFile(path) as wheel:
        metadata_paths = [name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise SystemExit(f"expected one METADATA entry in {path.name}")
        metadata = Parser().parsestr(wheel.read(metadata_paths[0]).decode("utf-8"))
    name = metadata["Name"].lower()
    version = metadata["Version"]
    if name in observed:
        raise SystemExit(f"duplicate wheel distribution: {name}")
    observed[name] = version
if observed != expected:
    raise SystemExit(f"wheelhouse contract mismatch: observed={observed!r}")
print(f"B4 H254 Torch wheelhouse validation PASS files={len(observed)}")
PY

[[ ! -e "$TARGET" && ! -L "$TARGET" ]] || die "target appeared during build"
mv -- "$BUILD_DIR" "$TARGET"
BUILD_DIR=""
printf 'B4 H254 Torch wheelhouse published: %s\n' "$TARGET"
