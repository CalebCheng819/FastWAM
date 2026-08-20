#!/usr/bin/env bash
set -Eeuo pipefail

die() {
  printf 'B4 H254 launcher error: %s\n' "$*" >&2
  exit 1
}

require_env() {
  local name="$1"
  [[ -n "${!name:-}" ]] || die "required environment variable is empty: ${name}"
}

require_regular() {
  local path="$1"
  [[ -f "$path" && ! -L "$path" ]] || die "expected regular non-symlink file: ${path}"
}

for name in \
  FASTWAM_RUN_ID FASTWAM_ATTEMPT_ID FASTWAM_GIT_COMMIT \
  FASTWAM_GIT_BUNDLE FASTWAM_PYTHON FASTWAM_RCLONE_CONFIG \
  FASTWAM_H_RAW_H5_REMOTE FASTWAM_H_INPUT_REMOTE FASTWAM_H_OUTPUT_REMOTE; do
  require_env "$name"
done

[[ "$FASTWAM_RUN_ID" =~ ^[a-z0-9][a-z0-9-]+$ ]] || die "invalid FASTWAM_RUN_ID"
[[ "$FASTWAM_ATTEMPT_ID" =~ ^attempt-[0-9]{3}$ ]] || die "invalid FASTWAM_ATTEMPT_ID"
[[ "$FASTWAM_GIT_COMMIT" =~ ^[0-9a-f]{40}$ ]] || die "FASTWAM_GIT_COMMIT must be a full Git commit"
[[ "$FASTWAM_PYTHON" == /* && -x "$FASTWAM_PYTHON" ]] || die "FASTWAM_PYTHON must be an absolute executable"
require_regular "$FASTWAM_GIT_BUNDLE"
require_regular "$FASTWAM_RCLONE_CONFIG"

EXPECTED_INPUT_REMOTE="eailab-hdd2:eailab/chengjuntao/fastwam/robofactory-multirobot/b4-h254-s42-20260820/inputs"
EXPECTED_RAW_REMOTE="eailab-hdd2:fkp-migrate/ailab-eailabagent-gpfs/chengjuntao/placefood_wam/Policy-Lightning/data/baai_tasks"
[[ "$FASTWAM_H_INPUT_REMOTE" == "$EXPECTED_INPUT_REMOTE" ]] || die "derived input remote drift"
[[ "$FASTWAM_H_RAW_H5_REMOTE" == "$EXPECTED_RAW_REMOTE" ]] || die "raw H5 remote drift"

if [[ "${FASTWAM_H_DRY_RUN:-0}" == "1" ]]; then
  printf 'B4 H254 launcher dry-run: run=%s attempt=%s commit=%s gpus=8 machines=1 accumulation=3\n' \
    "$FASTWAM_RUN_ID" "$FASTWAM_ATTEMPT_ID" "$FASTWAM_GIT_COMMIT"
  exit 0
fi

[[ -d /nvme && -w /nvme ]] || die "/nvme is not writable; submit with --store-host-nvme"

WORK_ROOT="/nvme/fastwam/${FASTWAM_RUN_ID}/${FASTWAM_ATTEMPT_ID}"
CODE_DIR="${WORK_ROOT}/code"
DATA_ROOT="${WORK_ROOT}/datasets/robofactory_multi_robot"
GAUSSIAN_DIR="${WORK_ROOT}/gaussian/compact-s42-13x28x40-fp16-meanalpha-v2"
MODEL_CACHE_ROOT="${WORK_ROOT}/model-cache"
WARMSTART_FILE="${WORK_ROOT}/weights/step_005000.pt"
OUTPUT_DIR="${WORK_ROOT}/output"
RCLONE_CONFIG="/dev/shm/fastwam-rclone-${FASTWAM_RUN_ID}-${FASTWAM_ATTEMPT_ID}.conf"

umask 077
mkdir -p "$WORK_ROOT" "$DATA_ROOT" "$GAUSSIAN_DIR" \
  "${MODEL_CACHE_ROOT}/DiffSynth-Studio/Wan-Series-Converted-Safetensors" \
  "$(dirname "$WARMSTART_FILE")" "$OUTPUT_DIR"
cp "$FASTWAM_RCLONE_CONFIG" "$RCLONE_CONFIG"
chmod 600 "$RCLONE_CONFIG"

cleanup() {
  local rc=$?
  if [[ -d "$OUTPUT_DIR" ]]; then
    rclone --config "$RCLONE_CONFIG" copy "$OUTPUT_DIR" "$FASTWAM_H_OUTPUT_REMOTE" \
      --transfers 4 --checkers 8 --stats 30s || true
  fi
  rm -f "$RCLONE_CONFIG"
  exit "$rc"
}
trap cleanup EXIT

git clone --quiet "$FASTWAM_GIT_BUNDLE" "$CODE_DIR"
actual_commit="$(git -C "$CODE_DIR" rev-parse HEAD)"
[[ "$actual_commit" == "$FASTWAM_GIT_COMMIT" ]] || die "bundle checkout commit drift: ${actual_commit}"
[[ -z "$(git -C "$CODE_DIR" status --porcelain)" ]] || die "bundle checkout is dirty"

MANIFEST="${CODE_DIR}/configs/transfer/b4_h254_raw_h5_paths.txt"
require_regular "$MANIFEST"
mapfile -t h5_relpaths < <(sed '/^[[:space:]]*$/d' "$MANIFEST")
[[ "${#h5_relpaths[@]}" -eq 24 ]] || die "raw H5 manifest must contain exactly 24 paths"
[[ "$(printf '%s\n' "${h5_relpaths[@]}" | sort -u | wc -l)" -eq 24 ]] || die "raw H5 manifest contains duplicates"

for rel in "${h5_relpaths[@]}"; do
  [[ "$rel" != /* && "$rel" != *".."* && "$rel" == *.h5 ]] || die "unsafe H5 relative path: ${rel}"
  mkdir -p "${DATA_ROOT}/$(dirname "$rel")"
  rclone --config "$RCLONE_CONFIG" copyto \
    "${FASTWAM_H_RAW_H5_REMOTE}/${rel}" "${DATA_ROOT}/${rel}" --stats 30s
done

STATS_FILE="${DATA_ROOT}/fastwam_multi_robot_n234_train_s42_stats_v2.json"
EMBEDS_DIR="${DATA_ROOT}/text_embeds_cache_n234"
VAE_FILE="${MODEL_CACHE_ROOT}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"
rclone --config "$RCLONE_CONFIG" copyto \
  "${FASTWAM_H_INPUT_REMOTE}/dataset-derived/stats/fastwam_multi_robot_n234_train_s42_stats_v2.json" \
  "$STATS_FILE" --stats 30s
rclone --config "$RCLONE_CONFIG" copy \
  "${FASTWAM_H_INPUT_REMOTE}/dataset-derived/text_embeds_cache_n234" \
  "$EMBEDS_DIR" --stats 30s
rclone --config "$RCLONE_CONFIG" copy \
  "${FASTWAM_H_INPUT_REMOTE}/gaussian/compact-s42-13x28x40-fp16-meanalpha-v2" \
  "$GAUSSIAN_DIR" --stats 30s
rclone --config "$RCLONE_CONFIG" copyto \
  "${FASTWAM_H_INPUT_REMOTE}/weights/step_005000.pt" "$WARMSTART_FILE" --stats 30s
rclone --config "$RCLONE_CONFIG" copyto \
  "${FASTWAM_H_INPUT_REMOTE}/model-cache/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors" \
  "$VAE_FILE" --stats 30s

[[ "$(find "$DATA_ROOT" -type f -name '*.h5' | wc -l)" -eq 24 ]] || die "staged H5 count is not 24"
[[ "$(find "$EMBEDS_DIR" -type f | wc -l)" -eq 6 ]] || die "staged text embedding count is not 6"
require_regular "$STATS_FILE"
require_regular "$WARMSTART_FILE"
require_regular "$VAE_FILE"
[[ "$(stat -c %s "$WARMSTART_FILE")" -eq 12047213728 ]] || die "warm-start byte size mismatch"
[[ "$(find "$GAUSSIAN_DIR" -type f | wc -l)" -eq 1590 ]] || die "Gaussian cache file count mismatch"

"$FASTWAM_PYTHON" - <<'PY'
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
    raise SystemExit(f"B4 H254 Python must be 3.10, got {sys.version.split()[0]}")
bad = {name: (md.version(name), want) for name, want in expected.items() if md.version(name) != want}
if bad:
    raise SystemExit(f"B4 H254 dependency contract mismatch: {bad}")
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() != 8:
    raise SystemExit(f"B4 H254 requires exactly 8 visible CUDA devices, got {torch.cuda.device_count()}")
import torchcodec
import torchvision
print(f"B4 H254 environment PASS python={sys.version.split()[0]} torch={torch.__version__} cuda_devices=8")
PY

export FASTWAM_B4_BASE_CHECKPOINT="$WARMSTART_FILE"
export FASTWAM_GAUSSIAN_CACHE_DIR="$GAUSSIAN_DIR"
export DIFFSYNTH_MODEL_BASE_PATH="$MODEL_CACHE_ROOT"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="${CODE_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

LOGICAL_STATS_ROOT="/cpfs/user/chengjuntao/datasets/robofactory_multi_robot"
cd "$CODE_DIR"
"$FASTWAM_PYTHON" -m accelerate.commands.launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_machines 1 --machine_rank 0 --num_processes 8 \
  scripts/train.py \
  task=robofactory_multi_robot_b4_phase_gripcontact_actft_224_1e-5 \
  +scale=robofactory_multi_robot_8gpu_h254_b4 \
  "data.train.root_dir=${DATA_ROOT}" \
  "data.val.root_dir=${DATA_ROOT}" \
  "data.train.pretrained_norm_stats=${STATS_FILE}" \
  "data.val.pretrained_norm_stats=${STATS_FILE}" \
  "data.train.stats_source_root=${LOGICAL_STATS_ROOT}" \
  "data.val.stats_source_root=${LOGICAL_STATS_ROOT}" \
  "data.train.text_embedding_cache_dir=${EMBEDS_DIR}" \
  "data.val.text_embedding_cache_dir=${EMBEDS_DIR}" \
  "data.train.gaussian_cache_dir=${GAUSSIAN_DIR}" \
  "data.val.gaussian_cache_dir=${GAUSSIAN_DIR}" \
  "output_dir=${OUTPUT_DIR}" \
  "wandb.name=${FASTWAM_RUN_ID}"
