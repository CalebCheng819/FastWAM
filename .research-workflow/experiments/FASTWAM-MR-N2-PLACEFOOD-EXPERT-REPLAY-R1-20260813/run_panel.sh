#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <output-root>" >&2
  exit 2
fi

output_root=$1
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source_root=$(cd -- "$script_dir/../../.." && pwd -P)
evaluation_code_commit=$(git -C "$source_root" rev-parse HEAD)
if [[ -n "$(git -C "$source_root" status --porcelain --untracked-files=no)" ]]; then
  echo "formal expert replay requires a clean tracked worktree: $source_root" >&2
  exit 2
fi
python_bin=/opt/venvs/gaudp-robofactory-py310/bin/python
aggregate="$output_root/aggregate.json"
logs_dir="$output_root/logs"

if [[ "$output_root" != /oss-chengjuntao/* ]]; then
  echo "output-root must be under /oss-chengjuntao" >&2
  exit 2
fi
if ! mountpoint -q /oss-chengjuntao || [[ ! -w /oss-chengjuntao ]]; then
  echo "/oss-chengjuntao is not a writable mount" >&2
  exit 3
fi
if [[ -e "$output_root" || -L "$output_root" ]]; then
  echo "refusing to overwrite existing output root: $output_root" >&2
  exit 4
fi

available_gpu_count=$(nvidia-smi --list-gpus | wc -l)
gpu_ids_raw=${FASTWAM_EVAL_GPU_IDS:-}
if [[ -n "$gpu_ids_raw" ]]; then
  IFS=',' read -r -a gpu_ids <<< "$gpu_ids_raw"
else
  gpu_ids=()
  for ((gpu = 0; gpu < available_gpu_count; gpu++)); do
    gpu_ids+=("$gpu")
  done
fi
if [[ ${#gpu_ids[@]} -lt 1 || ${#gpu_ids[@]} -gt 8 ]]; then
  echo "one through eight physical GPU IDs must be selected" >&2
  exit 5
fi
declare -A seen_gpu_ids=()
for gpu in "${gpu_ids[@]}"; do
  if [[ ! "$gpu" =~ ^[0-7]$ || "$gpu" -ge "$available_gpu_count" ]]; then
    echo "selected GPU is not exposed by nvidia-smi: $gpu" >&2
    exit 5
  fi
  if [[ -n "${seen_gpu_ids[$gpu]:-}" ]]; then
    echo "selected GPU is repeated: $gpu" >&2
    exit 5
  fi
  seen_gpu_ids[$gpu]=1
done

mkdir -p -- "$logs_dir"

run_worker() {
  local gpu=$1
  shift
  local token split index log
  for token in "$@"; do
    split=${token%%:*}
    index=${token##*:}
    log="$logs_dir/${split}-$(printf '%02d' "$index").log"
    if [[ -e "$log" || -L "$log" ]]; then
      echo "refusing to overwrite a worker log: $log" >&2
      return 5
    fi
    "$script_dir/run_one.sh" "$split" "$index" "$gpu" "$output_root" "$evaluation_code_commit" >"$log" 2>&1
  done
}

tokens=(
  train:0 train:1 train:2 train:3 train:4 train:5 train:6 train:7
  val:0 val:1 val:2 val:3 val:4 val:5 val:6 val:7
)
pids=()
for ((worker = 0; worker < ${#gpu_ids[@]}; worker++)); do
  gpu=${gpu_ids[$worker]}
  worker_tokens=()
  for ((index = worker; index < ${#tokens[@]}; index += ${#gpu_ids[@]})); do
    worker_tokens+=("${tokens[$index]}")
  done
  run_worker "$gpu" "${worker_tokens[@]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ $status -ne 0 ]]; then
  echo "one or more workers failed; aggregate was not written" >&2
  exit 6
fi

exec "$python_bin" -B "$script_dir/aggregate_results.py" \
  --root "$output_root" \
  --output "$aggregate" \
  --expected-evaluation-code-commit "$evaluation_code_commit"
