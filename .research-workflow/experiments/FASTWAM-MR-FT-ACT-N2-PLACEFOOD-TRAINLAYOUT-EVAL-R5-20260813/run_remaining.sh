#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <output-root>" >&2
  exit 2
fi

output_root=$1
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
python_bin=/opt/venvs/gaudp-robofactory-py310/bin/python
smoke_dir="$output_root/train/episode-00"
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

if [[ ! -d "$smoke_dir" || ! -f "$smoke_dir/summary.json" || ! -f "$smoke_dir/run_manifest.json" ]]; then
  echo "train episode 00 smoke output is not present" >&2
  exit 3
fi
if [[ -e "$aggregate" || -L "$aggregate" ]]; then
  echo "refusing to overwrite an existing aggregate: $aggregate" >&2
  exit 4
fi
if [[ -e "$logs_dir" || -L "$logs_dir" ]]; then
  echo "refusing to overwrite an existing logs directory: $logs_dir" >&2
  exit 4
fi

"$python_bin" -B - "$smoke_dir" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve(strict=True)
summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
rollout = summary.get("rollout")
if summary.get("status") != "COMPLETED" or manifest.get("status") != "terminal":
    raise SystemExit("smoke is not terminal and COMPLETED")
if not isinstance(rollout, dict) or rollout.get("status") != "completed":
    raise SystemExit("smoke rollout is not completed")
if int(rollout.get("steps", 0)) <= 0 or int(rollout.get("policy_queries", 0)) <= 0:
    raise SystemExit("smoke did not advance the environment and query the policy")
episode = manifest.get("episode")
if not isinstance(episode, dict) or episode.get("split") != "train" or int(episode.get("panel_index", -1)) != 0:
    raise SystemExit("smoke episode identity mismatch")
PY

available_gpu_count=$(nvidia-smi --list-gpus | wc -l)
gpu_ids_raw=${FASTWAM_EVAL_GPU_IDS:-}
gpu_count=${FASTWAM_EVAL_GPU_COUNT:-}

if [[ -n "$gpu_count" && ! "$gpu_count" =~ ^[1-8]$ ]]; then
  echo "FASTWAM_EVAL_GPU_COUNT must be an integer from 1 through 8" >&2
  exit 5
fi
if [[ -n "$gpu_ids_raw" ]]; then
  IFS=',' read -r -a gpu_ids <<< "$gpu_ids_raw"
  if [[ -n "$gpu_count" && "$gpu_count" -ne "${#gpu_ids[@]}" ]]; then
    echo "FASTWAM_EVAL_GPU_COUNT does not match FASTWAM_EVAL_GPU_IDS" >&2
    exit 5
  fi
  gpu_count=${#gpu_ids[@]}
else
  if [[ -z "$gpu_count" ]]; then
    gpu_count=$available_gpu_count
  fi
  gpu_ids=()
  for ((gpu = 0; gpu < gpu_count; gpu++)); do
    gpu_ids+=("$gpu")
  done
fi

if [[ ! "$gpu_count" =~ ^[1-8]$ ]]; then
  echo "the selected GPU count must be an integer from 1 through 8" >&2
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

mkdir -- "$logs_dir"

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
    "$script_dir/run_one.sh" "$split" "$index" "$gpu" "$output_root" >"$log" 2>&1
  done
}

tokens=(
  train:1 train:2 train:3 train:4 train:5 train:6 train:7
  val:0 val:1 val:2 val:3 val:4 val:5 val:6 val:7
)
pids=()
for ((worker = 0; worker < gpu_count; worker++)); do
  gpu=${gpu_ids[$worker]}
  worker_tokens=()
  for ((index = worker; index < ${#tokens[@]}; index += gpu_count)); do
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
  --expected-checkpoint \
  /oss-chengjuntao/artifacts/fastwam-action-n234-formal-r5-20260812/fastwam-act-n2-placefood-1k-s42-r5-20260812/checkpoints/weights/step_001000.pt \
  --expected-training-code-commit 1a690ab49246cbeb841618a86b5bd546f93ddd40 \
  --comparison "R5 PlaceFood-rf task-specific checkpoint on frozen train-layout and validation-layout panels" \
  --output "$aggregate"
