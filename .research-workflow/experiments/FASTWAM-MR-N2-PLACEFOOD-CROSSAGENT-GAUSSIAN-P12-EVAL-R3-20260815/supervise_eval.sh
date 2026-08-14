#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_root=$(git -C "$script_dir" rev-parse --show-toplevel)
train_root=${P12_TRAIN_ROOT:-/oss-chengjuntao/artifacts/fastwam-placefood-crossagent-gaussian-p12-s42-8g-r2-20260814}
teacher_output=${P12_TF_OUTPUT_ROOT:-/oss-chengjuntao/artifacts/fastwam-placefood-crossagent-gaussian-p12-paired-tf-20260815-r3}
step500_output=${P12_STEP500_OUTPUT_ROOT:-/oss-chengjuntao/artifacts/fastwam-p12-step000500-official-topp-h32-val8-20260815-r3}
step1000_output=${P12_STEP1000_OUTPUT_ROOT:-/oss-chengjuntao/artifacts/fastwam-p12-step001000-official-topp-h32-val8-20260815-r3}
record_root=${P12_EVAL_RECORD_ROOT:-/oss-chengjuntao/artifacts/fastwam-placefood-crossagent-gaussian-p12-eval-supervisor-20260815-r3}
runtime_root=${P12_EVAL_RUNTIME_ROOT:-/mnt/workspace/experiments/FASTWAM-P12-EVAL-SUPERVISOR-R3-20260815/runtime}
poll_seconds=${P12_EVAL_POLL_SECONDS:-120}
timeout_seconds=${P12_EVAL_TIMEOUT_SECONDS:-1209600}
memory_threshold_mib=${P12_EVAL_GPU_MEMORY_THRESHOLD_MIB:-1024}
teacher_script=$script_dir/run_teacher_forcing.sh
closedloop_script=$script_dir/run_closedloop_h32.sh

mkdir -p "$record_root" "$runtime_root"
exec 9>"$runtime_root/supervisor.lock"
if ! flock -n 9; then
  echo "another P12 evaluation supervisor owns $runtime_root/supervisor.lock" >&2
  exit 2
fi

started_epoch=$(date +%s)

write_state() {
  local state=$1
  local detail=${2:-}
  python3 - "$record_root/state.json" "$state" "$detail" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "state": sys.argv[2],
    "detail": sys.argv[3],
    "observed_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
}
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2) + "\n")
os.replace(temporary, path)
print(json.dumps(payload, sort_keys=True), flush=True)
PY
}

validate_offline() {
  python3 - "$teacher_output" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
assert (root / "terminal.status").read_text().strip() == "SUCCEEDED"
terminal = json.loads((root / "TERMINAL_STATUS.json").read_text())
comparison = json.loads((root / "comparison.json").read_text())
assert terminal.get("status") == "SUCCEEDED" and terminal.get("return_code") == 0
assert comparison.get("status") == "COMPLETED"
assert set(comparison.get("checkpoints", {})) == {"step_000500", "step_001000"}
for value in comparison["checkpoints"].values():
    assert value.get("states") == 263
    assert value.get("valid_pairs_h1") == 263
    assert value.get("valid_pairs_h5") == 1305
PY
}

validate_closedloop() {
  local root=$1
  python3 - "$root" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads((Path(sys.argv[1]) / "aggregate.json").read_text())
assert report.get("status") == "COMPLETE", report
assert report.get("expected_runs") == 8, report
assert report.get("operational_runs") == 8, report
assert isinstance(report.get("success_count"), int), report
PY
}

wait_for_idle_gpus() {
  local required=$1
  while true; do
    mapfile -t free_gpus < <(python3 - "$memory_threshold_mib" <<'PY'
import subprocess
import sys

threshold = int(sys.argv[1])
inventory = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"],
    check=True,
    capture_output=True,
    text=True,
).stdout
processes = subprocess.run(
    ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
    check=True,
    capture_output=True,
    text=True,
).stdout
busy = {line.split(",", 1)[0].strip() for line in processes.splitlines() if line.strip()}
for line in inventory.splitlines():
    index, uuid, memory = (part.strip() for part in line.split(","))
    if uuid not in busy and int(memory) <= threshold:
        print(index)
PY
    )
    if (( ${#free_gpus[@]} >= required )); then
      selected_gpus=("${free_gpus[@]:0:required}")
      return
    fi
    write_state WAITING_FOR_GPUS "required=$required free=${#free_gpus[@]}"
    sleep "$poll_seconds"
  done
}

for output in "$teacher_output" "$step500_output" "$step1000_output"; do
  if [[ -e "$output" || -L "$output" ]]; then
    case "$output" in
      "$teacher_output") validate_offline ;;
      *) validate_closedloop "$output" ;;
    esac
  fi
done

while true; do
  ready=1
  for step in 000500 001000; do
    checkpoint="$train_root/checkpoints/weights/step_${step}.pt"
    if [[ ! -s "$checkpoint" || ! -s "$checkpoint.COMPLETE" ]]; then
      ready=0
    fi
  done
  if (( ready )); then
    break
  fi
  if (( $(date +%s) - started_epoch >= timeout_seconds )); then
    write_state FAILED "timed out waiting for P12 checkpoints"
    exit 1
  fi
  write_state WAITING_FOR_CHECKPOINTS "$train_root"
  sleep "$poll_seconds"
done

if [[ ! -e "$teacher_output" ]]; then
  wait_for_idle_gpus 1
  write_state RUNNING_TEACHER "gpu=${selected_gpus[0]}"
  P12_TF_GPU=${selected_gpus[0]} bash "$teacher_script"
fi
validate_offline

if [[ ! -e "$step500_output" ]]; then
  wait_for_idle_gpus 4
  write_state RUNNING_CLOSEDLOOP "step=000500 gpus=${selected_gpus[*]}"
  P12_EVAL_STEP=000500 \
    P12_EVAL_GPUS="${selected_gpus[*]}" \
    P12_CLOSEDLOOP_OUTPUT_ROOT="$step500_output" \
    bash "$closedloop_script"
fi
validate_closedloop "$step500_output"

if [[ ! -e "$step1000_output" ]]; then
  wait_for_idle_gpus 4
  write_state RUNNING_CLOSEDLOOP "step=001000 gpus=${selected_gpus[*]}"
  P12_EVAL_STEP=001000 \
    P12_EVAL_GPUS="${selected_gpus[*]}" \
    P12_CLOSEDLOOP_OUTPUT_ROOT="$step1000_output" \
    bash "$closedloop_script"
fi
validate_closedloop "$step1000_output"
write_state COMPLETED "offline and both eight-episode closed-loop panels complete"
