#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

die() {
  printf 'P12_R4_DLC_EVAL_FATAL: %s\n' "$*" >&2
  exit 1
}

required_env=(
  P12_EXPERIMENT_ID P12_DISPLAY_NAME P12_EVAL_ROOT P12_EVALUATION_COMMIT
  P12_MODEL_ROOT P12_TRAIN_ROOT P12_TRAINING_COMMIT P12_TRAINING_JOB_ID
  P12_TF_OUTPUT_ROOT P12_STEP500_OUTPUT_ROOT P12_STEP1000_OUTPUT_ROOT
  P12_RECORD_ROOT P12_EVAL_PYTHON P12_PYTHON_EXTRA P12_EVAL_PANEL
  P12_ROBOFACTORY_ROOT P12_POLICY_LIGHTNING_ROOT P12_POLICY_LIGHTNING_COMMIT
  P12_NOPOSPLAT_CHECKPOINT P12_GRAPHICS_ROOT P12_VULKAN_LOADER
  P12_RUNTIME_SCRIPT P12_INTEGRITY_MODE
)
for name in "${required_env[@]}"; do
  [[ -n "${!name:-}" ]] || die "missing environment variable ${name}"
done

started_at=$(date --iso-8601=seconds)
started_epoch=$(date +%s)
final_state=FAILED
selected_profile=''

terminalize() {
  local command_rc=${1:-$?}
  local actual_rc=$command_rc
  local completed_at completed_epoch
  completed_at=$(date --iso-8601=seconds)
  completed_epoch=$(date +%s)
  if [[ "$final_state" != SUCCEEDED && "$actual_rc" -eq 0 ]]; then
    actual_rc=1
  fi
  mkdir -p -- "$P12_RECORD_ROOT"
  python3 - "$P12_RECORD_ROOT/worker-terminal.json" "$final_state" "$actual_rc" \
    "$started_at" "$completed_at" "$((completed_epoch - started_epoch))" \
    "$selected_profile" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
value = {
    "schema_version": "fastwam-p12-dlc-eval-worker-terminal-v1",
    "status": sys.argv[2],
    "return_code": int(sys.argv[3]),
    "started_at": sys.argv[4],
    "completed_at": sys.argv[5],
    "runtime_seconds": int(sys.argv[6]),
    "graphics_profile": sys.argv[7],
}
temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
os.replace(temporary, path)
PY
  trap - EXIT
  exit "$actual_rc"
}
trap terminalize EXIT

mountpoint -q /oss-chengjuntao || die '/oss-chengjuntao is not mounted'
test -w /oss-chengjuntao || die '/oss-chengjuntao is not writable'
mountpoint -q /cpfs/user/chengjuntao || die '/cpfs/user/chengjuntao is not mounted'
test -x "$P12_EVAL_PYTHON" || die 'evaluation Python is not executable'
test -f "$P12_RUNTIME_SCRIPT" || die 'runtime script path is missing'
test "$(readlink -f -- "$P12_RUNTIME_SCRIPT")" = "$(readlink -f -- "${BASH_SOURCE[0]}")" \
  || die 'runtime entrypoint identity mismatch'
test "$(git -C "$P12_EVAL_ROOT" rev-parse HEAD)" = "$P12_EVALUATION_COMMIT" \
  || die 'evaluation revision mismatch'
test -z "$(git -C "$P12_EVAL_ROOT" status --short)" || die 'evaluation worktree is dirty'
test "$(git -C "$P12_MODEL_ROOT" rev-parse HEAD)" = "$P12_TRAINING_COMMIT" \
  || die 'training revision mismatch'
test -z "$(git -C "$P12_MODEL_ROOT" status --short)" || die 'training worktree is dirty'
test "$(git -C "$P12_POLICY_LIGHTNING_ROOT" rev-parse HEAD)" = "$P12_POLICY_LIGHTNING_COMMIT" \
  || die 'Policy-Lightning revision mismatch'
test -z "$(git -C "$P12_POLICY_LIGHTNING_ROOT" status --short)" \
  || die 'Policy-Lightning worktree is dirty'
for output in "$P12_TF_OUTPUT_ROOT" "$P12_STEP500_OUTPUT_ROOT" "$P12_STEP1000_OUTPUT_ROOT"; do
  [[ ! -e "$output" && ! -L "$output" ]] || die "output already exists: ${output}"
done
for step in 000500 001000; do
  checkpoint="$P12_TRAIN_ROOT/checkpoints/weights/step_${step}.pt"
  test -s "$checkpoint" || die "checkpoint missing: ${checkpoint}"
  test -s "$checkpoint.COMPLETE" || die "checkpoint marker missing: ${checkpoint}.COMPLETE"
done

gpu_count=$("$P12_EVAL_PYTHON" -B -c 'import torch; print(torch.cuda.device_count())')
[[ "$gpu_count" == 8 ]] || die "expected exactly 8 visible GPUs, observed ${gpu_count}"

scratch_root=$(mktemp -d /tmp/fastwam-p12-r4-dlc.XXXXXXXX)
cleanup() {
  rm -rf -- "$scratch_root"
}
cleanup_and_terminalize() {
  local command_rc=$?
  cleanup
  terminalize "$command_rc"
}
trap cleanup_and_terminalize EXIT
ulimit -c 0
mkdir -m 0700 -p -- \
  "$scratch_root/xdg-cache" "$scratch_root/xdg-runtime" "$scratch_root/torch" \
  "$scratch_root/matplotlib" "$scratch_root/tmp" "$scratch_root/pycache" \
  "$scratch_root/graphics-probes" "$P12_RECORD_ROOT/logs"

export PYTHONPATH="$P12_PYTHON_EXTRA:$P12_MODEL_ROOT/src:$P12_EVAL_ROOT/experiments/robofactory:$P12_EVAL_ROOT:$P12_POLICY_LIGHTNING_ROOT:$P12_ROBOFACTORY_ROOT:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONFAULTHANDLER=1
export WANDB_MODE=offline
export XDG_CACHE_HOME="$scratch_root/xdg-cache"
export XDG_RUNTIME_DIR="$scratch_root/xdg-runtime"
export TORCH_HOME="$scratch_root/torch"
export MPLCONFIGDIR="$scratch_root/matplotlib"
export TMPDIR="$scratch_root/tmp"
export PYTHONPYCACHEPREFIX="$scratch_root/pycache"

graphics_keys=(
  LD_LIBRARY_PATH VK_ICD_FILENAMES VK_DRIVER_FILES
  __EGL_VENDOR_LIBRARY_FILENAMES __EGL_VENDOR_LIBRARY_DIRS
  __GLX_VENDOR_LIBRARY_NAME SAPIEN_VULKAN_LIBRARY_PATH FASTWAM_GL_SHIM_ROOT
  LIBGL_DRIVERS_PATH GBM_BACKEND MUJOCO_GL EGL_PLATFORM PYOPENGL_PLATFORM
  NVIDIA_DRIVER_CAPABILITIES
)
declare -A provider_graphics_present=()
declare -A provider_graphics_value=()
for name in "${graphics_keys[@]}"; do
  if [[ -v "$name" ]]; then
    provider_graphics_present["$name"]=1
    provider_graphics_value["$name"]="${!name}"
  else
    provider_graphics_present["$name"]=0
    provider_graphics_value["$name"]=''
  fi
done

restore_provider_graphics() {
  local key
  for key in "${graphics_keys[@]}"; do
    if [[ "${provider_graphics_present[$key]}" == 1 ]]; then
      export "$key=${provider_graphics_value[$key]}"
    else
      unset "$key"
    fi
  done
}

clear_explicit_graphics_selection() {
  unset VK_ICD_FILENAMES VK_DRIVER_FILES
  unset __EGL_VENDOR_LIBRARY_FILENAMES __EGL_VENDOR_LIBRARY_DIRS
  unset __GLX_VENDOR_LIBRARY_NAME SAPIEN_VULKAN_LIBRARY_PATH
  unset FASTWAM_GL_SHIM_ROOT LIBGL_DRIVERS_PATH GBM_BACKEND
}

apply_headless_contract() {
  export MUJOCO_GL=egl
  export EGL_PLATFORM=surfaceless
  export PYOPENGL_PLATFORM=egl
  export NVIDIA_DRIVER_CAPABILITIES=all
}

build_discovered_loader() {
  local candidate resolved entry
  local -a loader=()
  declare -A seen=()
  for candidate in \
    /usr/local/nvidia/lib64 /usr/local/nvidia/lib /usr/local/cuda/compat \
    /usr/local/cuda/lib64 /usr/local/cuda-12.8/lib64 \
    /usr/lib/x86_64-linux-gnu /usr/lib64 /lib/x86_64-linux-gnu; do
    [[ -d "$candidate" && ! -L "$candidate" ]] || continue
    resolved=$(readlink -f -- "$candidate")
    if [[ -z "${seen[$resolved]:-}" ]]; then
      loader+=("$resolved")
      seen["$resolved"]=1
    fi
  done
  IFS=:
  entry="${loader[*]}"
  unset IFS
  [[ -n "$entry" ]] || return 1
  export LD_LIBRARY_PATH="$entry"
}

first_regular_file() {
  local candidate
  for candidate in "$@"; do
    if [[ -f "$candidate" && ! -L "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

apply_graphics_profile() {
  local profile=$1
  local vk_manifest egl_manifest
  restore_provider_graphics
  case "$profile" in
    provider_native_headless)
      ;;
    provider_clean_headless)
      clear_explicit_graphics_selection
      ;;
    system_default_headless)
      clear_explicit_graphics_selection
      unset LD_LIBRARY_PATH
      ;;
    system_discovered_headless)
      clear_explicit_graphics_selection
      build_discovered_loader
      ;;
    system_manifest_headless)
      clear_explicit_graphics_selection
      build_discovered_loader
      vk_manifest=$(first_regular_file /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json) || return 2
      egl_manifest=$(first_regular_file /usr/share/glvnd/egl_vendor.d/10_nvidia.json /etc/glvnd/egl_vendor.d/10_nvidia.json) || return 2
      export VK_ICD_FILENAMES="$vk_manifest"
      export VK_DRIVER_FILES="$vk_manifest"
      export __EGL_VENDOR_LIBRARY_FILENAMES="$egl_manifest"
      export __GLX_VENDOR_LIBRARY_NAME=nvidia
      ;;
    system_discovered_sapien_loader)
      clear_explicit_graphics_selection
      build_discovered_loader
      test -f "$P12_VULKAN_LOADER" || return 2
      export SAPIEN_VULKAN_LIBRARY_PATH="$P12_VULKAN_LOADER"
      ;;
    *)
      return 2
      ;;
  esac
  apply_headless_contract
}

probe_program=$'import os\nfrom pathlib import Path\nfrom diagnose_place_food_fixed import _build_environment\nroot = Path(os.environ["P12_ROBOFACTORY_ROOT"])\nenvironment = _build_environment(root, "PlaceFood-rf")\nenvironment.close()\nprint("P12_R4_DLC_ENVIRONMENT_CONSTRUCTION_PROBE_PASS task=PlaceFood-rf device=0")\n'
profiles=(
  provider_native_headless provider_clean_headless system_default_headless
  system_discovered_headless system_manifest_headless
  system_discovered_sapien_loader
)
for profile in "${profiles[@]}"; do
  probe_log="$scratch_root/graphics-probes/${profile}.log"
  if ! apply_graphics_profile "$profile"; then
    printf 'P12_R4_DLC_GRAPHICS_PROFILE_SKIPPED profile=%s reason=unavailable\n' "$profile"
    continue
  fi
  set +e
  timeout --signal=TERM --kill-after=30s 180s env CUDA_VISIBLE_DEVICES=0 \
    "$P12_EVAL_PYTHON" -B -c "$probe_program" >"$probe_log" 2>&1
  probe_rc=$?
  set -e
  if [[ "$probe_rc" == 0 ]]; then
    selected_profile=$profile
    printf 'P12_R4_DLC_GRAPHICS_PROFILE_SELECTED profile=%s\n' "$profile"
    cp -- "$probe_log" "$P12_RECORD_ROOT/logs/graphics-${profile}.log"
    break
  fi
  printf 'P12_R4_DLC_GRAPHICS_PROFILE_REJECTED profile=%s rc=%s\n' "$profile" "$probe_rc" >&2
  cp -- "$probe_log" "$P12_RECORD_ROOT/logs/graphics-${profile}.log"
done
[[ -n "$selected_profile" ]] || die 'no GPU graphics profile could construct and close PlaceFood-rf'
apply_graphics_profile "$selected_profile" || die 'selected graphics profile became unavailable'

experiment_dir="$P12_EVAL_ROOT/.research-workflow/experiments/FASTWAM-MR-N2-PLACEFOOD-CROSSAGENT-GAUSSIAN-P12-EVAL-R3-20260815"
teacher_script="$experiment_dir/run_teacher_forcing.sh"
closedloop_script="$experiment_dir/run_closedloop_h32.sh"
test -f "$teacher_script" || die 'teacher-forcing script is missing'
test -f "$closedloop_script" || die 'closed-loop script is missing'

P12_TF_OUTPUT_ROOT="$P12_TF_OUTPUT_ROOT" \
P12_TF_GPU=0 \
bash "$teacher_script" >"$P12_RECORD_ROOT/logs/teacher-forcing.log" 2>&1

set +e
P12_EVAL_STEP=000500 \
P12_EVAL_GPUS='0 1 2 3' \
P12_CLOSEDLOOP_OUTPUT_ROOT="$P12_STEP500_OUTPUT_ROOT" \
bash "$closedloop_script" >"$P12_RECORD_ROOT/logs/closedloop-step000500.log" 2>&1 &
pid500=$!
P12_EVAL_STEP=001000 \
P12_EVAL_GPUS='4 5 6 7' \
P12_CLOSEDLOOP_OUTPUT_ROOT="$P12_STEP1000_OUTPUT_ROOT" \
bash "$closedloop_script" >"$P12_RECORD_ROOT/logs/closedloop-step001000.log" 2>&1 &
pid1000=$!
wait "$pid500"
rc500=$?
wait "$pid1000"
rc1000=$?
set -e
if [[ "$rc500" -ne 0 || "$rc1000" -ne 0 ]]; then
  tail -n 120 -- "$P12_RECORD_ROOT/logs/closedloop-step000500.log" >&2 || true
  tail -n 120 -- "$P12_RECORD_ROOT/logs/closedloop-step001000.log" >&2 || true
  die "closed-loop evaluation failed: step500=${rc500} step1000=${rc1000}"
fi

python3 - "$P12_RECORD_ROOT/evaluation-summary.json" "$P12_TF_OUTPUT_ROOT" \
  "$P12_STEP500_OUTPUT_ROOT" "$P12_STEP1000_OUTPUT_ROOT" "$selected_profile" <<'PY'
import json
import os
import sys
from pathlib import Path

destination = Path(sys.argv[1])
teacher = Path(sys.argv[2])
step500 = Path(sys.argv[3])
step1000 = Path(sys.argv[4])
graphics = sys.argv[5]
comparison = json.loads((teacher / "comparison.json").read_text())
aggregates = {}
for name, root in (("step_000500", step500), ("step_001000", step1000)):
    aggregate = json.loads((root / "aggregate.json").read_text())
    if aggregate.get("status") != "COMPLETE" or aggregate.get("operational_runs") != 8:
        raise RuntimeError(f"invalid closed-loop aggregate for {name}: {aggregate}")
    aggregates[name] = aggregate
value = {
    "schema_version": "fastwam-p12-r4-dlc-evaluation-summary-v1",
    "status": "COMPLETED",
    "graphics_profile": graphics,
    "teacher_forcing": comparison,
    "closed_loop": aggregates,
    "claim_limit": "Offline action error and DLC completion do not substitute for closed-loop success.",
}
temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
os.replace(temporary, destination)
PY

final_state=SUCCEEDED
printf 'P12_R4_DLC_EVAL_SCIENTIFIC_COMPLETE\n'
