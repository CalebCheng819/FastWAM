#!/usr/bin/env bash
set -euo pipefail

experiment_id=FASTWAM-MR-N2-PLACEFOOD-METRIC-P13-PAIRED-TF-R1-20260815
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
eval_root=$(git -C "$script_dir" rev-parse --show-toplevel)
model_root=${P13_MODEL_ROOT:-/mnt/workspace/experiments/FastWAM-p13-e5f20bb-20260815}
metric_cache=${P13_METRIC_CACHE_ROOT:?set P13_METRIC_CACHE_ROOT to the completed cache used by P13 training}
train_root=${P13_TRAIN_ROOT:-/oss-chengjuntao/artifacts/fastwam-placefood-metric-gaussian-p13-s42-8g-r1-20260815}
output=${P13_TF_OUTPUT_ROOT:-/oss-chengjuntao/artifacts/fastwam-placefood-metric-gaussian-p13-paired-tf-r1-20260815}
python=${P13_EVAL_PYTHON:-/opt/venvs/gaudp-robofactory-py310/bin/python}
panel=${P13_EVAL_PANEL:-/mnt/workspace/fastwam_eval_runtime/panels/robofactory_n234_s42_val8_v1.json}
dataset=${P13_DATASET_ROOT:-/oss-chengjuntao/cpfs-user-chengjuntao/datasets/robofactory_multi_robot}
robofactory=${P13_ROBOFACTORY_ROOT:-/mnt/workspace/fastwam_eval_runtime/RoboFactory-challenge-2d34fb3}
stats=${P13_STATS:-/oss-chengjuntao/artifacts/fastwam-nohash-inputs-20260809/fastwam_multi_robot_n234_train_s42_stats_cpfs_nohash_v1.json}
context=${P13_CONTEXT_FILE:-/oss-chengjuntao/cpfs-user-chengjuntao/datasets/robofactory_multi_robot/text_embeds_cache_n234_named_20260811/PlaceFood-rf.t5_len128.wan22ti2v5b.pt}
model_cache=${P13_MODEL_CACHE_ROOT:-/oss-chengjuntao/cpfs-user-chengjuntao/checkpoints/FastWAM/model-cache}
checkpoint=$train_root/checkpoints/weights/step_001000.pt
train_commit=e5f20bbf91477b82990e5c571d54305c639705c6
eval_commit=$(git -C "$eval_root" rev-parse HEAD)
gpu=${P13_TF_GPU:-0}

started_at=$(date --iso-8601=seconds)
started_epoch=$(date +%s)
final_state=FAILED

terminalize() {
  local command_rc=$?
  local completed_at completed_epoch actual_rc
  completed_at=$(date --iso-8601=seconds)
  completed_epoch=$(date +%s)
  actual_rc=$command_rc
  if [[ "$final_state" != SUCCEEDED && "$actual_rc" -eq 0 ]]; then
    actual_rc=1
  fi
  if [[ -d "$output" ]]; then
    python3 - "$output" "$experiment_id" "$final_state" "$actual_rc" \
      "$started_at" "$completed_at" "$((completed_epoch - started_epoch))" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
payload = {
    "schema_version": "fastwam-eval-terminal-v1",
    "experiment_id": sys.argv[2],
    "status": sys.argv[3],
    "return_code": int(sys.argv[4]),
    "started_at": sys.argv[5],
    "completed_at": sys.argv[6],
    "actual_runtime_seconds": int(sys.argv[7]),
}
(root / "TERMINAL_STATUS.json").write_text(json.dumps(payload, indent=2) + "\n")
(root / "terminal.status").write_text(payload["status"] + "\n")
PY
  fi
  trap - EXIT
  exit "$actual_rc"
}
trap terminalize EXIT

mountpoint -q /oss-chengjuntao
test -w /oss-chengjuntao
test ! -e "$output"
test -x "$python"
test -f "$eval_root/experiments/robofactory/diagnose_place_food_fixed.py"
test "$(git -C "$model_root" rev-parse HEAD)" = "$train_commit"
test -z "$(git -C "$model_root" status --short)"
test -z "$(git -C "$eval_root" status --short)"
test -s "$checkpoint"
test -f "$checkpoint.COMPLETE"
for path in "$panel" "$dataset" "$robofactory" "$stats" "$context" "$model_cache"; do
  test -e "$path"
done

python3 - "$metric_cache" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
assert root.is_dir(), root
assert (root / "COMPLETE").read_text().strip() == "complete"
manifest = json.loads((root / "manifest.json").read_text())
assert manifest.get("schema_name") == "fastwam.metric-geometry-cache", manifest
assert manifest.get("version") == 1, manifest
assert manifest.get("dtype") == "float16", manifest
assert manifest.get("frame_shape") == [13, 60, 80], manifest
assert int(manifest["data"]["frames"]) > 0, manifest
PY

mkdir -p "$output/logs"
env \
  CUDA_VISIBLE_DEVICES="$gpu" \
  PYTHONUNBUFFERED=1 \
  WANDB_MODE=offline \
  PYTHONPATH="$model_root/src:$eval_root/experiments/robofactory:$robofactory:${PYTHONPATH:-}" \
  "$python" "$eval_root/experiments/robofactory/diagnose_place_food_fixed.py" \
    --mode teacher-forcing \
    --formal-contract \
    --task PlaceFood-rf \
    --panel "$panel" \
    --dataset-root "$dataset" \
    --robofactory-root "$robofactory" \
    --gaussian-cache "$metric_cache" \
    --output-dir "$output/p13_step001000" \
    --episode-start 2 \
    --policy-seed 10002 \
    --teacher-start-timestep 5 \
    --max-teacher-states 263 \
    --initial-state raw \
    --checkpoint "$checkpoint" \
    --training-code-commit "$train_commit" \
    --evaluation-code-commit "$eval_commit" \
    --integrity-mode metadata_no_hash \
    --model-project-root "$model_root" \
    --action-architecture metric_gaussian_v5 \
    --gaussian-source metric_geometry \
    --stats "$stats" \
    --context-file "$context" \
    --model-cache-root "$model_cache" \
    --device cuda:0 \
    --teacher-device cuda:0 \
    --action-horizon 32 \
    --num-inference-steps 20 \
    >"$output/logs/p13_step001000.log" 2>&1

python3 - "$output" "$checkpoint" "$eval_commit" "$metric_cache" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
cell = root / "p13_step001000"
summary = json.loads((cell / "summary.json").read_text())
teacher = summary.get("teacher_forcing") or {}
contract = teacher.get("formal_contract") or {}
assert summary.get("status") == "COMPLETED", summary.get("status")
assert teacher.get("status") == "completed", teacher.get("status")
assert teacher.get("states_evaluated") == 263, teacher.get("states_evaluated")
assert contract.get("valid_pairs_h1") == 263, contract
assert contract.get("valid_pairs_h5") == 1305, contract
metrics = json.loads((cell / "teacher_forcing" / "phase_action_error_summary.json").read_text())
source = metrics["sources"]["all_states"]["live_denormalized_vs_expert"]

def group(horizon, name):
    value = source[horizon]["by_agent_and_group"][f"panda-0/{name}"]["mae"]
    assert math.isfinite(value) and value >= 0.0, (horizon, name, value)
    return value

result = {
    "schema_version": "fastwam-p13-offline-comparison-v1",
    "status": "COMPLETED",
    "checkpoint": {"path": sys.argv[2], "bytes": Path(sys.argv[2]).stat().st_size},
    "evaluation_code_commit": sys.argv[3],
    "training_code_commit": "e5f20bbf91477b82990e5c571d54305c639705c6",
    "gaussian_source": "metric_geometry",
    "metric_cache_root": sys.argv[4],
    "states": teacher["states_evaluated"],
    "valid_pairs_h1": contract["valid_pairs_h1"],
    "valid_pairs_h5": contract["valid_pairs_h5"],
    "robot0_arm_mae": {
        "h1": group("immediate", "arm"),
        "h5": group("prediction_horizon_5", "arm"),
        "full_horizon": group("full_horizon", "arm"),
    },
    "robot0_gripper_mae": {
        "h1": group("immediate", "gripper"),
        "h5": group("prediction_horizon_5", "gripper"),
        "full_horizon": group("full_horizon", "gripper"),
    },
    "gripper_sign_agreement": {
        key: source[horizon]["gripper_sign_agreement"]
        for key, horizon in (
            ("h1", "immediate"),
            ("h5", "prediction_horizon_5"),
            ("full_horizon", "full_horizon"),
        )
    },
    "p10_reference": {
        "robot0_arm_mae": {"h1": 0.0048300, "h5": 0.00558438, "full_horizon": 0.0119084},
        "closed_loop_h32": {"successes": 0, "episodes": 8, "grasped": 5},
    },
    "interpretation_limit": "Offline action error does not establish closed-loop task success.",
}
(root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
PY

final_state=SUCCEEDED
