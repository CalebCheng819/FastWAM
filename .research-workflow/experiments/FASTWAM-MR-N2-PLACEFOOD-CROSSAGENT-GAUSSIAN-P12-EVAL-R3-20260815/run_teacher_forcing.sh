#!/usr/bin/env bash
set -euo pipefail

experiment_id=FASTWAM-MR-N2-PLACEFOOD-CROSSAGENT-GAUSSIAN-P12-PAIRED-TF-R3-20260815
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
eval_root=$(git -C "$script_dir" rev-parse --show-toplevel)
model_root=${P12_MODEL_ROOT:-/mnt/workspace/experiments/FastWAM-p12-render-1181a37-20260814}
train_root=${P12_TRAIN_ROOT:-/oss-chengjuntao/artifacts/fastwam-placefood-crossagent-gaussian-p12-s42-8g-r2-20260814}
output=${P12_TF_OUTPUT_ROOT:-/oss-chengjuntao/artifacts/fastwam-placefood-crossagent-gaussian-p12-paired-tf-20260815-r3}
python=${P12_EVAL_PYTHON:-/mnt/workspace/venvs/fastwam-py310/bin/python}
python_extra=${P12_PYTHON_EXTRA:-/mnt/workspace/venvs/fastwam-gau0-eval-r7-py310-extra-20260813}
panel=${P12_EVAL_PANEL:-/mnt/workspace/fastwam_eval_runtime/panels/robofactory_n234_s42_val8_v1.json}
dataset=${P12_DATASET_ROOT:-/oss-chengjuntao/cpfs-user-chengjuntao/datasets/robofactory_multi_robot}
robofactory=${P12_ROBOFACTORY_ROOT:-/mnt/workspace/fastwam_eval_runtime/RoboFactory-challenge-2d34fb3}
stats=${P12_STATS:-/oss-chengjuntao/artifacts/fastwam-nohash-inputs-20260809/fastwam_multi_robot_n234_train_s42_stats_cpfs_nohash_v1.json}
context=${P12_CONTEXT_FILE:-/oss-chengjuntao/cpfs-user-chengjuntao/datasets/robofactory_multi_robot/text_embeds_cache_n234_named_20260811/PlaceFood-rf.t5_len128.wan22ti2v5b.pt}
model_cache=${P12_MODEL_CACHE_ROOT:-/oss-chengjuntao/cpfs-user-chengjuntao/checkpoints/FastWAM/model-cache}
gaussian_cache=${P12_GAUSSIAN_CACHE:-/oss-chengjuntao/fastwam-gaudp/robofactory_multi_robot/v2/noposplat-c944b498-4a35bc8c/builds/fastwam-8a035024af96-s42-20260801T230944Z/compact-s42-13x28x40-fp16-meanalpha-v2}
policy_lightning=${P12_POLICY_LIGHTNING_ROOT:-/mnt/workspace/Policy-Lightning}
noposplat_checkpoint=${P12_NOPOSPLAT_CHECKPOINT:-/mnt/workspace/checkpoints/noposplat/664ba9156f10a6203f0a0fad2f02c069c6894f4f/mixRe10kDl3dv_512x512.ckpt}
train_commit=1181a375c880a4a51df2ae78d533e16dde757465
eval_commit=$(git -C "$eval_root" rev-parse HEAD)
gpu=${P12_TF_GPU:-0}
steps=(000500 001000)

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
test -d "$python_extra"
test -f "$eval_root/experiments/robofactory/diagnose_place_food_fixed.py"
test "$(git -C "$model_root" rev-parse HEAD)" = "$train_commit"
test -z "$(git -C "$model_root" status --short)"
test -z "$(git -C "$eval_root" status --short)"
for path in \
  "$panel" "$dataset" "$robofactory" "$stats" "$context" "$model_cache" \
  "$gaussian_cache" "$gaussian_cache/COMPLETE" "$policy_lightning" \
  "$noposplat_checkpoint"; do
  test -e "$path"
done
test "$(git -C "$policy_lightning" rev-parse HEAD)" = c944b4989a89c99c69d2572ea870f6a04680f5e7
test -z "$(git -C "$policy_lightning" status --short)"
for step in "${steps[@]}"; do
  checkpoint="$train_root/checkpoints/weights/step_${step}.pt"
  test -s "$checkpoint"
  test -s "$checkpoint.COMPLETE"
done

mkdir -p "$output/logs"
for step in "${steps[@]}"; do
  checkpoint="$train_root/checkpoints/weights/step_${step}.pt"
  env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    WANDB_MODE=offline \
    PYTHONPATH="$python_extra:$model_root/src:$eval_root/experiments/robofactory:$eval_root:$policy_lightning:$robofactory:${PYTHONPATH:-}" \
    "$python" -B "$eval_root/experiments/robofactory/diagnose_place_food_fixed.py" \
      --mode teacher-forcing \
      --formal-contract \
      --task PlaceFood-rf \
      --panel "$panel" \
      --dataset-root "$dataset" \
      --robofactory-root "$robofactory" \
      --gaussian-cache "$gaussian_cache" \
      --output-dir "$output/p12_step${step}" \
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
      --action-architecture cross_agent_gaussian_v4 \
      --gaussian-source noposplat \
      --stats "$stats" \
      --context-file "$context" \
      --model-cache-root "$model_cache" \
      --policy-lightning-repo "$policy_lightning" \
      --noposplat-checkpoint "$noposplat_checkpoint" \
      --device cuda:0 \
      --teacher-device cuda:0 \
      --action-horizon 32 \
      --num-inference-steps 20 \
      >"$output/logs/p12_step${step}.log" 2>&1
done

python3 - "$output" "$train_root" "$eval_commit" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
train_root = Path(sys.argv[2])
results = {}
for step in ("000500", "001000"):
    cell = root / f"p12_step{step}"
    summary = json.loads((cell / "summary.json").read_text())
    teacher = summary.get("teacher_forcing") or {}
    contract = teacher.get("formal_contract") or {}
    assert summary.get("status") == "COMPLETED", summary.get("status")
    assert teacher.get("status") == "completed", teacher.get("status")
    assert teacher.get("states_evaluated") == 263, teacher.get("states_evaluated")
    assert contract.get("valid_pairs_h1") == 263, contract
    assert contract.get("valid_pairs_h5") == 1305, contract
    metrics = json.loads(
        (cell / "teacher_forcing" / "phase_action_error_summary.json").read_text()
    )
    source = metrics["sources"]["all_states"]["live_denormalized_vs_expert"]

    def group(horizon, name):
        value = source[horizon]["by_agent_and_group"][f"panda-0/{name}"]["mae"]
        assert math.isfinite(value) and value >= 0.0, (step, horizon, name, value)
        return value

    checkpoint = train_root / "checkpoints" / "weights" / f"step_{step}.pt"
    results[f"step_{step}"] = {
        "checkpoint": {"path": str(checkpoint), "bytes": checkpoint.stat().st_size},
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
    }

comparison = {
    "schema_version": "fastwam-p12-offline-comparison-v1",
    "status": "COMPLETED",
    "training_job_id": "dlc19rgpvuxr56b7",
    "training_code_commit": "1181a375c880a4a51df2ae78d533e16dde757465",
    "evaluation_code_commit": sys.argv[3],
    "action_architecture": "cross_agent_gaussian_v4",
    "gaussian_source": "noposplat",
    "online_at_deployment": True,
    "checkpoints": results,
    "p10_reference": {
        "robot0_arm_mae": {
            "h1": 0.0048300,
            "h5": 0.00558438,
            "full_horizon": 0.0119084,
        },
        "closed_loop_h32": {"successes": 0, "episodes": 8, "grasped": 5},
    },
    "interpretation_limit": "Offline action error does not establish closed-loop task success.",
}
(root / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
PY

final_state=SUCCEEDED
