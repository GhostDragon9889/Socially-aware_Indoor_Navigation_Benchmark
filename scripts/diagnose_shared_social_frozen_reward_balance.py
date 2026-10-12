#!/usr/bin/env python3
"""Compare saved frozen-D gains with bounded task terms; no dynamics inference."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_shared_social_training import identity  # noqa: E402

DIAGNOSTIC = ROOT / "resources/runtime_sources/style_improvement_20261008/shared_social_S3R_kinematic_discriminator_CPU_v1/result.json"
DIAGNOSTIC_SHA = "7bde143b7f5c9d5021f7710171bc540d1936e52706bfa7063b986603c5704e40"
TRAINING = ROOT / "output/diffusion_amp/shared_social_budget12500_recovery_training_v2/result.json"
TRAINING_SHA = "87bf93cbe6c67a2d9e852cd7b8d9b095de525fe97019e6747fab4ec1427467bb"
REWARDS = ROOT / "CrowdSim/humanoid/training/rewards.py"
REWARDS_SHA = "fc595dd9c1b2012aa76f37dcecd4ec5d7dad50fbd923840736a77c24622a6430"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    here = args.output.resolve()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not here.exists(), "Fresh CPU-only output required")
    inputs = [identity(DIAGNOSTIC), identity(TRAINING)]
    require([row["sha256"] for row in inputs] == [DIAGNOSTIC_SHA, TRAINING_SHA], "Require the actual completed S3-R evidence")
    sources = [identity(REWARDS), identity(Path(__file__)), identity(ROOT / "scripts/diagnose_shared_social_training.py")]
    require(sources[0]["sha256"] == REWARDS_SHA, "Task equations changed")
    diagnostic, training = (json.loads(path.read_text()) for path in (DIAGNOSTIC, TRAINING))
    config = training["actual_configuration"]["configuration"]
    require(training["status"] == "passed_local_training" and training["training_loop_completed"] is True
            and training["cleanup_succeeded"] is True and training["cleanup_errors"] == []
            and training["completed_updates"] == 12500 and training["actual_training_transitions"] == 102400000
            and config["amp_reward_weight"] == 1. and config["task_reward_weight"] == .5
            and config["amp_reward_scale"] == 2. and config["direction_mode"] == "backwards_penalty",
            "Require the preserved actual S3-R reward recipe and completion")
    require(diagnostic["status"] == "passed" and diagnostic["physical_policy_windows"] == 8192
            and diagnostic["expert_windows"] == 4096 and diagnostic["original_logged_AMP_reward_trace_reproduced"] is False
            and diagnostic["heldout_quality_evaluation"] is False and diagnostic["current_training_recipe_changed"] is False,
            "Require the admitted complete frozen-D kinematic diagnostic")
    score_record = identity(diagnostic["scores"]["path"])
    require(score_record == diagnostic["scores"], "Saved frozen-D scores changed")
    inputs.append(score_record)
    scores = np.load(score_record["path"], allow_pickle=False)
    trajectory_bound = config["task_reward_weight"]  # exp(-2 delta), delta>=0, lies in (0,1].
    trajectory_direction_bound = 2 * trajectory_bound  # backwards_penalty lies in [-1,0].
    populations = {}
    for population, count in (("physical_policy", 8192), ("actual_AMASS_expert", 4096)):
        identity_scores = scores[population + "__identity__rewards"]
        require(identity_scores.shape == (count,) and identity_scores.dtype == np.float32
                and np.isfinite(identity_scores).all(), "Identity population incomplete")
        conditions = {}
        for condition in ("repeat_current", "stationary_pose", "rigid_translation_pose"):
            values = scores[population + "__" + condition + "__rewards"]
            require(values.shape == (count,) and values.dtype == np.float32 and np.isfinite(values).all(),
                    "Condition population incomplete")
            gain = config["amp_reward_weight"] * (values.astype(np.float64) - identity_scores.astype(np.float64))
            conditions[condition] = {"windows": count, "mean_weighted_AMP_gain": float(gain.mean()),
                "weighted_AMP_gain_quantiles": dict(zip(("min", "p05", "p25", "p50", "p75", "p95", "max"),
                    np.quantile(gain, (0., .05, .25, .5, .75, .95, 1.)).tolist(), strict=True)),
                "gain_exceeds_max_weighted_trajectory_improvement_count": int(np.count_nonzero(gain > trajectory_bound)),
                "gain_exceeds_max_weighted_trajectory_improvement_fraction": float(np.mean(gain > trajectory_bound)),
                "gain_exceeds_max_weighted_trajectory_direction_improvement_count": int(np.count_nonzero(gain > trajectory_direction_bound)),
                "gain_exceeds_max_weighted_trajectory_direction_improvement_fraction": float(np.mean(gain > trajectory_direction_bound))}
        populations[population] = conditions
    require(all(identity(row["path"]) == row for row in sources + inputs), "Sources or inputs changed during the read")
    require("torch" not in sys.modules and not any(name.split(".")[0] in {"isaacsim", "isaaclab", "omni", "pxr", "warp"}
                                                  for name in sys.modules), "CPU accounting imported Torch or an SDK")
    report = {"schema": "shared_social.frozen_reward_balance_CPU_accounting.v1", "status": "passed_saved_score_accounting",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "sources": sources, "inputs": inputs,
        "actual_recipe": {key: config[key] for key in ("amp_reward_weight", "task_reward_weight", "amp_reward_scale", "direction_mode")},
        "max_weighted_trajectory_improvement": trajectory_bound,
        "max_weighted_trajectory_direction_improvement_excluding_energy": trajectory_direction_bound,
        "bounds": "A difference between any two exp(-2delta) values cannot exceed1; backwards_penalty has range[-1,0]. Signed torque-power energy is excluded and has no bound inferred here.",
        "populations": populations, "scoring_phase": diagnostic["scoring_phase"],
        "limits": "Saved post-update12500 frozen D, not logged training rewards or episode2000 candidate. Same current actions; kinematic controls do not establish realizable contact/torque dynamics. Even gains above bounds do not prove total reward dominance or causation. No scales are changed or new policies trained.",
        "all_saved_windows_retained": True, "Torch_imported": False, "SDK_started": False,
        "total_reward_dominance_proven": False, "causal_training_failure_proven": False,
        "current_training_recipe_changed": False, "default_model_promoted": False, "paper_effect_achieved": False}
    here.mkdir(parents=True)
    result = here / "result.json"
    result.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    result.chmod(0o444)
    print(json.dumps({"result": identity(result), "populations": populations, "Torch_imported": False, "SDK_started": False}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
