#!/usr/bin/env python3
"""CPU stationary/rigid-translation controls on frozen completed S3-R inputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.analysis import amp_kinematic_controls as controls  # noqa: E402
from CrowdSim.humanoid.analysis import amp_temporal_diagnostic as temporal  # noqa: E402
from CrowdSim.humanoid.application import recovery_evaluation_admission as admission  # noqa: E402
from scripts import resume_shared_social_recovery_evaluation as io  # noqa: E402

HERE = ROOT / "docs/style_improvement_20261008/shared_social_S3R_kinematic_discriminator_CPU_v1"
PREVIOUS = ROOT / "docs/style_improvement_20261008/shared_social_S3R_temporal_discriminator_CPU_v1"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    admission.require(not output.exists(), "Preserve prior diagnostic outputs")
    previous = io.json.loads((PREVIOUS / "result.json").read_text())
    execution = io.json.loads((PREVIOUS / "actual_CPU_execution.json").read_text())
    admission.require(execution["diagnostic_actual_exit_code"] == 0
        and admission.readonly_identity(PREVIOUS / "result.json") == execution["result"],
        "Require the actual completed temporal diagnostic")
    for row in previous["sources"]:
        admission.require(admission.record(row["path"]) == row, "Previous numerical diagnostic source changed")
    documents, identities, _ = admission.completed_documents()
    inputs = [*identities.values(), documents["audit"]["checkpoint"], documents["audit"]["rollout"],
              execution["result"], admission.readonly_identity(PREVIOUS / "actual_CPU_execution.json"), previous["scores"]]
    sources = [admission.record(path) for path in (Path(__file__), Path(controls.__file__), Path(temporal.__file__),
        ROOT / "CrowdSim/humanoid/amp/observations.py", ROOT / "CrowdSim/humanoid/training/trainer.py",
        ROOT / "protomotions/utils/rotations.py")]
    live_plan = io.json.loads((ROOT / "docs/style_improvement_20261008/shared_social_episode_after_recovery_v4/owner_plan.json").read_text())
    protected = {Path(row["path"]).resolve() for row in live_plan["sources"] + live_plan["inputs"]}
    admission.require(output != ROOT and output not in protected
        and not any(output == path or output in path.parents for path in protected), "Do not write in the live training closure")
    for row in inputs:
        admission.require(admission.readonly_identity(row["path"]) == row, "Completed diagnostic input changed")
    output.mkdir(parents=True)
    import numpy as np
    import torch

    torch.set_num_threads(2)
    state = torch.load(documents["audit"]["checkpoint"]["path"], map_location="cpu", weights_only=True, mmap=True)
    archive = torch.load(documents["audit"]["rollout"]["path"], map_location="cpu", weights_only=True, mmap=True)
    admission.require(state["update_count"] == 12500 and state["physical_transition_count"] == 102400000,
                      "Select the preselected actual completed S3-R checkpoint")
    proxy = temporal.load_discriminator(state)
    rollout, recorded, expert = archive["collection"]["rollout"], archive["collection"]["diagnostics"], archive["expert"]
    admission.require(bool(torch.equal(rollout["history_episode_ids"], rollout["episode_ids"].unsqueeze(-1).expand(-1, -1, 10))),
                      "Actual PRE histories must remain within their recorded episodes")
    populations = {"physical_policy": (rollout["amp_history"].reshape(-1, 10, 195), rollout["applied_actions"].reshape(-1, 69)),
                   "actual_AMASS_expert": (expert["history"], expert["actions"])}
    admission.require(len(populations["physical_policy"][0]) == 8192 and len(expert["history"]) == 4096,
                      "Keep every actual physical/expert window, including failures")
    body_state = SimpleNamespace(**{name: recorded["common_" + name].reshape(-1, *recorded["common_" + name].shape[2:])
                                   for name in controls.BODY_FIELDS})
    oracle = {}
    # An independent full-state re-encoding uses POST snapshots solely to
    # check the feature-layout intervention. Scoring below stays on PRE inputs.
    for translating, name in ((False, "stationary_pose"), (True, "rigid_translation_pose")):
        current, encoded = controls.encode_pose_sequence(body_state, translating=translating)
        expected = controls.controls(current[:, None].expand(-1, 10, -1))[name]
        difference = (encoded - expected).abs()
        oracle[name] = {"physical_pose_rows": len(current), "encoded_history_rows": len(current) * 10,
            "max_abs_encoder_minus_feature_control": float(difference.max()),
            "max_abs_outside_key_positions": float(difference[..., :183].max()),
            "max_abs_CPU_encoder_vs_GPU_POST_current": float((current - recorded["post_amp_history"].reshape(-1, 10, 195)[:, 0]).abs().max()),
            "translation_position_subtraction_can_round_float32": translating,
            "dynamic_feasibility_verified": False}
        admission.require(bool(torch.isfinite(encoded).all()) and float(difference[..., :183].max()) <= 1e-5,
                          "Kinematic oracle changed a non-position feature unexpectedly")
    summaries, arrays = {}, {}
    with np.load(previous["scores"]["path"], allow_pickle=False) as baseline:
        for role, (history, actions) in populations.items():
            summaries[role], scored = controls.score(proxy, history, actions)
            for name, values in scored.items():
                for kind, value in values.items():
                    key = role + "__" + name + "__" + kind
                    arrays[key] = value.numpy()
                    if name in ("identity", "repeat_current"):
                        admission.require(np.array_equal(arrays[key], baseline[key]), "Original baseline scores changed")
    np.savez_compressed(output / "scores.npz", **arrays)
    for row in inputs + sources:
        admission.require(admission.record(row["path"]) == row, "Input/source changed during CPU scoring")
    admission.require(torch.cuda.is_initialized() is False
        and not any(name.startswith(("isaacsim", "isaaclab", "omni.")) for name in sys.modules), "CPU-only diagnostic required")
    report = {"schema": "shared_social.frozen_S3R_kinematic_discriminator_diagnostic.v1", "status": "passed",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "sources": sources, "inputs": inputs,
        "scores": admission.record(output / "scores.npz"), "summaries": summaries, "kinematic_oracle": oracle,
        "original_identity_and_repeat_scores_match_previous_exactly": True,
        "scored_history_action_pairs": 49152, "physical_policy_windows": 8192, "expert_windows": 4096,
        "actual_expert_unique_motion_ids": int(expert["motion_ids"].unique().numel()),
        "scoring_phase": previous["scoring_phase"], "oracle_phase": "POST raw bodies; never used as scored PRE history",
        "controls": {"stationary_pose": "constant current pose; root linear/angular and selected DOF velocities zero",
                     "rigid_translation_pose": "constant current pose; root linear velocity kept, angular/DOF velocities zero"},
        "actions": "original actual applied physical69/expert proxy69 kept under every control",
        "contacts_force_balance_or_dynamic_feasibility_verified": False,
        "heldout_quality_evaluation": False, "original_logged_AMP_reward_trace_reproduced": False,
        "causal_training_failure_claimed": False, "current_training_recipe_changed": False,
        "Torch_version": torch.__version__, "CUDA_initialized": False, "SDK_started": False,
        "paper_effect_achieved": False, "default_model_promoted": False}
    (output / "scores.npz").chmod(0o444)
    io.write_json(output / "result.json", report, readonly=True)
    print(io.json.dumps({"status": "passed", "result": admission.record(output / "result.json"),
                        "summaries": summaries, "kinematic_oracle": oracle}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
