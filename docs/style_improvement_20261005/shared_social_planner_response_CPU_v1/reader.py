"""Actual saved-plan associations; no model evaluation or causal attribution."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
LOCOMOTION_SOURCE = ROOT / "scripts/diagnose_shared_social_locomotion.py"
LOCOMOTION_SHA = "2c975fe2b5bbe82435ff29dee2ed9aefe219f91270b168f698eb9eebf8a745be"
EVIDENCE = ROOT / "docs/style_improvement_20261005/shared_social_evaluation_v1"
POST = ROOT / "output/diffusion_amp/shared_social_popart_terminal2000_evaluation_v1"


def raw_velocities(raw):
    if (not isinstance(raw, np.ndarray) or raw.ndim != 3 or raw.shape[1:] != (51, 2)
            or not len(raw) or raw.dtype.kind != "f" or not np.isfinite(raw).all()):
        raise ValueError("Require complete finite raw51 XY plans")
    return np.diff(raw.astype(np.float64), axis=1) * 10


def prefix_speeds(raw):
    velocities = raw_velocities(raw)
    points = raw.astype(np.float64)
    return {f"{seconds:g}s_{kind}_speed_mps": value
            for seconds, knots in ((.5, 5), (5., 50))
            for kind, value in (("net", np.linalg.norm(points[:, knots] - points[:, 0], axis=-1) / seconds),
                                ("arc", np.linalg.norm(velocities[:, :knots], axis=-1).mean(-1)))}


def interval_velocity(raw, selected, phase):
    velocity = raw_velocities(raw)
    if (selected.shape != phase.shape or selected.dtype.kind not in "iu" or phase.dtype.kind not in "iu"
            or not ((selected >= 0) & (selected < len(raw))).all() or not ((phase >= 1) & (phase <= 15)).all()):
        raise ValueError("Invalid saved plan selection or executed interval age")
    # End age3 belongs to raw interval0->1, not the following1->2 interval.
    return velocity[selected, (phase - 1) // 3]


def selected_rows(plans, phase):
    selected = np.empty(phase.shape, dtype=np.int64)
    for actor in range(4):
        rows = np.flatnonzero(plans["actor_ids"] == actor)
        controls = plans["control_step"][rows]
        selected[:, actor] = rows[np.searchsorted(controls, np.arange(6000), side="right") - 1]
    if not np.array_equal(np.arange(6000)[:, None] - plans["control_step"][selected] + 1, phase):
        raise ValueError("Selected raw rows disagree with independently checked target ages")
    return selected


def run(name, navigation, audit_owner, locomotion, helpers):
    result, audit, arrays, inputs = locomotion.admit(navigation, audit_owner)
    original = locomotion.analyze(result, audit, arrays, helpers)
    with np.load(arrays[0]["path"], allow_pickle=False) as archive:
        state = {key: archive[key] for key in ("raw_target_xy", "target_xy", "reference_target_velocity_xy",
                                              "episode_ids", "reset_mask", "terminated", "truncated")}
    with np.load(arrays[2]["path"], allow_pickle=False) as archive:
        plans = {key: archive[key] for key in ("actor_ids", "control_step", "raw_xy", "reference_xy", "reference_velocity_xy")}
    target_checks, phase = helpers[0].targets_from_saved_plans(state, plans)
    selected = selected_rows(plans, phase)
    # The arrays do not save plan episode IDs; associate by actual submission
    # control, and require a freshly submitted plan at every physical reset.
    controls, actors = plans["control_step"], plans["actor_ids"]
    if (not np.isin(actors, np.arange(4)).all() or not ((controls >= 0) & (controls < 6000)).all()
            or not np.array_equal(controls[selected][state["reset_mask"]],
                                  np.broadcast_to(np.arange(6000)[:, None], phase.shape)[state["reset_mask"]])):
        raise ValueError("Invalid plan actor/clock or reset without a new saved plan")
    plan_episodes = state["episode_ids"][controls, actors]
    if not np.array_equal(plan_episodes[selected], state["episode_ids"]):
        raise ValueError("Executed reference uses a previous-episode plan")
    raw_v = interval_velocity(plans["raw_xy"], selected, phase)
    raw_speed = np.linalg.norm(raw_v, axis=-1)
    ref_v = state["reference_target_velocity_xy"].astype(np.float64)
    ref_speed = np.linalg.norm(ref_v, axis=-1)
    metrics = {"raw_interval_speed_mps": raw_speed, "executed_reference_speed_mps": ref_speed,
               "raw_minus_reference_speed_mps": raw_speed - ref_speed,
               "reference_minus_raw_velocity_norm_mps": np.linalg.norm(ref_v - raw_v, axis=-1)}
    control_groups = {}
    masks = {"all": np.ones((6000, 4), bool), "last100s": np.broadcast_to(np.arange(6000)[:, None] >= 3000, (6000, 4)),
             "falls": state["terminated"], "first_postreset": state["reset_mask"]}
    masks.update({f"age{age}": phase == age for age in range(1, 16)})
    for label, mask in masks.items():
        control_groups[label] = {"actor_controls": int(mask.sum()),
                                "metrics": {key: helpers[0].summary(value[mask]) for key, value in metrics.items()},
                                "raw_speed_below_0p1_actor_controls": int((mask & (raw_speed < .1)).sum()),
                                "reference_speed_below_0p1_actor_controls": int((mask & (ref_speed < .1)).sum()),
                                "raw_speed_below_0p5_actor_controls": int((mask & (raw_speed < .5)).sum()),
                                "reference_speed_below_0p5_actor_controls": int((mask & (ref_speed < .5)).sum())}
    prefix = prefix_speeds(plans["raw_xy"])
    plan_groups = {}
    for actor in range(4):
        for label, time_mask in (("all", np.ones(len(controls), bool)), ("last100s", controls >= 3000)):
            mask = (actors == actor) & time_mask
            ref_mask = np.zeros((6000, 4), bool)
            ref_mask[3000 if label == "last100s" else 0:, actor] = True
            plan_groups[f"actor{actor}_{label}"] = {
                "registered_plan_rows": int(mask.sum()),
                "raw_prefix_metrics": {key: helpers[0].summary(value[mask]) for key, value in prefix.items()},
                "raw_0p5s_arc_speed_below_0p5_plan_rows": int((mask & (prefix["0.5s_arc_speed_mps"] < .5)).sum()),
                "executed_control_count": int(ref_mask.sum()),
                "reference_speed_mps": helpers[0].summary(ref_speed[ref_mask]),
                "reference_speed_below_0p5_controls": int((ref_mask & (ref_speed < .5)).sum())}
    for item in inputs:
        if locomotion.record(item["path"]) != item:
            raise ValueError("Actual completed inputs changed during planner CPU observation")
    return {"run": name, "inputs": inputs, "all_actor_controls": original["all_actor_controls"],
            "fall_actor_controls": original["fall_actor_controls"], "all_target_phase_max_abs_errors": target_checks,
            "control_groups": control_groups, "registered_plan_groups": plan_groups,
            "plan_episode_source": "Actual physical episode at registered plan control; not saved pre-control plan state",
            "all_failed_reset_contact_controls_retained": True}


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or hashlib.sha256(LOCOMOTION_SOURCE.read_bytes()).hexdigest() != LOCOMOTION_SHA:
        raise ValueError("Explicit CUDA-hidden process and exact reviewed locomotion reader required")
    from scripts import diagnose_shared_social_locomotion as locomotion
    output = Path(__file__).resolve().parent / "actual_diagnostic.json"
    if output.exists() or output.is_symlink():
        raise ValueError("Preserve original diagnostic output")
    helpers, sources = locomotion.load_helpers()
    sources.extend(locomotion.record(p) for p in (Path(__file__), LOCOMOTION_SOURCE,
                   ROOT / "CrowdSim/humanoid/analysis/support_phases.py", ROOT / "scripts/compare_shared_social_evidence.py"))
    runs = {
        "S0": run("S0", EVIDENCE / "navigation200s_execution.json", EVIDENCE / "navigation200s_CPU_audit_execution.json", locomotion, helpers),
        "S1": run("S1", POST / "navigation200s_execution.json", EVIDENCE / "popart_post_training_execution.json", locomotion, helpers),
    }
    if (any(locomotion.record(item["path"]) != item for item in sources)
            or any(name.split(".")[0] in locomotion.FORBIDDEN for name in sys.modules)):
        raise ValueError("Sources changed or CPU process imported model/simulator modules")
    report = {"schema": "shared_social.completed_planner_response_CPU_association.v1", "status": "completed_CPU_diagnostic",
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "sources": sources, "runs": runs,
              "definition": {"raw_control_velocity": "Same executed interval age1..15; linear raw51@10Hz derivative at lower=(age-1)//3, including left segment at age3/6/.../15",
                             "reference_velocity": "Saved executed reference_velocity at same post-control age+1 target contract",
                             "prefixes": "Registered raw0..0.5s and0..5s; net displacement/horizon separately from summed XY arc length/horizon",
                             "late": "Last100s controls>=3000; plan rows registered at control>=3000; units/counts differ",
                             "failures": "Every actual failure/reset retained; no model calls, altered paths or fitted alignment"},
              "GPU_used": False, "SDK_started": False, "models_loaded": False, "causality_established": False,
              "paper_effect_achieved": False, "default_model_promoted": False}
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "controls_per_run": 24000,
                      "mean_raw_speed": {name: value["control_groups"]["all"]["metrics"]["raw_interval_speed_mps"]["mean"] for name, value in runs.items()}}))


if __name__ == "__main__":
    main()
