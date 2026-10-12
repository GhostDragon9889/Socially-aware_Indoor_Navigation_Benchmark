#!/usr/bin/env python3
"""Read completed physical captures for support and command-response diagnostics.

This CPU reader includes failures. Same-control root/reference distances are
not the paper's unpublished global Path Error, and threshold onsets are not
validated footsteps. It never loads a policy or launches the simulator.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.analysis.support_phases import support_phases  # noqa: E402
from scripts.compare_shared_social_evidence import bound, record, require  # noqa: E402

HELPERS = (
    ("resources/runtime_sources/style_improvement_20261004/actual_tracking_contact_CPU_v1/diagnose_tracking_contact.py",
     "feddba4fa4a526c2c7295c485f4ba513180c77369e2bd43a2ceec5f0c32896b0"),
    ("resources/runtime_sources/style_improvement_20261004/reference_response_CPU_v1/diagnose_response.py",
     "180b8a54f9ef1291111389c2ab751718bbca0c37f954a70653f384491a8585ec"),
)
FORBIDDEN = {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"}
STATE_FIELDS = ("root_pos", "root_rot", "target_xy", "raw_target_xy", "reference_target_velocity_xy",
                "episode_ids", "reset_mask", "terminated", "truncated", "static_contact", "peer_contact",
                "reference_deviation_m", "route_vertex_reached")
GROUND_FIELDS = ("body_link_pos_w", "body_link_quat_w", "body_link_lin_vel_w", "common_body_names",
                 "control_step", "control_step_after", "physics_step_before", "physics_step_end", "physics_step",
                 "substep_valid_count", "actor_substep_valid_count", "valid", "episode_ids", "reset_mask",
                 "terminated", "truncated", "foot_names", "foot_floor_normal_force_w", "foot_contact_substeps",
                 "side_contact_substeps")


def load_helpers():
    modules, identities = [], []
    for index, (relative, digest) in enumerate(HELPERS):
        path = ROOT / relative
        source = path.read_bytes()
        require(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1
                and not path.stat().st_mode & 0o222 and hashlib.sha256(source).hexdigest() == digest,
                "Original frozen CPU response helper changed")
        module = types.ModuleType(f"frozen_locomotion_helper_{index}")
        module.__file__ = str(path)
        exec(compile(source, str(path), "exec"), module.__dict__)
        modules.append(module)
        identities.append(record(path))
    return modules, identities


def admit(navigation_receipt, audit_owner):
    identities = [record(navigation_receipt), record(audit_owner)]
    navigation, owner = (bound(identity) for identity in identities)
    require(navigation.get("schema") == "shared_social.actual_navigation200s_execution.v1"
            and navigation.get("actual_exit_code") == 0 and owner.get("actual_exit_code") == 0,
            "Require separately observed actual navigation and CPU-audit exits")
    if owner.get("schema") == "shared_social.actual_navigation_CPU_audit_execution.v1":
        audit_identity = owner["audit"]
    elif owner.get("schema") == "shared_social.actual_post_training_owner_exit.v1":
        pipeline = bound(owner["pipeline"])
        identities.append(owner["pipeline"])
        require(pipeline.get("status") == "passed_execution"
                and len(pipeline["steps"]) == 5
                and all(step.get("status") == "passed" and step.get("actual_exit_code") == 0
                        for step in pipeline["steps"])
                and owner["measurements"] == pipeline["measurements"],
                "Actual post-training child exits or measurement ownership differ")
        by_name = {step["name"]: step for step in pipeline["steps"]}
        audit_identity = owner["measurements"]["navigation_CPU_audit"]
        require(by_name["navigation200s"]["result"] == navigation["result"]
                == owner["measurements"]["navigation"]
                and by_name["navigation_CPU_array_audit"]["result"] == audit_identity,
                "Navigation/audit identities differ from their actual pipeline children")
    else:
        raise ValueError("Unsupported separately observed CPU audit owner")
    result, audit = bound(navigation["result"]), bound(audit_identity)
    identities.extend([navigation["result"], audit_identity])
    require(audit.get("status") == "passed" and audit["result"] == navigation["result"]
            and audit["execution_receipt"] == identities[0]
            and result.get("status") == "passed" and result.get("cleanup_succeeded") is True
            and result.get("cleanup_errors") == [] and result["actual_control_steps"] == 6000
            and result["physics"]["control_hz"] == 30 and result["physics"]["physics_hz"] == 120,
            "Complete same-protocol physical/audit result required")
    run = Path(navigation["result"]["path"]).parent
    arrays = [record(run / name) for name in ("physical_states.npz", "ground_support.npz", "reference_plans.npz")]
    require(arrays == audit["arrays"], "Actual raw arrays differ from completed CPU audit")
    for identity in arrays:
        path = Path(identity["path"])
        require(not path.is_symlink() and path.stat().st_nlink == 1 and not path.stat().st_mode & 0o222,
                "Require sealed independent completed raw arrays")
    identities.extend(arrays)
    return result, audit, arrays, identities


def analyze(result, audit, arrays, helpers):
    with np.load(arrays[0]["path"], allow_pickle=False) as archive:
        state = {key: archive[key] for key in STATE_FIELDS}
    with np.load(arrays[1]["path"], allow_pickle=False) as archive:
        ground = {key: archive[key] for key in GROUND_FIELDS}
    with np.load(arrays[2]["path"], allow_pickle=False) as archive:
        plans = {key: archive[key] for key in ("actor_ids", "control_step", "raw_xy", "reference_xy", "reference_velocity_xy")}
    bools = {"reset_mask", "terminated", "truncated", "static_contact", "peer_contact", "route_vertex_reached"}
    for key, value in state.items():
        shape = ((6000, 4, 3) if key == "root_pos" else (6000, 4, 4) if key in ("root_rot", "peer_contact")
                 else (6000, 4, 2) if key in ("target_xy", "raw_target_xy", "reference_target_velocity_xy") else (6000, 4))
        require(value.shape == shape and np.isfinite(value).all()
                and (key not in bools or value.dtype == np.bool_)
                and (key != "episode_ids" or value.dtype.kind in "iu"), "Invalid physical channel: " + key)
    names = ground["common_body_names"].tolist()
    require(names == result["state_capture"]["body_names"] and len(names) == 24 and names[0] == "Pelvis",
            "Require actual common SMPL body mapping")
    for key, width in (("body_link_pos_w", 3), ("body_link_quat_w", 4), ("body_link_lin_vel_w", 3)):
        require(ground[key].shape == (6000, 4, 4, 24, width) and ground[key].dtype == np.float32
                and np.isfinite(ground[key]).all(), "Invalid native body channel: " + key)
    before = int(ground["physics_step_before"][0])
    expected = {"control_step": np.arange(6000), "control_step_after": np.arange(1, 6001),
                "physics_step_before": before + np.arange(6000) * 4,
                "physics_step_end": before + np.arange(1, 6001) * 4,
                "physics_step": before + np.arange(1, 24001).reshape(6000, 4),
                "substep_valid_count": np.full(6000, 4), "actor_substep_valid_count": np.full((6000, 4), 4),
                "valid": np.ones(6000, bool),
                **{key: state[key] for key in ("episode_ids", "reset_mask", "terminated", "truncated")}}
    require(all(np.array_equal(ground[key], value) and ground[key].dtype.kind == value.dtype.kind
                for key, value in expected.items()), "Native control, substep, validity or reset clocks differ")
    delta = np.diff(state["episode_ids"], axis=0)
    require(np.isin(delta, (0, 1)).all() and state["reset_mask"][0].all()
            and np.array_equal(delta == 1, state["reset_mask"][1:])
            and np.array_equal((state["terminated"] | state["truncated"])[:-1], state["reset_mask"][1:]),
            "Episode/reset/terminal phase mismatch")
    require(np.array_equal(ground["body_link_pos_w"][:, -1, :, 0], state["root_pos"])
            and np.array_equal(ground["body_link_quat_w"][:, -1, :, 0], state["root_rot"])
            and np.allclose(np.linalg.norm(state["root_rot"], axis=-1), 1., atol=1e-3, rtol=0),
            "Native fourth-substep pelvis differs from same-control root or is not unit XYZW")
    require(ground["foot_names"].tolist() == ["L_Ankle", "L_Toe", "R_Ankle", "R_Toe"]
            and result["physical_contact_measurement"]["threshold_n"] == 1., "Contact mapping or threshold differs")
    force = ground["foot_floor_normal_force_w"]
    require(force.shape == (6000, 4, 4, 4, 1, 3) and force.dtype == np.float32
            and np.isfinite(force).all(), "Invalid native all-substep floor force")
    foot = np.linalg.norm(force.sum(axis=-2), axis=-1) > 1.
    side = foot.reshape(6000, 4, 4, 2, 2).any(-1)
    require(ground["foot_contact_substeps"].dtype == np.bool_ and ground["side_contact_substeps"].dtype == np.bool_
            and np.array_equal(foot, ground["foot_contact_substeps"])
            and np.array_equal(side, ground["side_contact_substeps"]), "Saved support flags differ from native 1N force")
    target_checks, phase = helpers[0].targets_from_saved_plans(state, plans)
    metrics, valid, direction, heading, idle = helpers[1].responses(state, ground["body_link_lin_vel_w"][:, -1, :, 0, :2])
    deviation = np.linalg.norm(state["target_xy"] - state["raw_target_xy"], axis=-1)
    require(np.allclose(deviation, state["reference_deviation_m"], atol=1e-5, rtol=0),
            "Saved adaptation deviation is not target-minus-raw-target")
    require(int(state["terminated"].sum()) == audit["fall_termination_actor_controls"]
            and state["route_vertex_reached"].sum(0).tolist() == audit["route_vertices_reached"]
            and int(valid.sum()) == audit["same_episode_position_intervals"], "Raw response counts differ from actual audit")
    metrics["causal_reference_minus_raw_target_m"] = deviation
    moving = metrics["reference_speed_mps"] >= .5
    stationary = metrics["native_speed_mps"] < .1
    all_metrics = {key: {**helpers[0].summary(value[np.isfinite(value)]),
                         "undefined_actor_controls": int((~np.isfinite(value)).sum())} for key, value in metrics.items()}
    return {"all_actor_controls": 24000, "all_target_phase_max_abs_errors": target_checks,
            "plan_age_control_counts": {str(age): int((phase == age).sum()) for age in range(1, 16)},
            "all_control_metrics": all_metrics, "reference_direction_undefined_actor_controls": int((~direction).sum()),
            "root_heading_undefined_actor_controls": int((~heading).sum()),
            "same_episode_previous_position_intervals": int(valid.sum()), "idle_previous_position_intervals": int(idle.sum()),
            "previous_position_intervals_undefined": int((~valid).sum()),
            "moving_command_actor_controls": int(moving.sum()),
            "moving_command_and_stationary_native_root_actor_controls": int((moving & stationary).sum()),
            "moving_stationary_fraction_of_moving_commands": float((moving & stationary).sum() / moving.sum()) if moving.any() else None,
            "moving_command_stationary_rule": "reference XY speed>=.5m/s and native fourth-substep pelvis XY speed<.1m/s; all failures retained",
            "native_support": support_phases(side, state["episode_ids"]),
            "fall_actor_controls": int(state["terminated"].sum()), "route_vertices_reached": state["route_vertex_reached"].sum(0).tolist(),
            "all_failed_contact_idle_reset_frames_retained": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--navigation-receipt", type=Path, required=True)
    parser.add_argument("--audit-owner-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Explicit CUDA-hidden CPU process required")
    require(not args.output.exists() and not args.output.is_symlink(), "Preserve existing diagnostics; use a new output")
    helpers, source_ids = load_helpers()
    source_ids.extend(record(path) for path in (Path(__file__), ROOT / "CrowdSim/humanoid/analysis/support_phases.py",
                                               ROOT / "scripts/compare_shared_social_evidence.py", ROOT / "CrowdSim/humanoid/foot_support.py"))
    result, audit, arrays, identities = admit(args.navigation_receipt, args.audit_owner_receipt)
    diagnostic = analyze(result, audit, arrays, helpers)
    require(not any(name.split(".")[0] in FORBIDDEN for name in sys.modules), "CPU reader imported model/simulator modules")
    require(all(record(identity["path"]) == identity for identity in source_ids + identities), "Inputs changed during CPU observation")
    report = {"schema": "shared_social.completed_locomotion_CPU_diagnostic.v1", "status": "completed_CPU_diagnostic",
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "sources": source_ids, "inputs": identities,
              "measurement": diagnostic, "CPU_elapsed_s": time.perf_counter() - started,
              "definition": "All6000 controls x4actors; same-control physical root and saved reference at plan age+1; native fourth120Hz substep velocity; original frozen response formulas",
              "limits": ["Reference/raw-target root distances are not paper global Path Error.",
                         "Threshold crossings are not validated footsteps; no debounce or failure filtering.",
                         "Longest double support is observed episode-bound duration, not gait classification.",
                         "No lag fit, causal inference, author equivalence or superiority claim."],
              "GPU_used": False, "SDK_started": False, "weights_loaded": False,
              "paper_effect_achieved": False, "default_model_promoted": False}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "falls": diagnostic["fall_actor_controls"],
                      "moving_stationary_fraction": diagnostic["moving_stationary_fraction_of_moving_commands"],
                      "double_support_fraction": diagnostic["native_support"]["phase_fractions"]["double_support"]}))


if __name__ == "__main__":
    main()
