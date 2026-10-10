#!/usr/bin/env python3
"""Descriptive gait checks from complete, measured120Hz foot-floor contacts.

These are local diagnostics, not Nav scores or a visual realism certificate.
No frames are removed. A stable exclusive-support phase must last>=50ms;
short phases remain in the raw support fractions and phase-duration summaries.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runs(values):
    bounds = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(values)]
    return [(int(values[start]), int(start), int(end)) for start, end in zip(bounds[:-1], bounds[1:])]


def summary(values):
    values = np.asarray(values)
    return None if not len(values) else {
        "count": len(values), "mean": float(values.mean()),
        "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.execution.read_text())
    result_path = args.run / "result.json"
    result = json.loads(result_path.read_text())
    if (receipt.get("actual_returncode") != 0
            or receipt.get("actual_returncode_source") != "subprocess.Popen.wait"
            or receipt.get("sources_unchanged") is not True
            or receipt.get("inputs_unchanged") is not True
            or receipt.get("result_sha256") != digest(result_path)
            or result.get("status") != "passed"
            or result.get("cleanup_succeeded") is not True
            or result["actual_controls"] != result["requested_controls"]):
        raise ValueError("Full source-bound completed physical run required")
    paths = {name: args.run / name for name in ("physical.npz", "ground_support.npz")}
    for name, path in paths.items():
        if receipt.get("capture_sha256", {}).get(name) != digest(path):
            raise ValueError("Capture differs from completed actual parent: " + name)
    with np.load(paths["physical.npz"], allow_pickle=False) as physical, np.load(paths["ground_support.npz"], allow_pickle=False) as ground:
        n = result["actual_controls"]
        names = list(ground["common_body_names"])
        pelvis = names.index("Pelvis")
        if not (len(ground["control_step"]) == n
                and (ground["substep_valid_count"] == 4).all()
                and np.array_equal(ground["control_step"], np.arange(n))
                and np.array_equal(ground["control_step_after"], np.arange(1, n+1))
                and np.array_equal(physical["body_positions"], ground["body_link_pos_w"][:, -1])
                and np.array_equal(physical["root_rotations_xyzw"], ground["body_link_quat_w"][:, -1, :, pelvis])
                and names == result["physics"]["body_names"]
                and not ground["reset_mask"][1:].any()
                and not ground["terminated"].any()
                and not ground["truncated"].any()):
            raise ValueError("Actual physical/support pose, clock or reset alignment differs")
        side = ground["side_contact_substeps"].copy()
        foot = ground["foot_contact_substeps"].copy()
        velocity = ground["foot_body_origin_vel_w"].copy()
        root_rotation = ground["body_link_quat_w"][:, :, :, pelvis].copy()
        root_velocity = ground["body_link_lin_vel_w"][:, :, :, pelvis, :2].copy()
    if side.dtype != bool or side.shape != (n, 4, len(result["initial_reference_motion_ids"]), 2):
        raise ValueError("Boolean native contact clock[T,4,B,2] required")
    if not np.isfinite(velocity).all():
        raise ValueError("Nonfinite measured foot body-origin velocity")
    actors = []
    for actor in range(side.shape[2]):
        q = root_rotation[:, :, actor].reshape(-1, 4)
        v = root_velocity[:, :, actor].reshape(-1, 2)
        x, y, z, w = q.T
        forward = np.stack((1-2*(y*y+z*z), 2*(x*y+w*z)), -1)
        forward_norm = np.linalg.norm(forward, axis=-1)
        if not (np.isfinite(q).all() and np.isfinite(v).all()
                and (forward_norm > 1e-6).all()):
            raise ValueError("Finite native pelvis rotation/velocity and defined forward projection required")
        forward /= forward_norm[:, None]
        root_speed = np.linalg.norm(v, axis=-1)
        forward_speed = (forward*v).sum(-1)
        moving = root_speed >= .2
        backward = moving & (forward_speed < -.1)
        alignment = forward_speed[moving] / root_speed[moving]
        backward_phases = [end-start for code, start, end in runs(backward.astype(int)) if code == 1]
        contact = side[:, :, actor].reshape(-1, 2)
        code = contact[:, 0].astype(int) + 2 * contact[:, 1].astype(int)
        phases = runs(code)
        stable_single = [(value, start, end) for value, start, end in phases if value in (1, 2) and end-start >= 6]
        alternating = [(a, b) for a, b in zip(stable_single[:-1], stable_single[1:]) if a[0] != b[0]]
        durations = [(b[1]-a[1])/120 for a, b in alternating]
        speed = np.linalg.norm(velocity[:, :, actor, :, :2], axis=-1)
        supported = foot[:, :, actor]
        actors.append({
            "actor": actor,
            "all_substeps_included": len(code),
            "support_fraction": dict(zip(("none", "left_only", "right_only", "double"),
                                         [float(np.mean(code == value)) for value in range(4)])),
            "support_phase_duration_s": {
                str(value): summary([(end-start)/120 for v, start, end in phases if v == value]) for value in range(4)
            },
            "stable_single_support_minimum_duration_s": .05,
            "stable_exclusive_phases": len(stable_single),
            "alternating_stable_single_support_pairs": len(alternating),
            "opposite_single_support_onset_interval_s": summary(durations),
            "median_opposite_support_onset_frequency_hz": None if not durations else float(1/np.median(durations)),
            "supported_foot_body_origin_speed_proxy_m_s": summary(speed[supported]),
            "actual_pelvis_direction": {
                "source": "All120Hz native pelvis quaternions and XY linear velocities; local+X forward projection",
                "substeps_included": len(q),
                "moving_speed_threshold_m_s": .2,
                "backward_forward_speed_threshold_m_s": -.1,
                "moving_substeps": int(moving.sum()),
                "moving_alignment_cosine": summary(alignment),
                "backward_fraction_all_substeps": float(backward.mean()),
                "backward_fraction_moving_substeps": None if not moving.any() else float(backward[moving].mean()),
                "maximum_continuous_backward_s": max(backward_phases, default=0)/120,
                "diagnostic_only": True,
                "turning_and_reactive_motion_included": True,
            },
        })
    report = {
        "schema": "crowdsim.trace_pace_measured_gait_diagnostics.v1",
        "paper_metric": False, "visual_realism_certified": False,
        "diagnostic_only_no_gait_success_threshold": True,
        "actual_controls": n, "physics_substeps_per_actor": n*4,
        "input_result_sha256": digest(result_path),
        "capture_sha256": {name: digest(path) for name, path in paths.items()},
        "evaluation_source_sha256": digest(Path(__file__)),
        "threshold_normal_force_n": 1., "actors": actors,
        "limitations": [
            "Exclusive floor support phases are not heel strikes or an authored gait-label benchmark",
            "A body-origin velocity includes ankle/toe articulation; contact-patch slip and friction are not measured",
            "Frequency is descriptive across the full episode, including turns and stops",
            "Force support and alternation are evidence of physical gait, not proof of human perceptual realism",
        ],
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "completed", "actual_controls": n,
                      "median_opposite_support_onset_hz": [a["median_opposite_support_onset_frequency_hz"] for a in actors]}))


if __name__ == "__main__":
    main()
