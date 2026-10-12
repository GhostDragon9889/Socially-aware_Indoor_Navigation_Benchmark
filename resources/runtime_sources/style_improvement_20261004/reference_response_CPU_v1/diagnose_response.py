"""NumPy-only same-control reference/physical response associations; no lag fit."""
from __future__ import annotations

from protomotions.repository import repository_root

import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = repository_root(__file__)
ADMISSION_SOURCE = ROOT / "resources/runtime_sources/style_improvement_20261004/actual_tracking_contact_CPU_v1/diagnose_tracking_contact.py"
ADMISSION_SHA = "bbed5fbc10cf1efa24412e8e36651ed963a3f0ffd3f422e6df0bbfe097d312b8"
STATE_FIELDS = ("root_pos", "root_rot", "target_xy", "raw_target_xy", "reference_target_velocity_xy",
                "episode_ids", "reset_mask", "terminated", "truncated", "static_contact", "peer_contact")
GROUND_FIELDS = ("body_link_pos_w", "body_link_quat_w", "body_link_lin_vel_w", "common_body_names", "control_step",
                 "control_step_after", "physics_step_before", "physics_step", "physics_step_end", "substep_valid_count",
                 "valid", "episode_ids", "reset_mask", "terminated", "truncated")
FORBIDDEN = {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"}


def load_admission():
    info, source = ADMISSION_SOURCE.stat(), ADMISSION_SOURCE.read_bytes()
    if (not ADMISSION_SOURCE.is_file() or ADMISSION_SOURCE.is_symlink() or info.st_nlink != 1
            or info.st_mode & 0o222 or len(source) != 26264 or hashlib.sha256(source).hexdigest() != ADMISSION_SHA):
        raise ValueError("Frozen original CPU admission source identity mismatch")
    module = types.ModuleType("frozen_response_capture_admission")
    module.__file__ = str(ADMISSION_SOURCE)
    exec(compile(source, str(ADMISSION_SOURCE), "exec"), module.__dict__)
    return module


def wrapped(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def responses(state, native_velocity):
    root, reference, raw, velocity = (state[key][..., :2].astype(np.float64) for key in
                                     ("root_pos", "target_xy", "raw_target_xy", "reference_target_velocity_xy"))
    native_velocity = native_velocity.astype(np.float64)
    speed = np.linalg.norm(velocity, axis=-1)
    direction = speed >= .1
    unit = velocity / np.maximum(speed[..., None], np.finfo(np.float64).tiny)
    unit[~direction] = np.nan
    left = np.stack((-unit[..., 1], unit[..., 0]), -1)
    error, velocity_error = root-reference, native_velocity-velocity
    q = state["root_rot"].astype(np.float64)
    hx, hy = 1-2*(q[..., 1]**2+q[..., 2]**2), 2*(q[..., 3]*q[..., 2]+q[..., 0]*q[..., 1])
    heading_defined = np.hypot(hx, hy) > 1e-12
    heading = np.where(heading_defined, np.arctan2(hy, hx), np.nan)
    reference_heading = np.where(direction, np.arctan2(velocity[..., 1], velocity[..., 0]), np.nan)
    valid = np.zeros(speed.shape, bool)
    valid[1:] = ((state["episode_ids"][1:] == state["episode_ids"][:-1]) & ~state["reset_mask"][1:])
    displacement = np.full(speed.shape, np.nan)
    displacement[1:] = np.where(valid[1:], np.linalg.norm(np.diff(root, axis=0), axis=-1)*30, np.nan)
    metrics = {"reference_root_error_m": np.linalg.norm(error, axis=-1), "raw_root_error_m": np.linalg.norm(root-raw, axis=-1),
               "native_speed_mps": np.linalg.norm(native_velocity, axis=-1), "reference_speed_mps": speed,
               "velocity_error_norm_mps": np.linalg.norm(velocity_error, axis=-1),
               "velocity_error_along_mps": (velocity_error*unit).sum(-1), "velocity_error_left_mps": (velocity_error*left).sum(-1),
               "position_error_along_m": (error*unit).sum(-1), "position_error_left_m": (error*left).sum(-1),
               "actual_heading_rad": heading, "reference_motion_direction_rad": reference_heading,
               "heading_minus_reference_direction_rad": wrapped(heading-reference_heading), "previous_interval_speed_mps": displacement}
    for name, values, angular in (("native_velocity_change_mps", native_velocity, False),
                                 ("reference_velocity_change_mps", velocity, False),
                                 ("actual_heading_change_rad", heading, True), ("reference_direction_change_rad", reference_heading, True)):
        delta = wrapped(np.diff(values, axis=0)) if angular else np.linalg.norm(np.diff(values, axis=0), axis=-1)
        metrics[name] = np.full(speed.shape, np.nan)
        metrics[name][1:] = np.where(valid[1:], delta, np.nan)
    return metrics, valid, direction, heading_defined, valid & (displacement < .1)


def analyze(name, original, prior, writer):
    identities, audit, result, _ = prior.admission(name, original)  # Genuine closure and whole-byte identities before NPZ.
    identities["ground_support"] = prior.admitted(audit["artifacts"]["ground_support"])
    ground_record = {key: identities["ground_support"][key] for key in ("path", "bytes", "sha256")}
    gpu = prior.read(identities["actual_GPU_owner"]["path"])
    capture = result["ground_support_measurement"]["capture"]
    if ground_record not in gpu["artifacts"] or any(capture[key] != ground_record[key] for key in ground_record):
        raise ValueError("Ground whole-byte identity differs from original genuine GPU owner/result/raw audit")
    with np.load(identities["physical_states"]["path"], allow_pickle=False) as archive:
        state = {key: archive[key] for key in STATE_FIELDS}
    with np.load(identities["ground_support"]["path"], allow_pickle=False) as archive:
        ground = {key: archive[key] for key in GROUND_FIELDS}
    with np.load(identities["reference_plans"]["path"], allow_pickle=False) as archive:
        plans = {key: archive[key] for key in ("actor_ids", "control_step", "raw_xy", "reference_xy", "reference_velocity_xy")}
    booleans = {"reset_mask", "terminated", "truncated", "static_contact", "peer_contact"}
    for key, value in state.items():
        shape = (6000, 4, 4) if key in ("root_rot", "peer_contact") else (6000, 4, 3) if key == "root_pos" else (6000, 4, 2) if key in (
            "target_xy", "raw_target_xy", "reference_target_velocity_xy") else (6000, 4)
        if (value.shape != shape or not np.isfinite(value).all() or (key in booleans and value.dtype != np.bool_)
                or (key == "episode_ids" and value.dtype.kind not in "iu")):
            raise ValueError("Physical shape/finite/dtype mismatch: " + key)
    names = ground["common_body_names"].tolist()
    if names != result["state_capture"]["body_names"] or len(names) != 24 or names[0] != "Pelvis":
        raise ValueError("Exact original common body names with Pelvis0 required")
    for key, width in (("body_link_pos_w", 3), ("body_link_quat_w", 4), ("body_link_lin_vel_w", 3)):
        value = ground[key]
        if value.shape != (6000, 4, 4, 24, width) or value.dtype != np.dtype("float32") or not np.isfinite(value).all():
            raise ValueError("Native all-substep body channel shape/FP32/finite mismatch: " + key)
    before = audit["counts"]["first_before_submitted_step"]
    expected = {"control_step": np.arange(6000), "control_step_after": np.arange(1, 6001),
                "physics_step_before": before+np.arange(6000)*4, "physics_step_end": before+np.arange(1, 6001)*4,
                "physics_step": before+np.arange(1, 24001).reshape(6000, 4), "substep_valid_count": np.full(6000, 4),
                "valid": np.ones(6000, bool), **{key: state[key] for key in ("episode_ids", "reset_mask", "terminated", "truncated")}}
    if any(not np.array_equal(ground[key], value) or ground[key].dtype.kind != value.dtype.kind for key, value in expected.items()):
        raise ValueError("Native control/substep clocks, validity, episodes or reset/terminal fields differ")
    if (not np.array_equal(ground["body_link_pos_w"][:, -1, :, 0], state["root_pos"])
            or not np.array_equal(ground["body_link_quat_w"][:, -1, :, 0], state["root_rot"])
            or not np.allclose(np.linalg.norm(state["root_rot"], axis=-1), 1., atol=1e-3, rtol=0)):
        raise ValueError("Native end-substep Pelvis pose differs from same-control physical root or is not unit XYZW")
    target_checks, phase = prior.targets_from_saved_plans(state, plans)
    native = ground["body_link_lin_vel_w"][:, -1, :, 0, :2]
    metrics, valid, direction, heading_defined, idle = responses(state, native)
    peer = (state["peer_contact"] | state["peer_contact"].swapaxes(1, 2)).any(-1)
    contact = peer | state["static_contact"]
    if (int(peer.sum()) != original["original_contact_counts"]["peer_contact_actor_frames"]
            or int(state["static_contact"].sum()) != original["original_contact_counts"]["static_contact_actor_frames"]
            or int(state["terminated"].sum()) != len(original["fall_events"])
            or int(valid.sum()) != original["pooled_legal_intervals"] or int(idle.sum()) != original["pooled_valid_idle_intervals"]):
        raise ValueError("Full actual fall/contact/same-episode/idle counts differ from original sealed audit")
    masks = {"all": np.ones((6000, 4), bool), **{f"phase_{p:02d}": phase == p for p in range(1, 16)},
             "terminated": state["terminated"], "reset_first_poststep": state["reset_mask"], "peer_contact": peer,
             "static_contact": state["static_contact"], "any_contact": contact, "no_contact": ~contact, "idle_previous_interval": idle}
    groups = {label: {"actor_controls": int(mask.sum()), "reference_direction_undefined": int((mask & ~direction).sum()),
                      "actual_heading_undefined": int((mask & ~heading_defined).sum()),
                      "metrics": {key: {**prior.summary(value[mask & np.isfinite(value)]), "undefined_controls": int((mask & ~np.isfinite(value)).sum())}
                                  for key, value in metrics.items()}} for label, mask in masks.items()}
    for frame in range(6000):
        for actor in range(4):
            row = {"run": name, "source_frame": frame, "completed_control": int(ground["control_step_after"][frame]),
                   "derived_time_s": (frame+1)/30, "actor": actor, "episode_id": int(state["episode_ids"][frame, actor]), "phase": int(phase[frame, actor]),
                   "reset_first_poststep": bool(state["reset_mask"][frame, actor]), "terminated": bool(state["terminated"][frame, actor]),
                   "truncated": bool(state["truncated"][frame, actor]), "static_contact": bool(state["static_contact"][frame, actor]), "peer_contact": bool(peer[frame, actor]),
                   "same_episode_previous_interval": bool(valid[frame, actor]), "idle_previous_interval": bool(idle[frame, actor]),
                   "reference_direction_defined": bool(direction[frame, actor]), "actual_heading_defined": bool(heading_defined[frame, actor]),
                   **{key: float(value[frame, actor]) if np.isfinite(value[frame, actor]) else None for key, value in metrics.items()}}
            for key, values in (("native_velocity", native), ("reference_velocity", state["reference_target_velocity_xy"])):
                row[key+"_x_mps"], row[key+"_y_mps"] = (float(x) for x in values[frame, actor])
            if writer.fieldnames is None:
                writer.fieldnames = list(row)
                writer.writeheader()
            writer.writerow(row)
    return {"input_identities": identities, "all_actor_controls": 24000, "phase_counts_sum": sum(groups[f"phase_{p:02d}"]["actor_controls"] for p in range(1, 16)),
            "all_target_phase_checks_max_abs_error": target_checks, "groups": groups,
            "all_failed_contact_idle_reset_frames_retained": True, "new_success_gate_or_promotion": False}


def main():
    started = time.perf_counter()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or any(module.split(".")[0] in FORBIDDEN for module in sys.modules):
        raise ValueError("Explicit CUDA-hidden CPU process without model/simulator imports required")
    prior = load_admission()
    source, admission = prior.record(__file__), prior.record(ADMISSION_SOURCE, readonly=True)
    producer, comparison = prior.record(prior.PRODUCER), prior.record(prior.COMPARISON, readonly=True)
    if admission["sha256"] != ADMISSION_SHA or producer["sha256"] != prior.PRODUCER_SHA or comparison["sha256"] != prior.COMPARISON_SHA:
        raise ValueError("Original source-bound admission, producer or comparison changed")
    output, csv_output = HERE / "actual_diagnostic.json", HERE / "all_actor_control_rows.csv"
    if any(path.exists() or path.is_symlink() for path in (output, csv_output)):
        raise ValueError("Preserve existing outputs; never overwrite")
    original = prior.read(prior.COMPARISON)
    with csv_output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=None)
        runs = {name: analyze(name, original["runs"][name], prior, writer) for name in prior.DP}
    for expected in [source, admission, producer, comparison, *(identity for run in runs.values() for identity in run["input_identities"].values())]:
        if prior.record(expected["path"], readonly=expected["mode"] == "0o444") != expected:
            raise ValueError("Source/input changed during actual CPU observation")
    if any(module.split(".")[0] in FORBIDDEN for module in sys.modules):
        raise ValueError("CPU reader imported model/simulator modules")
    report = {"schema": "reference_response.same_control_association.v1", "status": "completed_CPU_diagnostic",
              "execution_source": source, "frozen_admission_source": admission, "producer": producer, "original_comparison": comparison,
              "definition": {"time": "same30Hz control end; native Pelvis from fourth120Hz substep; target/refV at saved plan age+1",
                             "direction": "reference velocity norm>=.1m/s; below .1 undefined; actual heading=projected root+X with horizontal norm>1e-12",
                             "errors": "actual-reference; along reference unit velocity, left=(-uy,ux); XY norms unchanged, no phase search",
                             "differences": "successive control-end differences, same recorded episode and nonreset only; first/reset null; wrapped heading differences",
                             "heading": "body root heading versus reference motion direction, not an explicitly commanded AMP yaw",
                             "idle": "original same-episode previous root displacement speed<.1m/s; all rows remain"},
              "runs": runs, "all_actor_control_rows": prior.record(csv_output), "all_actor_control_rows_count": 72000,
              "CPU_elapsed_seconds": time.perf_counter()-started, "causality_established": False, "true_action_delay_estimated": False,
              "limits": ["Only response associations; no lag search, fitted delay, causal inference or model superiority.",
                         "Task observation/actions and reset pre-step root velocity were not independently archived.",
                         "Direction-undefined and difference-undefined rows remain in all/phase/failure/contact denominators."],
              "weights_read_or_models_loaded": False, "GPU_used": False, "SDK_started": False,
              "sealed34_outputs_read": False, "new_success_gate_or_promotion": False}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "rows": 72000, "phase_totals": {name: run["phase_counts_sum"] for name, run in runs.items()}}))


if __name__ == "__main__":
    main()
