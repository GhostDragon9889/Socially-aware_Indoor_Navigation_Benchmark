"""Read-only CPU audit of full22 captures and exact paired reference protocol.

Checks the existing primary metric; does not align poses, decompose errors,
remove fall frames, calculate MMD, or open the reserved34 test captures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from CrowdSim.humanoid.motion_tracking import body_tracking_errors, summarize_tracking_sequence
from CrowdSim.humanoid.style_evaluation import assert_parent_split_disjoint

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "output/diffusion_amp/motion_tracking_amass2848_backward_penalty_full_validation_v1"
POSE_FIELDS = ("root_pos", "root_rot", "root_vel", "root_ang_vel", "dof_pos", "dof_vel",
               "rigid_body_pos", "rigid_body_rot", "rigid_body_vel", "rigid_body_ang_vel")
PAIR_FIELDS = (*(f"reference_{name}" for name in POSE_FIELDS), "reference_frame", "reference_time_s",
               "metric_frame", "native_reset_frame", "trajectory_xy", "waypoint_query_phase_frame",
               "waypoint_held_at_clip_end", "action_valid")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_run(directory):
    directory = Path(directory).resolve()
    result_path = directory / "result.json"
    identity = record(result_path)
    result = read_json(result_path)
    assert result["status"] == "passed" and result["arguments"]["split"] == "validation"
    checkpoint = record(result["arguments"]["checkpoint"])
    assert checkpoint["sha256"] == result["preflight"]["checkpoint_sha256_recomputed"]
    if result["arguments"]["checkpoint_sha256"] is not None:
        assert checkpoint["sha256"] == result["arguments"]["checkpoint_sha256"]
    assert result["arguments"]["max_steps"] is None and result["arguments"]["max_clips"] is None
    assert len(result["sequences"]) == len({s["motion_id"] for s in result["sequences"]}) == 22
    assert result["protocol"]["runtime_teleports"] == 0
    assert result["protocol"]["fall_reset_within_sequence"] is False
    assert result["protocol"]["initial_frame_excluded_from_errors"] is True
    primary_path = ROOT / "CrowdSim/humanoid/motion_tracking.py"
    assert result["selected_source_sha256"][str(primary_path.relative_to(ROOT))] == sha256(primary_path)
    manifest_path = Path(result["preflight"]["manifest"])
    assert sha256(manifest_path) == result["preflight"]["manifest_sha256"]
    heldout = read_json(manifest_path)
    parents = assert_parent_split_disjoint(heldout)
    actual_train = result["controller_provenance"]["dataset"]
    assert actual_train["split"] == "train" and sha256(actual_train["manifest"]) == actual_train["manifest_sha256"]
    train_parents = assert_parent_split_disjoint(read_json(actual_train["manifest"]))["train"]
    assert not train_parents & parents["validation"]
    ids = {row["motion_id"] for row in heldout["splits"]["validation"]["clips"]}
    assert {row["motion_id"] for row in result["sequences"]} == ids
    clips, primary = [], []
    for sequence in result["sequences"]:
        path = Path(sequence["capture"])
        assert path.resolve().parent == directory / "captures"
        identity_capture = record(path)
        assert identity_capture["sha256"] == sequence["capture_sha256"]
        window = sequence["tracking_window"]
        assert window["initial_frame"] == 9 and window["full_clip_requested"] is True
        assert sequence["app_or_simulation_stopped"] is False
        with np.load(path, allow_pickle=False) as arrays:
            total = len(arrays["reference_frame"])
            steps = total - 1
            assert steps == window["available_steps"] == window["requested_steps"] == sequence["observed_metric_steps"]
            assert arrays["reference_frame"].dtype == np.int64
            assert np.array_equal(arrays["reference_frame"], np.arange(9, 9 + total))
            np.testing.assert_allclose(arrays["reference_time_s"], arrays["reference_frame"] / 30, rtol=0, atol=1e-12)
            assert np.array_equal(arrays["metric_frame"], np.r_[False, np.ones(steps, dtype=bool)])
            assert np.array_equal(arrays["native_reset_frame"], ~arrays["metric_frame"])
            assert np.array_equal(arrays["action_valid"], arrays["metric_frame"])
            assert np.array_equal(arrays["waypoint_query_phase_frame"], np.r_[9, np.arange(9, 9 + steps)])
            assert arrays["episode_id"].shape == (total, 1) and arrays["episode_id"].dtype == np.int64
            assert np.all(arrays["episode_id"] == arrays["episode_id"][0])
            assert arrays["amp_history"].shape == (total, 1, 10, 195)
            assert arrays["trajectory_xy"].shape == (total, 1, 10, 2)
            for key in arrays.files:
                if arrays[key].dtype.kind in "fc":
                    assert np.isfinite(arrays[key]).all(), (sequence["motion_id"], key)
            calculated = body_tracking_errors(arrays["rigid_body_pos"][:, 0], arrays["rigid_body_rot"][:, 0],
                                               arrays["reference_rigid_body_pos"][:, 0], arrays["reference_rigid_body_rot"][:, 0])
            for key, value in calculated.items():
                np.testing.assert_allclose(value, arrays[key], rtol=0, atol=1e-12)
            summary = summarize_tracking_sequence(calculated, metric_mask=arrays["metric_frame"], fall_mask=arrays["fall_mask"],
                expected_steps=window["available_steps"], eligible=window["eligible"], full_clip_requested=window["full_clip_requested"],
                threshold_m=result["arguments"]["failure_threshold_m"])
            for key, value in summary.items():
                if isinstance(value, float):
                    assert math.isclose(value, sequence[key], abs_tol=1e-12, rel_tol=0)
                else:
                    assert value == sequence[key], (sequence["motion_id"], key)
            mask = arrays["metric_frame"]
            primary.append({"steps": steps, "position_sum": float(calculated["mean_body_position_error_m"][mask].sum()),
                            "rotation_sum": float(calculated["mean_body_rotation_error_rad"][mask].sum()), **summary})
        clips.append({"motion_id": sequence["motion_id"], "source_path": sequence["source_path"], "capture": identity_capture,
                      "metric_frames": steps, "success": summary["tracking_success"], "had_fall": summary["sequence_had_fall"]})
    total_steps = sum(row["steps"] for row in primary)
    assert total_steps == 11982
    aggregate = {"measured_sequences": 22, "complete_success_denominator": sum(p["in_full_sequence_success_denominator"] for p in primary),
        "tracking_successes": sum(p["tracking_success"] is True for p in primary),
        "sequences_with_observed_tracking_failure": sum(p["tracking_failed"] for p in primary),
        "sequences_with_fall": sum(p["sequence_had_fall"] for p in primary),
        "fall_onset_events": sum(p["fall_onset_events"] for p in primary), "observed_metric_steps": total_steps,
        "frame_weighted_mean_body_position_error_m": sum(p["position_sum"] for p in primary) / total_steps,
        "frame_weighted_mean_body_rotation_error_rad": sum(p["rotation_sum"] for p in primary) / total_steps}
    aggregate["tracking_success_fraction"] = aggregate["tracking_successes"] / aggregate["complete_success_denominator"]
    for key, value in aggregate.items():
        assert math.isclose(value, result["aggregate"][key], abs_tol=1e-12, rel_tol=0), key
    assert identity == record(result_path)
    return {"result": identity, "checkpoint": checkpoint, "clips": clips, "primary_aggregate_verified": aggregate,
            "training_and_validation_parent_sources_disjoint": True,
            "all_fall_frames_retained": True, "initial_reset_frames_excluded": 22,
            "within_clip_style_interval_count_only": total_steps - 22}, result


def compare_pair(a_dir, b_dir, a, b):
    for key in ("protocol", "physics", "layout", "reference_velocity_contract"):
        assert a[key] == b[key], key
    differences = {key: {"baseline": a["arguments"].get(key), "candidate": b["arguments"].get(key)}
                   for key in set(a["arguments"]) | set(b["arguments"]) if a["arguments"].get(key) != b["arguments"].get(key)}
    assert set(differences) == {"checkpoint", "checkpoint_sha256", "output"}
    pairs = []
    for left, right in zip(a["sequences"], b["sequences"], strict=True):
        for key in ("motion_id", "source_path", "tracking_window", "observed_metric_steps", "initial_translation_xyz_m"):
            assert left[key] == right[key]
        with np.load(left["capture"], allow_pickle=False) as x, np.load(right["capture"], allow_pickle=False) as y:
            assert set(x.files) == set(y.files)
            for key in PAIR_FIELDS:
                assert x[key].dtype == y[key].dtype and np.array_equal(x[key], y[key]), (left["motion_id"], key)
            for key in POSE_FIELDS:
                assert np.array_equal(x[key][0], y[key][0]), (left["motion_id"], "initial physical " + key)
        pairs.append({"motion_id": left["motion_id"], "reference_target_phase_and_initial_state_all_bitwise_equal": True})
    return {"argument_differences": differences, "paired_clips": pairs, "paired_metric_frames": 11982,
            "source_changes": {key: {"baseline": a["selected_source_sha256"].get(key), "candidate": b["selected_source_sha256"].get(key)}
                for key in set(a["selected_source_sha256"]) | set(b["selected_source_sha256"])
                if a["selected_source_sha256"].get(key) != b["selected_source_sha256"].get(key)},
            "initial_inputs_are_paired_physical_rollouts_are_not_assumed_equal": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=BASELINE)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--preparation-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or (not args.preparation_only and args.candidate is None):
        parser.error("Fresh output and explicit candidate are required")
    before, a = validate_run(args.baseline)
    document = {"schema": "amass_root_tracking_independent_protocol_audit.v1", "status": "passed",
                "CPU_only": True, "SDK_started": False, "GPU_used": False, "heldout34_test_outputs_read": False,
                "candidate_tracking_actual_verified": not args.preparation_only, "baseline": before,
                "script": record(__file__), "primary_metric_modified": False, "style_or_decomposition_calculated": False}
    if not args.preparation_only:
        assert args.candidate.resolve() != args.baseline.resolve()
        candidate, b = validate_run(args.candidate)
        document.update(candidate=candidate, paired_protocol=compare_pair(args.baseline, args.candidate, a, b))
    with args.output.open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "passed", "candidate_tracking_actual_verified": not args.preparation_only}))


if __name__ == "__main__":
    main()
