"""CPU toe-origin diagnostics after the unchanged full22 metric admission.

Joint orientation/position mathematics are read from the existing frozen
decomposition report. Foot measurements remain geometric proxies.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STYLE_MATH = ROOT / "output/isolated_training/mixed_state_action_terrain_state_prior_pacer_reward_v2/code/CrowdSim/humanoid/style_evaluation.py"
STYLE_SHA = "b09fa58caaf877d89b69ca788538c2808e3a3d8d91410277ea07b5df3bcdaca2"
SPEED_EDGES = (0., .5, 1., 1.5, 2., 3., float("inf"))
BASELINE_RESULT = ROOT / "output/diffusion_amp/state_prior_tracking2000_v1/result.json"
BASELINE_SHA = "61e72e346471f3e1866115b05c6d913d5ea49ca448af665bc6067479a5a97413"
REGISTRATION = ROOT / "docs/style_improvement_20261003/state_prior_pacer_reward_evaluation_v1/registration.json"


def record(path):
    path = Path(path).absolute()
    if not path.is_file() or path.is_symlink():
        raise ValueError("Regular source required: " + str(path))
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def require(value, message):
    if not value:
        raise ValueError(message)


def style_module():
    require(record(STYLE_MATH)["sha256"] == STYLE_SHA, "Frozen toe proxy mathematics changed")
    spec = importlib.util.spec_from_file_location("fixed_toe_proxy_math", STYLE_MATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def toe_intervals(positions, style):
    """Original world XY toe-origin speed and original near-floor stance mask."""
    spec = style.StyleSpec()
    spec.validate()
    positions = np.asarray(positions)
    require(positions.ndim == 3 and positions.shape[1:] == (24, 3)
            and len(positions) >= 3 and positions.dtype.kind == "f"
            and np.isfinite(positions).all(), "Finite post-step [T>=3,24,3] positions required")
    toes = positions[:, [style.BODY_NAMES.index(name) for name in ("L_Toe", "R_Toe")]].astype(np.float64)
    velocity = np.diff(toes, axis=0) / spec.dt_s
    near = toes[..., 2] - spec.ground_height_m <= spec.toe_near_ground_height_m
    stance = near[:-1] & near[1:] & (np.abs(velocity[..., 2]) <= spec.toe_vertical_speed_max_m_s)
    return {"speed": np.linalg.norm(velocity[..., :2], axis=-1), "stance": stance,
            "xy_velocity": velocity[..., :2], "end_height": toes[1:, :, 2] - spec.ground_height_m}


def summarize_intervals(arrays, selected, style):
    spec = style.StyleSpec()
    rows = []
    for side, name in enumerate(("L_Toe", "R_Toe")):
        actual = selected & arrays["actual_stance"][:, side]
        reference = selected & arrays["reference_stance"][:, side]
        common = actual & reference
        row = {"toe": name, "all_intervals": int(selected.sum()),
               "actual_proxy_intervals": int(actual.sum()), "reference_proxy_intervals": int(reference.sum()),
               "common_proxy_intervals": int(common.sum()),
               "actual_proxy_speed_m_s": style.summary(arrays["actual_speed"][actual, side]),
               "reference_proxy_speed_m_s": style.summary(arrays["reference_speed"][reference, side]),
               "actual_speed_on_common_proxy_m_s": style.summary(arrays["actual_speed"][common, side]),
               "reference_speed_on_common_proxy_m_s": style.summary(arrays["reference_speed"][common, side]),
               "xy_velocity_difference_on_common_proxy_m_s": style.summary(arrays["xy_velocity_difference"][common, side]),
               "actual_proxy_fraction_above_fixed_threshold": float(np.mean(arrays["actual_speed"][actual, side] > spec.toe_sliding_speed_threshold_m_s)) if actual.any() else None,
               "reference_proxy_fraction_above_fixed_threshold": float(np.mean(arrays["reference_speed"][reference, side] > spec.toe_sliding_speed_threshold_m_s)) if reference.any() else None,
               "actual_negative_toe_origin_end_heights": int((arrays["actual_end_height"][selected, side] < 0).sum()),
               "reference_negative_toe_origin_end_heights": int((arrays["reference_end_height"][selected, side] < 0).sum())}
        rows.append(row)
    return {"interval_count": int(selected.sum()), "fall_endpoint_intervals_retained": int((selected & arrays["fall_interval"]).sum()),
            "by_toe": rows}


def diagnose(input_dir, decomposition_path, style_path):
    """Consume completed admitted metrics; never infer future capture success."""
    source_path = input_dir / "result.json"
    registration = json.loads(REGISTRATION.read_text())
    future_dir = Path(registration["tracking"]["parsed"]["output"])
    require(input_dir == BASELINE_RESULT.parent or input_dir == future_dir,
            "Only the fixed old state-prior baseline or registered new validation capture is accepted")
    source = json.loads(source_path.read_text())
    decomposition = json.loads(decomposition_path.read_text())
    evaluation = json.loads(style_path.read_text())
    require(source["arguments"]["split"] == "validation" and source["status"] == "passed",
            "Only completed validation capture is accepted; sealed test is forbidden")
    if input_dir == BASELINE_RESULT.parent:
        require(record(source_path)["sha256"] == BASELINE_SHA, "Historical baseline source changed")
    else:
        admission = json.loads((decomposition_path.parent / "actual_source_admission.json").read_text())
        require(admission.get("status") == "passed" and admission.get("metric_pass_is_quality_pass") is False
                and admission.get("schema") == "pacer_reward.actual_CPU_metric_source_admission.v1"
                and admission.get("metric") == "tracking_decomposition",
                "Actual new935 complete capture/source admission must already have passed")
        prepared = ROOT / "docs/style_improvement_20261003/state_prior_pacer_reward_CPU_metrics_preparation_v1"
        require(admission["runner"] == record(prepared / "run_CPU_metrics.py")
                and admission["catalog"] == record(prepared / "commands_CPU_metrics_v1.json"), "Actual admitted metrics runner/catalog differs")
        admitted = admission["actual_source_admission"]
        require(admitted["registered_protocol"] == record(REGISTRATION)
                and admitted["isolated_source_manifest"]["sha256"] == "1fd9b5e14dc377f753b04ecfe6ce93054b300d6f46c5cb341aa466f6ac887a21"
                and admitted["resources_full_sha256_recomputed"]["controller_checkpoint"]["sha256"] == source["arguments"]["checkpoint_sha256"],
                "New result checkpoint/source must match completed metric admission")
    require(decomposition["status"] == "passed" and decomposition["source_result"]["sha256"] == record(source_path)["sha256"]
            and decomposition["primary_aggregate_unchanged"] == source["aggregate"], "Frozen same-frame decomposition differs")
    matching = [item for item in evaluation["runs"].values() if item["sources"]["result"] == record(source_path)]
    require(len(matching) == 1, "Completed fixed style report must bind this actual result")
    style_item = matching[0]
    require(len(source["sequences"]) == 22 and source["aggregate"]["measured_sequences"] == 22
            and source["physics"]["control_hz"] == 30
            and "flat unobstructed physical floor" in source["layout"]["method"], "Fixed full22 flat 30Hz tracking required")
    style = style_module()
    require(tuple(source["physics"]["body_names"]) == style.BODY_NAMES, "Canonical captured body order required")
    parts, sources, rows = [], [], []
    for sequence in source["sequences"]:
        path = Path(sequence["capture"])
        require(path.parent == input_dir / "captures" and record(path)["sha256"] == sequence["capture_sha256"], "Capture identity differs")
        with np.load(path, allow_pickle=False) as data:
            metric = data["metric_frame"]
            require(metric.dtype == np.bool_ and metric.ndim == 1 and not metric[0] and metric[1:].all()
                    and np.array_equal(data["native_reset_frame"], ~metric) and np.unique(data["episode_id"]).size == 1,
                    "One excluded initial frame followed by complete same-episode post-step frames required")
            require(int(metric.sum()) == sequence["observed_metric_steps"] == sequence["expected_control_steps"]
                    and sequence["complete_clip"] is True and np.all(np.diff(data["reference_frame"]) == 1), "No phase skips or partial clips")
            actual = toe_intervals(data["rigid_body_pos"][metric, 0], style)
            reference = toe_intervals(data["reference_rigid_body_pos"][metric, 0], style)
            falls = data["fall_mask"][metric]
            require(falls.dtype == np.bool_, "Explicit fall labels required")
            arrays = {"actual_speed": actual["speed"], "reference_speed": reference["speed"],
                      "actual_stance": actual["stance"], "reference_stance": reference["stance"],
                      "xy_velocity_difference": np.linalg.norm(actual["xy_velocity"] - reference["xy_velocity"], axis=-1),
                      "actual_end_height": actual["end_height"], "reference_end_height": reference["end_height"],
                      "reference_planar_speed": np.linalg.norm(data["reference_root_vel"][metric][1:, 0, :2].astype(np.float64), axis=-1),
                      "reference_end_frame": data["reference_frame"][metric][1:],
                      "motion_id": np.full(len(actual["speed"]), sequence["motion_id"], dtype=np.int64),
                      "fall_interval": falls[:-1] | falls[1:]}
        parts.append(arrays)
        sources.append(record(path))
        rows.append({"motion_id": sequence["motion_id"], "source_path": sequence["source_path"],
                     "tracking_success_primary": sequence["tracking_success"],
                     "toe_proxy": summarize_intervals(arrays, np.ones(len(arrays["motion_id"]), dtype=bool), style)})
    combined = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    require(len(combined["motion_id"]) == style_item["generated"]["same_episode_transition_count"], "Original style interval count differs")
    pooled = combined["actual_speed"][combined["actual_stance"]]
    original = style_item["generated"]["toe_slide_near_ground_proxy"]
    require(len(pooled) == original["stance_foot_intervals"], "Original pooled toe proxy interval count differs")
    for key, value in style.summary(pooled).items():
        require(value == original["speed_m_s"][key] or (value is not None and np.isclose(value, original["speed_m_s"][key], atol=1e-12, rtol=0)), "Original pooled toe proxy statistic differs: " + key)
    speed_bins = []
    for low, high in zip(SPEED_EDGES[:-1], SPEED_EDGES[1:], strict=True):
        selected = (combined["reference_planar_speed"] >= low) & (combined["reference_planar_speed"] < high)
        speed_bins.append({"reference_speed_range_m_s_half_open": [low, high if np.isfinite(high) else None],
                           **summarize_intervals(combined, selected, style)})
    report = {"schema": "pacer_reward.same_frame_joint_toe_diagnostics.v1", "status": "passed",
              "CPU_only": True, "GPU_used": False, "SDK_started": False, "sealed34_outputs_read": False,
              "primary_metric_changed": False, "quality_gate": False, "author_equivalent_metric": False,
              "sources": {"result": record(source_path), "decomposition": record(decomposition_path),
                          "style_report": record(style_path), "style_math": record(STYLE_MATH), "captures": sources},
              "checkpoint_sha256": source["arguments"]["checkpoint_sha256"], "fixed_proxy_spec": asdict(style.StyleSpec()),
              "all_body_same_frame_diagnostics": decomposition["by_body"],
              "all_clip_same_frame_diagnostics": decomposition["by_clip"],
              "toe_proxy_aggregate": summarize_intervals(combined, np.ones(len(combined["motion_id"]), dtype=bool), style),
              "by_clip_toe_proxy": rows, "by_reference_speed_toe_proxy": speed_bins,
              "definitions": {"same_frame_joints": "Unmodified frozen decomposition report: world, heading, full-root and parent-relative SO(3); no time matching",
                              "toe_proxy": "World XY toe-body-origin finite difference / (1/30 s); both endpoints z<=.12 m and |vz|<=.2 m/s above fixed floor0",
                              "above_threshold": "Original fixed .1 m/s; descriptive only, no new pass/fail threshold",
                              "common_proxy": "Both actual and reference satisfy the same geometric mask on the identical adjacent post-step frames",
                              "sample_retention": "Initial reset frame excluded; first post-step-to-post-step interval retained; no cross-clip difference; all fall/failed/negative-height samples retained",
                              "contact_observed": False, "contact_patch_slip_observed": False,
                              "reference_speed_bins": "Original decomposition bins on saved reference_root_vel at interval end; no bin fitting",
                              "cannot_infer": "Toe-origin movement may include swing/rolling; geometric stance is not measured contact or sole slip"}}
    return report, combined


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--decomposition", type=Path)
    parser.add_argument("--style-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU diagnostics require CUDA_VISIBLE_DEVICES=''")
    if args.prepare_only:
        require(not any((args.input, args.decomposition, args.style_report, args.output)), "Preparation cannot read actual captures")
        style = style_module()
        print(json.dumps({"status": "prepared_only", "actual_metrics_computed": False,
                          "style_math": record(STYLE_MATH), "fixed_proxy_spec": asdict(style.StyleSpec())}))
        return 0
    require(all((args.input, args.decomposition, args.style_report, args.output)), "Supply completed admitted capture/metrics and a fresh output")
    require(not args.output.exists(), "Output must be fresh")
    report, arrays = diagnose(args.input.absolute(), args.decomposition.absolute(), args.style_report.absolute())
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "toe_intervals.npz", **arrays)
    report["interval_archive"] = record(args.output / "toe_intervals.npz")
    with (args.output / "diagnostics.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "passed", "output": str(args.output), "toe_proxy": report["toe_proxy_aggregate"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
