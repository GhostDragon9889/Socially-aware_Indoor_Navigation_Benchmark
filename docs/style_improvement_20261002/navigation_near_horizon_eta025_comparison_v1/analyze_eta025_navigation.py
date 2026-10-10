"""Read-only CPU raw navigation comparison; no style model or MMD calculation."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

DOCS = Path(__file__).resolve().parent.parent
ROOT = DOCS.parent.parent
RUNS = ROOT / "output/diffusion_amp"
PREVIOUS = RUNS / "shared_amass2848_backward_penalty_ground_support_200s_v1"
CURRENT = RUNS / "shared_amass2848_near_horizon_eta025_ground_support_200s_v1"


def helper():
    specification = importlib.util.spec_from_file_location("eta025_raw_phase_analysis", DOCS / "summarize_navigation_style.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def contact_events(mask, episodes, forces, actors):
    """Contiguous true controls split explicitly at either actor's reset."""
    sections, start = [], None
    for frame in range(len(mask) + 1):
        present = frame < len(mask) and mask[frame]
        continuous = (present and start is not None and frame > 0
                      and np.array_equal(episodes[frame, actors], episodes[frame - 1, actors]))
        if start is not None and not continuous:
            sections.append({"actors": actors, "first_source_frame": start, "last_source_frame": frame - 1,
                             "first_time_s": (start + 1) / 30, "last_time_s": frame / 30,
                             "control_frames": frame - start, "duration_s": (frame - start) / 30,
                             "episode_ids": episodes[start, actors].tolist(),
                             "maximum_normal_force_n": float(forces[start:frame].max())})
            start = None
        if present and start is None:
            start = frame
    return sections


def outcome(module, directory, audit):
    report = json.loads((directory / "result.json").read_text())
    assert report["status"] == audit["status"] == "passed"
    for key, filename in (("result", "result.json"), ("physical_states", "physical_states.npz"), ("ground_support", "ground_support.npz")):
        assert module.record(directory / filename) == audit["artifacts"][key]
    with np.load(directory / "physical_states.npz", allow_pickle=False) as data:
        physical = {key: data[key] for key in data.files}
    with np.load(directory / "ground_support.npz", allow_pickle=False) as data:
        ground = {key: data[key] for key in ("double_support_fraction", "no_support_fraction", "foot_normal_load_peak_n")}
    t, b = physical["root_pos"].shape[:2]
    assert (t, b) == (6000, 4)
    episodes = physical["episode_ids"]
    static_events, peer_events = [], []
    for actor in range(b):
        static_events.extend(contact_events(physical["static_contact"][:, actor], episodes,
                             physical["static_contact_max_force_n"][:, actor], [actor]))
        for peer in range(actor + 1, b):
            peer_events.extend(contact_events(physical["peer_contact"][:, actor, peer], episodes,
                               physical["peer_contact_max_force_n"][:, actor, peer], [actor, peer]))
    falls = []
    for frame, actor in zip(*np.nonzero(physical["terminated"]), strict=True):
        window = slice(max(0, frame - 15), min(t, frame + 16))
        root = physical["root_pos"][frame, actor]
        falls.append({"source_frame": int(frame), "time_s": float((frame + 1) / 30), "actor": int(actor),
                      "episode_id": int(episodes[frame, actor]), "root_xyz_m": root.tolist(),
                      "executed_target_error_m": float(np.linalg.norm(physical["target_xy"][frame, actor] - root[:2])),
                      "raw_target_error_m": float(np.linalg.norm(physical["raw_target_xy"][frame, actor] - root[:2])),
                      "reference_target_speed_mps": float(np.linalg.norm(physical["reference_target_velocity_xy"][frame, actor])),
                      "root_point_to_each_box_signed_distance_m": module.aabb_point_distance(root[None, :2], report["evaluation_scene"]["cuboids"])[0].tolist(),
                      "double_support_fraction_at_terminal": float(ground["double_support_fraction"][frame, actor]),
                      "no_foot_support_fraction_at_terminal": float(ground["no_support_fraction"][frame, actor]),
                      "maximum_floor_foot_load_n_at_terminal": float(ground["foot_normal_load_peak_n"][frame, actor].max()),
                      "static_contact_at_terminal": bool(physical["static_contact"][frame, actor]),
                      "peer_contact_at_terminal": bool(physical["peer_contact"][frame, actor].any()),
                      "static_contact_any_half_second_before_after": bool(physical["static_contact"][window, actor].any()),
                      "peer_contact_any_half_second_before_after": bool(physical["peer_contact"][window, actor].any()),
                      "next_post_step_reset_mask": None if frame + 1 == t else bool(physical["reset_mask"][frame + 1, actor]),
                      "next_post_step_episode_id": None if frame + 1 == t else int(episodes[frame + 1, actor])})
    assert len(falls) == report["fall_resets"]
    return {"sources": {"result": module.record(directory / "result.json"), "states": module.record(directory / "physical_states.npz"),
                        "ground": module.record(directory / "ground_support.npz")},
            "runtime_arguments": report["arguments"], "training_config": report["training_config"],
            "physics": report["physics"], "scene": report["evaluation_scene"],
            "models_and_resources": audit["sources"]["resources_full_sha256_recomputed"],
            "controls": t, "actor_frames": t * b, "duration_s": t / 30,
            "falls": len(falls), "fall_records": falls, "time_limit_resets": report["time_limit_resets"],
            "minimum_pelvis_height_m": report["minimum_height_m"],
            "route_vertices_reached": report["route_vertices_reached"], "total_vertices": sum(report["route_vertices_reached"]),
            "cumulative_four_vertex_cycles": sum(n // 4 for n in report["route_vertices_reached"]),
            "global_path_failures": report["global_path_failures"], "start_recoveries": report["successful_start_recoveries"],
            "static_contact_actor_frames": int(physical["static_contact"].sum()),
            "peer_contact_actor_frames": int(physical["peer_contact"].any(-1).sum()),
            "peer_contact_unordered_pair_frames": int(np.triu(physical["peer_contact"], 1).sum()),
            "static_contact_events": static_events, "peer_contact_events": peer_events,
            "ground_support_statistics": audit["support_statistics"], "ground_origin_speed_proxy": audit["body_origin_motion_proxy"],
            "COM": audit["COM_statistics"], "native_probe_flags_remain_false": True}, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Fresh report required; no historical report replacement")
    module = helper()
    old_audit_path = DOCS / "ground_support_200s_actual_verification.json"
    old_audit = json.loads(old_audit_path.read_text())
    new_audit = json.loads(args.ground_audit.read_text())
    previous, previous_result = outcome(module, PREVIOUS, old_audit)
    current, current_result = outcome(module, CURRENT, new_audit)
    argument_diff = {key: {"previous": previous["runtime_arguments"].get(key, "<missing>"),
                          "current": current["runtime_arguments"].get(key, "<missing>")}
                     for key in sorted(set(previous["runtime_arguments"]) | set(current["runtime_arguments"]))
                     if previous["runtime_arguments"].get(key, "<missing>") != current["runtime_arguments"].get(key, "<missing>")}
    assert set(argument_diff) == {"output", "near_horizon_waypoint_mass"}
    configuration_diff = {key: {"previous": previous["training_config"].get(key, "<missing>"),
                               "current": current["training_config"].get(key, "<missing>")}
                          for key in sorted(set(previous["training_config"]) | set(current["training_config"]))
                          if previous["training_config"].get(key, "<missing>") != current["training_config"].get(key, "<missing>")}
    assert configuration_diff == {"expert_sampling": {"previous": "<missing>", "current": None}}
    assert previous["models_and_resources"] == current["models_and_resources"]
    assert previous["physics"] == current["physics"] and previous["scene"] == current["scene"]
    for key in ("runtime_execution", "runtime_numerics", "physical_contact_measurement"):
        assert previous_result[key] == current_result[key]
    baseline_analysis_path = DOCS / "backward_penalty_navigation_style_v1/comparison.json"
    baseline_analysis = json.loads(baseline_analysis_path.read_text())
    previous_stagnation = baseline_analysis["stagnation"]
    assert previous_stagnation["saved_plans"] == module.record(PREVIOUS / "reference_plans.npz")
    # Only this raw function is called: no evaluation-model loading or MMD.
    current_stagnation = module.stagnation(CURRENT, new_audit)
    output = {"schema": "navisaaclab.near_horizon_raw_navigation_comparison.v1", "status": "passed",
              "CPU_only": True, "Kit_started": False, "GPU_used": False,
              "script": module.record(Path(__file__)), "unchanged_phase_analysis_helper": module.record(DOCS / "summarize_navigation_style.py"),
              "native_ground_audits": {"previous": module.record(old_audit_path), "current": module.record(args.ground_audit)},
              "previous_raw_phase_report": module.record(baseline_analysis_path),
              "runtime_argument_differences": argument_diff, "serialized_training_metadata_differences": configuration_diff,
              "single_declared_deployment_parameter": "near_horizon_waypoint_mass legacy0.0 to0.25",
              "same_physics_scene_models_and_common_training_config": True,
              "shared_seed_does_not_prove_bitwise_closed_loop_equivalence": True,
              "runs": {"eta0": previous, "eta025": current},
              "stagnation": {"eta0": previous_stagnation, "eta025": current_stagnation},
              "definitions": {"idle": "same-episode30Hz XY displacement speed<0.1m/s; reset jumps excluded",
                              "actual_contacts": "normal-force>1N strict threshold, union/max of4 physics substeps; floor/self excluded",
                              "events": "consecutive contact controls, split whenever either actor episode changes",
                              "cycles": "sum over actors floor(cumulative reached vertices/4); not uninterrupted episode success",
                              "source_time": "post-control source frame i=(i+1)/30 seconds",
                              "near_mass": "eta.25 gives first2 of10 timed waypoints total0.4 coefficient mass; total guidance20 unchanged"},
              "MMD_recomputed": False,
              "official_frozen_style_report": "../near_horizon_eta025_style_v1/report.md",
              "limitations": ["One seed and fixture, not author8scenes or statistical replication.",
                              "Normal-force contacts omit friction and contact-point velocities; foot-origin speed is a rolling-sensitive proxy, not true slip.",
                              "Root-point AABB distance is geometry only, not collider clearance or measured whole-body contact.",
                              "All falls/low-speed intervals are retained; time-limit/cycles do not certify stability.",
                              "AMASS trueCOM/contact/friction unknown; no reference equivalents invented.",
                              "Frozen prelaunch archive, not later opt-in on-disk source, binds this runtime."]}
    with args.output.open("x") as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "passed", "output": str(args.output),
                      "falls": current["falls"], "vertices": current["total_vertices"]}))


if __name__ == "__main__":
    main()
