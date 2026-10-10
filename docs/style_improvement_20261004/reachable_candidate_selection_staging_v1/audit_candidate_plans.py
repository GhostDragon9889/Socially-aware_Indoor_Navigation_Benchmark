"""Future CPU candidate reconstruction plus the unchanged f851 physical audit.

Preparation is source-only. Numerical imports occur only after genuine closed0
root admission. Geometry reuses the recorded selector's helper, never a model.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from candidate_evidence import HERE, admit, load, read, record, require, source_binding, verify

# Predeclared CPU/GPU FP32 tolerances; never tune after observing a new run.
GEOMETRY_ATOL = 5e-4  # metres / m/s; original150-step FP32 integration.
COST_ATOL = 2e-2
RTOL = 1e-5
EXTRAS = {"candidate_raw_xy", "candidate_reference_xy", "candidate_reference_velocity_xy",
          "candidate_goal_cost", "candidate_static_cost", "candidate_pair_cost", "candidate_joint_cost",
          "candidate_waypoints", "candidate_scene_ids"}
BASE_FIELDS = {"control_step", "actor_ids", "selected_sample", "raw_xy", "reference_xy",
               "reference_velocity_xy", "initial_overspeed", "speed_violation", "deviation_m"}


def audit_candidates(run, result, domain, checks, scene_configuration):
    import numpy as np
    import torch
    import yaml
    from CrowdSim.humanoid.training.reachable_selection import select_reachable_candidates
    from CrowdSim.humanoid.training.reference import CausalReferenceAdapter, ReferenceAdapterConfig

    require(not torch.cuda.is_initialized(), "Candidate audit cannot initialize CUDA")
    with np.load(run/"reference_plans.npz", allow_pickle=False) as payload:
        values = {key: payload[key] for key in payload.files}
    checks.require(set(values) == BASE_FIELDS | EXTRAS, "Exact original nine fields plus all nine candidate extras")
    n = len(values["control_step"])
    checks.require(n > 0 and n % 4 == 0, "Every complete joint plan has four actor rows")
    shapes = {"control_step": (n,), "actor_ids": (n,), "selected_sample": (n,),
              "raw_xy": (n, 51, 2), "reference_xy": (n, 151, 2), "reference_velocity_xy": (n, 151, 2),
              "initial_overspeed": (n,), "speed_violation": (n, 151), "deviation_m": (n, 151),
              "candidate_raw_xy": (n, 10, 51, 2), "candidate_reference_xy": (n, 10, 151, 2),
              "candidate_reference_velocity_xy": (n, 10, 151, 2), "candidate_waypoints": (n, 10, 2),
              "candidate_scene_ids": (n,), **{key: (n, 10) for key in EXTRAS if key.endswith("cost")}}
    integer = {"control_step", "actor_ids", "selected_sample", "candidate_scene_ids"}
    boolean = {"initial_overspeed", "speed_violation"}
    for key, value in values.items():
        dtype = np.dtype(np.int64 if key in integer else np.bool_ if key in boolean else np.float32)
        checks.require(value.shape == shapes[key] and value.dtype == dtype, "Candidate shape/dtype: " + key)
        if key not in integer | boolean:
            checks.require(np.isfinite(value).all(), "Every candidate retained and finite: " + key)
        if key.endswith("cost"):
            checks.require((value >= 0).all(), "Every archived cost nonnegative: " + key)
    controls = values["control_step"].reshape(-1, 4)
    starts = controls[:, 0]
    checks.require(np.array_equal(controls, np.broadcast_to(starts[:, None], controls.shape))
                   and starts[0] == 0 and starts[-1] < 6000 and (np.diff(starts) > 0).all()
                   and (np.diff(starts) <= 15).all(), "Complete ordered jointplans and original maximum15-step clock")
    checks.equal(values["actor_ids"], np.tile(np.arange(4), len(starts)), "Canonical0..3 actor rows in every joint plan")
    checks.require((values["candidate_scene_ids"] == 0).all(), "One original shared four-actor fixture scene")
    with np.load(run/"physical_states.npz", allow_pickle=False) as payload:
        physical = {key: payload[key] for key in ("root_pos", "episode_ids", "reset_mask", "terminated", "truncated",
                    "target_xy", "raw_target_xy", "reference_target_velocity_xy", "global_waypoints_xy", "route_vertex_reached")}
    # f851 independently audits complete shapes, reset/episode and ground fields.
    resets = physical["reset_mask"]
    checks.require(resets.shape == (6000, 4) and resets[0].all(), "All initial/reset actor controls retained")
    expected_replans = np.flatnonzero(resets.any(-1))
    arrivals = np.flatnonzero(physical["route_vertex_reached"][:-1].any(-1))+1
    checks.require(set(expected_replans) <= set(starts) and set(arrivals) <= set(starts), "Every reset and route-arrival forces next joint replan")
    scene_path = Path(result["arguments"]["scene_config"]).absolute()
    checks.require(str(scene_path) == scene_configuration["path"], "Scene path matches admitted source identity")
    verify(scene_configuration)
    description = yaml.safe_load(scene_path.read_text())
    geometry = {"scene_bounds_xy": description["bounds_xy"],
                "obstacle_centers_xy": [row["center"][:2] for row in description["cuboids"]],
                "obstacle_sizes_xy": [row["size"][:2] for row in description["cuboids"]]}
    adapter = CausalReferenceAdapter(ReferenceAdapterConfig(max_speed_mps=1.5, max_acceleration_mps2=2., initial_speed_policy="brake"))
    checks.require(result["reference_adaptation"]["configuration"] == adapter.describe()["configuration"], "Original full adapter configuration unchanged")
    times = torch.arange(1, 11, dtype=torch.float32)/2
    checks.equal(result["navigation_guidance"]["waypoint_times_s"], times.numpy(), "Original timed guides .5..5s")
    maxima = {"geometry": 0., "cost": 0., "executed_target": 0.}
    prior, prior_start = None, None
    for group, start in enumerate(starts):
        rows = slice(4*group, 4*group+4)
        tensors = {key: torch.from_numpy(value[rows]) for key, value in values.items()}
        raw, xy, velocity = (tensors[key] for key in ("candidate_raw_xy", "candidate_reference_xy", "candidate_reference_velocity_xy"))
        selected = tensors["selected_sample"]
        checks.require(bool(((selected >= 0) & (selected < 10)).all()) and bool((selected == selected[0]).all()), "Shared valid selected sample at control" + str(start))
        checks.equal(tensors["candidate_joint_cost"].numpy(), tensors["candidate_joint_cost"][0].expand(4, -1).numpy(), "Same archived joint cost per actor")
        checks.equal(selected.numpy(), tensors["candidate_joint_cost"].argmin(-1).numpy(), "Actual selected is saved GPU joint-cost argmin")
        checks.equal(raw[:, :, 0].numpy(), raw[:, :1, 0].expand(-1, 10, -1).numpy(), "Same actual t0 for all candidates")
        checks.equal(xy[:, :, 0].numpy(), xy[:, :1, 0].expand(-1, 10, -1).numpy(), "Same previous-reference t0 for all candidates")
        checks.equal(velocity[:, :, 0].numpy(), velocity[:, :1, 0].expand(-1, 10, -1).numpy(), "Same previous-reference velocity for all candidates")
        reset = resets[start]
        if reset.any():
            checks.close(xy[reset, 0, 0], raw[reset, 0, 0], "Reset initializes reference from saved current physical rawt0", atol=GEOMETRY_ATOL, rtol=RTOL)
        if prior is not None:
            delta = int(start-prior_start)
            alive = ~reset
            if alive.any():
                checks.close(xy[alive, 0, 0], prior["reference_xy"][alive, delta], "Ordinary replan inherits previous executed reference XY", atol=GEOMETRY_ATOL, rtol=RTOL)
                checks.close(velocity[alive, 0, 0], prior["reference_velocity_xy"][alive, delta], "Ordinary replan inherits previous executed reference velocity", atol=GEOMETRY_ATOL, rtol=RTOL)
                checks.close(raw[alive, 0, 0], physical["root_pos"][start-1, alive, :2], "Ordinary rawt0 matches previous captured physical root", atol=GEOMETRY_ATOL, rtol=RTOL)
                checks.equal(physical["episode_ids"][start, alive], physical["episode_ids"][prior_start, alive], "Continuity never crosses an actor episode")
        scored = select_reachable_candidates(raw, adapter, xy[:, 0, 0], velocity[:, 0, 0],
                    tensors["candidate_scene_ids"], tensors["candidate_waypoints"], times,
                    selection_domain=domain, waypoint_weight=20., agent_collision_weight=5., agent_collision_radius=.6,
                    scene_collision_weight=10., scene_collision_radius=.8, **geometry)
        original_peer = scored["peer_cost"]
        expected = {"candidate_reference_xy": scored["adapted_xy"], "candidate_reference_velocity_xy": scored["adapted_velocity"],
                    "candidate_goal_cost": scored["goal_cost"], "candidate_static_cost": scored["static_cost"],
                    "candidate_pair_cost": (original_peer.sum(0)+original_peer.sum(1))/2,
                    "candidate_joint_cost": scored["scene_sample_cost"]}
        for key, value in expected.items():
            kind = "cost" if key.endswith("cost") else "geometry"
            error = float((tensors[key]-value).abs().max())
            maxima[kind] = max(maxima[kind], error)
            checks.close(tensors[key], value, "All candidates original-helper reconstruction: " + key,
                         atol=COST_ATOL if kind == "cost" else GEOMETRY_ATOL, rtol=RTOL)
        checks.close(tensors["candidate_pair_cost"].sum(0), original_peer.sum((0, 1)), "Half-incident pair accounting conserves each unordered pair once", atol=COST_ATOL, rtol=RTOL)
        checks.close(tensors["candidate_joint_cost"][0], (20*tensors["candidate_goal_cost"]+10*tensors["candidate_static_cost"]+5*tensors["candidate_pair_cost"]).sum(0), "Archived weighted joint cost conservation", atol=COST_ATOL, rtol=RTOL)
        actors = torch.arange(4)
        picked = {"raw_xy": raw[actors, selected], "reference_xy": xy[actors, selected], "reference_velocity_xy": velocity[actors, selected],
                  "initial_overspeed": scored["initial_overspeed"][actors, selected], "speed_violation": scored["speed_violation"][actors, selected],
                  "deviation_m": scored["deviation_m"][actors, selected]}
        for key, value in picked.items():
            if value.dtype == torch.bool:
                checks.equal(tensors[key], value, "Selected adapter boolean flag: " + key)
            else:
                checks.close(tensors[key], value, "Selected raw/reference/velocity/deviation: " + key, atol=GEOMETRY_ATOL, rtol=RTOL)
        stop = int(starts[group+1]) if group+1 < len(starts) else 6000
        length = stop-int(start)
        checks.require(1 <= length <= 15, "All6000 executed controls covered exactly once by <=15-step plans")
        for key, reference in (("target_xy", picked["reference_xy"]), ("reference_target_velocity_xy", picked["reference_velocity_xy"]),
                               ("raw_target_xy", scored["raw_dense_xy"][actors, selected])):
            expected_frames = reference[:, 1:length+1].transpose(0, 1).numpy()
            error = float(np.abs(physical[key][start:stop]-expected_frames).max())
            maxima["executed_target"] = max(maxima["executed_target"], error)
            checks.close(physical[key][start:stop], expected_frames, "Every executed control matches its selected same-phase reference: " + key, atol=GEOMETRY_ATOL, rtol=RTOL)
        checks.close(tensors["candidate_waypoints"], physical["global_waypoints_xy"][start], "Saved original world guide points", atol=GEOMETRY_ATOL, rtol=RTOL)
        prior, prior_start = tensors, int(start)
    require(not torch.cuda.is_initialized(), "Pure CPU audit must leave CUDA uninitialized")
    return {"joint_plans": len(starts), "actor_plan_rows": n, "candidates_per_actor": 10,
            "all6000_controls_and24000_actor_controls_retained": True, "all_candidate_costs_and_unsafe_plans_retained": True,
            "maximum_absolute_reconstruction_errors": maxima, "geometry_atol": GEOMETRY_ATOL, "cost_atol": COST_ATOL, "rtol": RTOL,
            "CPU_GPU_FP32_bitwise_equivalence_claimed": False, "CPU_argmin_bitwise_equivalence_required": False,
            "GPU_saved_joint_argmin_strictly_verified": True,
            "initial_reset_root_velocity_independently_measured": False,
            "initial_reset_velocity_source": "saved candidate reference velocity at t0, identical across allS; original reset source contract; no independent pre-step rootV capture",
            "collision_costs_are_geometric_proxies_not_actual_contact": True}


def audit(reg, domain, owner):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Explicit CUDA-hidden CPU process required")
    require(not any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules), "Admission starts before Torch/SDK imports")
    admission = admit(reg, domain, owner)
    run = Path(reg["cases"][domain]["output"])
    verify(reg["support_sources"]["raw_math"])
    raw = load("common50_original_f851_raw", reg["support_sources"]["raw_math"]["path"])
    original = raw.verify_sources
    try:
        raw.verify_sources = lambda *_args, **_kwargs: admission
        checks = raw.Audit()
        report = raw.audit_capture(run, checks)
    finally:
        raw.verify_sources = original
    counts = report["counts"]
    require(counts["controls"] == 6000 and counts["actors"] == 4 and counts["duration_s"] == 200., "Original full4actor200s denominator required")
    candidates = audit_candidates(run, read(run/"result.json"), domain, checks, admission["scene_configuration"])
    contacts = report["original_obstacle_peer_channels"]
    gate = {"controls": counts["controls"], "actors": counts["actors"], "duration_s": counts["duration_s"],
            "falls": counts["falls"], "static_actor_controls": contacts["static_contact_actor_frames"],
            "peer_pair_controls": contacts["peer_contact_unordered_pair_frames"]}
    post = source_binding(reg, domain)
    require(post == {key: admission[key] for key in post}, "Registered sources changed during CPU read")
    for row in reg["auditor_sources"]:
        verify(row)
    report.update(schema="local_reference.common50_independent_CPU_audit.v1", status="passed", domain=domain,
                  actual_capture_independently_audited=True, actual_source_admission=admission,
                  physical_gate_counts=gate,
                  physical_zero_gate_passed=all(gate[key] == 0 for key in ("falls", "static_actor_controls", "peer_pair_controls")),
                  candidate_consistency=candidates, checks=checks.checks, execution_reader=record(__file__),
                  reader_registration=record(HERE/"future_audit_registration_v1.json"), source_post_observation=post,
                  original_fixture_math=reg["support_sources"]["raw_math"], raw_math_changed=False,
                  source_admission_hook_only=True, all_failed_contact_idle_reset_samples_retained=True,
                  style_or_physical_zero_used_for_admission=False, no_default_promotion_claimed=True,
                  reader_loaded_model_or_weight_tensors=False, SDK_started_by_reader=False, GPU_used_by_reader=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--domain", choices=("matched_raw50", "reachable50"), required=True)
    parser.add_argument("--owner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reg_path = HERE/"future_audit_registration_v1.json"
    record(reg_path, sealed=True)
    reg = read(reg_path)
    require(str(args.output.absolute()) == reg["cases"][args.domain]["audit"] and not args.output.exists(), "Preserve prior audits at the registered new output")
    try:
        report = audit(reg, args.domain, args.owner)
    except BaseException as error:
        report = {"schema": "local_reference.common50_independent_CPU_audit.v1", "status": "failed", "domain": args.domain,
                  "failure": repr(error), "actual_capture_independently_audited": False}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "counts": report["counts"], "candidate_consistency": report["candidate_consistency"],
                      "physical_gate_counts": report["physical_gate_counts"], "physical_zero_gate_passed": report["physical_zero_gate_passed"]}))


if __name__ == "__main__":
    main()
