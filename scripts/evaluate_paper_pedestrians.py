#!/usr/bin/env python3
"""CPU evaluation of recorded physical pedestrians against disjoint AMASS data.

Reads train_paper_amp.py physical_states.npz and its sibling result.json. Fixed
train-only K50/100 centers, deterministic sample identities, true Gaussian MMD,
and complete reset-aware physical statistics are written to a separate folder.
Explicit scene geometry proxies are reported separately from recorded contacts.
No Kit/GPU process, pretrained policy or author-match claim is part of this tool.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-states", "--states", type=Path, required=True)
    parser.add_argument("--runner-result", type=Path, help="defaults to physical_states.npz's sibling result.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/diffusion_amp/amass_manifest.json")
    parser.add_argument("--output-dir", type=Path, help="defaults to the capture folder's pedestrian_metrics subfolder")
    parser.add_argument("--reference-samples", type=int, default=2048, help="poses sampled independently from train and test")
    parser.add_argument("--generated-samples", type=int, default=2048, help="maximum evenly spaced time-major pose samples")
    parser.add_argument("--seed", type=int, default=7, help="same declared CPU seed for each reference split and train kmeans")
    parser.add_argument("--mmd-chunk-size", type=int, default=512)
    parser.add_argument("--kmeans-max-iterations", type=int, default=200)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--ground-height", type=float, default=0., help="explicit constant world-ground height in meters")
    parser.add_argument("--proxy-agent-radius", type=float, default=.3, help="explicit shared-world XY disc radius; geometry proxy only")
    parser.add_argument("--verify-package-hashes", action="store_true", help="recompute all multi-GB AMASS source hashes")
    parser.add_argument("--input-kind", choices=("physical_rollout", "synthetic_interface"), default="physical_rollout")
    args = parser.parse_args(argv)
    if args.reference_samples < 100 or min(args.generated_samples, args.mmd_chunk_size, args.kmeans_max_iterations, args.cpu_threads) < 1:
        parser.error("Reference samples must be >=100; generated/chunk/iteration/thread bounds must be positive")
    args.physical_states = args.physical_states.resolve()
    args.runner_result = (args.runner_result or args.physical_states.with_name("result.json")).resolve()
    args.manifest = args.manifest.resolve()
    args.output_dir = (args.output_dir or args.physical_states.parent / "pedestrian_metrics").resolve()
    for path in (args.physical_states, args.runner_result, args.manifest):
        if not path.is_file():
            parser.error(f"Required input is missing: {path}")
    if args.output_dir == args.physical_states.parent:
        parser.error("Use a separate metrics output directory; do not overwrite the runner's result.json")
    return args


def read_capture(states_path, runner_result):
    import numpy as np

    report = json.loads(Path(runner_result).read_text())
    capture = report.get("state_capture")
    if not isinstance(capture, dict):
        raise ValueError("Runner result must declare state_capture metadata; body ordering/frequency cannot be guessed")
    names = capture.get("body_names")
    if not isinstance(names, list) or len(names) != 24 or len(set(names)) != 24:
        raise ValueError("Runner state_capture requires 24 unique explicit body_names")
    if capture.get("frequency_hz") != 30:
        raise ValueError("This evaluation protocol requires a declared 30Hz physical capture")
    if "first post-step frame" not in capture.get("reset_mask", ""):
        raise ValueError("Runner reset_mask must declare first post-step frame of each new episode")
    coordinates = capture.get("coordinates", "")
    if "XYZW" not in coordinates or "world meters" not in coordinates:
        raise ValueError("Runner must explicitly declare world meters and XYZW quaternions")
    if report.get("status") != "passed":
        raise ValueError("Runner did not finish successfully; incomplete capture is not a completed evaluation")
    required = {"root_pos", "body_pos", "root_rot", "dof_pos", "target_xy", "episode_ids", "reset_mask"}
    with np.load(states_path, allow_pickle=False) as archive:
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Physical capture lacks required fields: {sorted(missing)}")
        optional = {"terminated", "truncated", "map_collision_mask", "agent_collision_mask",
                    "global_goal_xy", "plan_endpoint_goal_error", "static_contact", "peer_contact",
                    "static_contact_max_force_n", "peer_contact_max_force_n"}
        arrays = {key: archive[key] for key in required | (optional & set(archive.files))}
    root = arrays["root_pos"]
    if root.ndim != 3 or root.shape[-1] != 3 or root.shape[0] < 2 or root.shape[1] < 1:
        raise ValueError("root_pos requires [T>=2,B>=1,3]")
    shape = root.shape[:2]
    expected = {"root_pos": (*shape, 3), "body_pos": (*shape, 24, 3), "root_rot": (*shape, 4),
                "dof_pos": (*shape, 69), "target_xy": (*shape, 2), "global_goal_xy": (*shape, 2),
                "plan_endpoint_goal_error": shape, "episode_ids": shape, "reset_mask": shape,
                "static_contact": shape, "static_contact_max_force_n": shape,
                "peer_contact": (*shape, shape[1]), "peer_contact_max_force_n": (*shape, shape[1])}
    for key, value in arrays.items():
        if value.shape != expected.get(key, shape):
            raise ValueError(f"{key} shape {value.shape} differs from declared {expected.get(key, shape)}")
        if value.dtype.kind not in "biuf" or not np.isfinite(value).all():
            raise ValueError(f"{key} must be finite numeric data without Python objects")
    for key in ("reset_mask", "terminated", "truncated", "map_collision_mask", "agent_collision_mask", "static_contact", "peer_contact"):
        if key in arrays and arrays[key].dtype != np.bool_:
            raise ValueError(f"{key} requires a genuine boolean mask")
    if arrays["episode_ids"].dtype.kind not in "iu":
        raise ValueError("episode_ids requires integer [T,B]")
    if "plan_endpoint_goal_error" in arrays and (arrays["plan_endpoint_goal_error"] < 0).any():
        raise ValueError("plan_endpoint_goal_error is a distance and cannot be negative")
    if "Pelvis" not in names or not np.allclose(arrays["body_pos"][..., names.index("Pelvis"), :], root, atol=1e-5, rtol=0):
        raise ValueError("Captured root_pos disagrees with its explicitly named Pelvis body position")
    validation = {"array_shapes": {key: list(value.shape) for key, value in arrays.items()},
                  "body_names": names, "frequency_hz": 30, "root_matches_named_pelvis": True,
                  "allow_pickle": False, "reset_mask_marks": "before_frame"}
    contact_fields = {"static_contact", "peer_contact", "static_contact_max_force_n", "peer_contact_max_force_n"}
    if contact_fields & arrays.keys():
        if not contact_fields <= arrays.keys():
            raise ValueError("Pairwise contact capture must include all four recorded mask/force fields")
        import torch
        from CrowdSim.humanoid.evaluation import pairwise_contact_metrics
        pairwise_contact_metrics(shape=shape, valid=torch.ones(shape, dtype=torch.bool),
            boundaries=torch.zeros((shape[0]-1, shape[1]), dtype=torch.bool),
            **{key: arrays[key] for key in contact_fields},
            contact_contract=report.get("physical_contact_measurement"))
        validation["pairwise_contact_fields"] = "all four shapes, genuine boolean masks, finite nonnegative forces, strict threshold consistency and explicit contract verified"
    return arrays, report, validation


def evenly_spaced_indices(total, maximum):
    """Distinct integer endpoints-inclusive indices in flattened time-major order."""
    import numpy as np

    count = min(total, maximum)
    if count == 1:
        return np.array([0], dtype=np.int64)
    return np.linspace(0, total - 1, count, dtype=np.int64)


def _distance_statistics(value):
    import numpy as np

    values = np.asarray(value, dtype=np.float64)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Distance statistics require finite nonnegative [T,B]")
    return {"recorded_frame_agent_samples": values.size, "mean_m": float(values.mean()),
            "minimum_m": float(values.min()), "maximum_m": float(values.max()),
            "rms_m": float(np.sqrt(np.mean(values ** 2))),
            "p05_m": float(np.quantile(values, .05)), "median_m": float(np.median(values)),
            "p95_m": float(np.quantile(values, .95)),
            "first_recorded_m_per_agent": values[0].tolist(), "last_recorded_m_per_agent": values[-1].tolist()}


def summarize_goal_metrics(arrays, runner):
    """Keep tracking, sparse conditioning goals and route advancement distinct."""
    import numpy as np

    report = {"distance_to_recorded_goal_xy": None, "recorded_plan_endpoint_goal_residual": None,
              "unchanged_goal_distance_progress": None, "route_vertices_reached": runner.get("route_vertices_reached"),
              "global_path_failures": runner.get("global_path_failures"),
              "global_path_method": runner.get("global_path_method"), "global_arrival_rate": None,
              "goal_semantics": "Sparse diffusion-conditioning goal; arrival at a full route/global destination is not established by tracking error."}
    if report["route_vertices_reached"] is not None:
        vertices = np.asarray(report["route_vertices_reached"])
        if vertices.shape != (arrays["root_pos"].shape[1],) or vertices.dtype.kind not in "iu" or (vertices < 0).any():
            raise ValueError("route_vertices_reached requires one nonnegative integer count per actor")
        report["route_vertices_reached_total"] = int(vertices.sum())
    failures = report["global_path_failures"]
    if failures is not None and (type(failures) is not int or failures < 0):
        raise ValueError("global_path_failures requires a nonnegative integer recorded query-failure count")
    scene = runner.get("evaluation_scene")
    if scene is not None:
        report["goal_semantics"] += " In this fixture the recorded global_goal_xy is an A* arclength-lookahead point, which can change at replanning."
    if "global_goal_xy" in arrays:
        goal = np.asarray(arrays["global_goal_xy"], dtype=np.float64)
        positions = np.asarray(arrays["root_pos"], dtype=np.float64)
        if goal.shape != (*positions.shape[:2], 2) or not np.isfinite(goal).all():
            raise ValueError("global_goal_xy must be finite time-aligned [T,B,2]")
        distance = np.linalg.norm(positions[..., :2] - goal, axis=-1)
        report["distance_to_recorded_goal_xy"] = _distance_statistics(distance)
        same_goal = np.max(np.abs(goal[1:] - goal[:-1]), axis=-1) <= 1e-7
        same_episode = arrays["episode_ids"][1:] == arrays["episode_ids"][:-1]
        comparable = same_goal & same_episode & ~arrays["reset_mask"][1:]
        reduction = (distance[:-1] - distance[1:])[comparable]
        report["unchanged_goal_distance_progress"] = {
            "comparable_same_goal_episode_transitions": int(comparable.sum()), "goal_equality_tolerance_m": 1e-7,
            "distance_reduction_sum_m": float(reduction.sum()) if reduction.size else None,
            "distance_reduction_mean_m": float(reduction.mean()) if reduction.size else None,
            "goal_changes_within_episode_per_agent": ((~same_goal) & same_episode).sum(0).tolist(),
            "definition": "Previous minus current root-goal distance only while the recorded goal and episode stay unchanged; reset jumps excluded."}
    if "plan_endpoint_goal_error" in arrays:
        report["recorded_plan_endpoint_goal_residual"] = {
            **_distance_statistics(arrays["plan_endpoint_goal_error"]),
            "definition": "Recorded distance from the held diffusion plan's final XY point to its conditioning goal; not physical root arrival error.",
            "weighting": "One value per recorded physical frame; a plan can repeat until replanning."}
    report["notes"] = "A* failure can set the conditioning goal to the current root. Small goal distance alone does not establish route/global arrival."
    return report


def geometry_proxy_metrics(arrays, runner, *, agent_radius_m=.3, ground_height=0., fall_height_m=.45, fall_up_z=.1):
    """Use the same declared cuboid footprints; retain actual contacts separately."""
    import numpy as np
    import torch
    from CrowdSim.humanoid.evaluation import planar_agent_overlap, rollout_metrics

    if not np.isfinite(agent_radius_m) or agent_radius_m <= 0:
        raise ValueError("proxy-agent-radius must be a finite positive explicit radius")
    map_flags = agent_flags = None
    map_definition = None
    scene_description = runner.get("evaluation_scene")
    if scene_description is not None:
        from CrowdSim.humanoid.reconstruction_scene import ReconstructionScene

        scene = ReconstructionScene(scene_description, device="cpu")
        xy = scene.tensor(arrays["root_pos"][..., :2])
        inside_obstacle = scene.occupied(xy)
        outside_domain = ~scene.in_domain(xy)
        map_flags = inside_obstacle | outside_domain
        map_definition = {"method": "Shared fixture ReconstructionScene.occupied XY root-point cuboid footprint OR outside declared bounds; no PhysX contact data",
                          "scene_name": scene_description.get("name"), "bounds_xy": scene_description["bounds_xy"],
                          "cuboids": scene_description.get("cuboids", []), "outside_domain_is_overlap": True,
                          "inside_cuboid_footprint_frame_agents": int(inside_obstacle.sum()),
                          "outside_domain_frame_agents": int(outside_domain.sum()),
                          "z_geometry": "XY projection: no full-body footprint, vertical contact, or shape intersection test"}
    if runner.get("physics", {}).get("isolated_environments") is False:
        agent_flags = planar_agent_overlap(torch.as_tensor(arrays["root_pos"]), radius_m=agent_radius_m)
    report = {"map_root_point_geometry_proxy": None, "agent_xy_disc_geometry_proxy": None,
              "actual_pair_contacts_available": "agent_collision_mask" in arrays,
              "notes": "Geometry proxies are separate estimates. Missing actual contact masks remain None in rollout_metrics."}
    if map_flags is None and agent_flags is None:
        return report
    stats = rollout_metrics(arrays["root_pos"], arrays["root_rot"], episode_ids=arrays["episode_ids"].astype(np.int64),
                            reset_mask=arrays["reset_mask"], reset_mask_marks="before_frame", dt=1 / 30,
                            ground_height=ground_height, fall_height_m=fall_height_m, fall_up_z=fall_up_z,
                            map_collision_mask=map_flags, agent_collision_mask=agent_flags,
                            map_collision_definition="explicit XY root-point cuboid/outside-domain proxy",
                            agent_collision_definition="explicit shared-world XY disc overlap proxy")
    if map_flags is not None:
        report["map_root_point_geometry_proxy"] = {**stats["map_collisions"], **map_definition}
    if agent_flags is not None:
        report["agent_xy_disc_geometry_proxy"] = {**stats["agent_collisions"], "radius_m": agent_radius_m,
                                                 "shared_world_declared": True, "self_pairs_excluded": True}
    return report


def main(argv=None):
    args = parse_args(argv)
    started = time.monotonic()
    import numpy as np
    import torch

    from CrowdSim.humanoid.amass_corpus import AMASSCorpus
    from CrowdSim.humanoid.amp.observations import PACER_BODY_NAMES
    from CrowdSim.humanoid.evaluation import PoseEmbeddingSpec, fit_pose_clusters, gaussian_mmd, pose_embedding, rollout_metrics
    from CrowdSim.humanoid.training.manifests import validate_training_manifest
    from protomotions.utils.checkpoint import atomic_write_text

    torch.set_num_threads(args.cpu_threads)
    arrays, runner, capture_validation = read_capture(args.physical_states, args.runner_result)
    manifest_gate = validate_training_manifest(args.manifest, verify_package_hashes=args.verify_package_hashes)
    train = AMASSCorpus(args.manifest, "train", device="cpu", seed=args.seed)
    test = AMASSCorpus(args.manifest, "test", device="cpu", seed=args.seed)
    training_state, _, _, train_ids, train_frames = train.sample_reference(args.reference_samples)
    test_state, _, _, test_ids, test_frames = test.sample_reference(args.reference_samples)
    train_features = pose_embedding(training_state.rigid_body_pos, training_state.root_rot, body_names=PACER_BODY_NAMES)
    test_features = pose_embedding(test_state.rigid_body_pos, test_state.root_rot, body_names=PACER_BODY_NAMES)
    total_frames, agents = arrays["root_pos"].shape[:2]
    flattened = evenly_spaced_indices(total_frames * agents, args.generated_samples)
    frame_indices, agent_indices = flattened // agents, flattened % agents
    generated_features = pose_embedding(arrays["body_pos"][frame_indices, agent_indices],
                                        arrays["root_rot"][frame_indices, agent_indices],
                                        body_names=capture_validation["body_names"])
    if not np.isfinite(args.ground_height):
        raise ValueError("ground-height must be a finite explicit scalar")
    definition = runner.get("physics", {}).get("fall_definition", {})
    fall_height = definition.get("pelvis_ground_height_less_than_m", .45)
    fall_up = definition.get("root_up_z_less_than", .1)
    physical = rollout_metrics(arrays["root_pos"], arrays["root_rot"], target_xy=arrays["target_xy"],
                               episode_ids=arrays["episode_ids"].astype(np.int64), reset_mask=arrays["reset_mask"],
                               reset_mask_marks="before_frame", dt=1 / 30, ground_height=args.ground_height,
                               fall_height_m=fall_height, fall_up_z=fall_up,
                               map_collision_mask=arrays.get("map_collision_mask"),
                               agent_collision_mask=arrays.get("agent_collision_mask"),
                               map_collision_definition=runner.get("map_collision_definition", "recorded boolean collision mask; definition not declared"),
                               agent_collision_definition=runner.get("agent_collision_definition", "recorded boolean collision mask; definition not declared"),
                               static_contact=arrays.get("static_contact"), peer_contact=arrays.get("peer_contact"),
                               static_contact_max_force_n=arrays.get("static_contact_max_force_n"),
                               peer_contact_max_force_n=arrays.get("peer_contact_max_force_n"),
                               contact_contract=runner.get("physical_contact_measurement"))
    goal_metrics = summarize_goal_metrics(arrays, runner)
    proxies = geometry_proxy_metrics(arrays, runner, agent_radius_m=args.proxy_agent_radius,
                                    ground_height=args.ground_height, fall_height_m=fall_height, fall_up_z=fall_up)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_provenance, test_provenance = train.provenance(), test.provenance()
    source_id = f"manifest_sha256={train_provenance['manifest_sha256']};train_package_sha256={train_provenance['source_sha256']}"
    models = fit_pose_clusters(train_features, split="train", source_id=source_id, seed=args.seed,
                               max_iterations=args.kmeans_max_iterations)
    clusters = {}
    for k, model in models.items():
        path = args.output_dir / f"train_pose_clusters_k{k}.json"
        model.save(path)
        clusters[str(k)] = {"train_reference": model.score(train_features, split="train"),
                            "test_reference": model.score(test_features, split="test"),
                            "generated": model.score(generated_features, split="generated"),
                            "artifact": {"path": str(path), "bytes": path.stat().st_size, "sha256": file_hash(path)},
                            "iterations": model.iterations}
    mmd = gaussian_mmd(generated_features, test_features, bandwidth=1.0, chunk_size=args.mmd_chunk_size)
    reference_mmd = gaussian_mmd(train_features, test_features, bandwidth=1.0, chunk_size=args.mmd_chunk_size)
    report = {
        "schema": "navisaaclab.pedestrian_evaluation_report.v1", "status": "evaluated",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "input_kind": args.input_kind,
        "device": "cpu", "cpu_threads": args.cpu_threads,
        "author_experiment_reproduced": False, "author_parameters_reproduced": False, "convergence_claim": False,
        "sources": {"physical_states": {"path": str(args.physical_states), "bytes": args.physical_states.stat().st_size,
                                        "sha256": file_hash(args.physical_states)},
                    "runner_result": {"path": str(args.runner_result), "sha256": file_hash(args.runner_result)},
                    "train": train_provenance, "test": test_provenance},
        "manifest_gate": manifest_gate, "capture_validation": capture_validation,
        "embedding_choices": asdict(PoseEmbeddingSpec()),
        "reference_sampling": {"seed_per_split": args.seed, "samples_per_split": args.reference_samples,
            "algorithm": "AMASSCorpus.sample_reference: uniformly select declared clip, then an eligible 30Hz control frame",
            "body_names": list(PACER_BODY_NAMES),
            "train": {"split": "train", "motion_ids": train_ids.tolist(), "frame_indices": train_frames.tolist()},
            "test": {"split": "test", "motion_ids": test_ids.tolist(), "frame_indices": test_frames.tolist()}},
        "generated_sampling": {"maximum_samples": args.generated_samples, "available_frames": total_frames,
            "agents": agents, "available_poses": total_frames * agents, "selected_poses": len(flattened),
            "algorithm": "np.linspace(0,T*B-1,min(T*B,maximum),dtype=int64); flattened time-major T then B",
            "flat_indices": flattened.tolist(), "frame_indices": frame_indices.tolist(), "agent_indices": agent_indices.tolist(),
            "samples_per_agent": np.bincount(agent_indices, minlength=agents).tolist(),
            "pose_filter": "None: reset-first, terminal, fall and otherwise finite captured poses remain eligible"},
        "cluster_choices": {"fit_split": "train", "ks": [50, 100], "seed": args.seed,
                            "initialization": "kmeans++", "maximum_iterations": args.kmeans_max_iterations, "tolerance": 1e-7},
        "cluster_scores": clusters, "generated_vs_test_gaussian_mmd": mmd,
        "train_vs_test_gaussian_mmd_reference": reference_mmd, "rollout_metrics": physical,
        "goal_condition_metrics": goal_metrics, "geometry_proxies": proxies,
        "recorded_termination_masks": {"terminated_true_frames": int(arrays["terminated"].sum()) if "terminated" in arrays else None,
                                      "truncated_true_frames": int(arrays["truncated"].sum()) if "truncated" in arrays else None},
        "runner_context": {key: runner.get(key) for key in (
            "method", "arguments", "actual_control_steps", "fall_resets", "time_limit_resets", "state_capture", "evaluation_dataset",
            "evaluation_scene", "route_vertices_reached", "global_path_failures", "global_path_method",
            "physical_contact_measurement", "amp_reward_timestamp")},
        "ground_height_m": args.ground_height,
        "elapsed_cpu_analysis_s": time.monotonic() - started,
        "implementation_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in (
            ROOT / "scripts/evaluate_paper_pedestrians.py", ROOT / "CrowdSim/humanoid/evaluation.py",
            ROOT / "CrowdSim/humanoid/reconstruction_scene.py")},
        "limitations": [
            "The author did not publish complete 15-point IDs, scale, cluster/sampling protocol or MMD kernel/bandwidth; these choices are explicit local reconstruction settings.",
            "AMASS source hashes are declarations unless verify-package-hashes was enabled; package metadata/size and split-source disjointness were verified.",
            "Generated K50/100 assignments use the same centers fitted only on train. Test/reference/generated data never refit centers.",
            "MMD is a sampled per-pose Gaussian biased V-statistic at fixed sigma=1, not a recovered author motion-sequence metric.",
            "Coverage and entropy depend on sample count and the deterministic sampling schedule.",
            "Absent actual contact fields remain None; no geometric proxy is silently substituted. Recorded pairwise contacts are strict normal-force threshold events with an explicit all-substep maximum contract.",
            "Contact actor frames, unordered pair frames and reset-aware contiguous incidences are local counts; the authors' collision aggregation protocol is unknown.",
            "Scene cuboid/outside-domain root-point and shared-world XY-disc proxies are explicit separate estimates, not actual PhysX pair contact.",
            "Recorded global_goal_xy is a planner-conditioning goal; in the fixture it is an A* lookahead point. Tracking and endpoint residuals do not establish arrival at a full route/global destination.",
            "Observed episodes can be censored; no navigation success contract is supplied, so success_rate remains None.",
            "This single recorded capture does not supply the authors' eight-scene protocol or Animated People/SFM+AMP baseline.",
        ],
    }
    if args.input_kind == "synthetic_interface":
        report["limitations"].insert(0, "SYNTHETIC INTERFACE VALIDATION: supplied capture is not an actual simulated physical rollout.")
    atomic_write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", args.output_dir / "evaluation.json")
    lines = ["# 行人姿态与物理记录评估", "", f"输入类型：`{args.input_kind}`；生成时间：{report['generated_at_utc']}。",
             "", f"CPU 读取 {total_frames} 帧 × {agents} 人，选取 {len(flattened)} 个生成姿态；AMASS train/test 各 {args.reference_samples} 个姿态，seed={args.seed}。",
             "", "## 固定 train cluster 与 held-out", "", "| K | train coverage | test coverage | generated coverage | generated entropy |", "| --- | --- | --- | --- | --- |"]
    for k, scores in clusters.items():
        lines.append(f"| {k} | {scores['train_reference']['coverage']:.6f} | {scores['test_reference']['coverage']:.6f} | {scores['generated']['coverage']:.6f} | {scores['generated']['normalized_pose_entropy']:.6f} |")
    lines.extend(["", f"Generated ↔ test Gaussian MMD²={mmd['mmd_squared']:.9f}，MMD={mmd['mmd']:.9f}，固定 σ=1.0，biased V-statistic。",
                  f"Train ↔ test 参考 MMD²={reference_mmd['mmd_squared']:.9f}，MMD={reference_mmd['mmd']:.9f}。", "",
                  "## 完整物理序列统计", "", f"XY 路程（排除 episode/reset 跳跃）：{physical['walked_distance_xy_m_per_agent']} m。",
                  f"跌倒 flag 事件起点：{physical['falls']['observed_event_starts']}；受影响已观测 episode 比例：{physical['falls']['affected_episode_fraction']:.6f}。",
                  f"同刻 XY 跟踪平均误差：{physical['plan_tracking']['mean_xy_error_m']:.6f} m；RMSE：{physical['plan_tracking']['rmse_xy_m']:.6f} m。",
                  f"Episode ID 与 reset mask 边界不一致数：{physical['boundary_source_disagreement_count']}。",
                  f"地图碰撞：{'未知（无记录 mask）' if physical['map_collisions'] is None else physical['map_collisions']}；行人碰撞：{'未知（无记录 mask）' if physical['agent_collisions'] is None else physical['agent_collisions']}。",
                  "", "## 记录目标、计划终点与路线推进", "",
                  f"记录目标语义：{goal_metrics['goal_semantics']}",
                  f"已记录 route vertices reached：{goal_metrics['route_vertices_reached']}；A* query failures：{goal_metrics['global_path_failures']}。",
                  "全局/完整路线到达率保持未知；近时刻轨迹跟踪误差不代表到达。"])
    if goal_metrics["distance_to_recorded_goal_xy"] is not None:
        goal_distance = goal_metrics["distance_to_recorded_goal_xy"]
        lines.append(f"Physical root 到记录的 planner condition XY 目标距离：平均 {goal_distance['mean_m']:.6f} m，末帧各人 {goal_distance['last_recorded_m_per_agent']} m。")
    else:
        lines.append("NPZ 缺 global_goal_xy，记录目标距离未知。")
    if goal_metrics["recorded_plan_endpoint_goal_residual"] is not None:
        endpoint = goal_metrics["recorded_plan_endpoint_goal_residual"]
        lines.append(f"Diffusion 计划末端到记录目标 residual：平均 {endpoint['mean_m']:.6f} m，p95 {endpoint['p95_m']:.6f} m；按物理记录帧计，计划可能重复。")
    else:
        lines.append("NPZ 缺 plan_endpoint_goal_error，计划终点 residual 未知。")
    if physical["actual_contacts"] is not None:
        actual_contacts = physical["actual_contacts"]
        lines.extend(["", "## 记录的真实法向接触", "",
                      f"测量合同：{actual_contacts['measurement_contract']}。",
                      f"静态障碍 body 接触：{actual_contacts['static']}。",
                      f"行人 body 接触：{actual_contacts['peer']}。",
                      "actor-frame、无序 pair-frame 和按 reset/episode 分段的连续事件分别统计；作者论文的接触聚合协议未知。"])
    lines.extend(["", "## 显式几何代理", "",
                  f"静态 cuboid XY root-point/domain proxy：{proxies['map_root_point_geometry_proxy']}。",
                  f"共享场景 XY circle proxy：{proxies['agent_xy_disc_geometry_proxy']}。",
                  "这些代理单独报告；实际 PhysX contact 缺失仍为未知。",
                  "", "## 解释边界", "", "- 不是作者参数/作者权重/八场景 benchmark 的严格复现；不判断稳定、收敛或论文数字一致性。",
                  "- MMD 使用显式局部 45D 单帧姿态和固定核；节点、尺度、抽样量会影响数值。",
                  "- 两份 cluster artifact 只来自训练数据。完整源 hash、采样 motion/frame/agent 索引和未知项见 [evaluation.json](evaluation.json)。",
                  "- reset mask 使用 before_frame：每个新 episode 的首个 post-step 状态。物理路径仅计算同 episode 的相邻记录。"])
    if args.input_kind == "synthetic_interface":
        lines.insert(2, "**本报告仅验证合成输入接口；不是实际物理仿真结果。**")
    atomic_write_text("\n".join(lines) + "\n", args.output_dir / "report.md")
    print(json.dumps({"status": "evaluated", "input_kind": args.input_kind, "output_dir": str(args.output_dir),
                      "generated_samples": len(flattened), "reference_samples_per_split": args.reference_samples,
                      "mmd_squared": mmd["mmd_squared"], "boundary_disagreements": physical["boundary_source_disagreement_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
