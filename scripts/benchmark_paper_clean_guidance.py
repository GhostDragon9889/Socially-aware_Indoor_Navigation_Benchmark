"""CPU replay of clean guidance with actual captured states and static A* goals.

This does not simulate a new rollout. Capture-derived velocities are backward
finite differences, because historical NPZs did not retain native root velocity.
A* supplies only sparse conditions; every output path comes from the frozen DDPM.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODES = ("legacy_posterior", "trace_clean", "trace_clean_variance")


def capture_context(arrays, frame):
    """Infer a masked 3s context without crossing a physical episode reset."""
    position, quaternion, episode = (np.asarray(arrays[key]) for key in ("root_pos", "root_rot", "episode_ids"))
    if (position.ndim != 3 or position.shape[-1] != 3
            or quaternion.shape != (*position.shape[:2], 4) or episode.shape != position.shape[:2]
            or not isinstance(frame, int) or not 90 <= frame < len(position)
            or not np.isfinite(position).all() or not np.isfinite(quaternion).all()):
        raise ValueError("Finite world root positions, XYZW quaternions and a frame with 3s history required")
    indices = np.arange(frame - 90, frame + 1, 3)
    # Every retained velocity uses adjacent native30Hz frames, not 10Hz samples.
    previous = np.maximum(indices - 1, 0)
    available = ((episode[indices] == episode[frame]) & (episode[previous] == episode[indices])
                 & (indices[:, None] > 0))
    if not available[-1].all():
        raise ValueError("Choose a frame whose current root velocity does not cross a reset")
    q = quaternion[indices]
    yaw = np.arctan2(2 * (q[..., 3] * q[..., 2] + q[..., 0] * q[..., 1]),
                     1 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2))
    velocity = (position[indices, :, :2] - position[previous, :, :2]) * 30.
    history = np.concatenate((position[indices, :, :2], velocity, yaw[..., None]), axis=-1).transpose(1, 0, 2).astype(np.float32)
    mask = available.T
    current = history[:, -1].copy()
    history = np.where(mask[..., None], history, 0.)
    batch = len(current)
    if batch < 2:
        raise ValueError("Shared-scene comparison needs at least two actors")
    others = np.asarray([[j for j in range(batch) if j != i] for i in range(batch)])
    neighbors, neighbor_mask = history[others], mask[others]
    distance = np.linalg.norm(neighbors[:, :, -1, :2] - current[:, None, :2], axis=-1)
    neighbor_mask &= (distance <= 15.)[..., None]
    return {"states": current, "history": history, "history_mask": mask,
            "neighbors": neighbors, "neighbor_mask": neighbor_mask}


def scene_plan_metrics(states, xy, waypoints, times, scene_description):
    """Sparse target adherence and joint-plan geometric proxies, not contacts."""
    from scripts.train_orca_diffusion import polygon_surface_gap

    states, xy, waypoints, times = [np.asarray(value, dtype=np.float64) for value in (states, xy, waypoints, times)]
    batch = len(states)
    if (xy.shape != (batch, 51, 2) or waypoints.shape != (batch, len(times), 2)
            or not all(np.isfinite(value).all() for value in (states, xy, waypoints, times))
            or not np.allclose(xy[:, 0], states[:, :2], atol=1e-5, rtol=0)):
        raise ValueError("Finite current+50future XY with matching current positions and timed conditions required")
    selected = np.rint(times / .1).astype(np.int64)
    error = np.linalg.norm(xy[:, selected] - waypoints, axis=-1)
    first = xy[:, 5] - xy[:, 0]
    goal = waypoints[:, -1] - states[:, :2]
    norm = np.maximum(np.linalg.norm(goal, axis=-1), 1e-8)
    progress = (first * goal).sum(-1) / norm
    cosine = (first * goal).sum(-1) / np.maximum(norm * np.linalg.norm(first, axis=-1), 1e-8)
    polygons = []
    for box in scene_description["cuboids"]:
        center, size = np.asarray(box["center"])[:2], np.asarray(box["size"])[:2]
        lower, upper = center - size / 2, center + size / 2
        polygons.append(np.asarray([[lower[0], lower[1]], [upper[0], lower[1]], [upper[0], upper[1]], [lower[0], upper[1]]]))
    obstacle_gap = polygon_surface_gap(xy[:, 1:], polygons, radius=.4)
    bounds = np.asarray(scene_description["bounds_xy"])
    domain_gap = np.minimum(xy[:, 1:] - bounds[:2], bounds[2:] - xy[:, 1:]).min(-1) - .4
    obstacle_gap = np.minimum(obstacle_gap, domain_gap)
    joint_distance = np.linalg.norm(xy[:, None, 1:] - xy[None, :, 1:], axis=-1)
    joint_distance[np.arange(batch), np.arange(batch)] = np.inf
    neighbor_gap = joint_distance.min(1) - .8
    speed = np.linalg.norm(np.diff(xy, axis=1), axis=-1) / .1
    acceleration = np.linalg.norm(np.diff(xy, n=2, axis=1), axis=-1) / .01
    return {"endpoint_error_mean_m": float(error[:, -1].mean()), "timed_waypoint_error_mean_m": float(error.mean()),
            "first_half_second_goal_progress_mean_m": float(progress.mean()),
            "first_half_second_goal_direction_cosine_mean": float(cosine.mean()),
            "per_actor_endpoint_error_m": error[:, -1].tolist(), "per_actor_first_half_second_progress_m": progress.tolist(),
            "obstacle_collision_point_fraction": float((obstacle_gap < -1e-4).mean()),
            "joint_plan_neighbor_collision_point_fraction": float((neighbor_gap < -1e-4).mean()),
            "minimum_obstacle_surface_gap_m": float(obstacle_gap.min()),
            "minimum_joint_neighbor_surface_gap_m": float(neighbor_gap.min()),
            "max_segment_speed_m_s": float(speed.max()), "max_segment_acceleration_m_s2": float(acceleration.max()),
            "scope": "0.4m discs,50future points against static cuboids/domain and jointly predicted actors; no physical contact or navigation success measurement"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, default=ROOT / "output/diffusion_amp/shared_obstacle_corrected_timed_1000")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "output/diffusion_amp/social_diffusion_training/step_1000.ckpt")
    parser.add_argument("--expected-checkpoint-sha256", default="69d4336d002c84fa848a93a65dd622e5222a663fac63f61b3e94c6872b8c9d15")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/diffusion_amp/clean_guidance_physical_context_cpu")
    parser.add_argument("--frame", type=int, default=120)
    parser.add_argument("--guidance-weights", type=float, nargs="+", default=[1., 5., 10., 20.])
    parser.add_argument("--guidance-modes", choices=MODES, nargs="+", default=list(MODES))
    parser.add_argument("--agent-collision-weights", type=float, nargs="+", default=[0.])
    parser.add_argument("--agent-collision-radius", type=float, default=.3)
    parser.add_argument("--scene-collision-weights", type=float, nargs="+", default=[0.])
    parser.add_argument("--scene-collision-radius", type=float, default=.5)
    parser.add_argument("--waypoint-counts", type=int, nargs="+", choices=(1, 10), default=[1, 10])
    parser.add_argument("--trajectory-speed-weights", type=float, nargs="+", default=[0.])
    parser.add_argument("--max-trajectory-speed-mps", type=float, default=3.)
    parser.add_argument("--trajectory-accel-weight", type=float, default=0.)
    parser.add_argument("--trajectory-accel-weights", type=float, nargs="+", default=None)
    parser.add_argument("--max-trajectory-accel-mps2", type=float, default=2.)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args(argv)
    accel_weights = args.trajectory_accel_weights if args.trajectory_accel_weights is not None else [args.trajectory_accel_weight]
    if (not 1 <= args.cpu_threads <= 4 or not 1 <= len(args.guidance_weights) <= 6
            or not 1 <= len(args.agent_collision_weights) <= 6
            or not 1 <= len(args.scene_collision_weights) <= 6
            or not 1 <= len(args.trajectory_speed_weights) <= 6
            or not 1 <= len(accel_weights) <= 6
            or any(not math.isfinite(w) or not 0 <= w <= 100 for w in (*args.guidance_weights, *args.agent_collision_weights, *args.scene_collision_weights, *args.trajectory_speed_weights, *accel_weights))
            or any(not math.isfinite(r) or not 0 < r <= 1. for r in (args.agent_collision_radius, args.scene_collision_radius))
            or any(not math.isfinite(limit) or limit <= 0 for limit in (args.max_trajectory_speed_mps, args.max_trajectory_accel_mps2))):
        parser.error("Use at most6 finite weights0..100 and at most4 CPU threads")
    if (args.output_dir / "result.json").exists():
        parser.error("Use a new output directory; historical artifacts are never overwritten")
    sys.path.insert(0, str(ROOT))
    import torch

    from CrowdSim.humanoid.diffusion import PaperDiffusionPlanner
    from CrowdSim.humanoid.orca_corpus import sha256
    from CrowdSim.humanoid.reconstruction_scene import ReconstructionScene
    from protomotions.utils.checkpoint import atomic_write_text
    from scripts.benchmark_paper_waypoint_guidance import noise_fingerprint

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    checkpoint_sha = sha256(args.checkpoint)
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise ValueError("Frozen social checkpoint SHA mismatch")
    source = args.capture_dir / "physical_states.npz"
    result_file = args.capture_dir / "result.json"
    runner = json.loads(result_file.read_text())
    if runner.get("status") != "passed" or runner["state_capture"]["frequency_hz"] != 30:
        raise ValueError("A complete30Hz actual physical capture is required")
    with np.load(source, allow_pickle=False) as artifact:
        arrays = {key: artifact[key] for key in ("root_pos", "root_rot", "episode_ids", "route_vertex_xy")}
    context = capture_context(arrays, args.frame)
    scene = ReconstructionScene(runner["evaluation_scene"], device="cpu")
    current = torch.as_tensor(context["states"])
    destination = torch.as_tensor(arrays["route_vertex_xy"][args.frame])
    distance = torch.linalg.vector_norm(scene.routes - destination[:, None], dim=-1)
    route_indices = distance.argmin(-1)
    if not bool((distance.min(-1).values < 1e-4).all()):
        raise ValueError("Captured route destination does not match the declared static scene")
    points, _, _ = scene.route_waypoints(current[:, :2], route_indices)
    context["raster"] = scene.raster(current)
    report = {"schema": "navisaaclab.clean_guidance_physical_context_cpu.v1", "status": "running",
              "checkpoint_sha256": checkpoint_sha, "capture_sha256": sha256(source), "runner_result_sha256": sha256(result_file),
              "frame_index": args.frame, "captured_time_s": (args.frame + 1) / 30., "seed": args.seed,
              "device": "cpu", "cpu_threads": args.cpu_threads, "dtype": "float32", "compile": False,
              "diffusion_steps": 100, "goal_source": "Fresh static A* timed conditions from actual captured roots to recorded route vertices; no future oracle, no trajectory replacement",
              "velocity_source": "Backward native30Hz finite differences; exact native root velocities were not captured",
              "history_available_per_actor": context["history_mask"].sum(-1).tolist(),
              "neighbor_count_per_actor": context["neighbor_mask"][:, :, -1].sum(-1).tolist(),
              "route_indices": route_indices.tolist(), "waypoints_xy": points.tolist(),
              "astar_failures": scene.global_path_failures, "start_recoveries": scene.successful_start_recoveries,
              "conditions": [], "limitations": ["One fixed captured context, not a new physical rollout or author benchmark", "No trajectory ADE ground truth: different goals change the intended future", "Disc probes do not measure full humanoid contacts"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def publish():
        report["elapsed_s"] = time.monotonic() - started
        atomic_write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", args.output_dir / "result.json")

    planner = PaperDiffusionPlanner(checkpoint_path=args.checkpoint, device="cpu", compile_forward=False)
    report["trained_steps"] = planner.trained_steps
    baseline_noise = None
    for count in args.waypoint_counts:
        waypoint = points[:, -1:] if count == 1 else points
        times = torch.tensor([5.]) if count == 1 else torch.arange(.5, 5.1, .5)
        for mode in args.guidance_modes:
            combinations = [(goal, peer, geometry, speed, accel) for goal in args.guidance_weights for peer in args.agent_collision_weights
                            for geometry in args.scene_collision_weights for speed in args.trajectory_speed_weights
                            for accel in accel_weights if mode != "legacy_posterior" or (peer == 0 and geometry == 0 and speed == 0 and accel == 0)]
            for weight, peer_weight, scene_weight, speed_weight, accel_weight in combinations:
                entry = {"waypoint_count": count, "guidance_mode": mode, "guidance_weight": weight,
                         "agent_collision_weight": peer_weight, "agent_collision_radius_m": args.agent_collision_radius,
                         "scene_collision_weight": scene_weight, "scene_collision_radius_m": args.scene_collision_radius,
                         "trajectory_speed_weight": speed_weight, "max_trajectory_speed_mps": args.max_trajectory_speed_mps,
                         "trajectory_accel_weight": accel_weight, "max_trajectory_accel_mps2": args.max_trajectory_accel_mps2}
                report["conditions"].append(entry)
                began = time.monotonic()
                try:
                    with noise_fingerprint(torch) as noise:
                        plan = planner.plan(**context, waypoints=waypoint, waypoint_times=times, seed=args.seed,
                                            guidance_mode=mode, guidance_weight=weight,
                                            scene_ids=torch.zeros(len(current), dtype=torch.long),
                                            agent_collision_weight=peer_weight, agent_collision_radius=args.agent_collision_radius,
                                            scene_collision_weight=scene_weight, scene_collision_radius=args.scene_collision_radius,
                                            scene_bounds_xy=scene.bounds, obstacle_centers_xy=scene.centers[:, :2], obstacle_sizes_xy=scene.sizes[:, :2],
                                            trajectory_speed_weight=speed_weight, max_trajectory_speed_mps=args.max_trajectory_speed_mps,
                                            trajectory_accel_weight=accel_weight, max_trajectory_accel_mps2=args.max_trajectory_accel_mps2)
                    baseline_noise = noise if baseline_noise is None else baseline_noise
                    if noise != baseline_noise or noise["draw_count"] != 100:
                        raise RuntimeError("Same100DDPM fixed-noise stream required")
                    entry.update(status="finite", noise=noise,
                                 **scene_plan_metrics(current, plan.xy.numpy(), waypoint, times, scene.description))
                    prediction_path = args.output_dir / f"{mode}_k{count}_w{weight:g}_peer{peer_weight:g}_scene{scene_weight:g}_speed{speed_weight:g}_accel{accel_weight:g}_r{args.agent_collision_radius:g}.npz"
                    np.savez_compressed(prediction_path, xy=plan.xy.numpy(), velocity=plan.velocity.numpy(), yaw=plan.yaw.numpy(),
                                        current=current.numpy(), waypoints=waypoint.numpy(), waypoint_times=times.numpy())
                    entry["prediction_sha256"] = sha256(prediction_path)
                except Exception as error:
                    entry.update(status="failed", error=f"{type(error).__name__}: {error}")
                entry["elapsed_s"] = time.monotonic() - began
                publish()
                print(json.dumps({key: entry.get(key) for key in ("waypoint_count", "guidance_mode", "guidance_weight", "agent_collision_weight", "scene_collision_weight", "trajectory_speed_weight", "trajectory_accel_weight", "status", "endpoint_error_mean_m", "first_half_second_goal_progress_mean_m", "obstacle_collision_point_fraction", "joint_plan_neighbor_collision_point_fraction", "max_segment_speed_m_s", "error")}), flush=True)
    report.update(status="completed", checkpoint_unchanged=sha256(args.checkpoint) == checkpoint_sha,
                  cuda_initialized=torch.cuda.is_initialized(),
                  source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in (Path(__file__), ROOT / "CrowdSim/humanoid/diffusion/paper.py")})
    publish()
    if not report["checkpoint_unchanged"] or report["cuda_initialized"]:
        raise RuntimeError("Read-only CPU planning invariant failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
