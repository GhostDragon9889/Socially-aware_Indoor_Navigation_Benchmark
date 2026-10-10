"""CPU-only fixed-noise waypoint guidance sweep on real ORCA heldout windows.

Reads the frozen independently trained social diffusion checkpoint. No training,
Isaac Sim import, CUDA inference, compilation, or runtime configuration changes.
Geometry checks compare a single open-loop plan against recorded expert neighbors;
they are not a closed-loop navigation-success measurement.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOCIAL_SHA256 = "69d4336d002c84fa848a93a65dd622e5222a663fac63f61b3e94c6872b8c9d15"


def oracle_waypoints(future, count):
    """True future positions are an oracle input, never a fair prediction metric."""
    future = np.asarray(future)
    if (count not in (1, 10) or future.ndim != 3 or future.shape[1:] != (52, 5)
            or not np.isfinite(future).all()):
        raise ValueError("Use finite future[B,52,5] and1or10 oracle waypoints")
    indices = np.array([49]) if count == 1 else np.arange(4, 50, 5)
    return future[:, indices, :2], (indices.astype(np.float32) + 1) * .1


def direction_metrics(states, future, predicted_xy):
    """Five-second endpoints and the first0.5s direction, with explicit timing."""
    states, future, predicted_xy = (np.asarray(value, dtype=np.float64)
                                     for value in (states, future, predicted_xy))
    batch = len(states)
    if (states.shape != (batch, 5) or future.shape != (batch, 52, 5)
            or predicted_xy.shape != (batch, 51, 2) or batch < 1
            or not all(np.isfinite(value).all() for value in (states, future, predicted_xy))):
        raise ValueError("Finite states[B,5], future[B,52,5], predicted_xy[B,51,2] required")
    current = states[:, :2]
    if not np.allclose(predicted_xy[:, 0], current, atol=1e-5, rtol=0):
        raise ValueError("Plan frame0 must be the actual current world XY")
    goal = future[:, 49, :2]
    goal_vector = goal - current
    expert_first = future[:, 4, :2] - current
    predicted_first = predicted_xy[:, 5] - current

    def cosine(left, right):
        denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
        return np.divide(np.sum(left * right, axis=-1), denominator,
                         out=np.zeros(batch), where=denominator > 1e-8)

    goal_unit = goal_vector / np.maximum(np.linalg.norm(goal_vector, axis=-1)[:, None], 1e-8)
    first_progress = np.sum(predicted_first * goal_unit, axis=-1)
    speed = np.linalg.norm(np.diff(predicted_xy, axis=1), axis=-1) / .1
    expert_xy = np.concatenate((current[:, None], future[:, :50, :2]), axis=1)
    expert_speed = np.linalg.norm(np.diff(expert_xy, axis=1), axis=-1) / .1
    error = np.linalg.norm(predicted_xy[:, 1:] - future[:, :50, :2], axis=-1)
    acceleration = np.linalg.norm(np.diff(predicted_xy, n=2, axis=1), axis=-1) / .01
    rows = [{"endpoint_error_m": float(error[i, -1]), "ade_m": float(error[i].mean()),
             "expert_goal_distance_m": float(np.linalg.norm(goal_vector[i])),
             "first_half_second_goal_progress_m": float(first_progress[i]),
             "first_half_second_goal_direction_cosine": float(cosine(predicted_first, goal_vector)[i]),
             "first_half_second_expert_direction_cosine": float(cosine(predicted_first, expert_first)[i]),
             "first_half_second_displacement_m": float(np.linalg.norm(predicted_first[i])),
             "max_segment_speed_m_s": float(speed[i].max()),
             "max_segment_acceleration_m_s2": float(acceleration[i].max()),
             "expert_max_segment_speed_m_s": float(expert_speed[i].max())}
            for i in range(batch)]
    return {"per_case": rows,
            "endpoint_error_mean_m": float(error[:, -1].mean()),
            "ade_mean_m": float(error.mean()),
            "first_half_second_goal_progress_mean_m": float(first_progress.mean()),
            "first_half_second_goal_direction_cosine_mean": float(cosine(predicted_first, goal_vector).mean()),
            "first_half_second_expert_direction_cosine_mean": float(cosine(predicted_first, expert_first).mean()),
            "max_segment_speed_m_s": float(speed.max()),
            "max_segment_acceleration_m_s2": float(acceleration.max()),
            "segment_speed_over_2m_s_fraction": float((speed > 2.0001).mean()),
            "expert_max_segment_speed_m_s": float(expert_speed.max()),
            "direction_zero_displacement_definition": "cosine0 if either compared displacement is zero",
            "timing": "current at frame0; future index4 at0.5s; future index49 at5.0s"}


def select_heldout_windows(corpus, cases, seed):
    """Choose one real conditional window per disjoint scene before reusing scenes."""
    generator = np.random.default_rng(seed)
    scenes = sorted({entry["scene_id"] for entry in corpus.episodes})
    chosen = []
    for row in range(cases):
        scene_id = scenes[row % len(scenes)]
        candidates = np.asarray([i for i, window in enumerate(corpus.windows)
                                 if corpus.episodes[window[0]]["scene_id"] == scene_id])
        chosen.append(int(generator.choice(candidates)))
    return np.asarray(chosen, dtype=np.int64)


@contextmanager
def noise_fingerprint(torch):
    original = torch.randn
    evidence = {"draw_count": 0, "shapes": []}
    digest = hashlib.sha256()

    def captured(*args, **kwargs):
        value = original(*args, **kwargs)
        if value.device.type != "cpu":
            raise RuntimeError("This benchmark must never draw CUDA noise")
        evidence["draw_count"] += 1
        evidence["shapes"].append(list(value.shape))
        digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
        return value

    try:
        with patch.object(torch, "randn", captured):
            yield evidence
    finally:
        evidence["sha256"] = digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "output/diffusion_amp/social_diffusion_training/step_1000.ckpt")
    parser.add_argument("--expected-checkpoint-sha256", default=SOCIAL_SHA256)
    parser.add_argument("--manifest", type=Path,
                        default=ROOT / "data/diffusion_amp/orca_social_pilot_v1/manifest.json")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "output/diffusion_amp/waypoint_guidance_cpu")
    parser.add_argument("--guidance-weights", type=float, nargs="+", default=[0., 1., 5., 20., 100.])
    parser.add_argument("--guidance-mode", choices=("legacy_posterior", "trace_clean", "trace_clean_variance"),
                        default="legacy_posterior")
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--waypoint-count", type=int, choices=(1, 10), default=1,
                        help="One oracle5s endpoint or ten oracle points at0.5..5s; extra future information is not fair prediction improvement")
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--deadline-seconds", type=int, default=720)
    args = parser.parse_args(argv)
    if not 1 <= args.cases <= 32 or not 1 <= args.cpu_threads <= 4 or not 1 <= args.deadline_seconds <= 900:
        parser.error("Use1..32cases,1..4CPU threads, and a1..900second deadline")
    if (not 1 <= len(args.guidance_weights) <= 6
            or any(not math.isfinite(weight) or not 0 <= weight <= 100 for weight in args.guidance_weights)):
        parser.error("Use at most6 finite guidance weights in0..100")
    if (args.output_dir / "guidance_result.json").exists():
        parser.error("Choose a new output directory; existing guidance reports are not overwritten")
    sys.path.insert(0, str(ROOT))
    import torch

    from CrowdSim.humanoid.diffusion import PaperDiffusionPlanner
    from CrowdSim.humanoid.orca_corpus import ORCACorpus, sha256
    from protomotions.utils.checkpoint import atomic_write_text
    from scripts.train_orca_diffusion import offline_plan_metrics

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    actual_sha = sha256(args.checkpoint)
    if actual_sha != args.expected_checkpoint_sha256:
        raise ValueError("Frozen checkpoint SHA mismatch; refusing inference")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"schema": "navisaaclab.waypoint_guidance_cpu.v1", "status": "running",
              "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": actual_sha,
              "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256(args.manifest),
              "device": "cpu", "cpu_threads": args.cpu_threads, "compile": False,
              "dtype": "float32", "diffusion_steps": 100, "seed": args.seed,
              "cases": args.cases, "split": args.split, "waypoint_count": args.waypoint_count, "guidance_mode": args.guidance_mode, "guidance": [],
              "scope": "Read-only open-loop tiny heldout sweep; target is the recorded ORCA expert5s endpoint; not physical navigation success or author reproduction"}

    def publish():
        report["elapsed_s"] = time.monotonic() - started
        atomic_write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                          args.output_dir / "guidance_result.json")

    try:
        planner = PaperDiffusionPlanner(checkpoint_path=args.checkpoint, device="cpu", compile_forward=False)
        if planner.config.diffusion_steps != 100 or any(p.dtype != torch.float32 for p in planner.model.parameters()):
            raise ValueError("Only unchanged100-step FP32 model accepted")
        variances = planner.model.posterior_variance[1:].numpy()
        report["direct_guidance_stability"] = {
            "maximum_guided_posterior_variance": float(variances.max()),
            "nonexpansive_weight_upper_bound": float(2. * args.waypoint_count / variances.max()),
            "meaning": "For distinct waypoint indices, the direct mean-goal residual multiplier is1-weight*variance/K. Bound applies only to legacy_posterior, not clean denoiser-Jacobian guidance or navigation safety."}
        corpus = ORCACorpus(args.manifest, args.split, device="cpu", seed=args.seed)
        indices = select_heldout_windows(corpus, args.cases, args.seed)
        sample_np = corpus.numpy_batch(args.cases, indices=indices)
        sample = corpus.diffusion_batch(args.cases, indices=indices)
        report["windows"] = {key: sample_np[key].tolist() if isinstance(sample_np[key], np.ndarray) else sample_np[key]
                             for key in ("window_indices", "scene_ids", "episode_ids", "agent_ids", "frame_indices", "neighbor_ids")}
        report["condition_checks"] = {"real_neighbor_count_per_case": sample_np["neighbor_mask"][:, :, -1].sum(-1).tolist(),
                                      "obstacle_raster_pixels_per_case": (sample_np["raster"][:, 1] > .5).sum((-1, -2)).tolist()}
        expert_xy = np.concatenate((sample_np["states"][:, None, :2], sample_np["future"][:, :50, :2]), axis=1)
        report["expert_reference"] = {**direction_metrics(sample_np["states"], sample_np["future"], expert_xy),
                                      **offline_plan_metrics(corpus, sample_np, expert_xy[:, 1:])}
        inputs = {key: sample[key] for key in ("states", "history", "neighbors", "raster", "history_mask", "neighbor_mask")}
        points, point_times = oracle_waypoints(sample_np["future"], args.waypoint_count)
        inputs.update(waypoints=torch.as_tensor(points), waypoint_times=torch.as_tensor(point_times), seed=args.seed)
        report["oracle_waypoints"] = {"world_xy": points.tolist(), "times_s": point_times.tolist(),
                                      "extra_future_information": args.waypoint_count > 1,
                                      "fair_predictive_quality_comparison": False}
        baseline_noise = None
        for weight in args.guidance_weights:
            if time.monotonic() - started > args.deadline_seconds:
                report["status"] = "partial_deadline"
                break
            entry = {"guidance_weight": weight,
                     "maximum_direct_residual_multiplier_abs": float(np.abs(1. - weight * variances / args.waypoint_count).max())}
            report["guidance"].append(entry)
            began = time.monotonic()
            noise = None
            try:
                with noise_fingerprint(torch) as noise:
                    plan = planner.plan(**inputs, guidance_weight=weight, guidance_mode=args.guidance_mode)
                entry["noise"] = noise
                baseline_noise = noise if baseline_noise is None else baseline_noise
                entry["noise_identical_to_first"] = noise == baseline_noise
                if not entry["noise_identical_to_first"] or noise["draw_count"] != 100:
                    raise RuntimeError("Fixed100DDPM noise stream changed")
                xy = plan.xy.numpy()
                entry.update(status="finite", **direction_metrics(sample_np["states"], sample_np["future"], xy),
                             **offline_plan_metrics(corpus, sample_np, xy[:, 1:]))
                file = args.output_dir / f"guidance_{weight:g}.npz"
                np.savez_compressed(file, xy=xy, yaw=plan.yaw.numpy(), velocity=plan.velocity.numpy(),
                                    time=plan.time.numpy(), target_xy=sample_np["future"][:, :50, :2],
                                    states=sample_np["states"], window_indices=indices)
                entry["prediction_artifact"] = {"path": str(file.resolve()), "sha256": sha256(file)}
            except Exception as error:
                entry.update(status="failed", error=f"{type(error).__name__}: {error}")
            if noise is not None:
                entry["noise"] = noise
            entry["seconds"] = time.monotonic() - began
            publish()
            print(json.dumps({key: entry.get(key) for key in ("guidance_weight", "status", "seconds", "endpoint_error_mean_m", "ade_mean_m", "first_half_second_goal_progress_mean_m", "obstacle_collision_point_fraction", "neighbor_collision_point_fraction", "max_segment_speed_m_s", "error")}), flush=True)
        else:
            report["status"] = "completed"
        report["checkpoint_unchanged"] = sha256(args.checkpoint) == actual_sha
        report["cuda_initialized"] = torch.cuda.is_initialized()
        report["implementation_sha256"] = {str(path.relative_to(ROOT)): sha256(path)
                                            for path in (Path(__file__).resolve(), ROOT / "CrowdSim/humanoid/diffusion/paper.py")}
        if not report["checkpoint_unchanged"] or report["cuda_initialized"]:
            raise RuntimeError("Read-only CPU benchmark invariant failed")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        publish()
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
