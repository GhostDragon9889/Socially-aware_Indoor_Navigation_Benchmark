"""Independent CPU pose-only diversity diagnostic; not an author score.

Reuses frozen 15-keypoint local_pose, fits only the frozen held-out AMASS
reference, and leaves the existing 90-dimensional evaluations untouched.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

from protomotions.repository import repository_root
ROOT = repository_root(__file__)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
REF = ROOT / "resources/runtime_sources/style_improvement_20261002/backward_penalty_style_v1/heldout_reference.json"
FROZEN = {
    "scripts/evaluate_pedestrian_style.py": "82ee083fd16d25ddc39a4c7541d884dfef13a34bb68abc99ba3d45195665f26f",
    "CrowdSim/humanoid/style_evaluation.py": "b09fa58caaf877d89b69ca788538c2808e3a3d8d91410277ea07b5df3bcdaca2",
}
REF_SHA = '6e9ed2fa9949336acf2c7d01f0ede77f952832bbf770d1c1800c4864500c91c2'


def record(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def poses(value):
    array = np.asarray(value)
    require(array.ndim == 2 and array.shape[1] == 45 and len(array) > 0 and
            array.dtype.kind == "f" and np.isfinite(array).all(), "Pose samples require finite nonempty floating [N,45]")
    return array.astype(np.float64, copy=False)


def capped_indices(count, cap=2048):
    require(type(count) is int and count > 0 and type(cap) is int and cap > 0,
            "Sample counts/cap must be positive integers, not booleans")
    return np.arange(count, dtype=np.int64) if count <= cap else np.linspace(0, count-1, cap, dtype=np.int64)


def array_sha(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.tobytes()).hexdigest()


def cluster_statistics(samples, centers):
    samples, centers = poses(samples), poses(centers)
    k = len(centers)
    require(k >= 2, "At least two AMASS clusters required for normalized entropy")
    labels, distances = [], []
    # Chunk bounded allocation; assignment is for every generated sample,
    # including stationary/fallen poses. Only MMD and reference fit use a cap.
    for start in range(0, len(samples), 2048):
        chunk = samples[start:start+2048]
        distance2 = np.maximum(0, (chunk*chunk).sum(-1)[:, None] +
                               (centers*centers).sum(-1)[None] - 2*chunk@centers.T)
        label = distance2.argmin(-1)
        labels.append(label)
        distances.append(np.sqrt(distance2[np.arange(len(label)), label]))
    labels, distances = np.concatenate(labels), np.concatenate(distances)
    counts = np.bincount(labels, minlength=k)
    p = counts[counts > 0]/len(samples)
    return {"samples_assigned": len(samples), "clusters": k, "occupied_clusters": int((counts > 0).sum()),
            "coverage_occupied_over_K": float((counts > 0).sum()/k),
            "normalized_entropy": float(-(p*np.log(p)).sum()/math.log(k)), "assignment_counts": counts.tolist(),
            "nearest_center_distance": {"mean": float(distances.mean()), "median": float(np.median(distances)),
                                        "p95": float(np.quantile(distances, .95)), "max": float(distances.max())}}


def _distance2(samples, centers):
    return np.maximum(0, (samples*samples).sum(-1)[:, None] +
                      (centers*centers).sum(-1)[None] - 2*samples@centers.T)


def _centroid_update(sample, labels, distance2, k):
    counts = np.bincount(labels, minlength=k)
    updated = np.zeros((k, sample.shape[1]), dtype=sample.dtype)
    np.add.at(updated, labels, sample)
    populated = counts > 0
    updated[populated] /= counts[populated, None]
    empty = np.flatnonzero(~populated)
    if len(empty):
        # Deterministic farthest distinct real reference poses. Ties use
        # original sample index; never introduce synthetic centers.
        farthest = np.lexsort((np.arange(len(sample)), -distance2[np.arange(len(sample)), labels]))
        used = {tuple(value) for value in updated[populated]}
        candidates = []
        for index in farthest:
            identity = tuple(sample[index])
            if identity not in used:
                used.add(identity)
                candidates.append(index)
                if len(candidates) == len(empty):
                    break
        require(len(candidates) == len(empty), "Insufficient distinct empty-cluster replacements")
        updated[empty] = sample[candidates]
    return updated


def _lloyd_reference(sample, k, rng):
    """Explicit kmeans++/Lloyd; private NumPy RNG, no external dependency."""
    centers = [sample[int(rng.integers(len(sample)))].copy()]
    nearest = _distance2(sample, np.asarray(centers))[:, 0]
    for _ in range(1, k):
        require(np.isfinite(nearest).all() and nearest.sum() > 0, "Reference initialization lost distinct finite poses")
        index = int(rng.choice(len(sample), p=nearest/nearest.sum()))
        centers.append(sample[index].copy())
        nearest = np.minimum(nearest, _distance2(sample, sample[index:index+1])[:, 0])
    centers = np.asarray(centers)
    for iteration in range(1, 301):
        d2 = _distance2(sample, centers)
        label = d2.argmin(-1)
        updated = _centroid_update(sample, label, d2, k)
        movement = np.linalg.norm(updated-centers, axis=-1).max()
        centers = updated
        if movement <= 1e-4:
            break
    final = _distance2(sample, centers)
    labels = final.argmin(-1)
    require(len(np.unique(labels)) == k, "Reference fit retained empty clusters")
    return centers, {"iterations": iteration, "inertia_sum_squared_distance": float(final[np.arange(len(sample)), labels].sum())}


def fit_reference(reference, *, clusters=(50, 100), seed=7, n_init=10, cap=2048):
    reference = poses(reference)
    require(type(seed) is int and 0 <= seed < 2**32 and type(n_init) is int and n_init > 0,
            "Seed and n_init must be explicit valid integers")
    require(isinstance(clusters, tuple) and clusters and len(set(clusters)) == len(clusters) and
            all(type(k) is int and k >= 2 for k in clusters), "Distinct integer cluster sizes >=2 required")
    indices = capped_indices(len(reference), cap)
    sample = reference[indices]
    require(len(np.unique(sample, axis=0)) >= max(clusters), "Fixed reference has insufficient distinct poses for requested K")
    centers, restart_metadata = {}, {}
    for k in clusters:
        candidates, metadata = [], []
        for restart in range(n_init):
            value, row = _lloyd_reference(sample, k, np.random.default_rng(np.random.SeedSequence([seed, k, restart])))
            candidates.append(value)
            metadata.append(row)
        best = min(range(n_init), key=lambda index: (metadata[index]["inertia_sum_squared_distance"], index))
        centers[k] = candidates[best]
        restart_metadata[str(k)] = {"selected_restart_zero_based": best, "restarts": metadata}
    return centers, {"fit_reference_only": True, "fit_sample_count": len(sample), "available_reference_poses": len(reference),
        "fit_indices": indices.tolist(), "fit_indices_sha256": array_sha(indices), "fit_samples_sha256": array_sha(sample),
        "random_state": seed, "n_init": n_init, "algorithm": "explicit NumPy kmeans++ / Lloyd",
        "restart_seed_scheme": "default_rng(SeedSequence([seed,K,restart_zero_based]))", "max_iter": 300,
        "convergence": "maximum centroid Euclidean movement <= absolute 1e-4", "restarts": restart_metadata,
        "centers": {str(k): {"shape": list(value.shape), "float64_sha256": array_sha(value)} for k, value in centers.items()},
        "generated_samples_used_for_fit": False, "feature_standardization_or_reweighting_after_local_pose": False}


def compare_pose_only(generated, reference, centers, *, cap=2048):
    from CrowdSim.humanoid.style_evaluation import gaussian_motion_mmd2

    generated, reference = poses(generated), poses(reference)
    gi, ri = capped_indices(len(generated), cap), capped_indices(len(reference), cap)
    return {"full_inclusive_pose_count": len(generated),
        "clusters": {str(k): cluster_statistics(generated, value) for k, value in centers.items()},
        "pose_only_biased_gaussian_MMD2": {"value": gaussian_motion_mmd2(generated[gi], reference[ri], sigma=1.),
            "sigma": 1., "dimensions": 45, "estimator": "biased empirical Gaussian MMD squared",
            "maximum_samples_each": cap, "generated_samples": len(gi), "reference_samples": len(ri),
            "generated_indices_sha256": array_sha(gi), "reference_indices_sha256": array_sha(ri),
            "sampling": "deterministic evenly spaced indices of inclusive population; no speed/upright/outlier filter"}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="NAME=completed4x200s navigation directory")
    parser.add_argument("--context-run", action="append", default=[], help="NAME=completed4x200s different-scene case; never a paired method comparison")
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory; historical reports cannot be overwritten")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output must be fresh")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA_VISIBLE_DEVICES must be empty")
    for path, expected in FROZEN.items():
        require(record(ROOT/path)["sha256"] == expected, "Frozen feature/selection reader changed")
    require(record(REF)["sha256"] == REF_SHA, "Frozen c274 reference changed")
    import torch
    from CrowdSim.humanoid.style_evaluation import StyleSpec, KEYPOINTS, sequence_style_samples
    from scripts.evaluate_pedestrian_style import load_reference, assert_heldout_disjoint
    from scripts.evaluate_paper_pedestrians import read_capture
    torch.set_num_threads(4)
    artifact = json.loads(REF.read_text())
    require(artifact["selection"]["split"] == "validation", "Only frozen validation reference allowed; no34 test captures")
    spec = StyleSpec()
    require(artifact["metric_spec"] == json.loads(json.dumps(asdict(spec))), "Frozen scale/pose extraction specification differs")
    manifest = Path(artifact["manifest"]["path"])
    require(record(manifest) == artifact["manifest"], "Frozen reference manifest changed")
    reference_document = json.loads(manifest.read_text())
    body, rotation, sample_hashes, provenance = load_reference(manifest, artifact["selection"])
    require(sample_hashes == artifact["actual_selected_array_sha256"], "Frozen selected actual arrays changed")
    episodes = np.ones(body.shape[:2], dtype=np.int64)
    resets = np.zeros(body.shape[:2], dtype=bool)
    resets[0] = True
    reference = sequence_style_samples(body, rotation, episode_ids=episodes, reset_mask=resets, spec=spec)
    centers, fit = fit_reference(reference["local_pose"])
    report = {"schema": "local.pose_only_15kp_diversity_candidate.v1", "status": "passed", "CPU_only": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "Kit_started": False, "CUDA_initialized": False,
        "source": {"script": record(__file__), "frozen_implementations": {path: record(ROOT/path) for path in FROZEN},
                   "reference_artifact": record(REF)},
        "reference": artifact, "reference_load_provenance": provenance,
        "method": {"feature": "unchanged sequence_style_samples.local_pose only; no velocity coordinates",
            "dimensions": 45, "dt_s": 1/30, "keypoints": list(KEYPOINTS), "feet_mapping": "L_Ankle/R_Ankle body origins, not toes/soles",
            "scale": reference["definitions"]["chain_scale"], "fit": fit,
            "source_paper": "https://arxiv.org/html/2606.26265v1#S4.SS3",
            "paper_section": "IV-C / TableIII", "author_scale_cluster_settings_or_bandwidth_known": False,
            "numerical_comparison_to_TableIII_valid": False},
        "reference_self_calibration": compare_pose_only(reference["local_pose"], reference["local_pose"], centers), "runs": {}}
    for pair, context_only in [(value, False) for value in args.run] + [(value, True) for value in args.context_run]:
        name, separator, path = pair.partition("=")
        require(separator and name and name not in report["runs"], "Unique NAME=directory required")
        directory = Path(path).resolve()
        states, result = directory/"physical_states.npz", directory/"result.json"
        arrays, runner, capture = read_capture(states, result)
        require(runner["status"] == "passed" and runner.get("error") is None and runner["actual_control_steps"] == 6000 and
                runner["arguments"]["num_envs"] == 4 and arrays["body_pos"].shape[:2] == (6000, 4),
                "Only actual complete4x200s navigation captures allowed")
        declared = Path(runner["state_capture"]["path"])
        require((ROOT/declared if not declared.is_absolute() else declared).resolve() == states,
                "Result declared a different capture")
        training_manifest = Path(runner["arguments"]["manifest"])
        training_manifest = ROOT/training_manifest if not training_manifest.is_absolute() else training_manifest
        disjoint = assert_heldout_disjoint(artifact["selection"], json.loads(training_manifest.read_text()), reference_document)
        generated = sequence_style_samples(arrays["body_pos"], arrays["root_rot"], body_names=capture["body_names"],
            episode_ids=arrays["episode_ids"], reset_mask=arrays["reset_mask"], spec=spec)
        report["runs"][name] = {"sources": {"states": record(states), "result": record(result),
                                               "training_manifest": record(training_manifest)},
            "capture_validation": capture, "reference_parent_disjointness": disjoint,
            "raw_source_frames_read_without_failure_filter": 6000, "source_actor_frames": 24000,
            "included_pose_valid_interval_endpoints": len(generated["local_pose"]),
            "excluded_first_frame_actor_samples": 4,
            "excluded_cross_reset_interval_endpoints": generated["summary"]["excluded_reset_transitions"],
            "all_post_step_poses_included": False,
            "poses": compare_pose_only(generated["local_pose"], reference["local_pose"], centers),
            "same_episode_intervals": generated["summary"]["same_episode_transition_count"],
            "cross_reset_intervals_excluded": generated["summary"]["excluded_reset_transitions"],
            "outcomes": {key: runner.get(key) for key in ("fall_resets", "route_vertices_reached")},
            "context_only_different_scene": context_only,
            "same_scene_method_comparison": not context_only,
            "scene_as_published": runner.get("evaluation_scene"),
            "native_scene_live_observation_as_published": runner.get("native_scene_live_observation"),
            "fall_upright_idle_or_outlier_filter_applied": False}
    report["package_versions"] = {"numpy": np.__version__, "torch": torch.__version__, "sklearn_required": False}
    report["CUDA_initialized"] = torch.cuda.is_initialized()
    report["limits"] = ["Paper-inspired local diagnostic, not an author implementation or TableIII score; unknown author scale/KMeans/bandwidth remain explicit.",
        "Coverage/entropy assign all legal same-episode interval endpoint poses including failures; first samples and reset-boundary endpoints are excluded by the unchanged helper. MMD uses capped deterministic inclusive samples; all failure poses are not guaranteed to enter the cap.",
        "Pose-only metrics ignore timing/velocity, navigation progress and contact quality. Standing still can have small pose-only MMD without good locomotion.",
        "Window overlap and adjacent poses are dependent; no statistical confidence intervals or person-disjoint/public-pretraining independence claim.",
        "Reference-self MMD is exactly zero by construction, not a separate AMASS holdout score comparable to paper reference-row MMD.",
        "Any context-only native scene row changes environment and deployment source; no paired method improvement or cross-scene causal claim.",
        "No old90D scores, dictionaries, physical model, data, deployed controller or production code changed."]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    np.savez(args.output/"reference_basis.npz", **{f"centers_{k}": value for k, value in centers.items()})
    (args.output/"heldout_reference.json").write_bytes(REF.read_bytes())
    print(json.dumps({"status": report["status"], "reference_self": report["reference_self_calibration"],
                      "runs": {name: row["poses"] for name, row in report["runs"].items()}}, indent=2))


if __name__ == "__main__":
    main()
