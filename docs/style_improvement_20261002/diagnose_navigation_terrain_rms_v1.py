"""Read-only saved-RMS and post-step terrain-feature reconstruction diagnostic.

Reconstruction uses archived body poses and the declared cuboids. It does not
recover the original pre-action observation or evaluate another controller.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DOCS = Path(__file__).resolve().parent
OUT = DOCS / "navigation_terrain_rms_review_v1"
NAMES = ("guidance20", "mixed2000", "state_action2000_nav")


def load(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).resolve()
    content = path.read_bytes()
    return {"path": str(path), "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    require(values.size > 0 and np.isfinite(values).all(), "Nonempty finite statistics required")
    return {"count": len(values), "min": float(values.min()), "median": float(np.median(values)),
            "mean": float(values.mean()), "p95": float(np.quantile(values, .95)), "max": float(values.max())}


def diagnostic(raw, mean, variance, clip):
    # Matches training RunningNormalizer.normalize float32 casting exactly.
    z = (raw - mean.float()) / torch.sqrt(variance.float() + 1e-5)
    clipped = z.abs() > clip
    nonzero = raw != 0
    return {"coordinate_count": raw.numel(), "raw_feature": stats(raw.numpy()),
            "unclipped_abs_normalized": stats(z.abs().numpy()),
            "coordinate_clip_fraction": float(clipped.float().mean()),
            "actor_frame_any_coordinate_clipped_fraction": float(clipped.any(-1).float().mean()),
            "nonzero_feature_coordinates": int(nonzero.sum()),
            "nonzero_coordinate_clip_fraction": float(clipped[nonzero].float().mean()) if nonzero.any() else None,
            "coordinate_sha256": tensor_sha(raw), "normalizer_mutated": False}


def main():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only diagnostic required")
    torch.set_num_threads(4)
    from CrowdSim.humanoid.amp.observations import PACER_BODY_NAMES, build_task_observation
    from CrowdSim.humanoid.reconstruction_scene import ReconstructionScene

    comp_path = DOCS / "state_action_navigation_style_v1/comparison.json"
    ground_path = DOCS / "state_action_navigation_style_v1/native_ground_comparison.json"
    comp, grounds = load(comp_path), load(ground_path)
    public_path = ROOT / "data/pretrained_models/paper_references/pacer/output/release/pacer_no_shape/Humanoid.pth"
    public = torch.load(public_path, map_location="cpu", weights_only=True, mmap=True)
    public_mean = public["running_mean_std"]["running_mean"][377:1401].double().clone()
    public_var = public["running_mean_std"]["running_var"][377:1401].double().clone()
    rows, tensors = {}, {}
    for name in NAMES:
        source = comp["runs"][name]["models"]["controller"]
        path = Path(source["path"])
        require(path.stat().st_size == source["bytes"], "Reviewed checkpoint changed size")
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        require(state["format"] == "navisaaclab_method_reconstruction_training", "Wrong checkpoint format")
        config = state["config"]
        require(config["normalize_observations"] and config["task_dim"] == 1044, "Task contract mismatch")
        normalizer = state["normalizers"]["task_normalizer"]
        mean, var = normalizer["mean"][20:].clone(), normalizer["variance"][20:].clone()
        count = float(normalizer["count"])
        require(mean.shape == var.shape == (1024,) and torch.isfinite(mean).all() and
                torch.isfinite(var).all() and (var >= 0).all(), "Invalid saved terrain moments")
        cap = config["warm_start_normalizer_count_cap"]
        require(cap == 200000. and state["provenance"]["initialization"]["observation_value_moments_transferred"],
                "Expected declared warm-start terrain moments/count cap")
        r = cap / count
        zero_only_mean = public_mean * r
        zero_only_var = (public_var + public_mean.square()) * r - zero_only_mean.square()
        rows[name] = {"checkpoint_as_bound_by_completed_evaluation": source,
            "checkpoint_full_SHA_recomputed_this_diagnostic": False, "checkpoint_stat_bytes_confirmed": True,
            "update_count": state["update_count"], "physical_transition_count": state["physical_transition_count"],
            "normalization_clip": config["normalization_clip"], "count": count,
            "warm_start_count_cap": cap, "warm_start_public_moment_weight": r,
            "terrain_mean": stats(mean.numpy()), "terrain_standard_deviation": stats(var.sqrt().numpy()),
            "terrain_mean_sha256": tensor_sha(mean), "terrain_variance_sha256": tensor_sha(var),
            "zero_training_terrain_moment_model": {
                "formula": "r=n0/N; mean=r*public_mean; variance=r*(public_variance+public_mean^2)-mean^2",
                "mean_max_error": float((mean-zero_only_mean).abs().max()),
                "variance_max_error": float((var-zero_only_var).abs().max()),
                "near_equal_at_1e_minus10": bool(torch.allclose(mean, zero_only_mean, atol=1e-10, rtol=0) and
                                                torch.allclose(var, zero_only_var, atol=1e-10, rtol=0))},
            "artificial_uniform_obstacle_probes_not_actual_observations": {
                str(height): diagnostic(torch.full((1, 1024), -5 * height), mean, var, config["normalization_clip"])
                for height in (0., .2, .5, 1., 2.)}}
        tensors[name] = (mean, var, config["normalization_clip"])
        del state

    reconstructed = {}
    for name in NAMES:
        result = load(comp["runs"][name]["result"]["path"])
        original_audit = load(grounds["runs"][name]["original_audit"]["path"])
        require(original_audit["status"] == "passed" and
                original_audit["artifacts"]["result"] == comp["runs"][name]["result"], "Actual ground source binding differs")
        sample_indices = np.arange(0, 6000, 30)
        with np.load(original_audit["artifacts"]["ground_support"]["path"], allow_pickle=False) as ground:
            names = tuple(ground["common_body_names"].tolist())
            require(names == PACER_BODY_NAMES, "Unknown captured body order")
            positions = torch.from_numpy(ground["body_link_pos_w"][sample_indices, -1].copy()).flatten(0, 1)
            quaternions = torch.from_numpy(ground["body_link_quat_w"][sample_indices, -1].copy()).flatten(0, 1)
            velocities = torch.from_numpy(ground["body_link_lin_vel_w"][sample_indices, -1].copy()).flatten(0, 1)
            angular = torch.from_numpy(ground["body_link_ang_vel_w"][sample_indices, -1].copy()).flatten(0, 1)
        with np.load(original_audit["artifacts"]["physical_states"]["path"], allow_pickle=False) as physical:
            require(np.array_equal(positions[:, 0].numpy(), physical["root_pos"][sample_indices].reshape(-1, 3)) and
                    np.array_equal(quaternions[:, 0].numpy(), physical["root_rot"][sample_indices].reshape(-1, 4)),
                    "Final physics-substep capture disagrees with saved common root")
            dof = torch.from_numpy(physical["dof_pos"][sample_indices].copy()).flatten(0, 1)
            dof_vel = torch.from_numpy(physical["dof_vel"][sample_indices].copy()).flatten(0, 1)
        require(bool(((positions[:, 0, :2] >= 2) & (positions[:, 0, :2] <= 22)).all()),
                "Ground-floor domain assumption must be reviewed")
        state = SimpleNamespace(rigid_body_pos=positions, rigid_body_rot=quaternions,
            rigid_body_vel=velocities, rigid_body_ang_vel=angular, dof_pos=dof, dof_vel=dof_vel)
        scene = ReconstructionScene(result["evaluation_scene"], device="cpu")
        # The positive 64m terrain is zero-height here. All declared cuboids
        # have base zero. Uses original box height function, not a collider query.
        trajectory = positions[:, 0, None, :2].expand(-1, 10, -1)
        raw = build_task_observation(state, trajectory, scene.obstacle_height)[:, 20:]
        mean, var, clip = tensors[name]
        reconstructed[name] = {"capture_ground_source": original_audit["artifacts"]["ground_support"],
            "capture_physical_source": original_audit["artifacts"]["physical_states"],
            "post_step_sample_count": len(positions), "saved_control_indices": sample_indices.tolist(),
            "last_substep_root_bitwise_equal_to_physical_capture": True,
            "head_orientation_from_actual_last_substep_not_root_yaw_surrogate": True,
            "original_pre_action_task_observation_archived": False,
            "ground_assumption": "Known zero-height positive64m terrain and declared exact cuboid top heights; no live PhysX/raycast query",
            "trajectory20_placeholder_not_evaluated": True,
            "reconstructed_terrain_under_own_RMS": diagnostic(raw, mean, var, clip),
            "same_reconstructed_terrain_under_public_RMS_counterfactual": diagnostic(raw, public_mean, public_var, clip),
            "comparison_is_same_post_step_features_not_another_physical_run": True}

    output = {"schema": "navigation.saved_terrain_rms_readonly_review.v1", "status": "passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "CPU_only": True,
        "Kit_started": False, "CUDA_initialized": torch.cuda.is_initialized(), "sealed34_read": False,
        "business_or_weights_or_normalizer_buffers_mutated": False,
        "sources": {"comparison": record(comp_path), "native_ground_objects": record(ground_path),
            "tool": record(__file__), "observation_formula": record(ROOT / "CrowdSim/humanoid/amp/observations.py"),
            "cuboid_height_formula": record(ROOT / "CrowdSim/humanoid/reconstruction_scene.py"),
            "normalizer_formula": record(ROOT / "CrowdSim/humanoid/training/trainer.py")},
        "public_PACER_terrain_standard_deviation": stats(public_var.sqrt().numpy()),
        "saved_normalizers": rows, "post_step_feature_reconstruction": reconstructed,
        "mixed_and_state_action_terrain_suffix_byte_equal": all(torch.equal(tensors["mixed2000"][i], tensors["state_action2000_nav"][i]) for i in (0, 1)),
        "causal_conclusion": "Flat observations dilute inherited terrain variance; this is shared by the zero-fall baseline and both new models. Identical mixed/state-action terrain moments cannot alone explain their different navigation outcomes.",
        "next_experiment": "Worth an explicitly tagged low-cost counterfactual or matched new-training terrain-statistics preservation ablation; do not replace a deployed model's learned feature scaling silently. Record actual pre-action raw features and clipping first.",
        "limits": ["Saved final-substep poses reconstruct post-step inputs, not original pre-action observations or online clip rates.",
            "Box geometry height function reproduces declared observation geometry, not independent collision correctness.",
            "Artificial uniform probes have no measured scene occupancy probability.",
            "This diagnostic does not change observations, actor weights, training, routes or physics and cannot establish a physical cause."]}
    OUT.mkdir(exist_ok=False)
    with (OUT / "audit.json").open("x") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": output["status"], "same_mixed_stateaction_suffix": output["mixed_and_state_action_terrain_suffix_byte_equal"],
        "rows": {key: {"std_median": row["terrain_standard_deviation"]["median"], "zero_model": row["zero_training_terrain_moment_model"]} for key, row in rows.items()},
        "reconstructed_clip_fraction": {key: row["reconstructed_terrain_under_own_RMS"]["coordinate_clip_fraction"] for key, row in reconstructed.items()}}, indent=2))


if __name__ == "__main__":
    main()
