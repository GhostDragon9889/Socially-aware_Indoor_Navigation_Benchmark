"""NumPy-only, speed-conditioned motion-quality diagnostics with explicit proxies.

These are local diagnostics, not the unidentified implementation of NavIsaacLab
Table III. No training samples fit a score; the caller must supply a fixed,
held-out reference and disclose its identities.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import posixpath
import re

import numpy as np

BODY_NAMES = ("Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head", "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand")
KEYPOINTS = ("Pelvis", "L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle", "L_Shoulder",
    "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "Neck", "Head")
CHAINS = (("Pelvis", "Torso", "Spine", "Chest", "Neck", "Head"),
          ("Pelvis", "L_Hip", "L_Knee", "L_Ankle"), ("Pelvis", "R_Hip", "R_Knee", "R_Ankle"))


def parent_source_identity(clip):
    """Project a source or a frame-segment alias to its original motion file.

    Directory aliases are not resolved by guessing. An explicit parent must
    agree with the source basename/suffix contract, so metadata cannot rename
    a held-out parent into a new training identity.
    """
    def path_text(value):
        if not isinstance(value, str) or not value.strip() or value != value.strip() or "\x00" in value:
            raise ValueError("Motion source identity must be a nonempty path string")
        return value.replace("\\", "/")

    if not isinstance(clip, dict) or "source_path" not in clip:
        raise ValueError("Motion metadata requires an explicit source_path")
    source = path_text(clip["source_path"])
    parent = posixpath.normpath(source)
    segment = None
    if "#" in source:
        match = re.fullmatch(r"(.+)#frames=(\d+):(\d+)", source)
        if not match or "#" in match[1] or int(match[2]) >= int(match[3]):
            raise ValueError("Motion segment source must end in valid #frames=start:stop bounds")
        parent = posixpath.normpath(match[1])
        segment = (int(match[2]), int(match[3]))
    if "parent_source_path" in clip:
        explicit = path_text(clip["parent_source_path"])
        if "#" in explicit or posixpath.normpath(explicit) != parent:
            raise ValueError("Explicit parent_source_path differs from the motion source parent")
    bound_keys = ("parent_control_frame_start", "parent_control_frame_stop")
    if any(key in clip for key in bound_keys):
        bounds = tuple(clip.get(key) for key in bound_keys)
        if any(type(value) is not int for value in bounds) or segment != bounds:
            raise ValueError("Declared parent control-frame bounds differ from #frames source identity")
    if segment is not None and "frames" in clip:
        if type(clip["frames"]) is not int or clip["frames"] != segment[1]-segment[0]:
            raise ValueError("Segment frame count differs from its half-open parent control-frame bounds")
    return parent


def assert_parent_split_disjoint(document):
    """Reject train/held-out overlap across the entire manifest, before sampling."""
    if not isinstance(document, dict) or not isinstance(document.get("splits"), dict):
        raise ValueError("Motion manifest requires explicit splits")
    parents = {}
    for split in ("train", "validation", "test"):
        entry = document["splits"].get(split)
        if entry is None:
            if split == "train":
                raise ValueError("Motion manifest requires an explicit training split")
            continue
        if not isinstance(entry, dict) or not isinstance(entry.get("clips"), list):
            raise ValueError("Motion split requires an explicit clip list")
        parents[split] = {parent_source_identity(clip) for clip in entry["clips"]}
    for split in ("validation", "test"):
        overlap = parents["train"] & parents.get(split, set())
        if overlap:
            raise ValueError(f"Motion manifest leaks parent sources between train and {split}: {sorted(overlap)[:3]}")
    return parents


@dataclass(frozen=True)
class StyleSpec:
    dt_s: float = 1/30
    ground_height_m: float = 0.
    toe_near_ground_height_m: float = .12
    toe_vertical_speed_max_m_s: float = .2
    toe_sliding_speed_threshold_m_s: float = .1
    motion_velocity_time_scale_s: float = .1
    motion_kernel_sigma: float = 1.
    speed_bin_lower_edges_m_s: tuple = (0., .1, .5, 1., 1.5, 2., 3.)

    def validate(self):
        for name, value in asdict(self).items():
            if name == "speed_bin_lower_edges_m_s":
                continue
            if type(value) not in (int, float) or not math.isfinite(value) or (name != "ground_height_m" and value <= 0):
                raise ValueError(f"{name} must be a finite numeric scalar with its stated positive bound")
        if any(type(value) not in (int, float) for value in self.speed_bin_lower_edges_m_s):
            raise ValueError("Speed-bin edges must be numeric and cannot be boolean")
        edges = np.asarray(self.speed_bin_lower_edges_m_s, dtype=np.float64)
        if edges.ndim != 1 or len(edges) < 2 or edges[0] != 0 or not np.isfinite(edges).all() or (np.diff(edges) <= 0).any():
            raise ValueError("Speed-bin lower edges must start at zero and strictly increase")


def _numeric(value, name, shape=None):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.isfinite(array).all() or (shape is not None and array.shape != shape):
        raise ValueError(f"{name} must be finite numeric data" + (f" with shape{shape}" if shape is not None else ""))
    return array.astype(np.float64)


def summary(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Statistics cannot discard nonfinite measurements")
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "p05": None, "p95": None, "max": None}
    return {"count": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "p05": float(np.quantile(values, .05)), "p95": float(np.quantile(values, .95)), "max": float(values.max())}


def wasserstein_1d(left, right):
    """Exact empirical one-dimensional Wasserstein distance, no fitted bins."""
    left, right = np.sort(_numeric(left, "left").reshape(-1)), np.sort(_numeric(right, "right").reshape(-1))
    if not len(left) or not len(right):
        raise ValueError("Wasserstein distance requires two nonempty samples")
    points = np.sort(np.concatenate((left, right)))
    widths = np.diff(points)
    left_cdf = np.searchsorted(left, points[:-1], side="right") / len(left)
    right_cdf = np.searchsorted(right, points[:-1], side="right") / len(right)
    return float(np.sum(widths * np.abs(left_cdf - right_cdf)))


def gaussian_motion_mmd2(left, right, *, sigma=1.):
    """Biased empirical Gaussian MMD squared, with explicit fixed bandwidth."""
    left, right = _numeric(left, "left features"), _numeric(right, "right features")
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[1] or not min(len(left), len(right)):
        raise ValueError("Motion features require two nonempty [N,D] matrices with common D")
    if type(sigma) not in (int, float) or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("Kernel sigma must be positive and finite")
    def kernel(a, b):
        squared = np.maximum(0, (a*a).sum(1)[:, None]+(b*b).sum(1)[None]-2*a@b.T)
        return np.exp(-squared/(2*sigma*sigma)).mean()
    return max(0., float(kernel(left, left)+kernel(right, right)-2*kernel(left, right)))


def sequence_style_samples(body_pos, root_quat_xyzw, *, episode_ids, reset_mask, body_names=BODY_NAMES,
                           spec=StyleSpec(), body_masses_kg=None, body_com_world=None, foot_contact_mask=None):
    """Return metric samples on legal [t-1,t] intervals; reset jumps are excluded.

    body_pos[T,B,24,3] contains body/joint origins, not body COMs or sole contact
    points. Physical speed uses root displacement/dt. Local pose derivatives
    subtract Pelvis and each frame's yaw, then divide by a fixed per-actor median
    head-chain + mean-leg-chain length. This separates local articulation changes
    from global locomotion speed/yaw and avoids pose-dependent frame scaling.
    """
    spec.validate()
    names = tuple(body_names)
    if len(names) != 24 or set(names) != set(BODY_NAMES):
        raise ValueError("body_names requires an exact named 24-body SMPL bijection")
    pos = _numeric(body_pos, "body_pos")
    if pos.ndim != 4 or pos.shape[-2:] != (24, 3) or pos.shape[0] < 3 or pos.shape[1] < 1:
        raise ValueError("body_pos requires [T>=3,B>=1,24,3]")
    t, b = pos.shape[:2]
    q = _numeric(root_quat_xyzw, "root_quat_xyzw", (t, b, 4))
    if not np.allclose(np.linalg.norm(q, axis=-1), 1., atol=1e-3, rtol=0):
        raise ValueError("Root quaternions must be unit XYZW")
    episodes, resets = np.asarray(episode_ids), np.asarray(reset_mask)
    if episodes.shape != (t, b) or episodes.dtype.kind not in "iu" or resets.shape != (t, b) or resets.dtype != np.bool_:
        raise ValueError("episode_ids must be integer and reset_mask genuine bool [T,B]")
    changes = episodes[1:] != episodes[:-1]
    if (np.diff(episodes, axis=0) < 0).any() or not np.array_equal(changes, resets[1:]):
        raise ValueError("Before-frame reset masks must match nondecreasing episode-ID changes")
    valid = ~changes & ~resets[1:]
    if not valid.any():
        raise ValueError("No same-episode transitions exist")
    root = pos[..., names.index("Pelvis"), :]
    chain_lengths = []
    for chain in CHAINS:
        values = pos[..., [names.index(name) for name in chain], :]
        chain_lengths.append(np.linalg.norm(np.diff(values, axis=-2), axis=-1).sum(-1))
    scale = np.median(chain_lengths[0] + .5*(chain_lengths[1] + chain_lengths[2]), axis=0)
    if (scale <= 1e-6).any():
        raise ValueError("Kinematic chain scale must be nondegenerate")
    relative = pos[..., [names.index(name) for name in KEYPOINTS], :] - root[..., None, :]
    yaw = np.arctan2(2*(q[..., 3]*q[..., 2] + q[..., 0]*q[..., 1]), 1-2*(q[..., 1]**2+q[..., 2]**2))
    c, s = np.cos(yaw)[..., None], np.sin(yaw)[..., None]
    local = np.stack((c*relative[..., 0]+s*relative[..., 1], -s*relative[..., 0]+c*relative[..., 1], relative[..., 2]), -1)
    local /= scale[None, :, None, None]
    velocity = np.diff(local, axis=0)/spec.dt_s
    speed = np.linalg.norm(np.diff(root[..., :2], axis=0), axis=-1)/spec.dt_s
    toe = pos[..., [names.index("L_Toe"), names.index("R_Toe")], :]
    toe_velocity = np.diff(toe, axis=0)/spec.dt_s
    near = (toe[..., 2] - spec.ground_height_m) <= spec.toe_near_ground_height_m
    stance = near[:-1] & near[1:] & (np.abs(toe_velocity[..., 2]) <= spec.toe_vertical_speed_max_m_s)
    toe_speed = np.linalg.norm(toe_velocity[..., :2], axis=-1)
    toe_count = stance.sum(-1)
    slide = (toe_speed * stance).sum(-1)/np.maximum(1, toe_count)
    acceleration = np.zeros_like(speed)
    acceleration[1:] = np.sqrt(np.mean(np.sum((np.diff(velocity, axis=0)/spec.dt_s)**2, axis=-1), axis=-1))
    acceleration_valid = np.zeros_like(valid)
    acceleration_valid[1:] = valid[1:] & valid[:-1]
    metric_values = {
        "pelvis_height_m": root[1:, :, 2]-spec.ground_height_m,
        "pelvis_height_over_chain_scale": (root[1:, :, 2]-spec.ground_height_m)/scale,
        "root_up_z": 1-2*(q[1:, :, 0]**2+q[1:, :, 1]**2),
        "local_pose_velocity_rms_chain_s": np.sqrt(np.mean(np.sum(velocity**2, axis=-1), axis=-1)),
        "local_pose_acceleration_rms_chain_s2": acceleration,
        "toe_slide_near_ground_proxy_m_s": slide,
        "toe_near_ground_proxy_supported": (toe_count > 0).astype(float),
        "toe_near_ground_proxy_double_support": (toe_count == 2).astype(float),
        "toe_min_origin_height_m": np.min(toe[1:, :, :, 2], axis=-1)-spec.ground_height_m,
        "toe_origins_below_ground_fraction": np.mean(toe[1:, :, :, 2] < spec.ground_height_m, axis=-1),
    }
    masks = {name: np.ones(valid.shape, dtype=bool) for name in metric_values}
    masks["local_pose_acceleration_rms_chain_s2"] = acceleration_valid
    masks["toe_slide_near_ground_proxy_m_s"] = toe_count > 0
    actual_com, mass_proxy = None, None
    if body_masses_kg is not None:
        masses = _numeric(body_masses_kg, "body_masses_kg")
        try:
            masses = np.broadcast_to(masses, (b, 24))
        except ValueError:
            raise ValueError("body_masses_kg must broadcast to [B,24] in declared body order") from None
        if (masses <= 0).any():
            raise ValueError("Body masses must be strictly positive")
        normalized = masses/masses.sum(-1, keepdims=True)
        mass_proxy = (pos[..., 2]*normalized).sum(-1)-spec.ground_height_m
        metric_values["mass_weighted_joint_origin_height_proxy_m"] = mass_proxy[1:]
        masks["mass_weighted_joint_origin_height_proxy_m"] = np.ones(valid.shape, dtype=bool)
        if body_com_world is not None:
            coms = _numeric(body_com_world, "body_com_world", pos.shape)
            actual_com = (coms[..., 2]*normalized).sum(-1)-spec.ground_height_m
            metric_values["measured_body_com_weighted_height_m"] = actual_com[1:]
            masks["measured_body_com_weighted_height_m"] = np.ones(valid.shape, dtype=bool)
    elif body_com_world is not None:
        raise ValueError("True COM height requires explicit matching body masses")
    measured_slide = None
    if foot_contact_mask is not None:
        contact = np.asarray(foot_contact_mask)
        if contact.shape != (t, b, 2) or contact.dtype != np.bool_:
            raise ValueError("foot_contact_mask requires measured boolean [T,B,2]")
        paired = contact[1:] & contact[:-1]
        measured_slide = summary(toe_speed[valid][paired[valid]])
    end, actor = np.nonzero(valid)
    return {"speed_m_s": speed[valid], "metrics": {name: values[valid] for name, values in metric_values.items()},
        "metric_valid": {name: values[valid] for name, values in masks.items()},
        "local_pose": local[1:][valid].reshape(-1, 45), "local_pose_velocity": velocity[valid].reshape(-1, 45),
        "end_frame_indices": end+1, "actor_indices": actor,
        "summary": {"same_episode_transition_count": int(valid.sum()), "excluded_reset_transitions": int(changes.sum()),
            "valid_acceleration_triplets": int(acceleration_valid.sum()), "chain_scale_m_per_actor": scale.tolist(),
            "physical_root_speed_m_s": summary(speed[valid]),
            "toe_slide_near_ground_proxy": {"stance_foot_intervals": int(stance[valid].sum()),
                "all_foot_intervals": int(valid.sum())*2, "speed_m_s": summary(toe_speed[valid][stance[valid]]),
                "sliding_above_threshold_fraction": float(np.mean(toe_speed[valid][stance[valid]] > spec.toe_sliding_speed_threshold_m_s)) if stance[valid].any() else None},
            "measured_foot_contact_body_origin_speed_m_s": measured_slide,
            "true_body_com_height_available": actual_com is not None,
            "mass_weighted_joint_origin_height_proxy_available": mass_proxy is not None,
            "metrics": {name: summary(values[valid & masks[name]]) for name, values in metric_values.items()}},
        "spec": asdict(spec), "body_names": list(names),
        "definitions": {"speed": "Physical Pelvis XY displacement / dt; reset intervals excluded",
            "chain_scale": "Per-actor median head-to-pelvis chain length plus mean pelvis-to-ankle chain; body origins, not full anthropometric height",
            "pose_derivatives": "Heading-aligned, Pelvis-relative, fixed-chain-scale articulation; global translation/yaw removed",
            "toe_slide": "World XY toe-body-origin speed when both interval endpoints near ground and |toe vz| small; geometric proxy, not contact-patch slip",
            "true_com": "Available only from explicitly captured world body COM locations and masses; joint/body origins alone are insufficient"}}


def compare_speed_conditioned(generated, reference, *, spec=StyleSpec(), minimum_bin_samples=30, maximum_motion_samples=256):
    """Compare scalar articulation metrics within fixed physical-speed bins.

    Equal-bin mean and reference-weighted mean use supported common bins only;
    omitted reference mass is reported. Unconditional scores remain visible.
    This is descriptive distribution distance, not a statistical independence
    assumption or author-equivalent score.
    """
    spec.validate()
    if type(minimum_bin_samples) is not int or minimum_bin_samples < 1:
        raise ValueError("minimum_bin_samples must be a positive integer")
    if type(maximum_motion_samples) is not int or maximum_motion_samples < minimum_bin_samples:
        raise ValueError("maximum_motion_samples must be an integer >= minimum_bin_samples")
    if generated["spec"] != asdict(spec) or reference["spec"] != asdict(spec):
        raise ValueError("Generated and held-out reference samples must use exactly the same metric specification")
    edges = np.asarray((*spec.speed_bin_lower_edges_m_s, np.inf))
    gs, rs = generated["speed_m_s"], reference["speed_m_s"]
    fields = sorted(set(generated["metrics"]) & set(reference["metrics"]))
    bins = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        gm, rm = (gs >= lower)&(gs < upper), (rs >= lower)&(rs < upper)
        row = {"lower_m_s": float(lower), "upper_m_s": float(upper) if np.isfinite(upper) else None,
               "generated_count": int(gm.sum()), "reference_count": int(rm.sum()), "metrics": {}}
        selected_count = min(int(gm.sum()), int(rm.sum()), maximum_motion_samples)
        row["pose_velocity_gaussian_mmd2"] = None
        if selected_count >= minimum_bin_samples:
            gi = np.flatnonzero(gm)[np.linspace(0, int(gm.sum())-1, selected_count, dtype=np.int64)]
            ri = np.flatnonzero(rm)[np.linspace(0, int(rm.sum())-1, selected_count, dtype=np.int64)]
            gf = np.concatenate((generated["local_pose"][gi], spec.motion_velocity_time_scale_s*generated["local_pose_velocity"][gi]), -1)
            rf = np.concatenate((reference["local_pose"][ri], spec.motion_velocity_time_scale_s*reference["local_pose_velocity"][ri]), -1)
            row["pose_velocity_gaussian_mmd2"] = {"value": gaussian_motion_mmd2(gf, rf, sigma=spec.motion_kernel_sigma),
                "sample_count_each": selected_count, "generated_sample_indices": gi.tolist(), "reference_sample_indices": ri.tolist(),
                "features": "45 fixed-chain-scale local positions + 45 local velocities multiplied by explicit time scale", "feature_width": 90,
                "velocity_time_scale_s": spec.motion_velocity_time_scale_s, "sigma": spec.motion_kernel_sigma, "estimator": "biased MMD squared including diagonal"}
        for field in fields:
            left = generated["metrics"][field][gm & generated["metric_valid"][field]]
            right = reference["metrics"][field][rm & reference["metric_valid"][field]]
            row["metrics"][field] = {"generated": summary(left), "reference": summary(right),
                "wasserstein_1d": wasserstein_1d(left, right) if min(len(left), len(right)) >= minimum_bin_samples else None}
        bins.append(row)
    totals = {}
    for field in fields:
        supported = [row for row in bins if row["metrics"][field]["wasserstein_1d"] is not None]
        weights = np.array([row["metrics"][field]["reference"]["count"] for row in supported], dtype=float)
        distance = np.array([row["metrics"][field]["wasserstein_1d"] for row in supported], dtype=float)
        left = generated["metrics"][field][generated["metric_valid"][field]]
        right = reference["metrics"][field][reference["metric_valid"][field]]
        totals[field] = {"unconditional_wasserstein_1d": wasserstein_1d(left, right) if len(left) and len(right) else None,
            "equal_supported_bin_mean_wasserstein_1d": float(distance.mean()) if len(distance) else None,
            "reference_weighted_supported_bin_mean_wasserstein_1d": float(np.average(distance, weights=weights)) if len(distance) else None,
            "supported_bins": len(supported), "reference_sample_mass_covered": float(weights.sum()/len(right)) if len(right) else None}
    motion_rows = [row for row in bins if row["pose_velocity_gaussian_mmd2"] is not None]
    return {"speed_bins": bins, "minimum_metric_samples_per_bin": minimum_bin_samples,
        "maximum_balanced_motion_samples_per_bin": maximum_motion_samples,
        "equal_supported_bin_mean_pose_velocity_mmd2": float(np.mean([row["pose_velocity_gaussian_mmd2"]["value"] for row in motion_rows])) if motion_rows else None,
        "motion_supported_bins": len(motion_rows),
        "physical_speed_distribution_wasserstein_m_s": wasserstein_1d(gs, rs), "aggregate_metrics": totals,
        "author_parameters": False, "pass_threshold": None,
        "qualification": "Physical-speed matching reduces speed-mixture confounding; turning, terrain and task distributions remain distinct. Unsupported bins are unknown, not zero error."}
