"""Deterministic AMASS pedestrian curation without inventing semantic labels.

The paper names locomotion classes but does not publish clip IDs or thresholds.
These thresholds are a reconstruction policy. Numeric source names receive a
kinematic proxy label, never an asserted AMASS/BABEL annotation. Original
MotionLib IDs and the held-out lists are preserved.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import torch


SCHEMA = "navisaaclab.amass_reconstruction.v1"
POSITIVE = re.compile(r"walk|jog|run|sprint|scamper|turn|side.?step|look|stand|idle|static|transition", re.I)
NEGATIVE = re.compile(r"jump|crawl|climb|stair|beam|handrail|cartwheel|somersault|kick|punch|box(?:ing)?|dance|chacha|skip|hop|squat|kneel|(?:^|[^a-z])sit(?:ting|down)?(?:$|[^a-z])|lying|lie.?down|throw|catch|lift|knock|motorcycle|ballet|balett|tennis|gesture|accident|damage|greeting|stretch|drink|pick|put.?down|basketball|golf|swim|exercise|(?:^|[^a-z])rom(?:$|[^a-z])", re.I)
FIELDS = ("gts", "grs", "gvs", "gavs", "dps", "dvs", "lrs")


@dataclass(frozen=True)
class CurationConfig:
    target_train_clips: int = 2848
    seed: int = 7
    minimum_frames: int = 91
    control_hz: int = 30
    upright_fraction_minimum: float = .95
    root_height_min_m: float = .65
    root_height_max_m: float = 1.25
    root_up_z_minimum: float = .75
    maximum_speed_p99_m_s: float = 4.5
    maximum_acceleration_p99_m_s2: float = 30.
    maximum_dof_speed_p99_rad_s: float = 30.
    maximum_quaternion_norm_error: float = .005
    standing_speed_threshold_m_s: float = .12
    running_speed_threshold_m_s: float = 1.8
    standing_fraction_maximum: float = .15
    standing_hand_height_max_m: float = .45
    minimum_moving_extent_m: float = .75
    minimum_moving_path_length_m: float = 1.5
    fingerprint_resolution: float = .0001
    resample_50hz: bool = True
    dof_velocity_frame: str = "parent_joint"
    segment_long_clips: bool = True
    segment_frames: int = 250
    maximum_segments_per_source: int = 4
    excluded_datasets: tuple[str, ...] = ("BMLhandball", "DanceDB", "DFaust_67", "MPI_Limits")
    allow_unlabeled_datasets: tuple[str, ...] = ("CMU", "BMLmovi")

    def __post_init__(self):
        integer_fields = ("target_train_clips", "seed", "minimum_frames", "control_hz", "segment_frames", "maximum_segments_per_source")
        for key in integer_fields:
            value = getattr(self, key)
            if type(value) is not int or value < (0 if key == "seed" else 1):
                raise ValueError(f"{key} must be an integer in its valid range")
        for key, value in asdict(self).items():
            if key in {*integer_fields, "excluded_datasets", "allow_unlabeled_datasets", "resample_50hz", "segment_long_clips", "dof_velocity_frame"}:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        for key in ("upright_fraction_minimum", "root_up_z_minimum", "standing_fraction_maximum"):
            if getattr(self, key) > 1:
                raise ValueError(f"{key} cannot exceed one")
        if self.root_height_min_m >= self.root_height_max_m:
            raise ValueError("Root height bounds are reversed")
        if self.standing_speed_threshold_m_s >= self.running_speed_threshold_m_s:
            raise ValueError("Standing and running thresholds are reversed")
        if not all(isinstance(item, str) and item for item in self.excluded_datasets):
            raise ValueError("excluded_datasets must contain dataset names")
        if type(self.resample_50hz) is not bool or type(self.segment_long_clips) is not bool:
            raise ValueError("resample_50hz and segment_long_clips must be bool")
        if self.dof_velocity_frame not in {"parent_joint", "child_joint"}:
            raise ValueError("dof_velocity_frame must be parent_joint or child_joint")
        if self.control_hz != 30:
            raise ValueError("Existing AMASSCorpus/Diffusion stride contracts require control_hz=30")
        if self.segment_frames < 250 or self.segment_frames < self.minimum_frames:
            raise ValueError("Long continuous segments require at least250frames (8.3s)")
        if self.maximum_segments_per_source > 4:
            raise ValueError("At most four non-overlapping segments per source are supported")
        if not all(isinstance(item, str) and item for item in self.allow_unlabeled_datasets):
            raise ValueError("allow_unlabeled_datasets must contain dataset names")


def origin(source):
    """Parse the actual AMASS dataset/subject namespace from the stored path."""
    parts = str(source).replace("\\", "/").split("/")
    try:
        index = parts.index("smpl")
    except ValueError as exc:
        raise ValueError(f"Source lacks a documented smpl/dataset/subject origin: {source}") from exc
    if len(parts) < index + 4:
        raise ValueError(f"Source lacks a subject namespace: {source}")
    return {"dataset": parts[index + 1], "subject_namespace": "/".join(parts[index + 1:index + 3])}


def _numpy(value):
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _canonical_quaternion(q):
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    sign = np.take_along_axis(q, np.abs(q).argmax(-1)[..., None], axis=-1)
    return np.where(sign < 0, -q, q)


def _yaw(q):
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))


def canonical_fingerprint(clip, dt, resolution=.0001):
    """Hash full clips after removing initial XY/yaw and quaternion sign.

    Includes frame count, genuine dt, whole-body geometry, root orientation and
    non-root local rotations. This catches quantized rigid-placement duplicates;
    it is not a guarantee against temporal crops or approximately repeated takes.
    """
    positions = np.asarray(_numpy(clip["gts"]), dtype=np.float64).copy()
    local = np.asarray(_numpy(clip["lrs"]), dtype=np.float64)
    root_q = np.asarray(_numpy(clip["grs"])[:, 0], dtype=np.float64).copy()
    yaw0 = float(_yaw(root_q[0]))
    positions[..., :2] -= positions[0, 0, :2].copy()
    c, s = math.cos(yaw0), math.sin(yaw0)
    x, y = positions[..., 0].copy(), positions[..., 1].copy()
    positions[..., 0], positions[..., 1] = c*x + s*y, -s*x + c*y
    # Premultiply the root by the inverse initial heading (xyzw convention).
    z, w = -math.sin(yaw0/2), math.cos(yaw0/2)
    qx, qy, qz, qw = root_q.T.copy()
    root_q[:, 0], root_q[:, 1] = w*qx - z*qy, w*qy + z*qx
    root_q[:, 2], root_q[:, 3] = w*qz + z*qw, w*qw - z*qz
    digest = hashlib.sha256()
    digest.update(f"amass-canonical-v1:{len(positions)}:{round(float(dt), 8)}:{resolution}".encode())
    for array in (positions, _canonical_quaternion(root_q), _canonical_quaternion(local[:, 1:])):
        digest.update(np.rint(array/resolution).astype("<i4").tobytes())
    return digest.hexdigest()


def assess_clip(clip, dt, config, *, body_names, require_control_dt=True):
    """Assess every frame; return independent quality failures and proxy class."""
    arrays = {key: _numpy(clip[key]) for key in FIELDS}
    count = len(arrays["gts"])
    expected = {"gts": (count, 24, 3), "grs": (count, 24, 4), "gvs": (count, 24, 3),
                "gavs": (count, 24, 3), "dps": (count, 69), "dvs": (count, 69), "lrs": (count, 24, 4)}
    failures = []
    if count < config.minimum_frames or (math.isfinite(dt) and (count - 1)*dt < (config.minimum_frames - 1)/config.control_hz - 1e-6):
        failures.append("too_short")
    if not math.isfinite(dt) or dt <= 0:
        return {"failures": failures + ["invalid_dt"]}
    if require_control_dt and abs(dt - 1/config.control_hz) > 1e-7:
        failures.append("requires_real_resampling")
    if any(value.shape != expected[key] for key, value in arrays.items()):
        return {"failures": failures + ["shape_contract"]}
    if any(not np.isfinite(value).all() for value in arrays.values()):
        return {"failures": failures + ["nonfinite"]}
    position, rotation = arrays["gts"][:, 0], arrays["grs"][:, 0]
    norm_error = max(float(np.max(np.abs(np.linalg.norm(arrays[key], axis=-1) - 1))) for key in ("grs", "lrs"))
    up = 1 - 2 * np.square(rotation[:, :2]).sum(-1)
    upright = float(np.mean((position[:, 2] >= config.root_height_min_m) & (position[:, 2] <= config.root_height_max_m) & (up >= config.root_up_z_minimum)))
    speed = np.linalg.norm(arrays["gvs"][:, 0, :2], axis=-1)
    median_speed, speed_p99 = (float(np.quantile(speed, q)) for q in (.5, .99))
    acceleration = np.linalg.norm(np.diff(arrays["gvs"][:, 0, :2], axis=0)/dt, axis=-1)
    accel_p99 = float(np.quantile(acceleration, .99)) if len(acceleration) else 0.
    dof_p99 = float(np.quantile(np.abs(arrays["dvs"]), .99))
    heading = np.unwrap(_yaw(rotation))
    yaw_span = float(np.ptp(heading))
    cosine, sine = np.cos(heading), np.sin(heading)
    lateral = np.abs(-sine*arrays["gvs"][:, 0, 0] + cosine*arrays["gvs"][:, 0, 1])
    lateral_fraction = float(np.median(lateral/np.maximum(speed, .05)))
    hand_indices = [body_names.index(name) for name in ("L_Hand", "R_Hand")]
    hand_height = float(np.quantile(arrays["gts"][:, hand_indices, 2] - position[:, None, 2], .5))
    extent = float(np.max(np.linalg.norm(position[:, :2] - position[0, :2], axis=-1)))
    path_length = float(np.linalg.norm(np.diff(position[:, :2], axis=0), axis=-1).sum())
    if norm_error > config.maximum_quaternion_norm_error:
        failures.append("quaternion_norm")
    if upright < config.upright_fraction_minimum:
        failures.append("upright_fraction")
    if speed_p99 > config.maximum_speed_p99_m_s:
        failures.append("root_speed_p99")
    if accel_p99 > config.maximum_acceleration_p99_m_s2:
        failures.append("root_acceleration_p99")
    if dof_p99 > config.maximum_dof_speed_p99_rad_s:
        failures.append("dof_speed_p99")
    if median_speed < config.standing_speed_threshold_m_s:
        category = "standing_looking_like"
        if hand_height > config.standing_hand_height_max_m:
            failures.append("standing_hands_high")
    elif median_speed >= config.running_speed_threshold_m_s:
        category = "running_like"
    elif lateral_fraction > .55:
        category = "sidestepping_like"
    elif yaw_span >= .8:
        category = "turning_like"
    else:
        category = "walking_like"
    if category != "standing_looking_like" and (extent < config.minimum_moving_extent_m or path_length < config.minimum_moving_path_length_m):
        failures.append("moving_translation_extent")
    feet = [body_names.index(name) for name in ("L_Ankle", "L_Toe", "R_Ankle", "R_Toe")]
    foot_z = arrays["gts"][:, feet, 2]
    inferred_floor = float(np.quantile(foot_z, .02))
    near_floor = foot_z <= inferred_floor + .05
    foot_speed = np.linalg.norm(arrays["gvs"][:, feet, :2], axis=-1)
    foot_sliding = float(np.mean(foot_speed[near_floor])) if near_floor.any() else None
    return {"failures": failures, "frames": count, "dt": dt,
            "upright_fraction": upright, "median_speed_m_s": median_speed,
            "root_speed_p99_m_s": speed_p99, "root_acceleration_p99_m_s2": accel_p99,
            "dof_speed_p99_rad_s": dof_p99, "quaternion_norm_error_max": norm_error,
            "root_yaw_span_rad": yaw_span, "lateral_speed_fraction_median": lateral_fraction,
            "hand_height_relative_pelvis_median_m": hand_height,
            "root_planar_extent_m": extent, "root_planar_path_length_m": path_length,
            "near_floor_foot_planar_speed_proxy_m_s": foot_sliding,
            "inferred_floor_m": inferred_floor, "style_class": category,
            "style_class_source": "kinematic_proxy", "semantic_label_verified": False,
            "diffusion_eligible": count >= 250}


def slerp_xyzw(first, second, fraction):
    """Shortest-arc normalized SLERP, including antipodal/near-equal endpoints."""
    first = torch.nn.functional.normalize(first, dim=-1)
    second = torch.nn.functional.normalize(second, dim=-1)
    dot = (first*second).sum(-1, keepdim=True)
    second = torch.where(dot < 0, -second, second)
    dot = dot.abs().clamp(max=1)
    while fraction.ndim < first.ndim:
        fraction = fraction.unsqueeze(-1)
    angle = torch.acos(dot)
    denominator = torch.sin(angle).clamp_min(1e-8)
    spherical = (torch.sin((1-fraction)*angle)*first + torch.sin(fraction*angle)*second)/denominator
    linear = first + fraction*(second-first)
    return torch.nn.functional.normalize(torch.where(dot > .9995, linear, spherical), dim=-1)


def joint_angular_velocity_xyzw(local_rotations, dt, *, frame):
    """Local angular velocity with explicit parent/child joint coordinates.

    Parent: q_next*q_current^-1 (current ProtoMotions AMASS converter).
    Child: q_current^-1*q_next (PACER and measured IsaacLab3 D6 coordinates).
    This is angular velocity, not the derivative of an exp-map coordinate.
    """
    from protomotions.utils.rotations import quat_angle_axis, quat_conjugate, quat_mul_norm

    if frame not in {"parent_joint", "child_joint"}:
        raise ValueError("Unknown joint angular-velocity frame")
    if isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Joint velocity needs finite positive dt")
    if local_rotations.ndim != 3 or local_rotations.shape[-1] != 4 or not torch.isfinite(local_rotations).all():
        raise ValueError("Expected finite local xyzw rotations[T,J,4]")
    norms = torch.linalg.vector_norm(local_rotations, dim=-1, keepdim=True)
    if torch.any(norms < 1e-8):
        raise ValueError("Joint quaternions must have nonzero norm")
    local = local_rotations/norms
    result = local.new_zeros((*local.shape[:-1], 3))
    if len(local) < 2:
        return result
    inverse = quat_conjugate(local[:-1], w_last=True)
    delta = (quat_mul_norm(inverse, local[1:], w_last=True) if frame == "child_joint" else
             quat_mul_norm(local[1:], inverse, w_last=True))
    angle, axis = quat_angle_axis(delta, w_last=True)
    result[:-1] = axis*angle.unsqueeze(-1)/dt
    result[-1] = result[-2]
    return result


def resample_clip(clip, source_dt, control_hz, kinematic_info, *, dof_velocity_frame="parent_joint"):
    """Actually sample root/local rotations at 30Hz, FK and recompute velocities.

    The last target grid point never extends beyond the source duration. Local
    rotations use shortest-arc SLERP; root XYZ uses linear interpolation. DOF
    exponential maps are extracted from rotations rather than interpolating
    discontinuous principal logarithms. Contacts are only kinematic proxies.
    """
    from protomotions.components.pose_lib import (
        compute_angular_velocity, compute_forward_kinematics_from_transforms,
        compute_kinematics_velocities, extract_qpos_from_transforms,
    )
    from protomotions.utils.rotations import quaternion_to_matrix, matrix_to_quaternion

    if not math.isfinite(source_dt) or source_dt <= 0 or type(control_hz) is not int or control_hz <= 0:
        raise ValueError("Resampling requires positive genuine source dt and integer control Hz")
    count = len(clip["gts"])
    if count < 2 or any(not torch.isfinite(clip[key]).all() for key in FIELDS):
        raise ValueError("Resampling requires at least two finite poses")
    # First/middle/last prove the original cache and local rotations use this rig.
    sample = torch.tensor(sorted({0, count//2, count - 1}))
    source_local = quaternion_to_matrix(clip["lrs"][sample], w_last=True)
    fk_positions, _ = compute_forward_kinematics_from_transforms(kinematic_info, clip["gts"][sample, 0], source_local)
    source_fk_error = float(torch.linalg.vector_norm(fk_positions - clip["gts"][sample], dim=-1).max())
    if source_fk_error > .001:
        raise ValueError(f"Source local-rotation/neutral-rig FK mismatch {source_fk_error:.6g}m; reconvert source explicitly")
    duration = (count - 1)*source_dt
    target_count = int(math.floor(duration*control_hz + 1e-6)) + 1
    locations = torch.arange(target_count, dtype=torch.float64)/(control_hz*source_dt)
    locations = locations.clamp(max=count - 1)
    lower = locations.floor().long()
    upper = (lower + 1).clamp(max=count - 1)
    blend = (locations-lower).float()
    root = clip["gts"][lower, 0] + blend[:, None]*(clip["gts"][upper, 0] - clip["gts"][lower, 0])
    local = slerp_xyzw(clip["lrs"][lower], clip["lrs"][upper], blend)
    matrices = quaternion_to_matrix(local, w_last=True)
    positions, rotations = compute_forward_kinematics_from_transforms(kinematic_info, root, matrices)
    velocities, angular = compute_kinematics_velocities(positions, rotations, control_hz, velocity_max_horizon=1)
    qpos = extract_qpos_from_transforms(kinematic_info, root, matrices, multi_dof_decomposition_method="exp_map")
    if dof_velocity_frame == "parent_joint":
        dof_velocities = compute_angular_velocity(matrices[:, 1:], control_hz, velocity_max_horizon=1).reshape(target_count, 69)
    else:
        dof_velocities = joint_angular_velocity_xyzw(local[:, 1:], 1/control_hz, frame=dof_velocity_frame).reshape(target_count, 69)
    result = {"gts": positions, "grs": matrix_to_quaternion(rotations, w_last=True),
              "gvs": velocities, "gavs": angular, "dps": qpos[:, 7:], "dvs": dof_velocities,
              "lrs": local, "contacts": (torch.linalg.vector_norm(velocities, dim=-1) < .15) & (positions[..., 2] < .1)}
    return result, {"source_frames": count, "target_frames": target_count,
                    "source_dt": source_dt, "target_dt": 1/control_hz,
                    "source_duration_s": duration, "target_duration_s": (target_count - 1)/control_hz,
                    "source_sample_fk_max_error_m": source_fk_error,
                    "orientation": "local xyzw shortest-arc SLERP; DOF exp_map from interpolated matrices",
                    "velocity": "fresh 30Hz forward differences/angular quaternion differences, horizon=1",
                    "dof_velocity_frame": dof_velocity_frame,
                    "contacts": "position/velocity threshold proxy, not PhysX contacts"}


def segment_spans(frame_count, config):
    """At most four separated, non-overlapping continuous windows per source.

    Prefer an8.3s diffusion window; retain a >=3s disjoint tail for AMP. Medium
    sources without an8.3s window may contribute >=3s AMP windows. Sources
    contribute windows OR the whole clip, never both. No padded/repeated time.
    """
    if not config.segment_long_clips or config.maximum_segments_per_source == 1:
        return [(0, frame_count)]
    if frame_count < 2*config.segment_frames:
        if frame_count >= config.segment_frames + config.minimum_frames:
            return [(0, config.segment_frames), (config.segment_frames, frame_count)]
        if 2*config.minimum_frames <= frame_count < config.segment_frames:
            count = min(frame_count//config.minimum_frames, config.maximum_segments_per_source)
            boundaries = np.rint(np.linspace(0, frame_count, count + 1)).astype(int)
            return [(int(start), int(stop)) for start, stop in zip(boundaries, boundaries[1:])]
        return [(0, frame_count)]
    available = frame_count//config.segment_frames
    count = min(available, config.maximum_segments_per_source)
    windows = np.rint(np.linspace(0, available - 1, count)).astype(int)
    spans = [(int(index)*config.segment_frames, (int(index)+1)*config.segment_frames) for index in windows]
    tail_start = available*config.segment_frames
    if len(spans) < config.maximum_segments_per_source and frame_count-tail_start >= config.minimum_frames:
        spans.append((tail_start, frame_count))
    return spans


def _refresh_segment_velocity(clip, control_hz, *, dof_velocity_frame="parent_joint"):
    from protomotions.components.pose_lib import compute_angular_velocity, compute_kinematics_velocities
    from protomotions.utils.rotations import quaternion_to_matrix

    result = dict(clip)
    result["gvs"], result["gavs"] = compute_kinematics_velocities(
        clip["gts"], quaternion_to_matrix(clip["grs"], w_last=True), control_hz, velocity_max_horizon=1)
    if dof_velocity_frame == "parent_joint":
        result["dvs"] = compute_angular_velocity(
            quaternion_to_matrix(clip["lrs"][:, 1:], w_last=True), control_hz, velocity_max_horizon=1).reshape(len(clip["gts"]), 69)
    else:
        result["dvs"] = joint_angular_velocity_xyzw(clip["lrs"][:, 1:], 1/control_hz, frame=dof_velocity_frame).reshape(len(clip["gts"]), 69)
    result["contacts"] = (torch.linalg.vector_norm(result["gvs"], dim=-1) < .15) & (result["gts"][..., 2] < .1)
    return result


def select_balanced(records, config):
    """Prefer navigation labels, then stable style/dataset/subject round robin."""
    groups = defaultdict(list)
    for record in records:
        tier = 1 if record.get("label_evidence", "kinematic_candidate_only") == "kinematic_candidate_only" else 0
        groups[(tier, record["style_class"], record["dataset"], record["subject_namespace"])].append(record)
    queues = {}
    for key, group in groups.items():
        group.sort(key=lambda row: hashlib.sha256(f"{config.seed}:{row['source_path']}".encode()).hexdigest())
        queues[key] = deque(group)
    # Rotating subjects prevents a long prolific subject from exhausting the pool.
    keys = sorted(queues, key=lambda key: hashlib.sha256(f"{config.seed}:{key}".encode()).hexdigest())
    chosen, standing = [], 0
    cap = int(config.target_train_clips*config.standing_fraction_maximum)
    for tier in (0, 1):
        while len(chosen) < config.target_train_clips:
            progress = False
            for key in keys:
                if key[0] != tier or not queues[key] or (key[1] == "standing_looking_like" and standing >= cap):
                    continue
                record = queues[key].popleft()
                chosen.append(record)
                standing += key[1] == "standing_looking_like"
                progress = True
                if len(chosen) == config.target_train_clips:
                    break
            if not progress:
                break
    # A shortfall must not turn a target-based standing count into an excessive
    # fraction of the actual dataset. It also cannot be filled by duplicates.
    moving = len(chosen) - standing
    actual_cap = (len(chosen) if config.standing_fraction_maximum == 1 else
                  int(moving*config.standing_fraction_maximum/(1-config.standing_fraction_maximum)))
    if standing > actual_cap:
        kept, count = [], 0
        for record in chosen:
            if record["style_class"] == "standing_looking_like":
                if count >= actual_cap:
                    continue
                count += 1
            kept.append(record)
        chosen = kept
    return sorted(chosen, key=lambda row: row["motion_id"])


def split_namespace_review(sources):
    """Fail on declared dataset/subject/source overlap; physical identity unknown."""
    namespaces = {}
    for split, names in sources.items():
        origins = [origin(name) for name in names]
        namespaces[split] = {"source": set(map(str, names)),
                             "dataset": {row["dataset"] for row in origins},
                             "subject_namespace": {row["subject_namespace"] for row in origins}}
    overlaps = {}
    splits = list(namespaces)
    for index, left in enumerate(splits):
        for right in splits[index + 1:]:
            for key in ("source", "dataset", "subject_namespace"):
                common = namespaces[left][key] & namespaces[right][key]
                if common:
                    overlaps[f"{left}/{right}/{key}"] = sorted(common)
    if overlaps:
        raise ValueError(f"Source namespace leakage: {overlaps}")
    return {"dataset_disjoint": True, "subject_namespace_disjoint": True,
            "physical_identity_across_datasets_verified": False,
            "pacer_pretraining_overlap_verified": False,
            "datasets": {key: sorted(value["dataset"]) for key, value in namespaces.items()}}


def _slice_clip(data, index):
    start, count = int(data["length_starts"][index]), int(data["motion_num_frames"][index])
    if start < 0 or count < 1 or start + count > len(data["gts"]):
        raise ValueError(f"Invalid clip boundaries at motion {index}")
    return {key: data[key][start:start + count] for key in FIELDS}


def _write_selected_bank(path, bank, selected, converted, control_hz, *, dof_velocity_frame):
    """Write a new, correctly indexed tensor-only train bank; never edit sources."""
    if path.exists():
        raise FileExistsError(f"Preserve existing bank: {path}")
    tensors = {key: [] for key in (*FIELDS, "contacts")}
    records = []
    for new_id, row in enumerate(selected):
        old_id = row.get("original_motion_id", row["motion_id"])
        if row["motion_id"] in converted:
            clip = converted[row["motion_id"]]
        else:
            clip = _slice_clip(bank, old_id)
            start, count = int(bank["length_starts"][old_id]), int(bank["motion_num_frames"][old_id])
            clip["contacts"] = bank["contacts"][start:start + count]
        for key in tensors:
            tensors[key].append(clip[key])
        records.append({**row, "motion_id": new_id, "original_motion_id": old_id})
    counts = torch.tensor([row["frames"] for row in records], dtype=torch.long)
    result = {key: torch.cat(values).contiguous() for key, values in tensors.items()}
    result.update(length_starts=counts.cumsum(0)-counts, motion_num_frames=counts,
                  motion_dt=torch.full((len(records),), 1/control_hz),
                  motion_lengths=(counts - 1)/control_hz,
                  motion_weights=torch.full((len(records),), 1/len(records)),
                  motion_files=tuple(row["source_path"] for row in records),
                  dof_velocity_frame=dof_velocity_frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return records, {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest(),
                     "clip_count": len(records), "diffusion_clip_count": sum(row["diffusion_eligible"] for row in records),
                     "dof_velocity_frame": dof_velocity_frame}


def curate_manifest(base_manifest, output, *, config=None, body_names, kinematic_info=None,
                    labels_by_source=None, progress=None):
    """Expand actual AMASS clips, genuinely resample eligible 50Hz takes if needed.

    The original multi-GB checksum is reused from the previously verified input
    manifest after a fresh byte-size check. Complete content hashing is not
    silently claimed. A new bank is written when actual rate conversion is used.
    A shortfall is explicit and never padded by duplication.
    """
    config = config or CurationConfig()
    base_path = Path(base_manifest).resolve()
    base = json.loads(base_path.read_text())
    if base.get("schema") != SCHEMA or not {"train", "validation", "test"}.issubset(base.get("splits", {})):
        raise ValueError("Expected the existing three-split AMASS manifest")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Preserve existing curation output: {output}")
    data, source_stats = {}, {}
    for split, entry in base["splits"].items():
        source = Path(entry["path"])
        source = source if source.is_absolute() else base_path.parent/source
        stat = source.stat()
        if stat.st_size != entry["bytes"]:
            raise ValueError(f"Source byte size changed: {source}")
        data[split] = torch.load(source, weights_only=True, map_location="cpu", mmap=True)
        source_stats[split] = {"path": str(source.resolve()), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                               "sha256": entry["sha256"], "sha256_recomputed": False,
                               "hash_evidence": f"Previously verified input manifest {base_path}"}
    separation = split_namespace_review({split: bank["motion_files"] for split, bank in data.items()})
    heldout_fingerprints = set()
    for split in ("validation", "test"):
        for index in range(len(data[split]["motion_files"])):
            clip = _slice_clip(data[split], index)
            if all(torch.isfinite(clip[key]).all() for key in ("gts", "grs", "lrs")):
                heldout_fingerprints.add(canonical_fingerprint(clip, float(data[split]["motion_dt"][index]), config.fingerprint_resolution))
    bank = data["train"]
    candidates, rejected, duplicates, known, converted = [], Counter(), [], {}, {}
    labels_by_source = labels_by_source or {}
    for index, source in enumerate(bank["motion_files"]):
        source_origin = origin(source)
        name = Path(source).name
        annotation = labels_by_source.get(str(source))
        description = annotation["description"] if annotation is not None else name
        if source_origin["dataset"] in config.excluded_datasets:
            rejected["excluded_dataset"] += 1
            continue
        if NEGATIVE.search(description):
            rejected["negative_description_label" if annotation else "negative_filename_label"] += 1
            continue
        positive = bool(POSITIVE.search(description))
        if not positive and (annotation is not None or source_origin["dataset"] not in config.allow_unlabeled_datasets):
            rejected["labeled_non_navigation" if annotation else "descriptive_filename_without_navigation_label"] += 1
            continue
        clip = _slice_clip(bank, index)
        source_dt = float(bank["motion_dt"][index])
        source_frames = len(clip["gts"])
        needs_resampling = abs(source_dt - 1/config.control_hz) > 1e-7
        if needs_resampling and (not config.resample_50hz or abs(source_dt - 1/50) > 1e-7):
            rejected["requires_real_resampling"] += 1
            continue
        if (source_frames - 1)*source_dt < (config.minimum_frames - 1)/config.control_hz - 1e-6:
            rejected["too_short_source"] += 1
            continue
        if any(not torch.isfinite(clip[key]).all() for key in FIELDS):
            rejected["nonfinite_source"] += 1
            continue
        norm_error = max(float((torch.linalg.vector_norm(clip[key], dim=-1) - 1).abs().max()) for key in ("grs", "lrs"))
        if norm_error > config.maximum_quaternion_norm_error:
            rejected["source_quaternion_norm"] += 1
            continue
        if kinematic_info is None:
            raise ValueError("Curated neutral-rig FK and fresh velocities need kinematic_info")
        # Even genuine30Hz sources are explicitly reconstructed, never silently
        # mixed with the old multi-horizon smoothed velocity convention.
        # motion_dt is float32 metadata for a nominal30/50Hz sampling clock.
        # Use that verified nominal source grid rather than accumulating its
        # sub-nanosecond representation error into interpolation boundaries.
        nominal_source_dt = 1/(50 if needs_resampling else config.control_hz)
        clip, resampling = resample_clip(clip, nominal_source_dt, config.control_hz, kinematic_info, dof_velocity_frame=config.dof_velocity_frame)
        resampling["source_dt_metadata"] = source_dt
        resampling["source_grid_policy"] = "verified nominal30/50Hz clock; real SLERP/FK resampling, float32 metadata rounding recorded"
        spans = segment_spans(len(clip["gts"]), config)
        for start, stop in spans:
            fragment = _refresh_segment_velocity({key: value[start:stop] for key, value in clip.items()}, config.control_hz, dof_velocity_frame=config.dof_velocity_frame)
            metrics = assess_clip(fragment, 1/config.control_hz, config, body_names=body_names)
            if metrics["failures"]:
                rejected.update(f"segment_{failure}" for failure in metrics["failures"])
                continue
            fingerprint = canonical_fingerprint(fragment, metrics["dt"], config.fingerprint_resolution)
            if fingerprint in heldout_fingerprints:
                rejected["heldout_canonical_duplicate"] += 1
                continue
            fragment_source = f"{source}#frames={start}:{stop}"
            if fingerprint in known:
                rejected["training_canonical_duplicate"] += 1
                duplicates.append({"original_motion_id": index, "source_path": fragment_source,
                                   "duplicate_of_candidate_id": known[fingerprint], "fingerprint": fingerprint})
                continue
            candidate_id = len(candidates)
            known[fingerprint] = candidate_id
            converted[candidate_id] = fragment
            metrics.pop("failures")
            source_frame_start = int(math.floor(start/(config.control_hz*nominal_source_dt) + 1e-5))
            source_frame_stop = int(math.ceil((stop - 1)/(config.control_hz*nominal_source_dt) - 1e-5)) + 1
            candidates.append({"motion_id": candidate_id, "original_motion_id": index,
                "source_path": fragment_source, "parent_source_path": str(source), **source_origin, **metrics,
                "parent_control_frame_start": start, "parent_control_frame_stop": stop,
                "parent_source_frame_start": source_frame_start, "parent_source_frame_stop": source_frame_stop,
                "parent_start_time_s": start/config.control_hz, "parent_end_time_s": (stop - 1)/config.control_hz,
                "parent_source_frames": source_frames, "parent_source_dt": source_dt,
                "canonical_sha256": fingerprint,
                "filename_navigation_label_match": bool(POSITIVE.search(name)),
                "semantic_label_verified": annotation is not None, "semantic_annotation": annotation,
                "source_frames": source_frames, "source_dt": source_dt,
                "dof_velocity_frame": config.dof_velocity_frame,
                "resampling": {**resampling, "source_rate_changed": needs_resampling},
                "label_evidence": "official_motion_description" if annotation else ("filename_keyword" if positive else "kinematic_candidate_only")})
        if progress is not None and index % 500 == 0:
            progress(index, len(bank["motion_files"]), len(candidates))
    selected = select_balanced(candidates, config)
    document = {"schema": SCHEMA, "author_clip_list": False, "seed": config.seed,
                "selection": {**asdict(config), "actual_author_ids_known": False,
                    "semantic_annotation_available": bool(labels_by_source), "positive_filename_regex": POSITIVE.pattern,
                    "semantic_annotations_sha256": hashlib.sha256(json.dumps(labels_by_source, sort_keys=True).encode()).hexdigest(),
                    "negative_filename_regex": NEGATIVE.pattern,
                    "candidate_policy": "descriptive sources require positive navigation/look/stand labels; supplied official descriptions supersede filenames; only declared numeric datasets may remain explicit unknown kinematic candidates",
                    "resampling": "allselectedclips reconstruct currentneutralFK from root/localrotations and fresh30Hz forward/angular velocities; 50Hz uses actual shortestarcSLERP",
                    "segmentation": "max4/source; prefer250frame(8.3s)Diffwindows; nonoverlappingmediumclips/tails >=91frames(3s) may contributeAMP-only windows; windowsORwholeclip, neverboth; no repeatedtimeinterval/padding",
                    "selection_order": "navigation filename/officialdescription candidates first, then explicitlyunknown candidates; stable seeded round robin over style/dataset/subject namespaces",
                    "expert_actions": base.get("selection", {}).get("expert_actions"),
                    "contact_quality": "near-floor speed is a kinematic proxy, not measured contact or a hard filter",
                    "heldout_policy": "original validation/test clip IDs retained exactly"},
                "splits": {key: value for key, value in base["splits"].items() if key != "train"},
                "curation": {"input_manifest": str(base_path),
                    "input_manifest_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
                    "source_stats": source_stats, "split_separation": separation,
                    "train_raw_clips": len(bank["motion_files"]), "eligible_unique_train_clips": len(candidates),
                    "requested_train_clips": config.target_train_clips, "selected_train_clips": len(selected),
                    "target_reached": len(selected) == config.target_train_clips,
                    "rejection_reason_counts_not_mutually_exclusive": dict(rejected),
                    "duplicates": duplicates, "heldout_fingerprints_checked": len(heldout_fingerprints),
                    "candidate_counts_by_style": dict(Counter(row["style_class"] for row in candidates)),
                    "selected_counts_by_style": dict(Counter(row["style_class"] for row in selected)),
                    "selected_counts_by_dataset": dict(Counter(row["dataset"] for row in selected)),
                    "selected_subject_namespaces": len({row["subject_namespace"] for row in selected}),
                    "selected_filename_labeled": sum(row["filename_navigation_label_match"] for row in selected),
                    "selected_official_description_labeled": sum(row["semantic_label_verified"] for row in selected),
                    "selected_unknown_proxy_labeled": sum(row["label_evidence"] == "kinematic_candidate_only" for row in selected),
                    "selected_resampled_clips": sum(row["resampling"]["source_rate_changed"] for row in selected),
                    "selected_parent_sources": len({row["parent_source_path"] for row in selected}),
                    "selected_segments_per_parent_max": max(Counter(row["parent_source_path"] for row in selected).values(), default=0),
                    "parent_source_paths_by_split": {"train": sorted({row["parent_source_path"] for row in selected}),
                        **{key: sorted(row["source_path"] for row in base["splits"][key]["clips"]) for key in ("validation", "test")}},
                    "limitations": ["Not the unpublished author clip list or semantic-label filtering.",
                        "Numeric AMASS takes are kinematic candidates, not verified walking annotations.",
                        "Dataset/subject namespaces are disjoint; physical identity across datasets is unknown.",
                        "Full-clip quantized duplicate hashing does not detect every cropped/repeated take.",
                        "Motion quality gates do not guarantee dynamic policy tracking or contacts.",
                        "Dataset balancing and quality thresholds change the source distribution."]}}
    train_entry = base["splits"]["train"]
    if selected:
        selected, train_entry = _write_selected_bank(output.parent/(output.stem + "_train.pt"), bank, selected, converted,
                                                     config.control_hz, dof_velocity_frame=config.dof_velocity_frame)
        document["curation"]["new_bank_hash_recomputed"] = True
    else:
        document["curation"]["new_bank_hash_recomputed"] = False
    document["splits"]["train"] = {**train_entry, "clips": selected, "clip_count": len(selected),
                                      "diffusion_clip_count": sum(row["diffusion_eligible"] for row in selected)}
    if not selected:
        raise ValueError("No genuine eligible training clips")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n")
    return document


def convert_curated_joint_velocities(manifest, output):
    """Version a selected bank using the measured native child-joint velocity.

    Clip membership/order, poses, all global velocities, contacts, timings and
    held-out resources remain byte-for-byte tensor equal. Only dvs and explicit
    convention/quality metadata change. Never overwrite the historical bank.
    """
    input_path, output = Path(manifest).resolve(), Path(output).resolve()
    document = json.loads(input_path.read_text())
    if document.get("schema") != SCHEMA:
        raise ValueError("Expected an AMASS reconstruction manifest")
    bank_path = Path(document["splits"]["train"]["path"])
    bank_path = bank_path if bank_path.is_absolute() else input_path.parent/bank_path
    target_bank = output.parent/(output.stem + "_train.pt")
    if output.exists() or target_bank.exists():
        raise FileExistsError("Velocity conversion preserves existing output artifacts")
    entry = document["splits"]["train"]
    if bank_path.stat().st_size != entry["bytes"]:
        raise ValueError("Selected source bank byte size changed")
    bank = torch.load(bank_path, weights_only=True, map_location="cpu", mmap=True)
    declared_frames = {value for value in (bank.get("dof_velocity_frame"), entry.get("dof_velocity_frame")) if value is not None}
    if len(declared_frames) > 1 or declared_frames - {"parent_joint", "child_joint"}:
        raise ValueError("Selected bank velocity declarations conflict or are unknown")
    if declared_frames == {"child_joint"}:
        raise ValueError("Selected bank already contains child-joint velocities; preserve it")
    converted = dict(bank)
    new_velocities = torch.empty_like(bank["dvs"])
    quality, intervals = [], []
    for row in entry["clips"]:
        index, count = row["motion_id"], row["frames"]
        if count != int(bank["motion_num_frames"][index]) or row["source_path"] != bank["motion_files"][index]:
            raise ValueError("Selected clip identity/frame contract changed")
        start, dt = int(bank["length_starts"][index]), float(row["dt"])
        velocity = joint_angular_velocity_xyzw(bank["lrs"][start:start + count, 1:], dt, frame="child_joint").reshape(count, 69)
        score = float(torch.quantile(velocity.abs().reshape(-1), .99))
        if not torch.isfinite(velocity).all() or score > document["selection"]["maximum_dof_speed_p99_rad_s"]:
            raise ValueError(f"Child-joint velocity quality fails at selected clip {index}")
        new_velocities[start:start + count] = velocity
        row["dof_speed_p99_rad_s"] = score
        row["dof_velocity_frame"] = "child_joint"
        row["resampling"]["dof_velocity_frame"] = "child_joint"
        quality.append(score)
        intervals.append((start, start + count))
    if sorted(intervals) != list(zip(bank["length_starts"].tolist(), (bank["length_starts"] + bank["motion_num_frames"]).tolist())) or sum(row["frames"] for row in entry["clips"]) != len(new_velocities):
        raise ValueError("Selected clips must cover the entire curated bank exactly once")
    converted["dvs"] = new_velocities
    converted["dof_velocity_frame"] = "child_joint"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, target_bank)
    digest = hashlib.sha256()
    with target_bank.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    restored = torch.load(target_bank, weights_only=True, map_location="cpu", mmap=True)
    equality = {}
    for key, value in bank.items():
        if key in {"dvs", "dof_velocity_frame"}:
            continue
        equality[key] = bool(torch.equal(value, restored[key])) if isinstance(value, torch.Tensor) else value == restored[key]
    if not all(equality.values()):
        raise ValueError("Velocity-only conversion changed another tensor/metadata field")
    difference = bank["dvs"] - restored["dvs"]
    review = {"source_manifest": str(input_path), "source_manifest_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
              "source_bank": str(bank_path), "source_bank_declared_sha256": entry["sha256"],
              "target_bank": str(target_bank), "target_bank_sha256": digest.hexdigest(),
              "target_bank_bytes": target_bank.stat().st_size,
              "source_bank_sha256_recomputed": False, "source_unchanged": True,
              "dof_velocity_frame": "child_joint", "formula": "axis_angle(q_t^-1 * q_(t+1))/dt; shortest rotation; finalframe carries previous velocity",
              "previous_formula": "axis_angle(q_(t+1) * q_t^-1)/dt in parentjointframe",
              "exp_map_coordinate_derivative": False, "world_gvs_gavs_unchanged": True,
              "all_other_fields_roundtrip_equal": equality,
              "dvs_delta_rms_rad_s": float(difference.square().mean().sqrt()),
              "dvs_delta_max_abs_rad_s": float(difference.abs().max()),
              "per_clip_dvs_component_p99_max_rad_s": max(quality),
              "per_clip_dvs_component_p99_median_rad_s": float(np.median(quality)),
              "child_velocity_quality_gate_all_passed": True,
              "heldout_entries_unchanged": True,
              "membership_order_parent_segments_poses_unchanged": True}
    document["selection"]["dof_velocity_frame"] = "child_joint"
    document["selection"]["dof_velocity_formula"] = review["formula"]
    document["curation"]["joint_velocity_conversion"] = review
    entry.update(path=str(target_bank), bytes=target_bank.stat().st_size, sha256=digest.hexdigest(),
                 dof_velocity_frame="child_joint")
    output.write_text(json.dumps(document, indent=2) + "\n")
    return document
