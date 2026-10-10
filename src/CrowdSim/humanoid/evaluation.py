"""CPU pedestrian metrics with explicit reconstruction choices.

The paper describes 15 root-centered, heading-aligned, scale-normalized points,
AMASS K=50/100 coverage/entropy and Motion-MMD, but does not release the complete
keypoint IDs, scale definition, kernel bandwidth or sampling protocol. The
choices below are reproducible local evaluation settings, not author metrics.
No simulator, training checkpoint or optional clustering package is imported.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path

import torch

from .amp.observations import PACER_BODY_NAMES


KEYPOINT_NAMES = (
    "Pelvis", "L_Hip", "R_Hip", "L_Knee", "R_Knee", "L_Ankle", "R_Ankle",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist", "Neck", "Head",
)


@dataclass(frozen=True)
class PoseEmbeddingSpec:
    keypoint_names: tuple[str, ...] = KEYPOINT_NAMES
    center_body: str = "Pelvis"
    height_definition: str = "head_to_mid_ankle_euclidean_length_per_frame"
    coordinates: str = "root-centered, inverse root XYZW heading, XYZ divided by height"
    feature_width: int = 45
    author_parameters: bool = False


def _tensor(value, name, *, dtype=torch.float64):
    # Set dtype while decoding Python floats/JSON, before any float32 rounding.
    result = torch.as_tensor(value, dtype=dtype)
    if result.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU; transfer explicitly before evaluation")
    result = result.detach().to(dtype=dtype)
    if not torch.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result


def _quaternions(value, leading):
    quaternion = _tensor(value, "root_quat_xyzw")
    if quaternion.shape != (*leading, 4):
        raise ValueError("root_quat_xyzw shape must match pose/root leading dimensions")
    if not torch.allclose(quaternion.norm(dim=-1), torch.ones(leading, dtype=quaternion.dtype), atol=1e-3, rtol=0):
        raise ValueError("root_quat_xyzw must contain unit XYZW quaternions")
    return quaternion


def pose_embedding(body_pos, root_quat_xyzw, *, body_names=PACER_BODY_NAMES,
                   keypoint_names=KEYPOINT_NAMES, height_m=None):
    """Return [...,45] embeddings for body_pos [...,24,3], preserving time/batch.

    Default height is the distance from Head to the midpoint of both Ankle body
    positions at each frame. This is positive, translation/yaw/scale invariant,
    but pose-dependent and explicitly a local choice. Supply positive height_m
    (broadcastable to leading dimensions) for a fixed anthropometric height;
    record that alternative scale in the saved cluster embedding_spec.
    """
    positions = _tensor(body_pos, "body_pos")
    names, selected = tuple(body_names), tuple(keypoint_names)
    if positions.ndim < 3 or positions.shape[-2:] != (24, 3) or not positions.numel():
        raise ValueError("body_pos must be nonempty [...,24,3]")
    if len(names) != 24 or len(set(names)) != 24 or set(names) != set(PACER_BODY_NAMES):
        raise ValueError("body_names must give the exact named 24-body SMPL bijection")
    if len(selected) != 15 or len(set(selected)) != 15 or not set(selected).issubset(names):
        raise ValueError("Exactly 15 distinct named SMPL keypoints must be explicitly selected")
    leading = positions.shape[:-2]
    q = _quaternions(root_quat_xyzw, leading)
    if height_m is None:
        ankle_midpoint = .5 * (positions[..., names.index("L_Ankle"), :] + positions[..., names.index("R_Ankle"), :])
        height = (positions[..., names.index("Head"), :] - ankle_midpoint).norm(dim=-1)
    else:
        height = _tensor(height_m, "height_m")
        try:
            height = torch.broadcast_to(height, leading)
        except RuntimeError:
            raise ValueError("height_m must broadcast to pose leading dimensions") from None
    if (height <= 1e-8).any():
        raise ValueError("Height normalization requires strictly positive nondegenerate height")
    relative = positions[..., [names.index(name) for name in selected], :] - positions[..., names.index("Pelvis"), None, :]
    yaw = torch.atan2(2 * (q[..., 3] * q[..., 2] + q[..., 0] * q[..., 1]),
                      1 - 2 * (q[..., 1].square() + q[..., 2].square()))
    cosine, sine = yaw.cos()[..., None], yaw.sin()[..., None]
    local = torch.stack((cosine * relative[..., 0] + sine * relative[..., 1],
                         -sine * relative[..., 0] + cosine * relative[..., 1], relative[..., 2]), -1)
    return (local / height[..., None, None]).flatten(-2)


def _features(value, name):
    features = _tensor(value, name)
    if features.ndim != 2 or not features.shape[0] or not features.shape[1]:
        raise ValueError(f"{name} requires a nonempty [N,D] matrix")
    return features


def _squared_distance(left, right):
    return (left.square().sum(1)[:, None] + right.square().sum(1)[None] - 2 * left @ right.T).clamp_min(0)


def _assign(features, centers, chunk_size=4096):
    assignments, distances = [], []
    for block in features.split(chunk_size):
        distance, index = _squared_distance(block, centers).min(dim=1)
        assignments.append(index)
        distances.append(distance)
    return torch.cat(assignments), torch.cat(distances)


@dataclass(frozen=True)
class PoseClusterModel:
    centers: torch.Tensor
    seed: int
    training_source: str
    training_sample_count: int
    embedding_spec: dict
    iterations: int
    training_split: str = "train"

    @classmethod
    def fit(cls, train_features, *, k, split, source_id, seed=7, max_iterations=200,
            tolerance=1e-7, embedding_spec=None):
        """Deterministic CPU kmeans++; fit is permitted only on declared train data.

        K=50/100 are the intended reported models. Other K>=2 support diagnostic
        fixtures. Held-out and generated samples can only be assigned via score.
        source_id should be the exact training manifest/package hash, not an
        unverified claim that a dataset contains the author's 2,848 clips.
        """
        if split != "train":
            raise ValueError("Cluster fitting requires split='train'; held-out/generated fitting is forbidden")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ValueError("An explicit training source_id is required")
        if isinstance(k, bool) or not isinstance(k, int) or k < 2 or max_iterations < 1:
            raise ValueError("K>=2 and positive maximum iterations are required")
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Kmeans tolerance must be finite and nonnegative")
        features = _features(train_features, "train_features")
        if len(features) < k or torch.unique(features, dim=0).shape[0] < k:
            raise ValueError("At least K distinct training reference poses are required")
        generator = torch.Generator().manual_seed(seed)
        first = int(torch.randint(len(features), (1,), generator=generator))
        centers = [features[first].clone()]
        nearest = _squared_distance(features, centers[0][None]).flatten()
        for _ in range(1, k):
            index = int(torch.multinomial(nearest, 1, generator=generator))
            centers.append(features[index].clone())
            nearest = torch.minimum(nearest, _squared_distance(features, centers[-1][None]).flatten())
        centers = torch.stack(centers)
        for iteration in range(1, max_iterations + 1):
            labels, distances = _assign(features, centers)
            counts = torch.bincount(labels, minlength=k)
            updated = torch.zeros_like(centers)
            updated.index_add_(0, labels, features)
            populated = counts > 0
            updated[populated] /= counts[populated, None]
            # Empty centers move to distinct worst-represented real train poses.
            empty = (~populated).nonzero().flatten()
            if len(empty):
                farthest = torch.argsort(distances, descending=True)[:len(empty)]
                updated[empty] = features[farthest]
            movement = (updated - centers).norm(dim=1).max()
            centers = updated
            if movement <= tolerance:
                break
        labels, _ = _assign(features, centers)
        if torch.unique(labels).numel() != k:
            raise ValueError("Kmeans retained empty reference clusters; choose more diverse training samples")
        spec = dict(embedding_spec) if embedding_spec is not None else asdict(PoseEmbeddingSpec())
        if spec.get("feature_width") != features.shape[1]:
            raise ValueError("Embedding specification width differs from training features")
        return cls(centers, seed, source_id, len(features), spec, iteration)

    def score(self, features, *, split="generated"):
        if split not in {"train", "validation", "test", "generated"}:
            raise ValueError("Declare train, validation, test or generated scoring split")
        values = _features(features, "features")
        if values.shape[1] != self.centers.shape[1]:
            raise ValueError("Scoring embedding width differs from the fixed training cluster model")
        labels, distances = _assign(values, self.centers)
        k = len(self.centers)
        counts = torch.bincount(labels, minlength=k)
        probability = counts.to(torch.float64) / len(values)
        positive = probability[probability > 0]
        return {"k": k, "split": split, "sample_count": len(values),
                "covered_clusters": int((counts > 0).sum()), "coverage": float((counts > 0).double().mean()),
                "normalized_pose_entropy": float(-(positive * positive.log()).sum() / math.log(k)),
                "cluster_counts": counts.tolist(),
                "mean_assigned_center_distance": float(distances.sqrt().mean()),
                "mean_assigned_center_squared_distance": float(distances.mean()),
                "cluster_fit_split": self.training_split, "cluster_training_source": self.training_source,
                "author_parameters": False}

    def save(self, path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        document = {"schema": "navisaaclab.pose_cluster.v1", "centers": self.centers.tolist(),
                    "seed": self.seed, "training_source": self.training_source,
                    "training_sample_count": self.training_sample_count, "embedding_spec": self.embedding_spec,
                    "iterations": self.iterations, "training_split": self.training_split}
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, destination)

    @classmethod
    def load(cls, path):
        document = json.loads(Path(path).read_text())
        expected = {"schema", "centers", "seed", "training_source", "training_sample_count", "embedding_spec", "iterations", "training_split"}
        if set(document) != expected or document["schema"] != "navisaaclab.pose_cluster.v1" or document["training_split"] != "train":
            raise ValueError("Invalid fixed train cluster artifact")
        centers = _features(document["centers"], "centers")
        if len(centers) < 2 or document["embedding_spec"].get("feature_width") != centers.shape[1]:
            raise ValueError("Invalid cluster count/embedding width")
        if not isinstance(document["training_source"], str) or not document["training_source"].strip():
            raise ValueError("Cluster artifact lacks training source")
        if document["training_sample_count"] < len(centers) or document["iterations"] < 1:
            raise ValueError("Cluster artifact lacks valid fit metadata")
        return cls(centers, document["seed"], document["training_source"], document["training_sample_count"],
                   document["embedding_spec"], document["iterations"])


def fit_pose_clusters(train_features, *, split, source_id, ks=(50, 100), **kwargs):
    """Build the two intended fixed reference models from a single train split."""
    return {k: PoseClusterModel.fit(train_features, k=k, split=split, source_id=source_id, **kwargs) for k in ks}


def gaussian_mmd(generated, reference, *, bandwidth=1.0, chunk_size=512):
    """Actual Gaussian-kernel biased MMD² and its nonnegative square root.

    k(x,y)=exp(-||x-y||²/(2*sigma²)); sigma is fixed by the caller before held-out
    scoring. No median heuristic, fitted covariance, cluster distance proxy or
    bandwidth search on validation/generated data is performed. All supplied
    samples are used; record any upstream subsampling seed/IDs independently.
    """
    x, y = _features(generated, "generated"), _features(reference, "reference")
    if x.shape[1] != y.shape[1]:
        raise ValueError("MMD embedding widths differ")
    if not math.isfinite(bandwidth) or bandwidth <= 0 or chunk_size < 1:
        raise ValueError("A positive finite fixed Gaussian bandwidth and chunk size are required")

    def kernel_mean(left, right):
        total = torch.zeros((), dtype=torch.float64)
        for a in left.split(chunk_size):
            for b in right.split(chunk_size):
                total += torch.exp(-_squared_distance(a, b) / (2 * bandwidth ** 2)).sum()
        return total / (len(left) * len(right))

    raw = kernel_mean(x, x) + kernel_mean(y, y) - 2 * kernel_mean(x, y)
    squared = float(raw.clamp_min(0))  # A biased Gaussian MMD is >=0; remove roundoff only.
    return {"mmd_squared": squared, "mmd": math.sqrt(squared), "raw_mmd_squared": float(raw),
            "kernel": "exp(-squared_euclidean_distance/(2*bandwidth^2))", "bandwidth": bandwidth,
            "estimator": "biased V-statistic including diagonal terms", "generated_samples": len(x),
            "reference_samples": len(y), "feature_width": x.shape[1], "author_parameters": False}


def planar_agent_overlap(root_pos, *, radius_m=.3):
    """[T,B] XY disc-overlap proxy in one shared world; not PhysX contacts.

    Call only for pedestrians sharing the same physical scene. B can also index
    isolated parallel worlds; this function cannot infer that distinction.
    """
    positions = _tensor(root_pos, "root_pos")
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("Agent overlap requires root_pos [T,B,3]")
    radius = _tensor(radius_m, "radius_m")
    try:
        radius = torch.broadcast_to(radius, (positions.shape[1],))
    except RuntimeError:
        raise ValueError("radius_m must be scalar or one radius per agent") from None
    if (radius <= 0).any():
        raise ValueError("Agent disc radii must be positive")
    distance = (positions[:, :, None, :2] - positions[:, None, :, :2]).norm(dim=-1)
    overlap = distance < radius[:, None] + radius[None, :]
    overlap[:, torch.arange(positions.shape[1]), torch.arange(positions.shape[1])] = False
    return overlap.any(-1)


def occupancy_collision_mask(root_pos, occupied_grid, *, origin_xy, resolution_m, outside_is_collision=True):
    """Root-point grid occupancy proxy; rows increase world Y and columns world X.

    occupied_grid is an explicit bool [height,width], not a grayscale image with
    implicit free/occupied/unknown conventions. The input grid origin is its
    bottom-left cell corner. No body geometry or footprint clearance is modeled.
    """
    positions = _tensor(root_pos, "root_pos")
    grid = torch.as_tensor(occupied_grid)
    if grid.device.type != "cpu" or grid.dtype != torch.bool or grid.ndim != 2 or not grid.numel():
        raise ValueError("occupied_grid must be a nonempty boolean CPU [height,width] grid")
    origin = _tensor(origin_xy, "origin_xy")
    if positions.shape[-1] != 3 or origin.shape != (2,) or not math.isfinite(resolution_m) or resolution_m <= 0:
        raise ValueError("Root XYZ, origin XY and positive finite grid resolution are required")
    cell = torch.floor((positions[..., :2] - origin) / resolution_m).long()
    x, y = cell.unbind(-1)
    inside = (x >= 0) & (x < grid.shape[1]) & (y >= 0) & (y < grid.shape[0])
    result = torch.full_like(inside, outside_is_collision)
    result[inside] = grid[y[inside], x[inside]]
    return result


def _bool_mask(value, shape, name):
    tensor = torch.as_tensor(value)
    if tensor.device.type != "cpu" or tensor.shape != shape or tensor.dtype != torch.bool:
        raise ValueError(f"{name} requires boolean CPU shape {tuple(shape)}")
    return tensor


def _flag_summary(flags, valid, boundaries):
    observed = flags & valid
    previous = torch.cat((torch.zeros_like(observed[:1]), observed[:-1]), 0)
    new_segment = torch.cat((torch.ones_like(boundaries[:1]), boundaries), 0)
    events = observed & (~previous | new_segment)
    segment_ids = torch.cat((torch.zeros_like(boundaries[:1], dtype=torch.long), boundaries.long().cumsum(0)), 0)
    episode_count = affected = 0
    for agent in range(valid.shape[1]):
        for segment in segment_ids[:, agent][valid[:, agent]].unique():
            indices = segment_ids[:, agent] == segment
            episode_count += 1
            affected += int(observed[indices, agent].any())
    return {"observed_flagged_frames": int(observed.sum()), "observed_frames": int(valid.sum()),
            "flagged_frame_fraction": float(observed.sum() / valid.sum()),
            "observed_event_starts": int(events.sum()), "observed_episodes": episode_count,
            "affected_observed_episodes": affected, "affected_episode_fraction": affected / episode_count}


def pairwise_contact_metrics(*, shape, valid, boundaries, static_contact=None, peer_contact=None,
                             static_contact_max_force_n=None, peer_contact_max_force_n=None,
                             contact_contract=None):
    """Recorded normal contacts; actor frames and unordered pair frames differ.

    Forces are maximum body-pair magnitudes across physics substeps. Episode
    changes of either participant break pair incidences. Missing evidence stays
    unknown; no occupancy or distance proxy is substituted. Directed views are
    combined with OR/max, while their disagreement remains visible.
    """
    values = (static_contact, peer_contact, static_contact_max_force_n, peer_contact_max_force_n)
    if all(value is None for value in values):
        return None
    if not isinstance(contact_contract, dict):
        raise ValueError("Recorded contacts require an explicit contact contract")
    threshold = contact_contract.get("threshold_n")
    if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold) or threshold <= 0
            or contact_contract.get("units") != "N"
            or contact_contract.get("components") != "normal forces only"
            or contact_contract.get("source") != "PhysX exact-path rigid_contact_view.get_contact_force_matrix"
            or contact_contract.get("substep_reduction") != "maximum body-pair magnitude over every physics substep; strict > threshold"
            or not isinstance(contact_contract.get("excluded"), list)
            or not all(isinstance(value, str) for value in contact_contract["excluded"])
            or not {"floor", "self articulation"} <= set(contact_contract["excluded"])
            or contact_contract.get("author_metric_equivalence", False) is not False):
        raise ValueError("Contact contract must declare normal N, strict threshold, all-substep maxima, and floor/self exclusions")
    substeps = contact_contract.get("physics_substeps_per_control")
    if substeps is not None and (type(substeps) is not int or substeps != 4):
        raise ValueError("This evaluation requires explicitly declared four physics substeps per control")
    for name, expected_dt in (("physics_dt_s", 1/120), ("control_dt_s", 1/30)):
        duration = contact_contract.get(name)
        if duration is not None and (isinstance(duration, bool) or not isinstance(duration, (int, float))
                or not math.isfinite(duration) or not math.isclose(duration, expected_dt, rel_tol=0, abs_tol=1e-9)):
            raise ValueError("Contact timestamps must match the declared 120Hz physics/30Hz capture protocol")

    def measured(mask, force, expected, name):
        if mask is None:
            if force is not None:
                raise ValueError(f"{name} force lacks its recorded boolean contact mask")
            return None, None
        flags = _bool_mask(mask, expected, name)
        if force is None:
            return flags, None
        forces = torch.as_tensor(force)
        if (forces.device.type != "cpu" or forces.shape != expected or not forces.is_floating_point()
                or not torch.isfinite(forces).all() or bool((forces < 0).any())):
            raise ValueError(f"{name} force must be finite nonnegative floating point CPU {tuple(expected)}")
        if not torch.equal(flags, forces > threshold):
            raise ValueError(f"{name} mask disagrees with strict force > threshold")
        return flags, forces

    def summarized(flags, available, changed):
        if not bool(available.any()):
            return {"observed_flagged_frames": 0, "observed_frames": 0, "flagged_frame_fraction": None,
                    "observed_event_starts": 0, "observed_episodes": 0,
                    "affected_observed_episodes": 0, "affected_episode_fraction": None}
        return _flag_summary(flags, available, changed)

    static, static_force = measured(static_contact, static_contact_max_force_n, shape, "static_contact")
    peer_shape = (*shape, shape[1])
    peer, peer_force = measured(peer_contact, peer_contact_max_force_n, peer_shape, "peer_contact")
    result = {"measurement_contract": contact_contract, "static": None, "peer": None,
              "author_metric_equivalence": False,
              "aggregation": "local recorded actor/pair frame and contiguous-incidence counts; author aggregate unknown"}
    if static is not None:
        statistics = summarized(static, valid, boundaries)
        result["static"] = {"actor_frame_statistics": statistics,
            "contact_actor_frames_per_agent": (static & valid).sum(0).tolist(),
            "maximum_body_pair_force_n": float(static_force[valid].max()) if static_force is not None else None,
            "definition": "Any body of an actor contacting a declared static collider; floor and self excluded; strict normal-force threshold"}
    if peer is not None:
        actors = torch.arange(shape[1])
        if bool(peer[:, actors, actors].any()) or (peer_force is not None and bool((peer_force[:, actors, actors] != 0).any())):
            raise ValueError("Peer contact diagonal must be zero: self contacts are excluded")
        directed_disagreements = (peer != peer.transpose(1, 2)) & valid[:, :, None] & valid[:, None, :]
        union = peer | peer.transpose(1, 2)
        union &= valid[:, :, None] & valid[:, None, :]
        actor_flags = union.any(-1)
        left, right = torch.triu_indices(shape[1], shape[1], offset=1)
        pair_flags = union[:, left, right]
        pair_valid = valid[:, left] & valid[:, right]
        pair_boundaries = boundaries[:, left] | boundaries[:, right]
        pair_summary = summarized(pair_flags, pair_valid, pair_boundaries)
        pair_force = None if peer_force is None else torch.maximum(peer_force, peer_force.transpose(1, 2))[:, left, right]
        maximum = float(pair_force[pair_valid].max()) if pair_force is not None and bool(pair_valid.any()) else None
        result["peer"] = {
            "actor_frame_statistics": summarized(actor_flags, valid, boundaries),
            "contact_actor_frames_per_agent": (actor_flags & valid).sum(0).tolist(),
            "unordered_pair_statistics": {
                "observed_contact_pair_frames": pair_summary["observed_flagged_frames"],
                "observed_pair_frames": pair_summary["observed_frames"],
                "contact_pair_frame_fraction": pair_summary["flagged_frame_fraction"],
                "contiguous_pair_contact_incidences": pair_summary["observed_event_starts"],
                "observed_pair_episode_segments": pair_summary["observed_episodes"],
                "affected_pair_episode_segments": pair_summary["affected_observed_episodes"],
                "affected_pair_episode_segment_fraction": pair_summary["affected_episode_fraction"]},
            "unordered_actor_pairs": list(map(list, zip(left.tolist(), right.tolist()))),
            "contact_pair_frames_per_pair": (pair_flags & pair_valid).sum(0).tolist(),
            "maximum_body_pair_force_n": maximum,
            "directed_boolean_disagreement_entries": int(directed_disagreements.sum()),
            "definition": "Unordered actor pair contacts if either exact-path directional view exceeds threshold; no sum of per-body forces",
            "pair_segment_boundary": "Episode/reset boundary of either actor or a gap in valid observations breaks contiguous contact incidences"}
    return result


def rollout_metrics(root_pos, root_quat_xyzw, *, target_xy=None, episode_ids=None, reset_mask=None,
                    reset_mask_marks="before_frame", valid_mask=None, dt=1 / 30, ground_height=0.,
                    fall_height_m=.45, fall_up_z=.1, map_collision_mask=None, agent_collision_mask=None,
                    map_collision_definition="provided boolean collision mask",
                    agent_collision_definition="provided boolean collision mask",
                    static_contact=None, peer_contact=None, static_contact_max_force_n=None,
                    peer_contact_max_force_n=None, contact_contract=None):
    """Time-aligned rollout metrics without counting reset teleportation as travel.

    reset_mask_marks='before_frame': reset_mask[t] starts a new episode at t.
    'after_frame': reset_mask[t] ends the episode after recorded state t, so the
    jump t->t+1 is excluded. episode_ids is an independent second boundary source.
    Collision masks must be supplied; absent evidence remains None, not zero.
    target_xy[t] must correspond to the physical root at the same recorded time.
    """
    positions = _tensor(root_pos, "root_pos")
    if positions.ndim != 3 or positions.shape[-1] != 3 or positions.shape[0] < 2 or not positions.shape[1]:
        raise ValueError("Rollout root_pos must be [T>=2,B>=1,3]")
    shape = positions.shape[:2]
    q = _quaternions(root_quat_xyzw, shape)
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(fall_height_m) or not math.isfinite(fall_up_z):
        raise ValueError("Finite thresholds and positive finite dt are required")
    if reset_mask_marks not in {"before_frame", "after_frame"}:
        raise ValueError("Declare reset_mask_marks as before_frame or after_frame")
    valid = torch.ones(shape, dtype=torch.bool) if valid_mask is None else _bool_mask(valid_mask, shape, "valid_mask")
    if not valid.any():
        raise ValueError("At least one valid observed frame is required")
    boundaries = torch.zeros((shape[0] - 1, shape[1]), dtype=torch.bool)
    id_boundaries = reset_boundaries = None
    if episode_ids is not None:
        ids = torch.as_tensor(episode_ids)
        if ids.device.type != "cpu" or ids.shape != shape or ids.dtype not in {torch.int32, torch.int64}:
            raise ValueError("episode_ids requires integer CPU [T,B]")
        id_boundaries = ids[1:] != ids[:-1]
        boundaries |= id_boundaries
    if reset_mask is not None:
        resets = _bool_mask(reset_mask, shape, "reset_mask")
        reset_boundaries = resets[1:] if reset_mask_marks == "before_frame" else resets[:-1]
        boundaries |= reset_boundaries
    transitions = valid[1:] & valid[:-1] & ~boundaries
    distance = (positions[1:, :, :2] - positions[:-1, :, :2]).norm(dim=-1)
    walked = (distance * transitions).sum(0)
    ground = _tensor(ground_height, "ground_height")
    try:
        ground = torch.broadcast_to(ground, shape)
    except RuntimeError:
        raise ValueError("ground_height must broadcast to [T,B]") from None
    up_z = 1 - 2 * q[..., :2].square().sum(-1)
    falls = (positions[..., 2] - ground < fall_height_m) | (up_z < fall_up_z)
    report = {"recorded_frames": shape[0], "agents": shape[1], "dt_s": dt,
              "observed_sample_span_s": (shape[0] - 1) * dt,
              "valid_transition_count_per_agent": transitions.sum(0).tolist(),
              "excluded_episode_boundary_transitions": int(boundaries.sum()),
              "episode_id_boundary_count": int(id_boundaries.sum()) if id_boundaries is not None else None,
              "reset_mask_boundary_count": int(reset_boundaries.sum()) if reset_boundaries is not None else None,
              "boundary_source_disagreement_count": int((id_boundaries != reset_boundaries).sum())
              if id_boundaries is not None and reset_boundaries is not None else None,
              "walked_distance_xy_m_per_agent": walked.tolist(), "walked_distance_xy_m_total": float(walked.sum()),
              "reset_mask_marks": reset_mask_marks, "falls": _flag_summary(falls, valid, boundaries),
              "fall_definition": {"pelvis_ground_height_less_than_m": fall_height_m, "root_up_z_less_than": fall_up_z},
              "minimum_pelvis_ground_height_m": float((positions[..., 2] - ground)[valid].min()),
              "map_collisions": None, "agent_collisions": None, "plan_tracking": None,
              "success_rate": None, "author_parameters": False,
              "notes": "Observed episodes may be censored; no navigation goal/success contract was supplied."}
    for name, mask, definition in (("map_collisions", map_collision_mask, map_collision_definition),
                                   ("agent_collisions", agent_collision_mask, agent_collision_definition)):
        if mask is not None:
            flags = _bool_mask(mask, shape, name)
            report[name] = {**_flag_summary(flags, valid, boundaries), "definition": definition}
    contacts = pairwise_contact_metrics(shape=shape, valid=valid, boundaries=boundaries,
        static_contact=static_contact, peer_contact=peer_contact,
        static_contact_max_force_n=static_contact_max_force_n, peer_contact_max_force_n=peer_contact_max_force_n,
        contact_contract=contact_contract)
    report["actual_contacts"] = contacts
    if contacts is not None:
        for name, kind in (("map_collisions", "static"), ("agent_collisions", "peer")):
            measured = contacts[kind]
            if measured is not None and report[name] is None:
                report[name] = {**measured["actor_frame_statistics"], "definition": measured["definition"],
                                "source": "recorded PhysX pairwise normal contact", "author_metric_equivalence": False}
    if target_xy is not None:
        target = _tensor(target_xy, "target_xy")
        if target.shape != (*shape, 2):
            raise ValueError("target_xy must be time-aligned [T,B,2]")
        squared = (positions[..., :2] - target).square().sum(-1)[valid]
        report["plan_tracking"] = {"samples": len(squared), "mean_xy_error_m": float(squared.sqrt().mean()),
                                   "rmse_xy_m": float(squared.mean().sqrt()), "maximum_xy_error_m": float(squared.sqrt().max()),
                                   "alignment": "physical root and explicit target at the same recorded timestamp"}
    return report
