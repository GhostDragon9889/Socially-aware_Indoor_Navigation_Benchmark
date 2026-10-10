"""Explicit, SDK-free expert-window priors for future AMP experiments.

This module does not modify a corpus, physical reset distribution, or trainer.
Sampling changes the expert prior unless the consumer explicitly applies p/q
importance weights. Short-window labels are kinematic proxies, not author IDs.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re
from typing import Mapping

import torch
from torch import Tensor


STRATA = ("low_speed", "fast", "sideways", "turning", "moving")
_SCHEMA = "navisaaclab.expert_window_sampler.v1"
_SEGMENT = re.compile(r"^(.*?)#frames=(\d+):(\d+)$")


def _number(value, name, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, not a boolean")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} is outside its declared range")
    return float(value)


def _integer(value, name, minimum):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class WindowThresholds:
    low_speed_m_s: float = .12
    fast_speed_m_s: float = 1.8
    lateral_fraction: float = .55
    net_turn_angle_rad: float = .2
    quaternion_norm_tolerance: float = 1e-3

    def validate(self):
        for name, value in asdict(self).items():
            _number(value, name, minimum=0)
        if not 0 < self.low_speed_m_s < self.fast_speed_m_s:
            raise ValueError("Speed thresholds require 0 < low < fast")
        if not 0 < self.lateral_fraction <= 1 or not 0 < self.net_turn_angle_rad <= math.pi:
            raise ValueError("Lateral fraction and net-turn threshold are invalid")
        if not 0 < self.quaternion_norm_tolerance <= .01:
            raise ValueError("Quaternion norm tolerance must be in (0,.01]")


@torch.no_grad()
def classify_windows(root_velocity_xy: Tensor, root_quaternion_xyzw: Tensor, *,
                     history_frames=10, future_frames=1, thresholds=None) -> Tensor:
    """Classify legal centers h-1..N-future-1 from real root kinematics.

    Speed/lateral ratios use the median of the complete history. Turning uses
    unwrapped net yaw change over that history, independent of whole-clip yaw
    span. History is exactly ten frames; a future frame is reserved by default.
    """
    if history_frames != 10 or type(history_frames) is not int:
        raise ValueError("This AMP contract requires exactly 10 history frames")
    _integer(future_frames, "future_frames", 1)
    thresholds = WindowThresholds() if thresholds is None else thresholds
    if not isinstance(thresholds, WindowThresholds):
        raise ValueError("thresholds must be WindowThresholds")
    thresholds.validate()
    for value, width in ((root_velocity_xy, 2), (root_quaternion_xyzw, 4)):
        if (not isinstance(value, Tensor) or value.ndim != 2 or value.shape[-1] != width
                or value.device.type != "cpu" or not value.is_floating_point() or not torch.isfinite(value).all()):
            raise ValueError("Root features require finite floating-point CPU tensors [N,2]/[N,4]")
    count = len(root_velocity_xy)
    if len(root_quaternion_xyzw) != count or count < history_frames+future_frames:
        raise ValueError("No legal window contains history10 and the required future frame")
    q = root_quaternion_xyzw
    if not torch.allclose(q.norm(dim=-1), torch.ones(count, dtype=q.dtype),
                          atol=thresholds.quaternion_norm_tolerance, rtol=0):
        raise ValueError("Root quaternion must be unit XYZW")
    angle = torch.atan2(2*(q[:, 3]*q[:, 2]+q[:, 0]*q[:, 1]), 1-2*(q[:, 1].square()+q[:, 2].square()))
    delta = angle[1:]-angle[:-1]
    heading = torch.cat((angle[:1], angle[:1]+torch.atan2(delta.sin(), delta.cos()).cumsum(0)))
    speed = root_velocity_xy.norm(dim=-1)
    lateral = (-heading.sin()*root_velocity_xy[:, 0]+heading.cos()*root_velocity_xy[:, 1]).abs()
    room = count-history_frames-future_frames+1

    def median_history(values):
        ordered = values.unfold(0, history_frames, 1)[:room].sort(-1).values
        return (ordered[:, 4]+ordered[:, 5])/2

    typical_speed = median_history(speed)
    typical_lateral = median_history(lateral/speed.clamp_min(.05))
    yaw_change = (heading[history_frames-1:count-future_frames]-heading[:room]).abs()
    labels = torch.full((room,), STRATA.index("moving"), dtype=torch.long)
    labels[yaw_change >= thresholds.net_turn_angle_rad] = STRATA.index("turning")
    labels[typical_lateral > thresholds.lateral_fraction] = STRATA.index("sideways")
    labels[typical_speed >= thresholds.fast_speed_m_s] = STRATA.index("fast")
    labels[typical_speed < thresholds.low_speed_m_s] = STRATA.index("low_speed")
    return labels


def _origin(record):
    source = record.get("source_path")
    if not isinstance(source, str) or not source:
        raise ValueError("Every expert clip needs an actual source_path")
    match = _SEGMENT.fullmatch(source)
    if "#" in source and match is None:
        raise ValueError("Unknown source segment suffix")
    parent = match[1] if match else source
    if record.get("parent_source_path", parent) != parent:
        raise ValueError("source_path and parent_source_path disagree")
    fields = parent.split("/processed/smpl/")
    parts = fields[1].split("/") if len(fields) == 2 else []
    if len(parts) < 3 or not all(parts[:2]):
        raise ValueError("Source must identify its actual processed/smpl dataset and subject namespace")
    dataset, subject = parts[0], "/".join(parts[:2])
    if record.get("dataset", dataset) != dataset or record.get("subject_namespace", subject) != subject:
        raise ValueError("Declared dataset/subject namespace disagrees with parent source")
    bounds = None
    if match:
        start, stop = int(match[2]), int(match[3])
        for key, expected in (("parent_control_frame_start", start), ("parent_control_frame_stop", stop)):
            if type(record.get(key)) is not int or record[key] != expected:
                raise ValueError("Parent control bounds must equal the source segment suffix")
        if not 0 <= start < stop or stop-start != record.get("frames"):
            raise ValueError("Segment length and parent control bounds disagree")
        bounds = (start, stop)
    elif any(key in record for key in ("parent_control_frame_start", "parent_control_frame_stop")):
        raise ValueError("Parent segment bounds require an explicit #frames source suffix")
    return parent, dataset, subject, bounds


def _capped_distribution(weights, cap):
    """Proportional weights with a strict feasible per-dataset probability cap."""
    if cap is None:
        return weights/weights.sum()
    _number(cap, "dataset_cap", minimum=0, maximum=1)
    if cap == 0 or len(weights)*cap < 1-1e-12:
        raise ValueError(f"Dataset cap{cap} is unsupported by only{len(weights)} datasets")
    result = torch.zeros_like(weights)
    available = torch.ones(len(weights), dtype=torch.bool)
    remaining = 1.
    for _ in range(len(weights)+1):
        active = weights[available]
        if not len(active):
            break
        proposed = remaining*active/active.sum()
        exceeding = proposed > cap+1e-12
        ids = available.nonzero().flatten()
        if not exceeding.any():
            result[ids] = proposed
            break
        fixed = ids[exceeding]
        result[fixed] = cap
        available[fixed] = False
        remaining -= len(fixed)*cap
    if not torch.isclose(result.sum(), torch.tensor(1., dtype=result.dtype), atol=1e-10, rtol=0):
        raise RuntimeError("Dataset probability construction did not normalize")
    return result


@dataclass(frozen=True)
class WindowSample:
    motion_ids: Tensor
    frame_indices: Tensor
    absolute_indices: Tensor
    history_indices: Tensor
    next_indices: Tensor
    probabilities: Tensor
    strata: tuple[str, ...]
    datasets: tuple[str, ...]
    parent_sources: tuple[str, ...]


@dataclass(frozen=True)
class _DatasetBucket:
    dataset: str
    probability: float
    parent_sources: tuple[str, ...]
    parent_windows: tuple[Tensor, ...]


class ExpertWindowSampler:
    """Hierarchy: window stratum -> dataset -> parent -> legal center.

    ``from_corpus`` only reads an existing train AMASSCorpus. All random draws
    use the caller's CPU torch.Generator. The caller must checkpoint state_dict
    explicitly; nothing is installed into the original corpus or reset path.
    """

    @classmethod
    def from_corpus(cls, corpus, *, target_weights: Mapping[str, float], generator,
                    thresholds=None, history_frames=10, future_frames=1,
                    dataset_weighting="sqrt_parent_support", dataset_cap=None,
                    minimum_parents_per_bucket=5, minimum_windows_per_bucket=100):
        sampler = cls.__new__(cls)
        sampler._build(corpus, target_weights=target_weights, generator=generator, thresholds=thresholds,
                       history_frames=history_frames, future_frames=future_frames,
                       dataset_weighting=dataset_weighting, dataset_cap=dataset_cap,
                       minimum_parents_per_bucket=minimum_parents_per_bucket,
                       minimum_windows_per_bucket=minimum_windows_per_bucket)
        return sampler

    def _build(self, corpus, **config):
        generator = config.pop("generator")
        if not isinstance(generator, torch.Generator) or generator.device.type != "cpu":
            raise ValueError("Supply an explicit CPU torch.Generator")
        self.generator = generator
        weights = config.pop("target_weights")
        if not isinstance(weights, Mapping) or set(weights)-set(STRATA):
            raise ValueError(f"target_weights may only contain{STRATA}")
        raw = {key: _number(weights.get(key, 0.), f"target_weights.{key}", minimum=0) for key in STRATA}
        total = sum(raw.values())
        if not math.isfinite(total) or total <= 0:
            raise ValueError("Target weights must have finite positive total mass")
        self.target_probability = torch.tensor([raw[key]/total for key in STRATA], dtype=torch.float64)
        supplied_thresholds = config.pop("thresholds")
        thresholds = WindowThresholds() if supplied_thresholds is None else supplied_thresholds
        if not isinstance(thresholds, WindowThresholds):
            raise ValueError("thresholds must be WindowThresholds")
        thresholds.validate()
        h, future = config["history_frames"], config["future_frames"]
        if type(h) is not int or h != 10:
            raise ValueError("Exactly10 history frames are required")
        _integer(future, "future_frames", 1)
        for name in ("minimum_parents_per_bucket", "minimum_windows_per_bucket"):
            _integer(config[name], name, 1)
        weighting, cap = config["dataset_weighting"], config["dataset_cap"]
        if not isinstance(weighting, str) or weighting not in {"sqrt_parent_support", "uniform"}:
            raise ValueError("Unknown dataset weighting")
        if cap is not None:
            _number(cap, "dataset_cap", minimum=0, maximum=1)
        self.config = {**config, "thresholds": asdict(thresholds), "target_weights": raw,
                       "normalized_target_probabilities": dict(zip(STRATA, self.target_probability.tolist(), strict=True))}
        self.config_sha256 = _digest(self.config)
        if getattr(corpus, "split", None) != "train":
            raise ValueError("Training expert sampler must not consume a held-out split")
        manifest_path = Path(corpus.manifest_path)
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict) or manifest != corpus.manifest or manifest.get("schema") != "navisaaclab.amass_reconstruction.v1":
            raise ValueError("Corpus manifest does not match its actual source file/schema")
        entry = manifest["splits"]["train"]
        if entry != corpus.entry:
            raise ValueError("Corpus train entry does not match its manifest")
        source_hash = entry.get("sha256")
        if not isinstance(source_hash, str) or not re.fullmatch("[0-9a-f]{64}", source_hash):
            raise ValueError("Training package needs an explicit SHA256 declaration")
        if not isinstance(entry.get("path"), str) or not entry["path"]:
            raise ValueError("Training package needs a real local path")
        source_path = Path(entry["path"])
        if not source_path.is_absolute():
            source_path = manifest_path.parent/source_path
        if type(entry.get("bytes")) is not int or source_path.stat().st_size != entry["bytes"]:
            raise ValueError("Training package size disagrees with its manifest")
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        self.source_provenance = {"manifest": str(manifest_path.resolve()), "manifest_sha256": self.manifest_sha256,
                                  "bank_path": str(source_path.resolve()), "bank_bytes_stat": source_path.stat().st_size,
                                  "bank_sha256_declared": source_hash, "bank_sha256_recomputed_here": False}
        heldout = {_origin(row)[0] for name, split in manifest["splits"].items() if name != "train" for row in split["clips"]}
        data = corpus.data
        for key in ("length_starts", "motion_num_frames", "motion_dt"):
            value = data.get(key)
            if not isinstance(value, Tensor) or value.ndim != 1 or value.device.type != "cpu":
                raise ValueError("Bank timing arrays must be one-dimensional CPU tensors")
        for key in ("length_starts", "motion_num_frames"):
            if data[key].dtype not in {torch.int32, torch.int64}:
                raise ValueError("Bank frame bounds must be integer tensors")
        velocity, rotation = data.get("gvs"), data.get("grs")
        if (not isinstance(velocity, Tensor) or velocity.ndim != 3 or velocity.shape[1:] != (24, 3)
                or not isinstance(rotation, Tensor) or rotation.shape != (len(velocity), 24, 4)):
            raise ValueError("Canonical24body world gvs/grs arrays are required")
        ids, names = set(), set()
        intervals = defaultdict(list)
        source_intervals = defaultdict(list)
        bank_intervals = []
        parent_meta = {}
        motions, frames, absolutes, strata, parents, datasets = [], [], [], [], [], []
        groups = defaultdict(list)
        feature_digest = hashlib.sha256()
        cursor = 0
        for row in entry["clips"]:
            index, count = row.get("motion_id"), row.get("frames")
            _integer(index, "motion_id", 0)
            _integer(count, "frames", h+future)
            parent, dataset, subject, bounds = _origin(row)
            if index in ids or row["source_path"] in names:
                raise ValueError("Duplicate motion ID/source in expert manifest")
            if index >= len(data["motion_files"]) or any(index >= len(data[key]) for key in ("length_starts", "motion_num_frames", "motion_dt")):
                raise ValueError("Motion ID is outside actual bank arrays")
            if parent in heldout:
                raise ValueError("Training and held-out parent sources overlap")
            if data["motion_files"][index] != row["source_path"] or int(data["motion_num_frames"][index]) != count:
                raise ValueError("Manifest motion source/count disagrees with actual bank")
            start, dt = int(data["length_starts"][index]), float(data["motion_dt"][index])
            declared_dt = _number(row.get("dt"), "clip.dt", minimum=0)
            if not math.isfinite(dt) or abs(dt-1/30) > 1e-7 or abs(declared_dt-dt) > 1e-7:
                raise ValueError("This sampler requires real30Hz bank timing matching the clip metadata")
            if start < 0 or start+count > len(velocity):
                raise ValueError("Actual clip bounds leave the bank")
            bank_intervals.append((start, start+count))
            if bounds is None:
                bounds = (0, count)
            intervals[parent].append(bounds)
            source_keys = ("parent_source_frame_start", "parent_source_frame_stop")
            if any(key in row for key in source_keys):
                low = _integer(row.get(source_keys[0]), source_keys[0], 0)
                high = _integer(row.get(source_keys[1]), source_keys[1], 1)
                if low >= high:
                    raise ValueError("Original source frame bounds must be a positive half-open interval")
                if "parent_source_frames" in row and high > _integer(row["parent_source_frames"], "parent_source_frames", 1):
                    raise ValueError("Original source bounds leave the parent")
                source_intervals[parent].append((low, high))
            meta = (dataset, subject, row.get("original_motion_id"))
            if "original_motion_id" in row:
                _integer(row["original_motion_id"], "original_motion_id", 0)
            if parent in parent_meta and parent_meta[parent] != meta:
                raise ValueError("One parent has conflicting dataset/subject/original-motion metadata")
            parent_meta[parent] = meta
            clip_v = velocity[start:start+count, 0, :2]
            clip_q = rotation[start:start+count, 0]
            labels = classify_windows(clip_v, clip_q, history_frames=h, future_frames=future, thresholds=thresholds)
            feature_digest.update(clip_v.detach().contiguous().numpy().tobytes())
            feature_digest.update(clip_q.detach().contiguous().numpy().tobytes())
            centers = torch.arange(h-1, count-future, dtype=torch.long)
            motions.append(torch.full_like(centers, index))
            frames.append(centers)
            absolutes.append(centers+start)
            strata.append(labels)
            parents.extend([parent]*len(centers))
            datasets.extend([dataset]*len(centers))
            for label in labels.unique().tolist():
                selected = (labels == label).nonzero().flatten()+cursor
                groups[(label, dataset, parent)].append(selected)
            cursor += len(centers)
            ids.add(index)
            names.add(row["source_path"])
        if not motions:
            raise ValueError("No actual training clips were declared")
        for parent, bounds in list(intervals.items())+list(source_intervals.items()):
            ordered = sorted(bounds)
            if any(left[1] > right[0] for left, right in zip(ordered, ordered[1:])):
                raise ValueError(f"Overlapping parent frame intervals:{parent}")
        ordered_bank = sorted(bank_intervals)
        if any(left[1] > right[0] for left, right in zip(ordered_bank, ordered_bank[1:])):
            raise ValueError("Selected motions overlap in actual bank frame storage")
        self.motion_ids, self.frame_indices = torch.cat(motions), torch.cat(frames)
        self.absolute_indices, self.stratum_ids = torch.cat(absolutes), torch.cat(strata)
        self.parents, self.datasets = tuple(parents), tuple(datasets)
        self.groups = {}
        self.expected_probabilities = torch.zeros(cursor, dtype=torch.float64)
        self.support = {}
        for label, name in enumerate(STRATA):
            by_dataset = defaultdict(list)
            for (key_label, key_dataset, parent), chunks in sorted(groups.items()):
                if key_label == label:
                    by_dataset[key_dataset].append((parent, torch.cat(chunks)))
            num_parents = sum(len(items) for items in by_dataset.values())
            num_windows = int((self.stratum_ids == label).sum())
            self.support[name] = {"parents": num_parents, "windows": num_windows,
                                  "datasets": {d: {"parents": len(items), "windows": sum(len(indices) for _, indices in items)} for d, items in sorted(by_dataset.items())}}
            if self.target_probability[label] == 0:
                continue
            if num_parents < config["minimum_parents_per_bucket"] or num_windows < config["minimum_windows_per_bucket"]:
                raise ValueError(f"Unsupported positive-weight bucket{name}:parents={num_parents},windows={num_windows}")
            ordered_datasets = sorted(by_dataset)
            dataset_weights = torch.tensor([len(by_dataset[d]) for d in ordered_datasets], dtype=torch.float64)
            dataset_weights = dataset_weights.sqrt() if weighting == "sqrt_parent_support" else torch.ones_like(dataset_weights)
            probabilities = _capped_distribution(dataset_weights, cap)
            buckets = []
            for dataset, probability in zip(ordered_datasets, probabilities.tolist(), strict=True):
                items = by_dataset[dataset]
                bucket = _DatasetBucket(dataset, probability, tuple(parent for parent, _ in items), tuple(indices for _, indices in items))
                buckets.append(bucket)
                for indices in bucket.parent_windows:
                    self.expected_probabilities[indices] = self.target_probability[label]*probability/len(items)/len(indices)
            self.groups[label] = tuple(buckets)
        if not torch.isclose(self.expected_probabilities.sum(), torch.tensor(1., dtype=torch.float64), atol=1e-10, rtol=0):
            raise RuntimeError("Hierarchy must give normalized per-center probabilities")
        registry = hashlib.sha256()
        registry.update(self.config_sha256.encode())
        registry.update(self.manifest_sha256.encode())
        for tensor in (self.motion_ids, self.frame_indices, self.absolute_indices, self.stratum_ids):
            registry.update(tensor.numpy().tobytes())
        registry.update(feature_digest.digest())
        self.registry_sha256 = registry.hexdigest()
        self.source_provenance["selected_root_feature_sha256"] = feature_digest.hexdigest()
        self.sampled_windows, self.sampled_batches = 0, 0

    def sample(self, batch) -> WindowSample:
        _integer(batch, "batch", 1)
        labels = torch.multinomial(self.target_probability, batch, replacement=True, generator=self.generator)
        selected = torch.empty(batch, dtype=torch.long)
        for label in labels.unique().tolist():
            positions = (labels == label).nonzero().flatten()
            buckets = self.groups[label]
            dataset_choices = torch.multinomial(torch.tensor([bucket.probability for bucket in buckets], dtype=torch.float64), len(positions), replacement=True, generator=self.generator)
            for dataset_index in dataset_choices.unique().tolist():
                destinations = positions[dataset_choices == dataset_index]
                bucket = buckets[dataset_index]
                parent_choices = torch.randint(len(bucket.parent_sources), (len(destinations),), generator=self.generator)
                for parent_index in parent_choices.unique().tolist():
                    target = destinations[parent_choices == parent_index]
                    windows = bucket.parent_windows[parent_index]
                    draws = torch.randint(len(windows), (len(target),), generator=self.generator)
                    selected[target] = windows[draws]
        absolute = self.absolute_indices[selected]
        self.sampled_windows += batch
        self.sampled_batches += 1
        return WindowSample(self.motion_ids[selected], self.frame_indices[selected], absolute,
                            absolute[:, None]-torch.arange(10), absolute+1,
                            self.expected_probabilities[selected], tuple(STRATA[i] for i in labels.tolist()),
                            tuple(self.datasets[i] for i in selected.tolist()), tuple(self.parents[i] for i in selected.tolist()))

    def describe(self):
        dataset_mass, parent_mass = defaultdict(float), defaultdict(float)
        for label, buckets in self.groups.items():
            for bucket in buckets:
                mass = float(self.target_probability[label])*bucket.probability
                dataset_mass[bucket.dataset] += mass
                for parent in bucket.parent_sources:
                    parent_mass[parent] += mass/len(bucket.parent_sources)
        return {"schema": _SCHEMA, "config": deepcopy(self.config), "config_sha256": self.config_sha256,
                "registry_sha256": self.registry_sha256, "source": deepcopy(self.source_provenance),
                "legal_centers": len(self.motion_ids), "support": deepcopy(self.support),
                "expected_dataset_probability": dict(sorted(dataset_mass.items())),
                "expected_parent_probability": dict(sorted(parent_mass.items())),
                "expected_center_probability_sum": float(self.expected_probabilities.sum()),
                "maximum_center_probability": float(self.expected_probabilities.max()),
                "center_distribution_ess": float(1/self.expected_probabilities.square().sum()),
                "sampled_windows": self.sampled_windows, "sampled_batches": self.sampled_batches,
                "rng": "caller-supplied CPU torch.Generator; state_dict must be explicitly checkpointed",
                "hierarchy": "short-window stratum -> declared dataset prior -> uniform parent -> uniform legal center",
                "purpose": "expert examples only; no reset/controller/corpus integration",
                "author_hyperparameters": False, "weights_change_expert_prior": True,
                "bank_hash_limit": "Full bank SHA is declared, not recomputed here; selected classification root tensors are actually hashed"}

    def state_dict(self):
        return {"schema": _SCHEMA, "config_sha256": self.config_sha256,
                "registry_sha256": self.registry_sha256, "manifest_sha256": self.manifest_sha256,
                "generator_state": self.generator.get_state().clone(),
                "sampled_windows": self.sampled_windows, "sampled_batches": self.sampled_batches}

    def load_state_dict(self, state):
        expected = set(self.state_dict())
        if not isinstance(state, dict) or set(state) != expected:
            raise ValueError("Sampler resume needs the complete exact state schema")
        for key in ("schema", "config_sha256", "registry_sha256", "manifest_sha256"):
            if state[key] != self.state_dict()[key]:
                raise ValueError(f"Sampler resume mismatched{key}")
        for key in ("sampled_windows", "sampled_batches"):
            _integer(state[key], key, 0)
        rng = state["generator_state"]
        if not isinstance(rng, Tensor) or rng.device.type != "cpu" or rng.dtype != torch.uint8 or rng.shape != self.generator.get_state().shape:
            raise ValueError("Sampler resume needs a valid CPU torch.Generator state")
        validation = torch.Generator(device="cpu")
        try:
            validation.set_state(rng)
        except RuntimeError as exc:
            raise ValueError("Sampler generator state is invalid") from exc
        self.generator.set_state(rng)
        self.sampled_windows, self.sampled_batches = state["sampled_windows"], state["sampled_batches"]
