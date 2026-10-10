"""Explicit fresh-experiment protection of public task terrain moments.

The first20 trajectory coordinates and the shared task count keep the existing
RunningNormalizer update. Only task[20:1044] mean/variance are protected.
There is no external statistics-file dependency when resuming or inferring.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import math

import torch

SCHEMA = "navisaaclab.task_terrain_statistics_policy.v1"
MODES = ("legacy", "preserve_declared_suffix")
PUBLIC_PACER_SHA256 = "bc0cb89b4d7ceaeb198339c34f3d736eb7449e7e479a7ee0f80d32563737d615"
PUBLIC_REFERENCE_COUNT = 61931520001.0
TASK_SLICE = [20, 1044]
STATE_KEYS = {"schema", "mode", "task_slice", "reference_initialization", "reference_count",
              "warm_start_task_count", "post_update_calls", "suffix_mean", "suffix_variance",
              "reference_statistics_sha256"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _counter(value, name):
    _require(type(value) is int and 0 <= value <= 2**53, f"{name} must be an exact nonnegative integer")
    return value


def _positive_float(value, name):
    _require(type(value) is float and math.isfinite(value) and value > 0, f"{name} must be a positive finite float")
    return value


def _task_state(normalizer):
    if isinstance(normalizer, dict):
        values = normalizer
    else:
        _require(getattr(normalizer, "clip", None) == 5., "Protected task statistics require clip5")
        values = normalizer.state_dict()
    _require(isinstance(values, dict) and set(values) == {"mean", "variance", "count"}, "Unknown task normalizer state")
    for name, shape in (("mean", (1044,)), ("variance", (1044,)), ("count", ())):
        value = values[name]
        _require(isinstance(value, torch.Tensor) and value.shape == shape and value.dtype == torch.float64
                 and value.layout == torch.strided and torch.isfinite(value).all(), f"Invalid task statistics {name}")
    _require(len({value.device for value in values.values()}) == 1, "Task statistics must share a device")
    _require((values["variance"] >= 0).all() and values["count"] > 0, "Invalid task variance/count")
    return values


def _initialization_record(initialization):
    _require(isinstance(initialization, dict), "Protected statistics require declared public initialization")
    _require(initialization.get("mode") == "transfer_warm_start_public_pacer_actor_critic"
             and initialization.get("actor_critic_task_transferred") is True
             and initialization.get("observation_value_moments_transferred") is True
             and initialization.get("sha256") == PUBLIC_PACER_SHA256,
             "Protected statistics require the verified complete public PACER warm start")
    path = initialization.get("path")
    _require(isinstance(path, str) and bool(path.strip()) and "://" not in path, "Invalid public initialization path provenance")
    return {key: initialization[key] for key in ("mode", "path", "sha256")}


def _statistics_sha256(mean, variance):
    digest = hashlib.sha256()
    for label, tensor in ((b"mean:f64le:1024\0", mean), (b"variance:f64le:1024\0", variance)):
        digest.update(label)
        digest.update(tensor.detach().cpu().contiguous().numpy().astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


def _validate_count_provenance(initialization, reference_count, warm_start_task_count):
    cap = initialization.get("normalizer_count_cap")
    _require(cap is None or (type(cap) in (int, float) and math.isfinite(cap) and cap > 0), "Invalid warm-start count cap provenance")
    expected_count = reference_count if cap is None else min(reference_count, float(cap))
    _require(warm_start_task_count == expected_count, "Warm-start count cap/provenance disagree")
    if cap is not None:
        counts = initialization.get("normalizer_counts_before_cap")
        _require(isinstance(counts, dict) and type(counts.get("task_normalizer")) is float
                 and counts["task_normalizer"] == reference_count, "Original public task count provenance changed")


class TerrainStatisticsPolicy:
    """No RNG/optimizer/replay; legacy performs no writes or new state saves."""

    def __init__(self, mode="legacy"):
        _require(type(mode) is str and mode in MODES, "Unknown task_terrain_statistics_mode")
        self.mode = mode
        self.bound = False
        self.post_update_calls = 0
        self._reference_initialization = None
        self._reference_count = None
        self._warm_start_task_count = None
        self._suffix_mean = self._suffix_variance = None
        self._reference_statistics_sha256 = None

    def require_ready(self):
        _require(self.mode == "legacy" or self.bound, "Protected task statistics are unbound; fresh public warm start is required")

    def require_fresh_warm_start(self, update_count):
        if self.mode != "legacy":
            _require(_counter(update_count, "update_count") == 0 and not self.bound,
                     "Protected task warm start must precede updates and cannot be repeated")

    @torch.no_grad()
    def bind_public_warm_start(self, task_normalizer, *, initialization, reference_count, warm_start_task_count):
        if self.mode == "legacy":
            return None
        self.require_fresh_warm_start(0)
        source = _initialization_record(initialization)
        reference_count = _positive_float(reference_count, "reference_count")
        warm_start_task_count = _positive_float(warm_start_task_count, "warm_start_task_count")
        _require(reference_count == PUBLIC_REFERENCE_COUNT and warm_start_task_count <= reference_count,
                 "Public original reference count and capped task count disagree")
        _validate_count_provenance(initialization, reference_count, warm_start_task_count)
        values = _task_state(task_normalizer)
        _require(float(values["count"]) == warm_start_task_count, "Warm-start task count does not match the current task normalizer")
        self._reference_initialization = deepcopy(source)
        self._reference_count = reference_count
        self._warm_start_task_count = warm_start_task_count
        self._suffix_mean = values["mean"][20:].detach().cpu().clone()
        self._suffix_variance = values["variance"][20:].detach().cpu().clone()
        self._reference_statistics_sha256 = _statistics_sha256(self._suffix_mean, self._suffix_variance)
        self.bound = True
        return self.describe()

    @torch.no_grad()
    def after_task_update(self, task_normalizer):
        if self.mode == "legacy":
            return None
        self.require_ready()
        values = _task_state(task_normalizer)
        values["mean"][20:].copy_(self._suffix_mean.to(values["mean"].device))
        values["variance"][20:].copy_(self._suffix_variance.to(values["variance"].device))
        self.post_update_calls += 1
        return self.describe()

    def validate_checkpoint(self, task_normalizer, *, update_count, physical_transition_count):
        if self.mode == "legacy":
            return None
        self.require_ready()
        _require(self.post_update_calls == _counter(update_count, "update_count"), "Protected statistics hook count differs from update_count")
        transitions = _counter(physical_transition_count, "physical_transition_count")
        _require(self._warm_start_task_count + transitions <= 2**53, "Task count exceeds exact float64 counter precision")
        values = _task_state(task_normalizer)
        _require(float(values["count"]) == self._warm_start_task_count + transitions, "Protected task global count disagrees with actual transitions")
        _require(torch.equal(values["mean"][20:].detach().cpu(), self._suffix_mean)
                 and torch.equal(values["variance"][20:].detach().cpu(), self._suffix_variance),
                 "Protected suffix does not match saved task moments")
        _require(_statistics_sha256(self._suffix_mean, self._suffix_variance) == self._reference_statistics_sha256,
                 "Protected reference moments changed")
        return self.describe()

    def state_dict(self):
        if self.mode == "legacy":
            return None
        self.require_ready()
        return {"schema": SCHEMA, "mode": self.mode, "task_slice": TASK_SLICE.copy(),
                "reference_initialization": deepcopy(self._reference_initialization), "reference_count": self._reference_count,
                "warm_start_task_count": self._warm_start_task_count, "post_update_calls": self.post_update_calls,
                "suffix_mean": self._suffix_mean.clone(), "suffix_variance": self._suffix_variance.clone(),
                "reference_statistics_sha256": self._reference_statistics_sha256}

    @classmethod
    def from_checkpoint(cls, saved_policy, *, config, initialization, saved_task_normalizer,
                        update_count, physical_transition_count):
        """Validate first; never replace task moments or load an external file.

        map_location=device may place policy tensors on CUDA. Validate each
        tensor before making the private1024-element CPU copy; no RNG draws.
        """
        policy = cls(getattr(config, "task_terrain_statistics_mode", "legacy"))
        if policy.mode == "legacy":
            _require(saved_policy is None, "Legacy checkpoint cannot contain protected terrain policy state")
            return policy
        _require(config.normalize_observations is True and config.normalization_clip == 5.
                 and config.task_dim == 1044 and config.self_dim == 357,
                 "Protected task configuration disagrees with public observation normalization")
        _require(isinstance(saved_policy, dict) and set(saved_policy) == STATE_KEYS, "Missing or unknown protected terrain policy state")
        _require(saved_policy["schema"] == SCHEMA and saved_policy["mode"] == policy.mode
                 and type(saved_policy["task_slice"]) is list and saved_policy["task_slice"] == TASK_SLICE
                 and all(type(value) is int for value in saved_policy["task_slice"]), "Invalid protected terrain policy schema/mode/slice")
        source = _initialization_record(initialization)
        _require(saved_policy["reference_initialization"] == source, "Protected statistics public initialization provenance changed")
        original_count = _positive_float(saved_policy["reference_count"], "reference_count")
        warm_count = _positive_float(saved_policy["warm_start_task_count"], "warm_start_task_count")
        _require(original_count == PUBLIC_REFERENCE_COUNT, "Public reference provenance count changed")
        cap = config.warm_start_normalizer_count_cap
        expected_warm_count = original_count if cap is None else min(original_count, float(cap))
        _require(warm_count == expected_warm_count and initialization.get("normalizer_count_cap") == cap,
                 "Warm-start count cap/config/provenance disagree")
        if cap is not None:
            counts = initialization.get("normalizer_counts_before_cap")
            _require(isinstance(counts, dict) and counts.get("task_normalizer") == original_count
                     and type(counts.get("task_normalizer")) is float, "Original public task count provenance changed")
        calls = _counter(saved_policy["post_update_calls"], "post_update_calls")
        tensors = []
        tensor_devices = []
        for name in ("suffix_mean", "suffix_variance"):
            value = saved_policy[name]
            _require(isinstance(value, torch.Tensor) and value.shape == (1024,) and value.dtype == torch.float64
                     and value.device.type in ("cpu", "cuda") and value.layout == torch.strided
                     and torch.isfinite(value).all(), f"Invalid protected {name}")
            tensor_devices.append(value.device)
            tensors.append(value.detach().cpu().clone())
        _require(tensor_devices[0] == tensor_devices[1], "Protected reference moments must share a transport device")
        _require((tensors[1] >= 0).all(), "Negative protected reference variance")
        _require(saved_policy["reference_statistics_sha256"] == _statistics_sha256(*tensors), "Protected statistic array SHA changed")
        policy._reference_initialization = deepcopy(source)
        policy._reference_count, policy._warm_start_task_count = original_count, warm_count
        policy.post_update_calls = calls
        policy._suffix_mean, policy._suffix_variance = tensors
        policy._reference_statistics_sha256 = saved_policy["reference_statistics_sha256"]
        policy.bound = True
        policy.validate_checkpoint(saved_task_normalizer, update_count=update_count,
                                   physical_transition_count=physical_transition_count)
        return policy

    def describe(self):
        return {"schema": SCHEMA, "mode": self.mode, "bound": self.bound,
                "task_slice": TASK_SLICE.copy(), "post_update_calls": self.post_update_calls,
                "reference_initialization": deepcopy(self._reference_initialization),
                "reference_count": self._reference_count, "warm_start_task_count": self._warm_start_task_count,
                "reference_statistics_sha256": self._reference_statistics_sha256,
                "global_count_and_prefix20_protected": False, "external_statistics_file_required": False}


def validate_terrain_statistics_checkpoint(saved_policy, **kwargs):
    """Shared actor/training admission; returns the validated small policy."""
    return TerrainStatisticsPolicy.from_checkpoint(saved_policy, **kwargs)
