"""Unintegrated prototype for an explicit fresh terrain-statistic ablation.

Preserves only task[20:1044] moments after the existing update. No actor,
trajectory[0:20], task count, optimizer, AMP/value statistics, or RNG changes.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import re

import numpy as np
import torch

SCHEMA = "staged_terrain_RMS_policy.v1"
MODES = {"legacy", "preserve_declared_suffix"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def normalizer_buffers(normalizer):
    mean, variance, count = normalizer.mean, normalizer.variance, normalizer.count
    require(isinstance(mean, torch.Tensor) and isinstance(variance, torch.Tensor) and isinstance(count, torch.Tensor), "Tensor statistics required")
    require(mean.shape == variance.shape == (1044,) and count.shape == () and
            mean.dtype == variance.dtype == count.dtype == torch.float64 and
            mean.device == variance.device == count.device and torch.isfinite(mean).all() and
            torch.isfinite(variance).all() and torch.isfinite(count).all() and
            (variance >= 0).all() and count > 0, "Finite float64 task1044 moments/count contract required")
    require(normalizer.clip == 5., "This candidate binds unchanged PACER clip5")
    return mean, variance, count


class StagedTerrainRMSPolicy:
    def __init__(self, mode="legacy", *, reference_record=None, fresh_experiment=False):
        require(mode in MODES, "Unknown terrain-statistics mode")
        require(type(fresh_experiment) is bool, "Explicit boolean fresh_experiment required")
        self.mode = mode
        self.calls = 0
        self.attached = False
        self.reference_record = None
        self.reference_count = None
        self.suffix_mean = self.suffix_variance = None
        if mode == "legacy":
            require(reference_record is None and not fresh_experiment, "Legacy must not declare a hidden reference override")
            return
        require(fresh_experiment, "Protected statistics require explicit fresh experiment, never silent learned-checkpoint conversion")
        require(isinstance(reference_record, dict) and set(reference_record) == {"path", "bytes", "sha256"} and
                type(reference_record["bytes"]) is int and reference_record["bytes"] > 0 and
                isinstance(reference_record["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", reference_record["sha256"]),
                "Exact declared statistic snapshot path/bytes/SHA required")
        require(file_record(reference_record["path"]) == reference_record, "Declared statistic snapshot changed")
        with np.load(reference_record["path"], allow_pickle=False) as archive:
            require(set(archive.files) == {"task_mean", "task_variance", "reference_count"}, "Unknown statistic snapshot schema")
            mean, variance, reference_count = archive["task_mean"], archive["task_variance"], archive["reference_count"]
        require(mean.dtype == variance.dtype == np.float64 and mean.shape == variance.shape == (1044,) and
                np.isfinite(mean).all() and np.isfinite(variance).all() and (variance >= 0).all(), "Invalid reference task1044 moments")
        require(reference_count.shape == () and reference_count.dtype == np.float64 and
                np.isfinite(reference_count) and reference_count > 0, "Positive float64 reference provenance count required")
        self.reference_count = float(reference_count)
        self.reference_record = deepcopy(reference_record)
        self.suffix_mean = torch.from_numpy(mean[20:].copy())
        self.suffix_variance = torch.from_numpy(variance[20:].copy())

    @torch.no_grad()
    def attach(self, normalizer):
        if self.mode == "legacy":
            return None
        require(not self.attached, "Already attached; do not reset protection counters")
        mean, variance, _ = normalizer_buffers(normalizer)
        require(torch.equal(mean[20:].cpu(), self.suffix_mean) and torch.equal(variance[20:].cpu(), self.suffix_variance),
                "Fresh initialized task suffix differs from declared reference; no silent overwrite at attach")
        self.attached = True
        return self.describe()

    @torch.no_grad()
    def after_update(self, normalizer):
        if self.mode == "legacy":
            return None
        require(self.attached, "Attach before the first real statistics update")
        mean, variance, _ = normalizer_buffers(normalizer)
        mean[20:].copy_(self.suffix_mean.to(mean.device))
        variance[20:].copy_(self.suffix_variance.to(variance.device))
        self.calls += 1
        return {"protected_dimensions": 1024, "post_update_calls": self.calls,
                "task_prefix_and_count_changed_by_this_hook": False}

    def describe(self):
        return {"schema": SCHEMA, "mode": self.mode, "reference_record": deepcopy(self.reference_record),
                "reference_statistics_count_not_global_task_count": self.reference_count,
                "terrain_suffix_half_open": [20, 1044], "clip": 5., "attached": self.attached,
                "post_update_calls": self.calls, "integrated_in_production": False,
                "legacy_default_unchanged": self.mode == "legacy", "counts_preserved": True,
                "model_optimizer_AMP_value_or_RNG_modified": False}

    def state_dict(self):
        if self.mode == "legacy":
            return None
        require(self.attached, "Cannot save unattached protected policy")
        return {"schema": SCHEMA, "mode": self.mode, "reference_record": deepcopy(self.reference_record),
                "reference_count": self.reference_count, "calls": self.calls, "suffix_mean": self.suffix_mean.clone(), "suffix_variance": self.suffix_variance.clone()}

    def validate_checkpoint(self, normalizer, *, expected_update_count):
        """Bind hook calls to the caller's actual completed training updates."""
        if self.mode == "legacy":
            return None
        require(self.attached and type(expected_update_count) is int and expected_update_count >= 0 and
                self.calls == expected_update_count, "Protection calls must match the completed trainer update_count")
        mean, variance, _ = normalizer_buffers(normalizer)
        require(torch.equal(mean[20:].cpu(), self.suffix_mean) and torch.equal(variance[20:].cpu(), self.suffix_variance),
                "Checkpoint statistics do not carry the declared protected suffix")
        return self.describe()

    def load_state_dict(self, state, normalizer, *, expected_update_count=None):
        if self.mode == "legacy":
            require(state is None, "Legacy cannot silently restore protected-statistics experiment")
            return None
        expected = {"schema", "mode", "reference_record", "reference_count", "calls", "suffix_mean", "suffix_variance"}
        require(isinstance(state, dict) and set(state) == expected and state["schema"] == SCHEMA and
                state["mode"] == self.mode and state["reference_record"] == self.reference_record and
                type(state["reference_count"]) is float and state["reference_count"] == self.reference_count and
                type(state["calls"]) is int and state["calls"] >= 0 and
                type(expected_update_count) is int and expected_update_count >= 0 and state["calls"] == expected_update_count,
                "Protection resume schema/source/config/counters changed")
        for name in ("suffix_mean", "suffix_variance"):
            value = state[name]
            require(isinstance(value, torch.Tensor) and value.dtype == torch.float64 and value.device.type == "cpu" and
                    value.shape == (1024,) and torch.isfinite(value).all() and torch.equal(value, getattr(self, name)), "Protection resume moments changed")
        mean, variance, _ = normalizer_buffers(normalizer)
        require(torch.equal(mean[20:].cpu(), self.suffix_mean) and torch.equal(variance[20:].cpu(), self.suffix_variance),
                "Checkpoint statistics do not carry the declared protected suffix")
        self.calls = state["calls"]
        self.attached = True
        return self.describe()
