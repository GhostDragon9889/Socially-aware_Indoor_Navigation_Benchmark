"""Read-only pre-action samples; no App/context/model constructor at import."""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile


def _array_record(value):
    return {
        "shape": list(value.shape),
        "dtype": value.dtype.str,
        "sha256_C_order": hashlib.sha256(value.tobytes(order="C")).hexdigest(),
    }


class PreActionObservationWriter:
    """Persist every selected successful actor call before the driver's step.

    call_index is zero-based actor.act invocation count, not a simulator counter.
    The JSONL row is fsynced after its same-directory atomic NPZ publication.
    Nothing here declares runtime completion or advances physics.
    """

    def __init__(self, path, *, stride=30):
        if type(stride) is not int or stride < 1:
            raise ValueError("Observation stride must be a positive integer")
        self.path = Path(path)
        if self.path.exists() or self.path.is_symlink():
            raise ValueError("Observation directory must be new; no overwrite")
        self.path.mkdir(parents=True, exist_ok=False)
        self.stride = stride
        self.samples = 0

    def publish(self, index, arrays, *, clip):
        import numpy as np

        if type(index) is not int or index < 0 or index % self.stride:
            raise ValueError("Selected call index disagrees with observer stride")
        for name, value in arrays.items():
            if value.dtype.kind == "O" or (
                value.dtype.kind in "fc" and not np.isfinite(value).all()
            ):
                raise ValueError(
                    "Observation array must be finite and nonobject: " + name
                )
        target = self.path / f"call_{index:08d}.npz"
        if target.exists():
            raise ValueError("Observation sample already exists")
        descriptor, temporary = tempfile.mkstemp(
            prefix=".pre-action-", suffix=".npz", dir=self.path
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                np.savez(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
            descriptor = os.open(self.path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            Path(temporary).unlink(missing_ok=True)
        records = {name: _array_record(value) for name, value in arrays.items()}
        moments = {
            name: records[name] for name in ("task_mean", "task_variance", "task_count")
        }
        row = {
            "schema": "paper_pre_action_observation_sample.v1",
            "call_index": index,
            "stride": self.stride,
            "call_index_is_native_counter": False,
            "raw_observation_capture_timing": "immediately before original actor.act and before driver world.step",
            "actor_act_called_once": True,
            "actor_outputs_returned_without_replacement": True,
            "normalized_z_source": "existing actor.task_normalizer.normalize(task_obs,clamp=False)",
            "normalization_clip": clip,
            "normalization_epsilon": 1e-5,
            "moments_SHA256": hashlib.sha256(
                json.dumps(moments, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "file": {
                "path": str(target.resolve()),
                "bytes": target.stat().st_size,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            },
            "arrays": records,
            "native_counter": None,
            "native_engine_time": None,
            "runtime_completion_claim": False,
        }
        with (self.path / "samples.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.samples += 1
        return row


class PreActionObservedActor:
    """Proxy one frozen baseline actor; return its exact output object."""

    def __init__(self, actor, writer):
        inference = getattr(actor, "_inference", None)
        if (
            inference is None
            or hasattr(inference, "baseline")
            or hasattr(inference, "style")
        ):
            raise ValueError(
                "Observer supports one baseline EvaluationActor, not arm blending"
            )
        config = actor.config
        if (
            config.self_dim != 357
            or config.task_dim != 1044
            or config.normalize_observations is not True
        ):
            raise ValueError(
                "Observer requires actual357/1044 normalized inference input contract"
            )
        self._observed_actor = actor
        self._normalizer = inference.task_normalizer
        self._writer = writer
        self._calls = 0

    def __getattr__(self, name):
        return getattr(self._observed_actor, name)

    def act(self, self_obs, task_obs, *, deterministic=True):
        if deterministic is not True:
            raise ValueError(
                "Pre-action observer supports deterministic evaluation only"
            )
        index = self._calls
        arrays = None
        if index % self._writer.stride == 0:
            import torch

            if (
                not isinstance(self_obs, torch.Tensor)
                or not isinstance(task_obs, torch.Tensor)
                or self_obs.ndim != 2
                or task_obs.ndim != 2
                or self_obs.shape[0] < 1
                or self_obs.shape != (task_obs.shape[0], 357)
                or task_obs.shape[1] != 1044
                or self_obs.dtype != torch.float32
                or task_obs.dtype != torch.float32
                or not torch.isfinite(self_obs).all()
                or not torch.isfinite(task_obs).all()
            ):
                raise ValueError(
                    "Pre-action observations require aligned finiteFP32[B357]/[B1044]"
                )
            with torch.no_grad():
                raw_self = self_obs.detach().clone()
                raw_task = task_obs.detach().clone()
                normalizer = self._normalizer
                task = raw_task.to(normalizer.mean.device)
                z = normalizer.normalize(task, clamp=False).detach().clone()
                clipped = normalizer.normalize(task).detach().clone()
                values = {
                    "self_obs": raw_self,
                    "task_obs": raw_task,
                    "task_z_unclamped": z,
                    "task_z_clamped": clipped,
                    "task_clip_mask": z != clipped,
                    "task_mean": normalizer.mean.detach().clone(),
                    "task_variance": normalizer.variance.detach().clone(),
                    "task_count": normalizer.count.detach().clone(),
                }
                arrays = {
                    name: value.detach().cpu().numpy().copy()
                    for name, value in values.items()
                }
        outputs = self._observed_actor.act(
            self_obs, task_obs, deterministic=deterministic
        )
        self._calls += 1
        if arrays is not None:
            self._writer.publish(index, arrays, clip=float(self._normalizer.clip))
        return outputs
