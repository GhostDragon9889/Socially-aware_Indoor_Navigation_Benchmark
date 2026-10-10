"""Explicit Pop-Art checkpoint family reader; retains the strict actor loader.

Only the caller-owned trainer-definition dispatch changes. The pinned Linux
descriptor, full SHA, weights_only/mmap, complete model/RMS validation and
actor-only copies retain the original implementation. No SDK or optimizer
restore. Source preparation is stdlib only; actual weight admission is separate.
"""
from __future__ import annotations

import sys
from types import ModuleType

from .social_evaluation_lifecycle import ROOT, record
from . import social_popart_training

PARENT = ROOT / "docs/style_improvement_20261004/social_checkpoint_inference_staging_v1/checkpoint_inference.py"
PARENT_SHA = "cd227f5d2de81e21da560dba97a1d2f564e125beba9e73d739548c5c1800a352"


def prepare_reader():
    parent = record(PARENT)
    if parent["sha256"] != PARENT_SHA or PARENT.stat().st_mode & 0o222:
        raise ValueError("Sealed strict checkpoint loader changed")
    adapter = record(__file__)
    trainer = social_popart_training.describe()
    module = ModuleType("_explicit_social_popart_checkpoint_reader")
    module.__file__ = str(PARENT)
    exec(compile(PARENT.read_bytes(), str(PARENT), "exec"), module.__dict__)
    original_sources, original_owned = module._sources, module._owned_module

    def sources():
        if record(__file__) != adapter or social_popart_training.describe() != trainer:
            raise ValueError("Explicit Pop-Art inference source changed")
        return {**original_sources(), "popart_reader": adapter,
                "popart_training_adapter": trainer["adapter"], "scalar_popart_helper": trainer["helper"]}

    def owned(name, key):
        sources()
        if key == "trainer":
            selected = "_social_popart_checkpoint_source_trainer"
            existing = sys.modules.get(selected)
            if existing is not None:
                if (getattr(existing, "__file__", None) != str(social_popart_training.TRAINER)
                        or getattr(existing, "_POPART_BINDING", None) != trainer
                        or getattr(existing, "FORMAT", None) != social_popart_training.FORMAT):
                    raise ValueError("Foreign Pop-Art trainer definitions in this reader process")
                existing._verify_popart_sources()
                return existing
            return social_popart_training.load_trainer(selected)
        return original_owned(name, key)

    module._sources, module._owned_module = sources, owned
    return module, {"parent": parent, "adapter": adapter, "training": trainer,
                    "checkpoint_validation_math_changed": False, "optimizer_restored": False}


class FrozenSocialPopArtCheckpointInference:
    """Explicit family facade; returns the original strict frozen actor owner."""

    @classmethod
    def from_checkpoint(cls, checkpoint, *, expected_sha256, device="cpu"):
        reader, binding = prepare_reader()
        instance = reader.FrozenSocialCheckpointInference.from_checkpoint(
            checkpoint, expected_sha256=expected_sha256, device=device)
        instance._metadata["explicit_PopArt_reader"] = binding
        return instance
