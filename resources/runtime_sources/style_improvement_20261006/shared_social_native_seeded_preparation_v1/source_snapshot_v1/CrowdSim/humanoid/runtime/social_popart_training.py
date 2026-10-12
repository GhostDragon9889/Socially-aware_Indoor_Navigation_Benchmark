"""Explicit, reversible Pop-Art derivation of the sealed shared trainer.

Stdlib import. The caller loads this trainer only after launching Isaac Sim.
The baseline files and installed SDK remain unchanged. The new checkpoint
format/contract require a matching explicit reader; legacy loading must fail.
"""
from __future__ import annotations

import hashlib
import sys
from types import ModuleType

from .social_evaluation_lifecycle import ROOT, record

TRAINER = ROOT / "resources/runtime_sources/style_improvement_20261004/social_trainer_staging_v1/social_trainer.py"
TRAINER_SHA = "6cb0d2d1bd7267b68cca92aa81d4ad30a808c4c6a52e7d4c808e6b00874ed936"
HELPER = ROOT / "CrowdSim/humanoid/training/value_normalization.py"
FORMAT = "local.neutral_smpl.social_CNN_PPO_AMP_training.popart.v1"


def derive_trainer(source):
    insertion = '''DERIVATIONS["update"].append((
    "            self.value_normalizer.update(target[:, None])",
    "            previous_value_RMS = {key: value.detach().clone() for key, value in self.value_normalizer.named_buffers()}\\n"
    "            self.value_normalizer.update(target[:, None])\\n"
    "            popart_adjustment = _SOCIAL_POPART_FUNCTION(self.model.critic[4], previous_value_RMS, dict(self.value_normalizer.named_buffers()))", 1))
DERIVATIONS["update"].append((
    "        return metrics",
    "        metrics[\\\"value_normalization\\\"] = popart_adjustment\\n        return metrics", 1))
_DERIVATION_RECORDS = derivation_records()
'''
    replacements = (
        ('FORMAT = "local.neutral_smpl.social_CNN_PPO_AMP_training.v1"', 'FORMAT = ' + repr(FORMAT)),
        ('_DERIVATION_RECORDS = derivation_records()\n', insertion),
        ('                     _SOCIAL_MIRROR_FUNCTIONS=_SOCIAL_MIRROR_FUNCTIONS, __name__=__name__)\n',
         '                     _SOCIAL_MIRROR_FUNCTIONS=_SOCIAL_MIRROR_FUNCTIONS, __name__=__name__,\n'
         '                     _SOCIAL_POPART_FUNCTION=_SOCIAL_POPART_FUNCTION)\n'),
        ('def _contract(config):\n    _verify_sources()\n',
         'def _contract(config):\n    _verify_sources()\n    _verify_popart_sources()\n'),
        ('            "derivations": _DERIVATION_RECORDS,\n',
         '            "derivations": _DERIVATION_RECORDS,\n'
         '            "value_normalization": _POPART_BINDING,\n'),
        ('        config = config.detached_copy()\n',
         '        config = config.detached_copy()\n'
         '        if config.normalize_values is not True:\n'
         '            raise ValueError("Pop-Art experiment requires normalized scalar values")\n'),
    )
    derived = source
    for before, after in replacements:
        if derived.count(before) != 1:
            raise ValueError("Ambiguous Pop-Art trainer migration")
        derived = derived.replace(before, after, 1)
    inverse = derived
    for before, after in reversed(replacements):
        if inverse.count(after) != 1:
            raise ValueError("Ambiguous Pop-Art migration inverse")
        inverse = inverse.replace(after, before, 1)
    if inverse != source:
        raise ValueError("Pop-Art trainer source is not exactly reversible")
    compile(derived, str(TRAINER) + ":popart", "exec")
    return derived, {"parent_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "derived_sha256": hashlib.sha256(derived.encode()).hexdigest(), "exact_inverse": True,
        "replacement_count": len(replacements), "numeric_change": "value RMS output preservation only",
        "actor_encoder_reward_GAE_sampling_changed": False}


def describe():
    identity = record(TRAINER)
    if identity["sha256"] != TRAINER_SHA or TRAINER.stat().st_mode & 0o222:
        raise ValueError("Sealed shared trainer changed")
    _, proof = derive_trainer(TRAINER.read_text())
    return {"strategy": "scalar_popart_output_preservation_v1", "checkpoint_format": FORMAT,
        "parent_trainer": identity, "derivation": proof, "adapter": record(__file__), "helper": record(HELPER),
        "epsilon": 1e-5, "Adam_state_modified": False, "baseline_weights_resumed": False,
        "author_method_equivalence_claimed": False}


def load_trainer(name):
    """Construct definitions, not a NN; entry invokes prepare after App startup."""
    if name in sys.modules:
        raise ValueError("Refuse a pre-existing entry-owned trainer module")
    binding = describe()
    source, _ = derive_trainer(TRAINER.read_text())
    from CrowdSim.humanoid.training.value_normalization import preserve_unnormalized_critic

    def verify():
        if describe() != binding:
            raise ValueError("Pop-Art numerical sources changed during training")

    def guarded_preservation(*args):
        verify()
        return preserve_unnormalized_critic(*args)

    module = ModuleType(name)
    # Original __file__ retains the original resource paths and exact collector
    # class origin. The new compiled source, format and mathematical derivation
    # are separately bound in every checkpoint and actual entry report.
    module.__file__ = str(TRAINER)
    module.__dict__.update(_SOCIAL_POPART_FUNCTION=guarded_preservation,
                          _verify_popart_sources=verify, _POPART_BINDING=binding)
    sys.modules[name] = module
    try:
        exec(compile(source, str(TRAINER) + ":popart", "exec"), module.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    return module
