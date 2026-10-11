"""Explicit, reversible Pop-Art derivation of the sealed shared trainer.

Stdlib import. The caller loads this trainer only after launching Isaac Sim.
The baseline files and installed SDK remain unchanged. The new checkpoint
format/contract require a matching explicit reader; legacy loading must fail.
"""
from __future__ import annotations

import ast
import hashlib
import sys
from types import ModuleType

from .social_evaluation_lifecycle import ROOT, record

TRAINER = ROOT / "docs/style_improvement_20261004/social_trainer_staging_v1/social_trainer.py"
TRAINER_SHA = "81a6319f2520b3be547f30454c88b088638d559d492bba52fe850d1c1d5ba96e"
HELPER = ROOT / "CrowdSim/humanoid/training/value_normalization.py"
LEGACY_FORMAT = "local.neutral_smpl.social_CNN_PPO_AMP_training.popart.v1"
FORMAT = "local.neutral_smpl.social_CNN_PPO_AMP_training.popart.sim61.v2"
HISTORICAL_ADAPTER = ROOT / "docs/style_improvement_20261006/shared_social_native_seeded_preparation_v1/source_snapshot_v1/CrowdSim/humanoid/runtime/social_popart_training.py"
HISTORICAL_ADAPTER_SHA = "66f8d37bd6e515f577c306f009d268111c860d847268626fdf01db0826152493"
NUMERICAL_PARENT = ROOT / "docs/style_improvement_20261003/staged_action_state_prior_v1/proposal/CrowdSim/humanoid/training/trainer.py"
NUMERICAL_PARENT_SHA = "707f355acbd8b5d33aa40c3299a2540610ea695ae4bb9705ede3498e27224f9d"
CURRENT_SOURCES = {
    "CrowdSim/humanoid/training/trainer.py": "1e5444f62ccc5214e3f635850d4b9b018b457918259ffbb1e999a05148f4eee7",
    "CrowdSim/humanoid/amp/observations.py": "000f3980a8f4d3c1fac929c57029dd05a9ce19b1d8ce82423dad273c9315e599",
}


def derive_trainer(source, *, format_name=LEGACY_FORMAT):
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
        ('FORMAT = "local.neutral_smpl.social_CNN_PPO_AMP_training.v1"', 'FORMAT = ' + repr(format_name)),
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
    """Read the archived v1 metadata; this never admits current runtime code.

    Historical receipts retain their original identities. The adapter bytes are
    verified in the immutable snapshot, rather than asserted to be live bytes.
    New training and inference definitions use describe_runtime() below.
    """
    import ast

    archived = record(HISTORICAL_ADAPTER)
    if archived["sha256"] != HISTORICAL_ADAPTER_SHA:
        raise ValueError("Historical Pop-Art adapter snapshot changed")
    tree = ast.parse(HISTORICAL_ADAPTER.read_text())
    # Execute only the pure derivation/description definitions, with the exact
    # original constants; no Torch, checkpoint or SDK is imported here.
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {"derive_trainer", "describe"}]
    if len(nodes) != 2:
        raise ValueError("Historical Pop-Art metadata definitions are incomplete")

    def historical_record(path):
        if str(path) == str(__file__):
            return {**archived, "path": str((ROOT / "CrowdSim/humanoid/runtime/social_popart_training.py").resolve())}
        return record(path)

    namespace = dict(hashlib=hashlib, TRAINER=TRAINER, TRAINER_SHA=TRAINER_SHA,
                     FORMAT=LEGACY_FORMAT, HELPER=HELPER, __file__=__file__, record=historical_record)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(HISTORICAL_ADAPTER), "exec"), namespace)
    return namespace["describe"]()


def derive_runtime(source):
    """Keep the four historical numerical methods and fix CPU RNG restoration.

    The live base class supplies helpers; each selected dependency is separately
    pinned. Later base-class freeze/regularization hooks cannot silently change
    the sealed Social PPO update or invalidate the Pop-Art insertion.
    """
    if (record(TRAINER)["sha256"] != TRAINER_SHA
            or hashlib.sha256(source.encode()).hexdigest() != TRAINER_SHA):
        raise ValueError("Sealed Social trainer changed")
    if record(NUMERICAL_PARENT)["sha256"] != NUMERICAL_PARENT_SHA:
        raise ValueError("Historical numerical trainer snapshot changed")
    for relative, sha in CURRENT_SOURCES.items():
        if record(ROOT / relative)["sha256"] != sha:
            raise ValueError("Current Social trainer dependency changed: " + relative)
    derived, popart = derive_trainer(source, format_name=FORMAT)
    owner = next(node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name == "_owned_module")
    replacements = [
        ('torch.full_like(model["log_std"], self.config.log_std)',
         'torch.full_like(state["model"]["log_std"], self.config.log_std)'),
        ('replay["actions"] * (1 - self.action_mask)',
         'replay["actions"] * (1 - self.action_mask.to(replay["actions"].device))'),
        (ast.get_source_segment(source, owner),
         'def _owned_module(name, relative):\n'
         '    return _SOCIAL_LOAD_STAGED(name, relative, PINS[relative])'),
        ('text = (REPO / "CrowdSim/humanoid/training/trainer.py").read_text()',
         'text = _SOCIAL_NUMERICAL_PARENT.read_text()'),
        ('if _sha(text.encode()) != PINS["CrowdSim/humanoid/training/trainer.py"]:',
         'if _sha(text.encode()) != _SOCIAL_NUMERICAL_PARENT_SHA:'),
    ]
    for relative, sha in CURRENT_SOURCES.items():
        # Parse the original dictionary to require the exact historical pin.
        declaration = next(node for node in ast.parse(source).body
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                and target.id == "PINS" for target in node.targets))
        old_sha = ast.literal_eval(declaration.value)[relative]
        replacements.append((f'"{relative}": "{old_sha}"', f'"{relative}": "{sha}"'))
    # The historical loader would send CPU generator buffers to CUDA. Retain
    # every other load instruction and let torch optimizer loaders place state.
    replacements.append(('    "load": [("\\\"navisaaclab_method_reconstruction_training\\\"", "FORMAT", 1),',
                         '    "load": [("map_location=device", "map_location=\\\"cpu\\\"", 1),\n'
                         '             ("\\\"navisaaclab_method_reconstruction_training\\\"", "FORMAT", 1),'))
    inverse_pairs = []
    for before, after in replacements:
        count = derived.count(before)
        expected = 2 if before.startswith(('text =', 'if _sha(text')) else 1
        if count != expected:
            raise ValueError("Ambiguous Social trainer runtime migration: " + before)
        derived = derived.replace(before, after)
        inverse_pairs.append((before, after, count))
    inverse = derived
    for before, after, count in reversed(inverse_pairs):
        if inverse.count(after) != count:
            raise ValueError("Ambiguous Social trainer runtime inverse")
        inverse = inverse.replace(after, before)
    original_popart, _ = derive_trainer(source, format_name=FORMAT)
    if inverse != original_popart:
        raise ValueError("Social runtime migration inverse differs")
    compile(derived, str(TRAINER) + ":sim61_popart", "exec")
    return derived, {"version": "social_popart.sim61.v2", "format": FORMAT,
        "derived_sha256": hashlib.sha256(derived.encode()).hexdigest(),
        "source_selector_exact_inverse": True, "popart": popart,
        "numerical_parent": record(NUMERICAL_PARENT),
        "current_dependencies": [record(ROOT / name) for name in CURRENT_SOURCES],
        "load_runner_tensors_on_CPU": True, "historical_checkpoint_implicitly_admitted": False}


def load_staged(name, relative, expected_sha):
    """Load separately owned, source-pinned definitions after path relocation."""
    import ast
    from pathlib import Path

    path = (ROOT / relative).resolve()
    identity = record(path)
    if identity["sha256"] != expected_sha:
        raise ValueError("Staged Social definition changed: " + relative)
    source = path.read_text()
    if name == "social_training_settings":
        before = 'Path(_base_settings.__file__).resolve() != _base_path:'
        after = 'Path(_base_settings.__file__).resolve() != _base_path.resolve():'
        if source.count(before) != 1:
            raise ValueError("Ambiguous staged configuration location guard")
        source = source.replace(before, after, 1)
    elif name == "social_symmetry":
        row = record(ROOT / "CrowdSim/humanoid/amp/observations.py")
        if row["sha256"] != CURRENT_SOURCES["CrowdSim/humanoid/amp/observations.py"]:
            raise ValueError("Social symmetry observation dependency changed")
        before = '10438, "a64dc2eba5c132cf68a248f2324ea20446dd264a2ea8f9f139faae4738f9aeba"'
        after = str(row["bytes"]) + ', "' + row["sha256"] + '"'
        if source.count(before) != 1:
            raise ValueError("Ambiguous Social symmetry source selector")
        source = source.replace(before, after, 1)
        # These are the only observation declarations consumed by symmetry.
        tree = ast.parse((ROOT / "CrowdSim/humanoid/amp/observations.py").read_text())
        declarations = [node for node in tree.body if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in {"PACER_BODY_NAMES", "PACER_DOF_NAMES"}
                    for target in node.targets)]
        if len(declarations) != 2:
            raise ValueError("Canonical PACER name declarations changed")
    proof = {"historical": identity, "derived_sha256": hashlib.sha256(source.encode()).hexdigest(),
             "version": "social_staged_definitions.sim61.v2"}
    selected = "_owned_sim61_" + name
    if selected in sys.modules:
        module = sys.modules[selected]
        if (Path(getattr(module, "__file__", "")).resolve() != path
                or getattr(module, "_SIM61_SOURCE_BINDING", None) != proof):
            raise ValueError("Foreign staged Social definitions in process")
        return module
    module = ModuleType(selected)
    module.__file__ = str(path)
    module._SIM61_SOURCE_BINDING = proof
    sys.modules[selected] = module
    try:
        exec(compile(source, str(path) + ":sim61", "exec"), module.__dict__)
    except BaseException:
        del sys.modules[selected]
        raise
    return module


def describe_runtime():
    if TRAINER.stat().st_mode & 0o222:
        raise ValueError("Sealed shared trainer must remain readonly")
    _, proof = derive_runtime(TRAINER.read_text())
    return {"strategy": "scalar_popart_output_preservation_v1", "checkpoint_format": FORMAT,
        "parent_trainer": record(TRAINER), "derivation": proof,
        "adapter": record(__file__), "helper": record(HELPER),
        "epsilon": 1e-5, "Adam_state_modified": False,
        "baseline_weights_resumed": False, "author_method_equivalence_claimed": False}


def load_trainer(name):
    """Construct definitions, not a NN; entry invokes prepare after App startup."""
    if name in sys.modules:
        raise ValueError("Refuse a pre-existing entry-owned trainer module")
    binding = describe_runtime()
    source, _ = derive_runtime(TRAINER.read_text())
    from CrowdSim.humanoid.training.value_normalization import preserve_unnormalized_critic

    def verify():
        if describe_runtime() != binding:
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
                          _verify_popart_sources=verify, _POPART_BINDING=binding,
                          _SOCIAL_NUMERICAL_PARENT=NUMERICAL_PARENT,
                          _SOCIAL_NUMERICAL_PARENT_SHA=NUMERICAL_PARENT_SHA,
                          _SOCIAL_LOAD_STAGED=load_staged)
    sys.modules[name] = module
    try:
        exec(compile(source, str(TRAINER) + ":popart", "exec"), module.__dict__)
    except BaseException:
        del sys.modules[name]
        raise
    return module
