"""Strict Pop-Art family selection for audits and actor-only inference.

Historical v1 metadata is reconstructed from the sealed trainer declaration.
It is never admitted by the current trainer's optimizer-resume interface.
"""
import ast
from dataclasses import asdict
import hashlib
from pathlib import Path
from types import SimpleNamespace

from social_navigation.artifact_identity import artifact_path, same_recorded_value
from . import social_popart_training as training
from .social_evaluation_lifecycle import record


def describe(format_name):
    if format_name == training.LEGACY_FORMAT:
        return training.describe()
    if format_name == training.FORMAT:
        return training.describe_runtime()
    raise ValueError("Unregistered Pop-Art checkpoint format")


def admit_binding(binding):
    if type(binding) is not dict:
        raise ValueError("Complete Pop-Art family binding is required")
    expected = describe(binding.get("checkpoint_format"))
    if not same_recorded_value(binding, expected, root=training.ROOT):
        raise ValueError("Pop-Art checkpoint family source binding changed")
    return expected


def legacy_contract(runtime, config):
    """Build v1's expected contract from immutable declarations, never saved data."""
    runtime._verify_sources()
    if record(training.TRAINER)["sha256"] != training.TRAINER_SHA:
        raise ValueError("Historical Social declaration changed")
    source, _ = training.derive_trainer(training.TRAINER.read_text())
    tree = ast.parse(source)
    assignments = {"PINS", "DERIVATIONS", "CONTRACT_SCHEMA"}
    functions = {"_method_source", "derivation_records", "_contract"}
    nodes = [node for node in tree.body if (
        isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in assignments for t in node.targets)
        or isinstance(node, ast.FunctionDef) and node.name in functions
        or isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "append"
        and isinstance(node.value.func.value, ast.Subscript)
        and isinstance(node.value.func.value.value, ast.Name) and node.value.func.value.value.id == "DERIVATIONS")]
    selected = ast.unparse(ast.Module(body=nodes, type_ignores=[]))
    old = "(REPO / 'CrowdSim/humanoid/training/trainer.py').read_text()"
    if selected.count(old) != 1:
        raise ValueError("Historical derivation selector changed")
    selected = selected.replace(old, "_NUMERICAL_PARENT.read_text()", 1)
    if record(training.NUMERICAL_PARENT)["sha256"] != training.NUMERICAL_PARENT_SHA:
        raise ValueError("Historical numerical source changed")
    namespace = dict(ast=ast, hashlib=hashlib, Path=Path, REPO=training.ROOT,
        HERE=training.TRAINER.parent, _NUMERICAL_PARENT=training.NUMERICAL_PARENT,
        _sha=lambda data: hashlib.sha256(data).hexdigest(), _policy=runtime._policy,
        _verify_sources=runtime._verify_sources, _verify_popart_sources=training.describe,
        _POPART_BINDING=training.describe(), _EXECUTION_ADAPTER_RECORD=record(training.TRAINER))
    exec(compile(selected, str(training.TRAINER) + ":historical_metadata_only", "exec"), namespace)

    def historical_record(path):
        local = artifact_path(path, root=training.ROOT)
        relative = str(local.relative_to(training.ROOT))
        if relative == "src/CrowdSim/humanoid/training/trainer.py":
            return {**record(training.NUMERICAL_PARENT), "path": str(local)}
        if relative == "src/CrowdSim/humanoid/amp/observations.py":
            # This identity is declared in the SHA-pinned original World and
            # trainer. The historical observation implementation is not run.
            return {"path": str(local), "bytes": 10438,
                    "sha256": namespace["PINS"]["CrowdSim/humanoid/amp/observations.py"]}
        actual = record(local)
        original_relative = str(Path(relative).relative_to("src")) if relative.startswith("src/") else relative
        expected = namespace["PINS"].get(original_relative)
        if expected is not None and actual["sha256"] != expected:
            raise ValueError("Historical metadata dependency changed: " + original_relative)
        return actual

    namespace["_record"] = historical_record
    namespace["_DERIVATION_RECORDS"] = namespace["derivation_records"]()
    return namespace["_contract"](config)


def select_inference_definitions(runtime, artifact):
    """Choose only metadata/tensor validators; never expose v1 resume methods."""
    if type(artifact) is not dict:
        raise ValueError("Complete Pop-Art checkpoint dictionary is required")
    expected = describe(artifact.get("format"))
    if artifact["format"] == training.FORMAT:
        return runtime

    def admit_metadata(state):
        runtime._verify_sources()
        if (type(state) is not dict or set(state) != runtime.CHECKPOINT_KEYS
                or state.get("format") != training.LEGACY_FORMAT
                or type(state["version"]) is not int or state["version"] != 1
                or type(state["provenance"]) is not dict):
            raise ValueError("Unknown/incomplete historical social checkpoint schema")
        contract = state["social_training_contract"]
        if type(contract) is not dict or contract.get("schema") != runtime.CONTRACT_SCHEMA:
            raise ValueError("Complete historical Social contract is required")
        config = runtime.SocialTrainingConfig.from_document(contract.get("configuration_document"))
        if (config.discriminator_input != "state_action"
                or not runtime._equal_json(state["config"], asdict(config))
                or not same_recorded_value(contract, legacy_contract(runtime, config), root=training.ROOT)
                or not same_recorded_value(contract.get("value_normalization"), expected, root=training.ROOT)):
            raise ValueError("Historical Social architecture/config/source derivation changed")
        return config

    return SimpleNamespace(torch=runtime.torch, SocialAMPNetwork=runtime.SocialAMPNetwork,
        _base=runtime._base, _settings=runtime._settings, _equal_json=runtime._equal_json,
        _verify_sources=runtime._verify_sources,
        SocialAMPTrainer=SimpleNamespace(_admit_metadata=admit_metadata, _tensor=runtime.SocialAMPTrainer._tensor))
