"""Independent social CNN/PPO/AMP connection; import requires Torch.

This source stage owns no simulator, group reset, curriculum or physical
checkpoint restoration. Its network/checkpoint paths have not been executed.
The original numerical training methods are reused from an exact source pin;
the four bounded method derivations below are recorded in each checkpoint.
"""
from __future__ import annotations

import ast
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import sys

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FORMAT = "local.neutral_smpl.social_CNN_PPO_AMP_training.v1"
CONTRACT_SCHEMA = "local.neutral_smpl.social_training_checkpoint_contract.v1"
CHECKPOINT_KEYS = {"format", "version", "config", "provenance", "model", "optimizer", "discriminator_optimizer",
                   "normalizers", "update_count", "physical_transition_count", "rng", "runner_state",
                   "physical_runner_state_included", "policy_replay", "social_training_contract"}
PINS = {
    "CrowdSim/humanoid/training/trainer.py": "707f355acbd8b5d33aa40c3299a2540610ea695ae4bb9705ede3498e27224f9d",
    "CrowdSim/humanoid/training/contracts.py": "26c7379ba60842c67f062fa5c1a3459ddfb6e0843e1b4952e1d078aa6f78cdad",
    "CrowdSim/humanoid/training/replay.py": "f023108348cfc0a2045138a97e09453ec6b489a900c2eb6da31467419184d27a",
    "CrowdSim/humanoid/training/terrain_statistics.py": "7c07e3a54617d40b72475bc40bf0b728516b7d4166fb99795c8a8bb69b0f78f3",
    "CrowdSim/humanoid/application/settings.py": "5254c5b89d75d3eef52de8b5463db12801e8605cb02bff4d95551963171dc7dc",
    "CrowdSim/humanoid/application/state_prior_initialization.py": "9520b35ddc77ace422d245521a020b6ade8ff791818299a451c2f09dc1351b3b",
    "CrowdSim/humanoid/amp/observations.py": "a64dc2eba5c132cf68a248f2324ea20446dd264a2ea8f9f139faae4738f9aeba",
    "docs/style_improvement_20261004/social_policy_staging_v1/social_policy.py": "2a4007f732f0854d05b784debcbd0798e0c587149ba8b63d623378ad1ab3ac15",
    "docs/style_improvement_20261004/social_training_settings_staging_v1/social_training_settings.py": "ed55eafaf29eade986e00dbbce5320b46e6729c247e606887cdb9f0b9eece452",
    "docs/style_improvement_20261004/social_symmetry_staging_v1/social_symmetry.py": "b59303fbaddb50941db190ead55e3caca060747278cff31e95838d1433744756",
    "docs/style_improvement_20261004/social_task_observation_staging_v1/social_task_observation.py": "6975d95f369e2c00f1fa2ef34474e14164c36c82f70e4fb83a925335ee276ee1",
    "docs/style_improvement_20261004/social_raster_torch_staging_v1/social_raster_torch.py": "234e6366a00301998349dbc837fae5878cac4c506e29ad605ab8b2023a6c5dbe",
    "CrowdSim/humanoid/training/symmetry.py": "af8dfd1cc04f42c03ee538c16543455466dea7067c6e04c3c494c16ce3e14373",
}

# Each tuple has an exact occurrence count. The metadata/load admission hooks
# are additions; the original save/load and update numerical bodies survive.
DERIVATIONS = {
    "__init__": [("self.model = PaperAMPNetwork(config).to(self.device)",
                  "self.model = SocialAMPNetwork(config).to(self.device)", 1)],
    "update": [("from .symmetry import mirror_policy_observations, mirror_actions",
                "mirror_policy_observations, mirror_actions = _SOCIAL_MIRROR_FUNCTIONS", 1)],
    "save": [("\"navisaaclab_method_reconstruction_training\"", "FORMAT", 1),
             ("        temporary = path.with_name(path.name + \".tmp\")",
              "        state[\"social_training_contract\"] = self._checkpoint_contract()\n"
              "        self._validate_checkpoint(state)\n"
              "        temporary = path.with_name(path.name + \".tmp\")", 1)],
    "load": [("\"navisaaclab_method_reconstruction_training\"", "FORMAT", 1),
             ("        trainer = cls(TrainingConfig(**state[\"config\"]), device=device, provenance=state[\"provenance\"])",
              "        config = cls._admit_metadata(state)\n"
              "        trainer = cls(config, device=device, provenance=state[\"provenance\"])\n"
              "        trainer._validate_checkpoint(state)", 1)],
}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _record(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": _sha(data)}


def _verify_sources():
    for relative, expected in PINS.items():
        if _record(REPO / relative)["sha256"] != expected:
            raise ValueError(f"Social trainer source identity changed: {relative}")


def _owned_module(name, relative):
    path = (REPO / relative).resolve()
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path:
            raise ValueError(f"Foreign staged module already imported: {name}")
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses needs the actual owned module.
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


def _method_source(text, name):
    tree = ast.parse(text)
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PaperAMPTrainer")
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == name)
    start = min([method.lineno] + [node.lineno for node in method.decorator_list])
    lines = text.splitlines(keepends=True)[start - 1:method.end_lineno]
    return "".join(line[4:] if line.startswith("    ") else line for line in lines)


def derivation_records():
    """Pure source metadata; this function does not execute derived functions."""
    text = (REPO / "CrowdSim/humanoid/training/trainer.py").read_text()
    if _sha(text.encode()) != PINS["CrowdSim/humanoid/training/trainer.py"]:
        raise ValueError("Original trainer source changed before derivation")
    records = {}
    for name, edits in DERIVATIONS.items():
        original = _method_source(text, name)
        derived = original
        for before, after, count in edits:
            # Method extraction removes four leading spaces from every line.
            before = "\n".join(line[4:] if line.startswith("    ") else line for line in before.split("\n"))
            after = "\n".join(line[4:] if line.startswith("    ") else line for line in after.split("\n"))
            if derived.count(before) != count:
                raise ValueError(f"Bounded social derivation occurrence changed: {name}")
            derived = derived.replace(before, after)
        reverse = derived
        for before, after, count in reversed(edits):
            before = "\n".join(line[4:] if line.startswith("    ") else line for line in before.split("\n"))
            after = "\n".join(line[4:] if line.startswith("    ") else line for line in after.split("\n"))
            if reverse.count(after) != count:
                raise ValueError(f"Bounded inverse occurrence changed: {name}")
            reverse = reverse.replace(after, before)
        if reverse != original or ast.dump(ast.parse(reverse)) != ast.dump(ast.parse(original)):
            raise ValueError(f"Social derivation does not invert exactly: {name}")
        compile(derived, str(HERE / "social_trainer.py") + f"::{name}", "exec")
        records[name] = {"original_sha256": _sha(original.encode()), "derived_sha256": _sha(derived.encode()),
                         "replacement_counts": [row[2] for row in edits],
                         "inverse_bytes_equal": True, "inverse_AST_equal": True}
    return records


# Only a future import executes this section. Current source preparation uses
# compile and Ruff without importing this file or Torch.
_EXECUTION_ADAPTER_RECORD = _record(HERE / "social_trainer.py")
_verify_sources()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from CrowdSim.humanoid.training import trainer as _base  # noqa: E402

for relative in PINS:
    if relative.startswith("CrowdSim/"):
        module_name = relative.removesuffix(".py").replace("/", ".")
        module = sys.modules.get(module_name)
        if module is not None and Path(module.__file__).resolve() != (REPO / relative).resolve():
            raise ValueError(f"Foreign training dependency imported: {module_name}")

_settings = _owned_module("social_training_settings", "docs/style_improvement_20261004/social_training_settings_staging_v1/social_training_settings.py")
_policy = _owned_module("social_policy", "docs/style_improvement_20261004/social_policy_staging_v1/social_policy.py")
_symmetry = _owned_module("social_symmetry", "docs/style_improvement_20261004/social_symmetry_staging_v1/social_symmetry.py")
SocialTrainingConfig = _settings.SocialTrainingConfig
torch = _base.torch
np = _base.np
_SOCIAL_MIRROR_FUNCTIONS = (_symmetry.mirror_raw_observations, _symmetry.mirror_actions)


class SocialAMPNetwork(_policy.SocialAMPActorCritic):
    """New CNN actor/critic plus exactly the original state-action D2019."""

    def __init__(self, config):
        if config.discriminator_input != "state_action":
            raise ValueError("Social stage requires state_action D2019; state-only is a separate experiment")
        super().__init__(task_encoder_sharing="shared")
        self.discriminator_input = config.discriminator_input
        self.discriminator = _base._mlp(2019, config.discriminator_layers, 1)

    disc_logits = _base.PaperAMPNetwork.disc_logits


_DERIVATION_RECORDS = derivation_records()


def _derive(name):
    text = (REPO / "CrowdSim/humanoid/training/trainer.py").read_text()
    if _sha(text.encode()) != PINS["CrowdSim/humanoid/training/trainer.py"]:
        raise ValueError("Original trainer changed before function compilation")
    source = _method_source(text, name)
    for before, after, count in DERIVATIONS[name]:
        before = "\n".join(line[4:] if line.startswith("    ") else line for line in before.split("\n"))
        after = "\n".join(line[4:] if line.startswith("    ") else line for line in after.split("\n"))
        if source.count(before) != count:
            raise ValueError(f"Unexpected social derivation count: {name}")
        source = source.replace(before, after)
    if _sha(source.encode()) != _DERIVATION_RECORDS[name]["derived_sha256"]:
        raise ValueError("Derived function source differs from its declared metadata")
    namespace = dict(vars(_base))
    namespace.update(SocialAMPNetwork=SocialAMPNetwork, FORMAT=FORMAT,
                     _SOCIAL_MIRROR_FUNCTIONS=_SOCIAL_MIRROR_FUNCTIONS, __name__=__name__)
    exec(compile(source, str(HERE / "social_trainer.py") + f"::{name}", "exec"), namespace)
    return namespace[name]


_original_init_with_social_factory = _derive("__init__")
_original_update_with_social_mirror = _derive("update")


def _equal_json(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def _contract(config):
    _verify_sources()
    if _record(HERE / "social_trainer.py") != _EXECUTION_ADAPTER_RECORD:
        raise ValueError("Social trainer file changed after this execution adapter was imported")
    amp_dim = 195 * (10 if config.amp_normalization == "per_history_coordinate" else 1)
    return {"schema": CONTRACT_SCHEMA, "configuration_document": config.to_document(),
            "architecture": _policy.architecture_contract(),
            "sources": [_record(REPO / relative) for relative in PINS],
            "execution_adapter": dict(_EXECUTION_ADAPTER_RECORD),
            "derivations": _DERIVATION_RECORDS,
            "discriminator": {"input": "state_action", "history": [10, 195], "actions": 69, "input_dimensions": 2019},
            "AMP_RMS_dimensions": amp_dim,
            "runner_state_contract": "opaque caller supplied dictionary or None; physical restoration not implemented"}


class SocialAMPTrainer(_base.PaperAMPTrainer):
    """Training mathematics connected to new inputs, with strict new resumes.

    Supplied runner state is opaque. This class cannot prove SDK simulation,
    group reset or physical restoration from merely receiving a dictionary.
    """

    def __init__(self, config, *, device="cpu", provenance=None):
        if type(config) is not SocialTrainingConfig:
            raise TypeError("An explicit new SocialTrainingConfig is required")
        config = config.detached_copy()
        if config.discriminator_input != "state_action":
            raise ValueError("This social trainer requires state_action D2019")
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("Original discriminator initialization requires the declared float32 default dtype")
        supplied = dict(provenance or {})
        initialization = {"strategy": "fresh_local_social_CNN_and_state_action_D", "seed": config.seed,
                          "weights_or_moments_transferred": False, "author_navisaaclab_weights": False}
        if "initialization" in supplied and not _equal_json(supplied["initialization"], initialization):
            raise ValueError("Social fresh initialization cannot inherit a legacy warm-start provenance")
        supplied["initialization"] = initialization
        _original_init_with_social_factory(self, config, device=device, provenance=supplied)

    @classmethod
    def prepare(cls, config, **kwargs):
        return cls(config, **kwargs)

    def warm_start_pacer(self, *args, **kwargs):
        raise ValueError("Social12308/633 architecture is fresh only; no old1044 warm start or strict=False migration")

    def update(self, rollout, expert_history, expert_actions):
        rewards = rollout.get("rewards")
        if not isinstance(rewards, torch.Tensor) or rewards.ndim != 2 or rewards.shape[1] != self.config.actor_count:
            raise ValueError("Social rollout actor count differs from the explicitly declared flat batch")
        return _original_update_with_social_mirror(self, rollout, expert_history, expert_actions)

    def _checkpoint_contract(self):
        return _contract(self.config)

    @classmethod
    def _admit_metadata(cls, state):
        """Called by derived load after format validation, before NN construction."""
        _verify_sources()
        if (type(state) is not dict or set(state) != CHECKPOINT_KEYS or type(state["format"]) is not str
                or state["format"] != FORMAT or type(state["version"]) is not int or state["version"] != 1
                or type(state["provenance"]) is not dict):
            raise ValueError("Unknown/incomplete social checkpoint schema before network construction")
        contract = state.get("social_training_contract")
        if type(contract) is not dict or contract.get("schema") != CONTRACT_SCHEMA:
            raise ValueError("Complete new social configuration/source contract is missing")
        config = SocialTrainingConfig.from_document(contract.get("configuration_document"))
        if config.discriminator_input != "state_action" or not _equal_json(state.get("config"), asdict(config)):
            raise ValueError("Saved social config/D shape disagrees with its complete declaration")
        if not _equal_json(contract, _contract(config)):
            raise ValueError("Social architecture/config/source derivation changed; no implicit migration")
        return config

    @staticmethod
    def _tensor(value, expected, label):
        if (not isinstance(value, torch.Tensor) or value.shape != expected.shape or value.dtype != expected.dtype
                or value.device.type == "meta" or not bool(torch.isfinite(value).all())):
            raise ValueError(f"Invalid full social tensor: {label}")

    def _optimizer(self, saved, optimizer, label, updates):
        expected = optimizer.state_dict()
        if type(saved) is not dict or set(saved) != {"state", "param_groups"} or type(saved["state"]) is not dict:
            raise ValueError(f"Invalid Adam schema: {label}")
        if type(saved["param_groups"]) is not list or len(saved["param_groups"]) != len(expected["param_groups"]):
            raise ValueError(f"Invalid Adam parameter groups: {label}")
        parameters = {}
        for actual, declared, live in zip(saved["param_groups"], expected["param_groups"], optimizer.param_groups, strict=True):
            if (type(actual) is not dict or set(actual) != set(declared)
                    or any(type(actual[key]) is not type(value) or actual[key] != value for key, value in declared.items())):
                raise ValueError(f"Adam parameter identities/options changed: {label}")
            parameters.update(zip(declared["params"], live["params"], strict=True))
        if any(type(key) is not int for key in saved["state"]) or not set(saved["state"]) <= set(parameters):
            raise ValueError(f"Unknown Adam parameter state: {label}")
        if (updates == 0 and saved["state"]) or (updates > 0 and set(saved["state"]) != set(parameters)):
            raise ValueError(f"Incomplete Adam state for update count: {label}")
        for key, values in saved["state"].items():
            if type(values) is not dict or set(values) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError(f"Incomplete Adam moments: {label}")
            for field in ("exp_avg", "exp_avg_sq"):
                self._tensor(values[field], parameters[key], f"{label}.{key}.{field}")
            step = values["step"]
            if (not isinstance(step, torch.Tensor) or step.shape != () or step.dtype != torch.float32
                    or not bool(torch.isfinite(step)) or float(step) < 1 or float(step) != int(float(step))
                    or bool((values["exp_avg_sq"] < 0).any())):
                raise ValueError(f"Invalid Adam step/second moment: {label}")

    def _validate_checkpoint(self, state):
        self._admit_metadata(state)
        if type(state["provenance"]) is not dict or not _equal_json(state["provenance"], self.provenance):
            raise ValueError("Saved social fresh provenance changed")
        updates, transitions = state["update_count"], state["physical_transition_count"]
        if (type(updates) is not int or type(transitions) is not int or min(updates, transitions) < 0
                or transitions % self.config.actor_count or (updates == 0) != (transitions == 0)):
            raise ValueError("Invalid social update/physical transition counters")
        model = self.model.state_dict()
        if not isinstance(state["model"], dict):
            raise ValueError("Complete social model state is required")
        if set(state["model"]) != set(model):
            raise ValueError("Social model keys differ; old1044/partial migration is forbidden")
        for key, value in state["model"].items():
            self._tensor(value, model[key], "model." + key)
        if not torch.equal(state["model"]["log_std"], torch.full_like(model["log_std"], self.config.log_std)):
            raise ValueError("Fixed social Gaussian log_std changed")
        normalizers = state["normalizers"]
        names = {"self_normalizer", "task_normalizer", "amp_normalizer", "value_normalizer"}
        if type(normalizers) is not dict or set(normalizers) != names:
            raise ValueError("Social resume requires exactly four RMS states")
        for name in names:
            expected = getattr(self, name).state_dict()
            actual = normalizers[name]
            if not isinstance(actual, dict):
                raise ValueError("Social RMS state must be a complete dictionary")
            if set(actual) != set(expected):
                raise ValueError("Social RMS fields differ")
            for key, value in actual.items():
                self._tensor(value, expected[key], name + "." + key)
            if bool((actual["variance"] < 0).any()) or float(actual["count"]) < 0:
                raise ValueError("Social RMS variance/count must be nonnegative")
            if name in {"self_normalizer", "task_normalizer", "value_normalizer"} and float(actual["count"]) != transitions:
                raise ValueError("Fresh social self/task/value RMS count must equal physical transitions")
        self._optimizer(state["optimizer"], self.optimizer, "policy", updates)
        self._optimizer(state["discriminator_optimizer"], self.discriminator_optimizer, "discriminator", updates)
        replay = state["policy_replay"]
        if self.policy_replay is None:
            if replay is not None:
                raise ValueError("Replay state is incompatible with disabled replay")
        else:
            fields = {"capacity", "history_shape", "size", "cursor", "accepted", "history", "actions"}
            if type(replay) is not dict or set(replay) != fields:
                raise ValueError("Complete social policy replay is required")
            size, cursor, accepted = replay["size"], replay["cursor"], replay["accepted"]
            if (type(replay["capacity"]) is not int or replay["capacity"] != self.policy_replay.capacity
                    or tuple(replay["history_shape"]) != (10, 195)
                    or any(type(value) is not int for value in (size, cursor, accepted))
                    or not 0 <= size <= self.policy_replay.capacity or not 0 <= cursor < self.policy_replay.capacity
                    or size != min(self.policy_replay.capacity, accepted)
                    or size < self.policy_replay.capacity and (cursor != size or accepted != size)):
                raise ValueError("Social replay configuration/counters differ")
            self._tensor(replay["history"], self.policy_replay.history[:size], "replay.history")
            self._tensor(replay["actions"], self.policy_replay.actions[:size], "replay.actions")
            if bool((replay["actions"].abs() > 1 + 1e-6).any()) or bool((replay["actions"] * (1 - self.action_mask)).any()):
                raise ValueError("Saved replay violates normalized/frozen PD action columns")
        rng = state["rng"]
        if type(rng) is not dict or set(rng) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
            raise ValueError("Complete Python/NumPy/CPU/CUDA RNG structure is required")
        random.Random(0).setstate(rng["python"])  # private state validation, no global draw.
        ns = rng["numpy"]
        if (type(ns) is not list or len(ns) != 5 or ns[0] != "MT19937" or type(ns[1]) is not list or len(ns[1]) != 624
                or any(type(value) is not int or not 0 <= value < 2 ** 32 for value in ns[1])
                or type(ns[2]) is not int or not 0 <= ns[2] <= 624 or type(ns[3]) is not int or ns[3] not in (0, 1)
                or type(ns[4]) is not float or not math.isfinite(ns[4])):
            raise ValueError("Invalid complete NumPy RNG state")
        np.random.RandomState(0).set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
        if type(rng["torch_cuda"]) is not list:
            raise ValueError("Saved CUDA RNG must be an explicit list")
        for value in [rng["torch_cpu"], *rng["torch_cuda"]]:
            if not isinstance(value, torch.Tensor) or value.ndim != 1 or value.dtype != torch.uint8 or not value.numel() or value.device.type == "meta":
                raise ValueError("Invalid materialized Torch RNG state")
        torch.Generator(device="cpu").set_state(rng["torch_cpu"].cpu())
        # CUDA generator bytes are saved, structurally checked, and restored by
        # the original restore_rng only on a CUDA host; no CPU restore claim.
        if (state["runner_state"] is not None and type(state["runner_state"]) is not dict
                or type(state["physical_runner_state_included"]) is not bool
                or state["physical_runner_state_included"] is not (state["runner_state"] is not None)):
            raise ValueError("Opaque runner state/inclusion flag disagree")

    save = _derive("save")
    load = _derive("load")
