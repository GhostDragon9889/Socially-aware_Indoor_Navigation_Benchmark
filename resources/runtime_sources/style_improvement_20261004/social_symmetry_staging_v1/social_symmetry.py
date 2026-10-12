"""Coupled raw self357/task12308 and canonical action69 sagittal reflection."""
from __future__ import annotations

import ast
import __future__
import hashlib
from pathlib import Path
import stat
from types import ModuleType

import torch

from protomotions.repository import repository_root
ROOT = repository_root(__file__)
PINS = {
    "symmetry": (ROOT / "CrowdSim/humanoid/training/symmetry.py", 4897, "af8dfd1cc04f42c03ee538c16543455466dea7067c6e04c3c494c16ce3e14373"),
    "observations": (ROOT / "CrowdSim/humanoid/amp/observations.py", 10586, "a64dc2eba5c132cf68a248f2324ea20446dd264a2ea8f9f139faae4738f9aeba"),
    "task": (ROOT / "resources/runtime_sources/style_improvement_20261004/social_task_observation_staging_v1/social_task_observation.py", 8512, '6975d95f369e2c00f1fa2ef34474e14164c36c82f70e4fb83a925335ee276ee1'),
}
DERIVED_AST_SHA256 = "1ba13634dd6649781bcff160c861de4058dd300721f713ca426894fc846eee86"
_DEPS = None


def _derive(tree):
    """Keep original body statements2,4..16; replace return and remove task1044."""
    funcs = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    original = funcs["mirror_policy_observations"]
    if len(original.body) != 20 or not isinstance(original.body[16], ast.Assign):
        raise ValueError("Original self derivation structure mismatch")
    helpers = [funcs[name] for name in ("_counterpart", "_name_permutation", "_validate_tensor", "mirror_actions")]
    kept = [original.body[2], *original.body[4:17]]
    if any(isinstance(node, ast.Name) and node.id == "task_obs" for statement in kept for node in ast.walk(statement)):
        raise ValueError("Task1044 must not remain in the self adaptation")
    args = ast.arguments(posonlyargs=[], args=original.args.args[:1], vararg=None,
                         kwonlyargs=original.args.kwonlyargs, kw_defaults=original.args.kw_defaults,
                         kwarg=None, defaults=[])
    adapted = ast.FunctionDef(name="_mirror_raw_self_source_adaptation", args=args,
                             body=kept+[ast.Return(value=ast.Name(id="mirrored_self", ctx=ast.Load()))],
                             decorator_list=[], returns=ast.Name(id="Tensor", ctx=ast.Load()))
    return ast.fix_missing_locations(ast.Module(body=helpers+[adapted], type_ignores=[]))


def _dependencies():
    global _DEPS
    if _DEPS is None:
        sources = {}
        for key, (path, size, sha) in PINS.items():
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or (key == "task" and info.st_mode & 0o222):
                raise ValueError("Invalid source: " + key)
            source = path.read_bytes()
            if len(source) != size or hashlib.sha256(source).hexdigest() != sha:
                raise ValueError("Source byte identity mismatch: " + key)
            sources[key] = source
        constants = ast.parse(sources["observations"])
        names = [node for node in constants.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id in ("PACER_BODY_NAMES", "PACER_DOF_NAMES") for target in node.targets)]
        if len(names) != 2:
            raise ValueError("Canonical PACER name declarations missing")
        original = ModuleType("_original_symmetry_with_declared_self_adaptation")
        original.__dict__.update(torch=torch, Tensor=torch.Tensor)
        exec(compile(ast.Module(body=names, type_ignores=[]), str(PINS["observations"][0]), "exec"), original.__dict__)
        if len(original.PACER_BODY_NAMES) != 24 or original.PACER_BODY_NAMES[0] != "Pelvis" or len(set(original.PACER_BODY_NAMES)) != 24 or len(set(original.PACER_DOF_NAMES)) != 69:
            raise ValueError("Canonical body/DOF names invalid")
        derived = _derive(ast.parse(sources["symmetry"]))
        identity = hashlib.sha256(ast.dump(derived, include_attributes=False).encode()).hexdigest()
        if identity != DERIVED_AST_SHA256:
            raise ValueError("Declared original self/action derivation identity mismatch")
        exec(compile(derived, str(PINS["symmetry"][0]), "exec", flags=__future__.annotations.compiler_flag), original.__dict__)
        task = ModuleType("_frozen_raw_social_task_mirror")
        task.__file__ = str(PINS["task"][0])
        exec(compile(sources["task"], task.__file__, "exec"), task.__dict__)
        _DEPS = original, task
    return _DEPS


def _tensor(value, dimension, name, device=None):
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] != dimension or not len(value):
        raise ValueError(name + " requires nonempty Tensor[B," + str(dimension) + "]")
    if value.dtype != torch.float32 or value.device.type == "meta" or not bool(torch.isfinite(value).all()):
        raise ValueError(name + " requires finite materialized float32")
    target = value.device if device is None else torch.device(device)
    if target.type == "cpu":
        target = torch.device("cpu")
    if target.type not in ("cpu", "cuda") or (target.type == "cuda" and target.index is None) or value.device != target:
        raise ValueError("Tensors must already share the declared CPU/indexed CUDA device")
    return target


def mirror_raw_observations(self_obs, task_obs, *, device=None):
    """Mirror canonical raw observations together, before the same external RMS."""
    target = _tensor(self_obs, 357, "self_obs", device)
    _tensor(task_obs, 12308, "task_obs", target)
    if len(self_obs) != len(task_obs):
        raise ValueError("Coupled self/task batch mismatch")
    original, task = _dependencies()
    return original._mirror_raw_self_source_adaptation(self_obs), task.mirror_raw_task(task_obs, device=target)


def mirror_actions(actions, *, device=None):
    """Reuse exact original canonical action reflection; no clipping, mask or detach."""
    _tensor(actions, 69, "actions", device)
    original, _ = _dependencies()
    return original.mirror_actions(actions)


def canonical_action_mask(*, device):
    """Hand/toe four triples frozen; no alternate57 subset is admitted."""
    original, _ = _dependencies()
    target = torch.device(device)
    if target.type == "cpu":
        target = torch.device("cpu")
    if target.type not in ("cpu", "cuda") or (target.type == "cuda" and target.index is None):
        raise ValueError("Explicit CPU/indexed CUDA device required")
    values = tuple(int(name.rsplit("_", 1)[0] not in {"L_Hand", "R_Hand", "L_Toe", "R_Toe"}) for name in original.PACER_DOF_NAMES)
    if sum(values) != 57:
        raise ValueError("Expected canonical57 effective coordinates")
    mask = torch.tensor(values, dtype=torch.float32, device=target)
    if not torch.equal(original.mirror_actions(mask[None]).abs()[0], mask):
        raise ValueError("Canonical physical mask is not mirror invariant")
    return mask


def effective_symmetry_loss(mean, mirrored_mean, action_mask, *, device=None):
    """Unweighted mean masked squared residual/57; both mean gradients are retained."""
    target = _tensor(mean, 69, "mean", device)
    _tensor(mirrored_mean, 69, "mirrored_mean", target)
    if mean.shape != mirrored_mean.shape:
        raise ValueError("Policy mean batches must match")
    expected = canonical_action_mask(device=target)
    if not isinstance(action_mask, torch.Tensor) or action_mask.shape != (69,) or action_mask.device != target or action_mask.dtype != torch.float32 or not torch.equal(action_mask, expected):
        raise ValueError("Exact canonical57-active float32 mask required")
    residual = mirrored_mean-mirror_actions(mean, device=target)
    if not bool(torch.isfinite(residual).all()):
        raise ValueError("Nonfinite symmetry residual")
    loss = (residual*action_mask).square().sum(-1).mean()/action_mask.sum()
    if not bool(torch.isfinite(loss)):
        raise ValueError("Nonfinite symmetry loss")
    return loss
