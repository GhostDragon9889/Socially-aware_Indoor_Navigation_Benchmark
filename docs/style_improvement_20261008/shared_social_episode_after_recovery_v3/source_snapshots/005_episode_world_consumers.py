"""Bind private expert/collector namespaces to the reviewed episode World.

The source-bound consumers originally admitted only the periodic-reference
World. Their equations, guards and classes are unchanged: this adapter selects
one separately pinned World before either consumer is constructed. Import and
source inspection are cold and never load Torch or an SDK.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
from pathlib import Path
import stat

from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record
from scripts import train_shared_social_episode_reference as candidate

ROOT = candidate.ROOT
HERE = ROOT / "docs/style_improvement_20261004/social_world_lifetime_dependency_migration_v1"
PINS = {
    "expert": (HERE / "expert_batch.py", 15295,
               "029697b3c44c2617515cb9239b2cf075fdd5d2bf62ee9a5161fbb744afd5f57f"),
    "collector": (HERE / "social_rollout_collector.py", 27268,
                  "4497ed99f0424a33148a31c137d2c7f705acd9a72f32b8eae6e66c833b321898"),
}
OLD_WORLD = ROOT / "docs/style_improvement_20261004/shared_group_world_owned_lifetime_revision_v2/shared_group_world.py"
METHOD_PINS = {
    "prepare_control": "0506a5e39ce2ea249af26ab8d98ffaeff652721e04253889197cfc3296c22c16",
    "step": "57fb3d8c04b43d3b54a67f923ea38bb8364127f4d7c5e0fa0c9bf9bb950f8c8b",
    "complete_transition": "699b3216e1d7b9d5397c8427138c2ad59c885e0378f42372a0bd7291267dfd73",
}


def _source(path, size, sha):
    item = record(path)
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_mode & 0o222
            or item["bytes"] != size or item["sha256"] != sha):
        raise ValueError("Exact readonly consumer/World source required: " + str(path))
    return item


def _assignment(path, name):
    statements = [node for node in ast.parse(path.read_bytes()).body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    if len(statements) != 1:
        raise ValueError("Require one exact source declaration: " + name)
    namespace = {"ROOT": ROOT, "STAGES": ROOT / "docs/style_improvement_20261004"}
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def describe():
    old = _source(OLD_WORLD, 39943, candidate.PARENT_WORLD_SHA)
    world = _source(candidate.WORLD, 40596, candidate.WORLD_SHA)
    consumers = {key: _source(*pin) for key, pin in PINS.items()}
    klass = next(node for node in ast.parse(candidate.WORLD.read_bytes()).body
                 if isinstance(node, ast.ClassDef) and node.name == "SharedGroupWorld")
    methods = {node.name: node for node in klass.body if isinstance(node, ast.FunctionDef)}
    actual = {name: hashlib.sha256(ast.dump(methods[name], include_attributes=False).encode()).hexdigest()
              for name in METHOD_PINS}
    if actual != METHOD_PINS or _assignment(PINS["collector"][0], "WORLD_METHOD_AST_PINS") != METHOD_PINS:
        raise ValueError("Episode World must preserve all original collector control/step/reset method ASTs")
    return {"schema": "shared_social.episode_world_consumer_admission.v1", "adapter": record(__file__),
            "original_World": old, "selected_World": world, "consumer_sources": consumers,
            "unchanged_World_method_ASTs": actual, "private_namespace_world_pin_changed": True,
            "origin_guards_disabled": False, "expert_sampling_math_changed": False,
            "collector_reward_control_history_math_changed": False,
            "original_consumer_files_modified": False}


def adapt(loaded, key):
    binding = describe()
    if key not in PINS or Path(getattr(loaded, "__file__", "")).resolve() != PINS[key][0]:
        raise ValueError("Only the actual pinned private expert/collector module can be adapted")
    expected = _assignment(PINS[key][0], "PINS")
    if loaded.PINS != expected:
        raise ValueError("Consumer dependency admission already changed")
    if key == "collector":
        if loaded._DEPS is not None or loaded.WORLD_METHOD_AST_PINS != METHOD_PINS:
            raise ValueError("Adapt the unused collector before dependency admission/construction")
        method = loaded.SocialRolloutCollector.__init__
        world_pin = (candidate.WORLD, 40596, candidate.WORLD_SHA, True)
    else:
        method = loaded.SocialAMASSExpert.__init__
        world_pin = (str(candidate.WORLD.relative_to(ROOT)), 40596, candidate.WORLD_SHA)
    if method.__globals__ is not loaded.__dict__:
        raise ValueError("Require the actual original consumer class namespace")
    loaded.PINS = {**expected, "world": world_pin}
    loaded.EPISODE_WORLD_ADMISSION = deepcopy(binding)
    return loaded
