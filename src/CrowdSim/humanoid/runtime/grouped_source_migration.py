"""Explicit source-bound Lab 3 EA ports of the sealed grouped training modules.

Only caller-owned Python namespaces are derived. Historical files and SDK
classes remain untouched. Source admission is not physical runtime evidence.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

from protomotions.repository import repository_root
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record

ROOT = repository_root(__file__)
MANIFEST = ROOT / "configs/compatibility/grouped_social_sim61_sources.json"
MANIFEST_SHA = "7b213a6a91c084b0ba0f35fb40823b85fab527343330acd04fc256ce50918baf"


def manifest():
    if record(MANIFEST)["sha256"] != MANIFEST_SHA or MANIFEST.stat().st_mode & 0o222:
        raise ValueError("Grouped Sim61 source registration changed")
    from .terminal_cache_source_migration import admit_dependency
    inventory = admit_dependency(json.loads(MANIFEST.read_text()))
    inventory['dependencies'] = {str(_path(key)): row for key, row in inventory['dependencies'].items()}
    return inventory


def _path(value):
    path = Path(value)
    if path.is_absolute():
        from social_navigation.artifact_identity import artifact_path
        from social_navigation.sdk_paths import relocate_sdk_path
        return artifact_path(relocate_sdk_path(path))
    return ROOT / path


def _check(row):
    path = _path(row["path"])
    item = record(path)
    if (item["bytes"], item["sha256"]) != (row["bytes"], row["sha256"]):
        raise ValueError("Grouped Sim61 dependency changed: " + str(path))
    return path


def describe():
    inventory = manifest()
    for row in inventory["dependencies"].values():
        _check(row)
    for row in inventory["modules"].values():
        _check(row["source"])
    return {"schema": "shared_social.grouped_source_migration.sim61.v2",
            "adapter": record(__file__), "registration": record(MANIFEST),
            "lifecycle": record(Path(__file__).with_name("grouped_sdk61_lifecycle.py")),
            "terminal_cache_migration": inventory["terminal_cache_migration"],
            "SDK_source_modified": False, "runtime_executed": False,
            "old_lifecycle_certificate_reused": False,
            "root_link_velocity_via_com_api": True}


def pins_for(path, pins=None):
    """Check each old declaration before selecting its reviewed current source."""
    inventory = manifest()
    key = str(Path(path).resolve().relative_to(ROOT))
    spec = inventory["modules"][key]
    _check(spec["source"])
    if pins is None:
        tree = ast.parse(_path(spec["source"]["path"]).read_bytes())
        nodes = [node for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "PINS" for t in node.targets)]
        scope = {"ROOT": ROOT, "STAGES": ROOT / "docs/style_improvement_20261004",
                 "LAB": Path(spec["lab"]), "API": Path(spec.get("api", "/unused")), "Path": Path}
        if len(nodes) != 1:
            raise ValueError("Require the sealed grouped PINS declaration")
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
        pins = scope["PINS"]
    if set(pins) != set(spec["pins"]):
        raise ValueError("Grouped dependency role set changed")
    result = {}
    for name, original in spec["pins"].items():
        pin = pins[name]
        # Episode consumers explicitly select another sealed World before use.
        selected_world = name == "world" and key.endswith(("expert_batch.py", "social_rollout_collector.py"))
        allowed = [original]
        if selected_world:
            allowed += [row[:len(original)] for row in inventory["world_choices"]]
        actual = [str(pin[0]), *pin[1:]]
        def equivalent(expected):
            return (_path(actual[0]).resolve() == _path(expected[0]).resolve()
                    and len(actual) == len(expected) and actual[1:] == expected[1:])
        if not any(equivalent(expected) for expected in allowed):
            raise ValueError("Unreviewed grouped dependency declaration: " + name)
        identity = inventory["dependencies"][str(_path(pin[0]).resolve())]
        current = _check(identity)
        # Relative-string declarations are used by the expert's ROOT / pin API.
        target = str(current.relative_to(ROOT)) if isinstance(pin[0], str) else current
        result[name] = (target, identity["bytes"], identity["sha256"], *pin[3:])
    return result


def _replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError("Grouped migration seam changed: " + old)
    return source.replace(old, new, 1)


def load(path, name, identity):
    """Load one reviewed source and attach only that module's runtime adapter."""
    path = Path(path).resolve()
    data = path.read_bytes()
    actual = {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    if actual != identity:
        raise ValueError("Grouped source changed before execution")
    if name in sys.modules:
        raise ValueError("Refuse a previously imported entry-owned module: " + name)
    spec = manifest()["modules"].get(str(path.relative_to(ROOT)))
    source = data.decode()
    if spec is not None:
        if (len(data), actual["sha256"]) != (spec["source"]["bytes"], spec["source"]["sha256"]):
            raise ValueError("Grouped module differs from its sealed source")
        if path.name == "shared_group_world.py":
            source = _replace(source, "            modules[key] = module\n",
                              "            _sim61_child(module)\n            modules[key] = module\n")
            source = _replace(source, "            from isaaclab_physx.physics import PhysxCfg\n",
                              "            from isaaclab_physx.physics import PhysxCfg\n            _sim61_verify_sdk()\n")
        if path.name == "expert_batch.py":
            source = _replace(source, "        sources = {}\n",
                              "        global PINS\n        PINS = _sim61_pins(PINS)\n        sources = {}\n")
        if path.name == "social_rollout_collector.py":
            source = _replace(source, "    global _DEPS\n",
                              "    global _DEPS, PINS\n")
            source = _replace(source, "    if _DEPS is None:\n",
                              "    if _DEPS is None:\n        PINS = _sim61_pins(PINS)\n")
    module = ModuleType(name)
    module.__file__, module.__package__ = str(path), name.rpartition(".")[0]
    module._sim61_child = adapt
    module._sim61_pins = lambda pins: pins_for(path, pins)
    module._sim61_verify_sdk = verify_sdk
    sys.modules[name] = module
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
        if spec is not None:
            adapt(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def verify_sdk():
    from isaaclab_physx.assets.articulation import Articulation

    method = Articulation.set_joint_position_target_index
    path = Path(method.__code__.co_filename).resolve()
    expected = "/project/EXP/IsaacLab/source/isaaclab/isaaclab/assets/articulation/base_articulation.py"
    identity = manifest()["dependencies"][str(_path(expected))]
    if path != _check(identity):
        raise ValueError("Grouped joint target writer belongs to another SDK source")


def adapt(module):
    path = Path(module.__file__).resolve()
    children = {
        "social_net_normal_sampler_staging_v1/net_normal_sampler.py":
            "04d21596a8f1fd54d9d9a43977c8a76137840a574fb4f04a73ed4ce94beb5d09",
        "nonfoot_net_normal_contact_staging_v1/nonfoot_contact.py":
            "9fa82a7771a2c875e1cc41610b0203e8c3d235a9c50f7668f78665ae373a863a",
    }
    for relative, digest in children.items():
        if path == ROOT / "docs/style_improvement_20261004" / relative:
            original = path.read_bytes()
            if hashlib.sha256(original).hexdigest() != digest or path.stat().st_mode & 0o222:
                raise ValueError("Grouped contact/sampler source changed")
            names = manifest()["dependencies"][str(ROOT / "src/CrowdSim/humanoid/amp/observations.py")]
            _check(names)
            source = _replace(original.decode(), "10438", str(names["bytes"]))
            source = _replace(source, "a64dc2eba5c132cf68a248f2324ea20446dd264a2ea8f9f139faae4738f9aeba", names["sha256"])
            exec(compile(source, str(path), "exec"), module.__dict__)
            return module
    spec = manifest()["modules"].get(str(path.relative_to(ROOT)))
    if spec is None:
        return module
    _check(spec["source"])
    if path.name == "state_adapter.py":
        # The reviewed SDK link writer submits its link buffer to the native
        # COM setter. Reuse the same explicit correction as the main backend.
        helper = ROOT / "src/protomotions/simulator/isaaclab/utils/runtime_compat.py"
        _check(manifest()["dependencies"][str(helper)])
        from protomotions.simulator.isaaclab.utils.runtime_compat import write_root_link_velocity
        if Path(write_root_link_velocity.__code__.co_filename).resolve() != helper.resolve():
            raise ValueError("Grouped velocity writer belongs to another source")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != spec["source"]["sha256"]:
            raise ValueError("Grouped state adapter changed before derivation")
        source = _replace(data.decode(),
            "asset.write_root_link_velocity_to_sim_index(root_velocity=velocity, env_ids=ids.clone(), full_data=False)",
            "_sim61_write_root_link_velocity(asset, velocity, env_ids=ids.clone())")
        source = _replace(source, '"write_root_link_velocity_to_sim_index",',
                          '"write_root_com_velocity_to_sim_index",')
        module._sim61_write_root_link_velocity = write_root_link_velocity
        exec(compile(source, str(path) + ":sim61", "exec"), module.__dict__)
    if path.name not in ("expert_batch.py", "social_rollout_collector.py"):
        module.PINS = pins_for(path, module.PINS)
    if path.name == "probe_shared_world.py":
        module._load = load
        module._verify_lifecycle_owner = lambda records: describe()
        from .grouped_sdk61_lifecycle import cleanup_owned_runtime
        module._cleanup_owned_runtime = cleanup_owned_runtime
    return module


def adapt_entry(module):
    """Bind current sources and preserve the original 120 Hz implicit PD mode."""
    adapt(module)
    module._verify_lifecycle_owner = lambda records: describe()
    setup = module._runner_setup

    def runner_setup(*args):
        values = setup(*args)
        values[-1].use_newton_actuators = False
        values[-1].physics.enable_external_forces_every_iteration = False
        return values

    module._runner_setup = runner_setup
    return module
