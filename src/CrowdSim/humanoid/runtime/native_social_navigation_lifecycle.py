"""Prepared terminal ownership for N3..15 in one original native AMP World.

The current N4 release owns one articulation and24 sensors. The native variant
owns N peer views plus one floor view, and clears borrowed provider/core aliases.
This adapter requires a separate actual native SDK smoke before full evaluation.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys
import textwrap
import weakref

from .social_evaluation_lifecycle import ROOT, record
from .social_navigation_lifecycle import FourActorNavigationLifecycle
from . import native_social_observation_producer

PARENT = ROOT / "CrowdSim/humanoid/runtime/social_navigation_lifecycle.py"
PARENT_SHA = "beb3973d1899a9b5b53ffbebc806097e10d198418f9c6bbc111ffd49f6225319"
NATIVE_PINS = {
    "CrowdSim.humanoid.runtime.dynbench_scene": ("CrowdSim/humanoid/runtime/dynbench_scene.py", "fd8f3cb6e84776af6b6bc7f6e5d527b24284dad2086ca23dc5c89d4f8ebe3870"),
    "CrowdSim.humanoid.runtime.native_grid_scene": ("CrowdSim/humanoid/runtime/native_grid_scene.py", "3fe1090b9c1e457c4c7ca00d4dbc3c07d7ab48ef1a5d1973ed8ae9d4f520881a"),
}
SDK61_NATIVE_PINS = {
    "CrowdSim.humanoid.runtime.dynbench_scene": (NATIVE_PINS["CrowdSim.humanoid.runtime.dynbench_scene"][0],
        "68bd2171ad9707c3f280c374d73538b8c532764a42e1f90335c6375090595d28"),
    "CrowdSim.humanoid.runtime.native_grid_scene": (NATIVE_PINS["CrowdSim.humanoid.runtime.native_grid_scene"][0],
        "b5563bd5a509cdb02ae33e57ad34ab7a0524825ad16e481ef631531d59308165"),
}

# This constructor-only branch leaves every sealed terminal method byte intact.
SDK61_PARENT_BRANCH = '''        if self._sdk61 is not None:
            # SDK61Lifecycle validates the loaded backend and releases contacts
            # before stop; the sealed Sim6.0 sensor teardown must not run here.
            report["evaluation_lifecycle"]["navigation_release_version"] = "sdk61_owned_contacts.v1"
            return
'''


def derive_native_release(source):
    historical = source
    if historical.count(SDK61_PARENT_BRANCH) == 1:
        historical = historical.replace(SDK61_PARENT_BRANCH, "", 1)
    if hashlib.sha256(historical.encode()).hexdigest() != PARENT_SHA:
        raise ValueError("Require the exact current N4 terminal lifecycle")
    cls = next(node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == "FourActorNavigationLifecycle")
    original = "\n\n".join(textwrap.dedent(ast.get_source_segment(source, node)) for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_detach_world", "_verify_contacts"}) + "\n"
    producer_old = ("            if producer is not None:\n"
                    "                if producer.world is not world or producer.sim is not simulator._sim:\n"
                    "                    raise RuntimeError(\"Navigation producer does not belong to this World\")\n"
                    "                producer.world = producer.sim = producer.sampler = None\n")
    replacements = [
        ("world.num_envs != 4", "world.num_envs != self.actor_count"),
        ("or world.scene_provider is not None or world.validation_scene is None",
         "or world.scene_provider is None or world.validation_scene is None"),
        ("Require the complete original N4 cuboid evaluation World", "Require the complete original N3..15 native evaluation World"),
        ("            simulator = world.simulator\n", "            self._guard_native_owners(world, producer)\n            simulator = world.simulator\n"),
        ("floor.num_actors != 4 or peer.num_envs != 4 or len(peer.views) != 4",
         "floor.num_actors != self.actor_count or peer.num_envs != self.actor_count or len(peer.views) != self.actor_count"),
        ("sorted(row[0] for row in rows) != [0, 1, 2, 3]", "sorted(row[0] for row in rows) != list(range(self.actor_count))"),
        ("len({id(ref()) for ref in self.contact_refs}) != 5", "len({id(ref()) for ref in self.contact_refs}) != self.actor_count + 1"),
        ("Five distinct original floor/peer contact views required", "N+1 distinct original native floor/peer contact views required"),
        ("one-floor/four-peer native measurement owners", "one-floor/N-peer native measurement owners"),
        (producer_old, "            self._release_native_aliases(world, producer, simulator._sim)\n"),
        ('("navigation_owned_contact_views", self.contact_refs, 5)',
         '("navigation_owned_contact_views", self.contact_refs, self.actor_count + 1)'),
    ]
    source = original
    for old, new in replacements:
        if source.count(old) != 1:
            raise ValueError("Ambiguous native terminal owner seam")
        source = source.replace(old, new, 1)
    inverse = source
    for old, new in reversed(replacements):
        if inverse.count(new) != 1:
            raise ValueError("Ambiguous native terminal inverse")
        inverse = inverse.replace(new, old, 1)
    if inverse != original:
        raise ValueError("Native terminal edits did not restore complete N4 methods")
    compile(source, "native_shared_social_terminal_ownership", "exec")
    return source, {"replacement_count": len(replacements), "exact_inverse": True,
        "parent_sha256": PARENT_SHA, "derived_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "one_asset_24_sensors_SDK_release_math_changed": False, "actual_SDK_execution_verified": False}


_source, _derivation = derive_native_release(PARENT.read_text())
_namespace = {"weakref": weakref, "sys": sys}
exec(compile(_source, "native_shared_social_terminal_ownership", "exec"), _namespace)


class NativeNavigationLifecycle(FourActorNavigationLifecycle):
    _detach_world = _namespace["_detach_world"]
    _verify_contacts = _namespace["_verify_contacts"]

    def __init__(self, report, *, actors):
        if type(actors) is not int or not 3 <= actors <= 15:
            raise ValueError("Native lifecycle supports exactly N3..15")
        super().__init__(report)
        self.actor_count = actors
        pins = SDK61_NATIVE_PINS if self._sdk61 is not None else NATIVE_PINS
        self._native_sources = {name: record(ROOT / relative) for name, (relative, sha) in pins.items()}
        if any(self._native_sources[name]["sha256"] != sha for name, (_, sha) in pins.items()):
            raise ValueError("Native ownership dependency changed")
        self._producer_binding = native_social_observation_producer.describe_migration()
        report["evaluation_lifecycle"].update(native_navigation_implementation=record(__file__),
            native_navigation_derivation=dict(_derivation), navigation_actor_count=actors,
            expected_owned_contact_views=actors + 1, native_sources=self._native_sources,
            native_producer=self._producer_binding, native_SDK_smoke_verified=False)
        if self._sdk61 is not None:
            report["evaluation_lifecycle"]["native_source_version"] = "native_navigation.sim61.v2"

    def _guard_native_owners(self, world, producer):
        for name, expected in self._native_sources.items():
            module = sys.modules.get(name)
            if (module is None or Path(getattr(module, "__file__", "")).resolve() != Path(expected["path"])
                    or record(expected["path"]) != expected):
                raise RuntimeError("Actual native terminal owner module differs: " + name)
        provider, scene = world.scene_provider, world.validation_scene
        if (type(provider) is not sys.modules["CrowdSim.humanoid.runtime.dynbench_scene"].DynBenchSceneProvider
                or type(scene) is not sys.modules["CrowdSim.humanoid.runtime.native_grid_scene"].NativeGridScene
                or scene.provider is not provider or provider._query is None
                or getattr(provider._counter, "__self__", None) is not world.simulator._sim):
            raise RuntimeError("Require this native World's exact provider/scene/query/context owners")
        if (native_social_observation_producer.describe_migration() != self._producer_binding
                or type(producer) is not native_social_observation_producer.RawSocialNativeEvaluationObservations
                or producer.world is not world or producer._provider is not provider or producer._scene is not scene
                or producer._core.world is not world or producer._core.sim is not world.simulator._sim
                or getattr(producer._core.sampler, "__self__", None) is not world
                or getattr(producer._core.describe, "__self__", None) is not producer):
            raise RuntimeError("Native observation facade/core/counter cycle is not owned by this World")

    def _release_native_aliases(self, world, producer, sim):
        if (producer.world is not world or producer._core.world is not world or producer._core.sim is not sim
                or getattr(world.scene_provider._counter, "__self__", None) is not sim):
            raise RuntimeError("Native aliases changed before terminal release")
        core = producer._core
        # Instance aliases only: SDK sources and class methods are unchanged.
        core.__dict__.pop("describe", None)
        core.world = core.sim = core.sampler = None
        producer.world = producer._core = producer._base_describe = producer._provider = producer._scene = None
        world.scene_provider._query = world.scene_provider._counter = None
        world.scene_provider = world.validation_scene = None
        self.report["native_provider_borrowed_query_counter_released"] = True
        self.report["native_observation_facade_core_cycle_released"] = True
