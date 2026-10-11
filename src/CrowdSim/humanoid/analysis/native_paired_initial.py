"""Versioned N3..15 pre-planner initial snapshots on admitted native maps.

The fixed N4 collector, RNG getters, archive writer and matching calculation
are reused in an owned module. Four exact map channels supplement body/RNG
channels. CPU fixtures never establish a live World or an actual process exit.
"""
from __future__ import annotations

from protomotions.repository import repository_root

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import ModuleType

import numpy as np

from . import paired_initial as original

ROOT = repository_root(__file__)
PARENT = ROOT / "CrowdSim/humanoid/analysis/paired_initial.py"
PARENT_SHA = "3395ebc1e4de385788fec738f49994af2887bb1b0af998363608c8625e17022d"
SCHEMA = "shared_social.before_planner_native_initial_condition.v1"
LIVE_KIND = "live_original_native_world"
LAYOUT = "row0=max_world_y,column0=min_world_x; origin=lower-left edge; no dynamic fitting"
MAP_FIELDS = {"native_scene_labels": ("_labels", np.dtype("uint8")),
              "native_scene_origin_xy": ("_origin", np.dtype("float64")),
              "native_scene_bounds_xy": ("_bounds", np.dtype("float64")),
              "native_scene_signed_distance_grid": ("_signed_grid", np.dtype("float32"))}
PINS = {"CrowdSim.humanoid.runtime.dynbench_scene":
            ("CrowdSim/humanoid/runtime/dynbench_scene.py", "c49282b44d38e65b92cbf71fa1909c3e87f551e8494793aa31c5f68246ac7a73"),
        "CrowdSim.humanoid.runtime.native_grid_scene":
            ("CrowdSim/humanoid/runtime/native_grid_scene.py", "d77e9b5702d47765e045a03f6e79f770c313dc8d4d0ad9e969aead81d40024f5"),
        "CrowdSim.humanoid.application.native_deployment":
            ("CrowdSim/humanoid/application/native_deployment.py", "af708cc18873424ed57b497473f35d17cecc2c3910b15f4c67ef4d8cd5ab49f0")}

# Match the versioned native SFM scene providers. This is a new capture
# provenance boundary; an archived v1 capture cannot certify these providers.
SOURCE_VERSION = "native_initial.current_scene_sources.v2"
CURRENT_SCENE_SHA = {
    "CrowdSim.humanoid.application.native_deployment": "a3a8097e5d63b03e5f96f24e6246c333b2103f3628c6209139e2aff2c564cd96",
    "CrowdSim.humanoid.runtime.dynbench_scene": "68bd2171ad9707c3f280c374d73538b8c532764a42e1f90335c6375090595d28",
    "CrowdSim.humanoid.runtime.native_grid_scene": "b5563bd5a509cdb02ae33e57ad34ab7a0524825ad16e481ef631531d59308165",
}


def restore_core_location(source):
    """Prove whole-file equality to the N4 core after undoing its root lookup."""
    if hashlib.sha256(source.encode()).hexdigest() == PARENT_SHA:
        return source
    restored = source
    for current, historical in (
        ("from protomotions.repository import repository_root\n\n", ""),
        ("ROOT = repository_root(__file__)", "ROOT = Path(__file__).resolve().parents[3]"),
    ):
        if restored.count(current) != 1:
            raise ValueError("Require the exact fixed N4 initial collector location migration")
        restored = restored.replace(current, historical, 1)
    if hashlib.sha256(restored.encode()).hexdigest() != PARENT_SHA:
        raise ValueError("Require the exact fixed N4 initial collector location inverse")
    return restored


def source_bindings():
    parent = original._identity(PARENT)
    try:
        restore_core_location(PARENT.read_text())
    except ValueError as error:
        raise ValueError("Fixed initial-state core changed") from error
    result = {"core": parent, "native_reader": original._identity(__file__),
              "source_version": SOURCE_VERSION, "historical_core_sha256": PARENT_SHA,
              "core_location_exact_inverse": True,
              "historical_scene_sha256": {role: row[1] for role, row in PINS.items()}}
    capture = original._identity(ROOT / "CrowdSim/humanoid/foot_capture.py")
    if capture["sha256"] != original.FOOT_CAPTURE_SHA:
        raise ValueError("Native initial foot capture changed")
    result["foot_capture"] = capture
    portable = original._identity(ROOT / "src/social_navigation/portable_usd.py")
    if portable["sha256"] != "a72a52ee08d5341f322b01907cd8ba407e5fc582399ad1bfd686c4b4e320a7b3":
        raise ValueError("Portable USD pairing source changed")
    result["portable_USD_pairing"] = portable
    for role, (name, sha) in PINS.items():
        row = original._identity(ROOT / name)
        if row["sha256"] != CURRENT_SCENE_SHA.get(role, sha):
            raise ValueError("Native initial dependency changed: " + role)
        result[role] = row
    return result


def derive_count_core(source):
    """Explicit reversible boundary/count edits; RNG and copy math retained."""
    restore_core_location(source)
    replacements = [
        ('    if (world.num_envs != 4 or not world.shared_world or world.scene_provider is not None\n'
         '            or world.validation_scene is None or ground_capture.world is not world):\n',
         '    if (type(world.num_envs) is not int or world.num_envs != _native_actor_count\n'
         '            or world.shared_world is not True or world.scene_provider is None\n'
         '            or world.validation_scene is None or world.validation_scene.provider is not world.scene_provider\n'
         '            or ground_capture.world is not world):\n', 1),
        ("original complete shared N4 fixture", "original complete shared N3..15 native World", 1),
        ("np.arange(4", "np.arange(_native_actor_count", 2),
        ("np.zeros(4", "np.zeros(_native_actor_count", 1),
        (".shape[0] != 4", ".shape[0] != _native_actor_count", 2),
        ("all four registered routes", "all registered native actor routes", 1),
        ("all four actors and at least two vertices", "all native actors and at least two vertices", 1),
        ('"live_original_world"', '"live_original_native_world"', 3),
        ("shapes.update(actor_ids=(4,), scene_ids=(4,),",
         "shapes.update(actor_ids=(_native_actor_count,), scene_ids=(_native_actor_count,),", 1),
    ]
    derived = source
    for old, new, count in replacements:
        if derived.count(old) != count:
            raise ValueError("Ambiguous native initial-count seam: " + old)
        derived = derived.replace(old, new)
    inverse = derived
    for old, new, count in reversed(replacements):
        if inverse.count(new) != count:
            raise ValueError("Ambiguous native initial-count inverse")
        inverse = inverse.replace(new, old)
    if inverse != source:
        raise ValueError("Native initial seams did not restore complete N4 source")
    changed = {"collect_snapshot", "capture_initial_condition", "validate_channels"}
    def helpers(value):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(value).body
                if isinstance(node, ast.FunctionDef) and node.name not in changed}
    if helpers(source) != helpers(derived):
        raise ValueError("Native count adapter changed a copy/RNG/archive/matching helper")
    compile(derived, "native_initial_count_core", "exec")
    return derived, {"parent_sha256": hashlib.sha256(source.encode()).hexdigest(),
                     "historical_parent_sha256": PARENT_SHA, "core_location_exact_inverse": True,
                     "literal_seams": len(replacements),
                     "literal_occurrences": sum(row[2] for row in replacements), "exact_inverse": True,
                     "unchanged_helpers_AST_equal": True, "changed_functions": sorted(changed),
                     "derived_sha256": hashlib.sha256(derived.encode()).hexdigest(),
                     "copy_RNG_archive_match_math_changed": False, "actual_SDK_execution_verified": False}


def count_backend(actors):
    """One private namespace per explicit actor count, with no shared mutation."""
    if type(actors) is not int or not 3 <= actors <= 15:
        raise ValueError("Native initial protocol requires an explicit N3..15")
    source_bindings()
    source, proof = derive_count_core(PARENT.read_text())
    module = ModuleType(f"_owned_native_initial_count_{actors}")
    module.__file__ = str(Path(__file__).resolve())
    module._native_actor_count = actors
    exec(compile(source, str(PARENT) + ":native_count", "exec"), module.__dict__)
    for name in ("COMMON_FIELDS", "WORLD_FIELDS", "NATIVE_FIELDS"):
        shapes = getattr(module, name)
        setattr(module, name, {key: (actors, *shape[1:]) if shape else () for key, shape in shapes.items()})
    module.SCHEMA = SCHEMA
    return module, proof


def map_arrays(scene):
    return {key: original._array(getattr(scene, field), key) for key, (field, _) in MAP_FIELDS.items()}


def native_protocol(actors, scene):
    return {"actors": actors, "map_shape_hw": list(scene._labels.shape), "resolution_m": .1,
            "layout": LAYOUT,
            "map_channels": {key: original._description(value) for key, value in map_arrays(scene).items()},
            "distance_field": "Original immutable NativeGridScene signed-grid proxy; no full-body clearance claim"}


def admit_native_resource_scene(request):
    """Recheck actual records and reconstruct only the admitted CPU grid."""
    from CrowdSim.humanoid.application.native_deployment import make_native_scene_factory, preflight_native_scene
    source_bindings()
    current = preflight_native_scene(request["configuration_file"]["path"], actors=request["actors"])
    if current != request:
        raise ValueError("Native initial request differs from current source-bound scene")
    return make_native_scene_factory(current)(current["configuration"], device="cpu")


def validate_channels(arrays, contract, *, expected_scene=None):
    protocol = contract["native_protocol"]
    actors = protocol["actors"]
    backend, _ = count_backend(actors)
    backend.validate_channels({key: value for key, value in arrays.items() if key not in MAP_FIELDS}, contract)
    shape = protocol["map_shape_hw"]
    if (type(shape) is not list or len(shape) != 2 or any(type(n) is not int or n < 2 for n in shape)
            or protocol["resolution_m"] != .1 or protocol["layout"] != LAYOUT
            or set(protocol["map_channels"]) != set(MAP_FIELDS)
            or not MAP_FIELDS.keys() <= arrays.keys()):
        raise ValueError("Missing explicit native map/count/channel protocol")
    shapes = {"native_scene_labels": tuple(shape), "native_scene_origin_xy": (2,),
              "native_scene_bounds_xy": (4,), "native_scene_signed_distance_grid": tuple(shape)}
    for key, (_, dtype) in MAP_FIELDS.items():
        value = original._array(arrays[key], key, shape=shapes[key])
        if value.dtype != dtype or original._description(value) != protocol["map_channels"][key]:
            raise ValueError("Native initial map channel dtype/bytes differ: " + key)
    labels, origin_xy, bounds = (arrays[key] for key in ("native_scene_labels", "native_scene_origin_xy", "native_scene_bounds_xy"))
    if (not np.isin(labels, (0, 205, 254)).all()
            or not np.allclose(bounds[:2], origin_xy, rtol=0, atol=2e-4)
            or not np.allclose(bounds[2:] - origin_xy, np.array(shape[::-1]) * .1, rtol=0, atol=2e-4)):
        raise ValueError("Native initial labels/frame differ; no coordinate fitting")
    if contract["capture_kind"] == LIVE_KIND:
        sources = source_bindings()
        if (contract.get("native_sources") != sources
                or contract["initial_capture_source"] != sources["native_reader"]
                or contract["native_capture_source"] != sources["foot_capture"]):
            raise ValueError("Native live initial source bindings differ")
        request = contract["native_resources"]
        if (request["actors"] != actors or contract["resources"]["scene"] != request["configuration_file"]
                or contract["native_count_derivation"] != count_backend(actors)[1]):
            raise ValueError("Native live initial count/scene/derivation bindings differ")
        selected = expected_scene or admit_native_resource_scene(contract["native_resources"])
        if protocol != native_protocol(actors, selected):
            raise ValueError("Native initial protocol differs from its actual admitted grid")
        if not original._equal_arrays({key: arrays[key] for key in MAP_FIELDS}, map_arrays(selected)):
            raise ValueError("Native initial arrays differ from actual admitted labels/SDF/frame")
        if (not np.array_equal(arrays["world_origins"], selected._origins.astype(np.float32))
                or not np.array_equal(arrays["scene_routes_xy"], selected._routes.astype(np.float32))):
            raise ValueError("Native live initial origins/routes differ from admitted absolute world XY")


def _archive_backend(contract, *, expected_scene=None):
    backend, _ = count_backend(contract["native_protocol"]["actors"])
    backend.validate_channels = lambda arrays, selected: validate_channels(arrays, selected, expected_scene=expected_scene)
    return backend


def publish_snapshot(output, arrays, contract):
    return _archive_backend(contract).publish_snapshot(output, arrays, contract)


def read_snapshot(path):
    receipt = json.loads(Path(path).read_text())
    if receipt.get("schema") != SCHEMA:
        raise ValueError("Wrong native paired initial schema")
    return _archive_backend(receipt["contract"]).read_snapshot(path)


def compare_initial_conditions(left, right, *, require_live=True):
    baseline, candidate = read_snapshot(left), read_snapshot(right)
    if require_live and any(row["contract"]["capture_kind"] != LIVE_KIND for row in (baseline, candidate)):
        raise ValueError("Require actual original native World captures; CPU fixtures cannot establish live pairing")
    if baseline["contract"] != candidate["contract"] or baseline["channels"] != candidate["channels"]:
        raise ValueError("Native planner arms do not share exact initial body/map/RNG/protocol/resources")
    return {"status": "native_initial_conditions_matched", "snapshots": [original._identity(left), original._identity(right)],
            "channel_count": len(baseline["channels"]), "actors": baseline["contract"]["native_protocol"]["actors"],
            "same_initial_body_map_and_RNG_bytes": True, "same_resources_protocol_and_RNG": True,
            "capture_kind": baseline["contract"]["capture_kind"], "actual_process_exit_verified": False,
            "navigation_quality_established": False, "author_experiment_reproduced": False}


def _run_owned_capture(backend, collect, pairing, validate, arguments):
    """Restore private callbacks on every exit, dropping their live owners.

    Functions retain their module globals. Leaving World/capture closures in
    that globals dictionary would keep native owners in a reference cycle.
    """
    names = ("collect_snapshot", "pairing_contract", "validate_channels")
    saved = tuple(getattr(backend, name) for name in names)
    try:
        for name, callback in zip(names, (collect, pairing, validate), strict=True):
            setattr(backend, name, callback)
        return backend.capture_initial_condition(*arguments)
    finally:
        for name, callback in zip(names, saved, strict=True):
            setattr(backend, name, callback)


def capture_initial_condition(world, ground_capture, output, report, args, provenance):
    """Live hook: no reseeding, restored state, contact query or physical step."""
    import sys
    from CrowdSim.humanoid.application.native_deployment import preflight_native_scene
    from CrowdSim.humanoid.runtime.dynbench_scene import DynBenchSceneProvider
    from CrowdSim.humanoid.runtime.native_grid_scene import NativeGridScene

    actors = world.num_envs
    request = preflight_native_scene(args.scene_config, actors=actors)
    expected_scene = admit_native_resource_scene(request)
    backend, proof = count_backend(actors)
    sources = source_bindings()
    for name, (relative, _) in PINS.items():
        module = sys.modules.get(name)
        if module is None or Path(getattr(module, "__file__", "")).resolve() != (ROOT / relative).resolve():
            raise ValueError("Actual native snapshot module origin differs: " + name)
    provider, scene = world.scene_provider, world.validation_scene
    if (type(provider) is not DynBenchSceneProvider or type(scene) is not NativeGridScene
            or scene.provider is not provider or provider._query is None
            or getattr(provider._counter, "__self__", None) is not world.simulator._sim
            or provider.describe()["native_map_evidence"] != request["native_map"]):
        raise ValueError("Native snapshot requires this live World's exact admitted provider/query/grid")
    core_collect, core_contract = backend.collect_snapshot, backend.pairing_contract

    def collect(selected_world, capture):
        if (selected_world is not world or capture is not ground_capture or source_bindings() != sources
                or not original._equal_arrays(map_arrays(scene), map_arrays(expected_scene))):
            raise ValueError("Native initial World/source/map changed during capture")
        arrays, clock = core_collect(selected_world, capture)
        arrays.update(map_arrays(scene))
        return arrays, clock

    def pairing(selected_report, selected_args, description, clock, rng_metadata, selected_provenance):
        contract = deepcopy(core_contract(selected_report, selected_args, description, clock, rng_metadata, selected_provenance))
        from social_navigation.portable_usd import stable_layer_receipts
        stable = stable_layer_receipts(provider._portable_layers, provider._portable_usd_receipt,
                                       source_records=provider._data['layers'])
        for bound_provider in (contract['scene']['provider'], contract['physics']['scene_geometry']['provider']):
            if bound_provider['portable_USD_location_derivation'] != provider._portable_usd_receipt:
                raise ValueError('Initial USD description differs from its live owner')
            bound_provider['portable_USD_location_derivation'] = deepcopy(stable)
        contract.update(native_protocol=native_protocol(actors, scene), native_resources=deepcopy(request),
                        native_sources=deepcopy(sources), native_count_derivation=proof)
        return contract

    def validate(arrays, contract):
        validate_channels(arrays, contract, expected_scene=expected_scene)
    result = _run_owned_capture(backend, collect, pairing, validate,
                               (world, ground_capture, output, report, args, provenance))
    if source_bindings() != sources:
        raise ValueError("Native snapshot source changed after publication; files retained")
    return result
