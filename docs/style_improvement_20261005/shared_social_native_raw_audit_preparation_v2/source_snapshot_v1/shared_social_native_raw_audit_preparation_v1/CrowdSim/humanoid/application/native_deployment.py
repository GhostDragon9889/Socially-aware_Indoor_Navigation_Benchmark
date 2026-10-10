"""Cold, explicit resource contract for one shared native DynBench scene."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import re

from CrowdSim.humanoid.runtime.dynbench_scene import DynBenchSceneProvider, file_record

SCHEMA = "crowdsim.native_pedestrian_scene.v1"


def _points(value, name):
    if (type(value) is not list or not value or any(type(row) is not list or len(row) != 2
            or any(type(x) not in (float, int) or not math.isfinite(x) for x in row) for row in value)):
        raise ValueError(f"{name} requires nonempty finite world XY coordinates")


def _bound_resource(value, directory):
    if (type(value) is not dict or set(value) != {"path", "sha256"}
            or type(value["path"]) is not str or "://" in value["path"]
            or type(value["sha256"]) is not str or not re.fullmatch("[0-9a-f]{64}", value["sha256"])):
        raise ValueError("Native resource requires a local path and explicit SHA256")
    path = (directory/value["path"]).resolve()
    record = file_record(path)
    if record["sha256"] != value["sha256"]:
        raise ValueError("Native scene resource changed: "+str(path))
    return record


def preflight_native_scene(path, *, actors):
    """Validate metadata without NumPy/Torch/SDK; actual arrays load at runtime."""
    import yaml
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    keys = {"schema", "scene", "preflight", "native_map", "origins_xy", "routes_xy", "clearance_m"}
    if (type(data) is not dict or not keys <= set(data) or set(data)-keys-{"probe_script_snapshot"}
            or data["schema"] != SCHEMA
            or type(actors) is not int or not 1 <= actors <= 15):
        raise ValueError("Expected explicit native scene schema and1..15 matched actors")
    _points(data["origins_xy"], "origins_xy")
    routes = data["routes_xy"]
    if type(routes) is not list or len(routes) != actors or len(data["origins_xy"]) != actors:
        raise ValueError("Native scene requires one spawn/route per actor")
    for route in routes:
        _points(route, "routes_xy")
        if len(route) < 2:
            raise ValueError("Native routes require at least two vertices")
    if len({len(route) for route in routes}) != 1:
        raise ValueError("Native route vertex counts must match")
    clearance = data["clearance_m"]
    if type(clearance) not in (float, int) or not math.isfinite(clearance) or clearance < .35:
        raise ValueError("Native clearance must be finite and at least .35m")
    preflight = _bound_resource(data["preflight"], path.parent)
    native = _bound_resource(data["native_map"], path.parent)
    snapshot = (_bound_resource(data["probe_script_snapshot"], path.parent)
                if "probe_script_snapshot" in data else None)
    provider = DynBenchSceneProvider.from_preflight(preflight["path"], expected_sha256=preflight["sha256"],
        probe_script_snapshot=None if snapshot is None else snapshot["path"])
    result = json.loads(Path(native["path"]).read_text())
    owner = result.get("floor_ownership", {})
    if (provider.scene_name != data["scene"] or result.get("scene") != data["scene"]
            or result.get("status") != "scene_resource_gate_passed"
            or result.get("preflight") != preflight or result.get("scene_resource_gate_passed") is not True
            or owner.get("floor_ownership_verified") is not True
            or owner.get("single_declared_floor_owner_verified") is not True
            or result.get("original_source_layers_unchanged") is not True):
        raise ValueError("Native scene requires the successful full-grid, source-bound single-floor resource gate")
    return {"schema": "native_scene_deployment_request.v1", "configuration": deepcopy(data),
            "configuration_file": file_record(path), "preflight": preflight, "native_map": native,
            "probe_script_snapshot": snapshot,
            "actors": actors, "tensor_arrays_validated": False, "SDK_started": False,
            "human_navigation_validated": False}


def make_native_scene_factory(request):
    """Load bound actual arrays after App startup; never launch another SDK."""
    def prepare(description, *, device):
        from CrowdSim.humanoid.runtime.native_grid_scene import NativeGridScene
        if description != request["configuration"]:
            raise ValueError("Native scene configuration changed between preflight and driver")
        if file_record(request["configuration_file"]["path"]) != request["configuration_file"]:
            raise ValueError("Native scene configuration file changed")
        preflight, native = request["preflight"], request["native_map"]
        snapshot = request["probe_script_snapshot"]
        if snapshot is not None and file_record(snapshot["path"]) != snapshot:
            raise ValueError("Selected historical probe source changed")
        provider = DynBenchSceneProvider.from_preflight(preflight["path"], expected_sha256=preflight["sha256"],
            probe_script_snapshot=None if snapshot is None else snapshot["path"])
        provider.admit_native_map(native["path"], expected_sha256=native["sha256"])
        return NativeGridScene.from_provider(provider, origins_xy=description["origins_xy"],
            routes_xy=description["routes_xy"], clearance_m=description["clearance_m"], device=device)
    return prepare
