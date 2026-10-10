"""CPU USD composition proof; no renderer, SDK app, or physics execution."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def record(path):
    p = Path(path)
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def matrices(stage):
    result = {}
    for time_name, time in (("default", Usd.TimeCode.Default()), ("t0", Usd.TimeCode(0))):
        cache = UsdGeom.XformCache(time)
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()):
            path = str(prim.GetPath())
            if not path.startswith("/Render") and UsdGeom.Xformable(prim):
                result[(time_name, path)] = tuple(tuple(row) for row in cache.GetLocalToWorldTransform(prim))
    return result


def properties(stage):
    values = {}
    for prim in Usd.PrimRange.AllPrims(stage.GetPrimAtPath("/Render")):
        for prop in prim.GetAuthoredProperties():
            if isinstance(prop, Usd.Attribute):
                values[str(prop.GetPath())] = (str(prop.GetTypeName()), str(prop.Get()), tuple(prop.GetTimeSamples()))
            else:
                values[str(prop.GetPath())] = tuple(str(v) for v in prop.GetTargets())
    return values


def main():
    spec = importlib.util.spec_from_file_location("CrowdSim.humanoid.runtime._native_visual_candidate", HERE / "native_visual_scene_candidate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    probe_path = ROOT / "docs/style_improvement_20261002/native_visual_placeholder_sdk_probe_v1/actual_empty_stage.json"
    failure_path = ROOT / "output/diffusion_amp/native_warehouse_recorded_replay_30s_v1/result.json"
    probe = json.loads(probe_path.read_text())
    failure = json.loads(failure_path.read_text())
    serialized_contract = failure["native_visual_scene"]
    # Recreate the actual entrypoint's in-memory contract. JSON round-trips
    # turn provider bounds tuples into lists; production prepares before
    # compose, rather than passing a deserialized report as a live contract.
    completed_source = json.loads(Path(serialized_contract["recorded_source_result"]["path"]).read_text())
    contract = module.prepare_native_visual_scene(serialized_contract["evaluation_scene"], completed_source)
    for key in ("recorded_source_result", "recorded_states"):
        contract[key] = serialized_contract[key]
    assert json.loads(json.dumps(contract)) == serialized_contract
    source = Usd.Stage.Open(contract["source_USD"]["path"])
    assert source and not source.GetCompositionErrors()
    original_matrices = matrices(source)
    before_layers = [record(entry["path"]) for entry in contract["source_layers"]]
    assert before_layers == contract["source_layers"]
    runtime = Usd.Stage.CreateInMemory("actual_SDK_root_text_CPU_reconstruction.usda")
    assert runtime.GetRootLayer().ImportFromString(probe["root_layer_text"])
    before_properties = properties(runtime)
    result = module.compose_native_visual_scene(runtime, contract)
    after_matrices = matrices(runtime)
    mismatches = [key for key, value in original_matrices.items() if after_matrices.get(key) != value]
    assert not mismatches
    assert properties(runtime) == before_properties
    after_layers = [record(entry["path"]) for entry in contract["source_layers"]]
    assert before_layers == after_layers
    assert all(not runtime.GetPrimAtPath(p).IsActive() for p in result["disabled_authored_PhysicsScene_paths"])
    assert all(not UsdPhysics.CollisionAPI(runtime.GetPrimAtPath(p)).GetCollisionEnabledAttr().Get() for p in result["disabled_collision_paths"])
    assert all(not UsdPhysics.RigidBodyAPI(runtime.GetPrimAtPath(p)).GetRigidBodyEnabledAttr().Get() for p in result["disabled_rigid_body_paths"])
    assert not runtime.GetPrimAtPath(module.FLOOR_PATH).HasAPI(UsdPhysics.CollisionAPI)
    report = {"schema": "native_render_namespace_candidate.actual_Warehouse_CPU_proof.v1", "status": "passed",
        "scope": "standalone OpenUSD reconstruction of actual SDK root layer text; session layer text was not captured; no live Kit render claim",
        "candidate": record(HERE / "native_visual_scene_candidate.py"), "actual_SDK_probe": record(probe_path),
        "original_failed_renderer_result": record(failure_path), "source_USD": contract["source_USD"],
        "source_layer_count": len(before_layers), "source_layer_records_before": before_layers, "source_layer_records_after": after_layers,
        "non_Render_Xformable_paths": len(original_matrices) // 2,
        "default_and_t0_matrices_compared_exactly": len(original_matrices), "matrix_mismatch_count": len(mismatches),
        "SDK_Render_authored_properties_preserved_exactly": len(before_properties),
        "source_Render_products_added": [], "source_layers_saved": False,
        "production_source": record(ROOT / "CrowdSim/humanoid/runtime/native_visual_scene.py"),
        "composition_report": result, "physics_steps": 0, "policy_forward_calls": 0, "rendered_frames": 0,
        "live_SDK_session_opinions_independently_reconstructed": False,
        "actual_render_success_verified": False}
    (HERE / "actual_warehouse_CPU_proof.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "source_layer_count", "non_Render_Xformable_paths", "default_and_t0_matrices_compared_exactly", "matrix_mismatch_count", "SDK_Render_authored_properties_preserved_exactly")}, indent=2))


if __name__ == "__main__":
    main()
