"""CPU-only USD/source/API proof; it never instantiates native Kit Context."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys
import threading

import numpy as np
from pxr import Usd, UsdGeom, UsdUtils

ROOT = Path(__file__).resolve().parents[3]
DOC = Path(__file__).resolve().parent
LAB = ROOT.parent / "IsaacLab-Protomotions/source/isaaclab/isaaclab"
SDK = ROOT.parent / "IsaacLab-Protomotions/.venv/lib/python3.12/site-packages/isaacsim"
STAGE_PATH = LAB / "sim/utils/stage.py"
CONTEXT_PATH = LAB / "sim/simulation_context.py"
USD_STUB = SDK / "extscache/omni.usd-1.15.2+6312fa25.lx64.r.cp312/omni/usd/_usd.pyi"


def record(path):
    path = Path(path).resolve()
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def transforms(stage):
    prims = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Xformable)]
    rows = []
    for time in (Usd.TimeCode.Default(), Usd.TimeCode(0.)):
        cache, digest = UsdGeom.XformCache(time), hashlib.sha256()
        for prim in prims:
            value = np.asarray(cache.GetLocalToWorldTransform(prim), np.float64)
            assert np.isfinite(value).all()
            digest.update(str(prim.GetPath()).encode() + b"\0")
            digest.update(value.tobytes(order="C"))
        rows.append({"time": str(time), "xformable_prims": len(prims), "path_and_matrix_C_order_SHA": digest.hexdigest()})
    return rows


def sdk_USD_function_bodies():
    """Execute exact public USD helper bodies with Kit sync explicitly disabled.

    No full IsaacLab/Kit module is imported. The real SDK-only sync function is
    not exercised; actual Timeline/context registration remains unverified.
    """
    tree = ast.parse(STAGE_PATH.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name in {"create_new_stage", "open_stage", "get_current_stage", "get_current_stage_id"}]
    namespace = {"_context": threading.local(), "_sync_isaacsim_stage_context": lambda: None}
    selected.insert(0, ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0))
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(STAGE_PATH), "exec"), namespace)
    return namespace


def main():
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    stage_tree = ast.parse(STAGE_PATH.read_text())
    check("no_undefined_attach_stage_to_usd_context_helper_assumed", not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "attach_stage_to_usd_context"
        for node in stage_tree.body))
    stub = ast.parse(USD_STUB.read_text())
    context = next(n for n in stub.body if isinstance(n, ast.ClassDef) and n.name == "UsdContext")
    methods = {n.name: n for n in context.body if isinstance(n, ast.FunctionDef)}
    check("actual_Context_has_async_attach_stage_with_callback", "attach_stage_with_callback" in methods)
    check("Context_attach_stage_not_assumed_present", "attach_stage" not in methods)
    check("callback_stage_id_annotation_int", ast.unparse(methods["attach_stage_with_callback"].args.args[1].annotation) == "int")
    simulation = ast.parse(CONTEXT_PATH.read_text())
    attach = [node for node in ast.walk(simulation) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == "attach_stage_with_callback"]
    check("SimulationContext_has_exact_cached_stage_attach_call", len(attach) == 1
          and ast.unparse(attach[0]) == "kit_context.attach_stage_with_callback(stage_cache.GetId(self.stage).ToLongInt())")
    public = sdk_USD_function_bodies()
    proof = json.loads((ROOT / "docs/style_improvement_20261002/native_map_clock_v8_proposal/actual_four_scene_USD_session_proof.json").read_text())
    rows = []
    for original in proof["sources"]:
        scene, source_record = original["scene"], original["source"]
        check(scene + ":source_file_matches_existing_exact_proof", record(source_record["path"]) == source_record)
        source = Usd.Stage.Open(source_record["path"])
        layer = source.GetRootLayer()
        ram_before = layer.ExportToString()
        source_range = [source.GetStartTimeCode(), source.GetEndTimeCode(), source.GetTimeCodesPerSecond()]
        original_transforms = transforms(source)
        check(scene + ":original_default_t0_match_frozen_real_proof", original_transforms == original["transforms"])
        runtime = public["create_new_stage"]()
        owned_root = runtime.GetRootLayer()
        check(scene + ":SDK_public_create_sets_threadlocal_and_anonymous_root", owned_root.anonymous
              and public["get_current_stage"]() == runtime and owned_root != layer)
        # Composition retains the original layer as a sublayer, preserving its
        # relative asset anchoring and original root prim paths.
        owned_root.subLayerPaths = [str(Path(source_record["path"]).resolve())]
        before_metadata = {"up_axis": str(UsdGeom.GetStageUpAxis(runtime)), "meters_per_unit": UsdGeom.GetStageMetersPerUnit(runtime),
                           "start": runtime.GetStartTimeCode(), "end": runtime.GetEndTimeCode(),
                           "tcps": runtime.GetTimeCodesPerSecond(), "fps": runtime.GetFramesPerSecond()}
        tcps, end = source.GetTimeCodesPerSecond(), max(source.GetEndTimeCode(), source.GetTimeCodesPerSecond())
        runtime.SetStartTimeCode(0.)
        runtime.SetEndTimeCode(end)
        runtime.SetTimeCodesPerSecond(tcps)
        runtime.SetFramesPerSecond(source.GetFramesPerSecond())
        UsdGeom.SetStageUpAxis(runtime, UsdGeom.GetStageUpAxis(source))
        UsdGeom.SetStageMetersPerUnit(runtime, UsdGeom.GetStageMetersPerUnit(source))
        default_prim = source.GetDefaultPrim()
        if default_prim:
            runtime.SetDefaultPrim(runtime.GetPrimAtPath(default_prim.GetPath()))
        check(scene + ":runtime_has_same_composition_errors_as_source", runtime.GetCompositionErrors() == source.GetCompositionErrors() == [])
        check(scene + ":root_sublayer_exact_single_absolute_original_path", owned_root.subLayerPaths == [source_record["path"]])
        check(scene + ":explicit_units_time_rates_and_default_prim_retained",
              UsdGeom.GetStageUpAxis(runtime) == UsdGeom.GetStageUpAxis(source)
              and UsdGeom.GetStageMetersPerUnit(runtime) == UsdGeom.GetStageMetersPerUnit(source)
              and runtime.GetTimeCodesPerSecond() == tcps and runtime.GetFramesPerSecond() == source.GetFramesPerSecond()
              and (str(runtime.GetDefaultPrim().GetPath()) if runtime.GetDefaultPrim() else None)
                  == (str(default_prim.GetPath()) if default_prim else None))
        check(scene + ":anonymous_identifier_supported_by_real_USD", Usd.Stage.IsSupportedFile(owned_root.identifier))
        opened = public["open_stage"](owned_root.identifier)
        check(scene + ":SDK_public_open_anonymous_identifier_retains_root_and_current_stage", opened.GetRootLayer() == owned_root
              and public["get_current_stage"]() == opened)
        cache_id = public["get_current_stage_id"]()
        check(scene + ":public_cache_id_is_valid_int_and_maps_to_current_stage", type(cache_id) is int and cache_id >= 0
              and UsdUtils.StageCache.Get().Find(Usd.StageCache.Id.FromLongInt(cache_id)) == opened)
        # Deliberate CPU ownership stress: writes to owned root after switching
        # the EditTarget to session model a *possible* writer bypass. They do
        # not claim to reproduce actual Kit asynchronous callbacks.
        opened.SetEditTarget(opened.GetSessionLayer())
        owned_root.endTimeCode = float(end + tcps)
        check(scene + ":explicit_owned_root_write_cannot_mutate_original_source_RAM", layer.ExportToString() == ram_before)
        owned_root.endTimeCode = float(end)
        opened.SetStartTimeCode(0.)
        opened.SetEndTimeCode(end)
        after = transforms(opened)
        check(scene + ":default_and_t0_transform_bytes_identical_after_wrapper_and_owned_root_writes", after == original_transforms)
        check(scene + ":original_source_range_and_RAM_unchanged", layer.ExportToString() == ram_before
              and [source.GetStartTimeCode(), source.GetEndTimeCode(), source.GetTimeCodesPerSecond()] == source_range)
        check(scene + ":original_source_file_fullSHA_unchanged", record(source_record["path"]) == source_record)
        rows.append({"scene": scene, "original_source": source_record, "source_original_range_start_end_tcps": source_range,
                     "source_RAM_SHA256": hashlib.sha256(ram_before.encode()).hexdigest(),
                     "anonymous_runtime_root_identifier": owned_root.identifier, "anonymous_session_identifier": opened.GetSessionLayer().identifier,
                     "source_absolute_sublayers": list(owned_root.subLayerPaths), "before_explicit_metadata": before_metadata,
                     "runtime_range_start_end_tcps": [opened.GetStartTimeCode(), opened.GetEndTimeCode(), opened.GetTimeCodesPerSecond()],
                     "default_prim_path": str(default_prim.GetPath()) if default_prim else None,
                     "public_cache_stage_id": cache_id, "public_helper_body_used": True, "SDK_Kit_sync_function_executed": False,
                     "real_Kit_Context_or_Timeline_instantiated": False, "CPU_owned_root_write_stress": True,
                     "default_and_t0_transforms": after, "all_source_layer_files_rehashed_in_this_tool": False})
    check("no_Kit_Lab_or_Torch_imported", not any(n in sys.modules for n in ("omni", "isaacsim", "isaaclab", "torch")))
    report = {"schema": "native_map_clock_v9_independent_anonymous_root_CPU_proof.v1",
              "status": "CPU_API_source_and_real_USD_ownership_proof_passed_native_pending", "CPU_only": True,
              "SDK_started": False, "GPU_used": False, "production_or_candidate_modified": False,
              "native_asynchronous_callback_cause_proved": False, "native_v9_execution_verified": False,
              "checks_passed": len(checks), "checks": checks, "standalone_USD_version": list(Usd.GetVersion()), "scenes": rows,
              "API_primary_sources": [record(STAGE_PATH), record(CONTEXT_PATH), record(USD_STUB)],
              "prior_all_layer_CPU_proof": record(ROOT / "docs/style_improvement_20261002/native_map_clock_v8_proposal/actual_four_scene_USD_session_proof.json"),
              "reader": record(__file__), "limits": [
                  "Actual Kit Context attach/Timeline root ownership behavior remains unexecuted in this CPU proof.",
                  "SDK public helper bodies run with the Kit-only context-sync function deliberately disabled, not full SDK initialization.",
                  "Four original source root files/default+t0 transforms and source RAM checked; complete file-layer inventory hashing belongs to the candidate owner.",
                  "Original USD as absolute sublayer; no TransferContent/flatten/CopySpec that could change relative asset anchoring."]}
    (DOC / "proof.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "checks_passed": len(checks), "actual_USD_scenes": len(rows),
                      "native_v9_execution_verified": False}))


if __name__ == "__main__":
    main()
