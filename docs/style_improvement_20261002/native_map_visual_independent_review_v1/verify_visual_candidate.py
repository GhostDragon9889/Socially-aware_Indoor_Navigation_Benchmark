"""Independent read-only CPU review of the frozen native visual candidate."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from pxr import Usd, UsdPhysics

ROOT = Path(__file__).resolve().parents[3]
DOC = Path(__file__).resolve().parent
CAND = ROOT / "docs/style_improvement_20261002/native_render_namespace_candidate_v1"
sys.path.insert(0, str(ROOT))


def record(path):
    path = Path(path).resolve()
    b = path.read_bytes()
    return {"path": str(path), "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dump(node):
    return ast.dump(node, include_attributes=False)


def main():
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    freeze = json.loads((CAND / "source_freeze.json").read_text())
    for row in freeze["files"]:
        check("frozen candidate evidence " + row["path"], all(record(ROOT / row["path"])[k] == row[k] for k in ("bytes", "sha256")))
    production = ROOT / freeze["production"]["path"]
    check("production before bytes remain c824", all(record(production)[k] == freeze["production"][k] for k in ("bytes", "sha256")))
    before = ast.parse((CAND / "native_visual_scene_before.py").read_text())
    after = ast.parse((CAND / "native_visual_scene_candidate.py").read_text())
    old = {n.name: n for n in before.body if isinstance(n, ast.FunctionDef)}
    new = {n.name: n for n in after.body if isinstance(n, ast.FunctionDef)}
    for key in ("prepare_native_visual_scene", "verify_native_visual_sources"):
        check("full unchanged contract AST " + key, dump(old[key]) == dump(new[key]))
    a, b = copy.deepcopy(old["compose_native_visual_scene"]), copy.deepcopy(new["compose_native_visual_scene"])
    old_if = [n for n in a.body if isinstance(n, ast.If)]
    replacements = {"if not source_roots or stage.GetPrimAtPath(FLOOR_PATH)": old_if[2],
                    "if source in session.subLayerPaths or source in root_layer.subLayerPaths": old_if[4]}
    restored = []
    for node in b.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("overlap", "render_admission", "root_layer") for t in node.targets):
            continue
        if isinstance(node, ast.If):
            for prefix, original in replacements.items():
                if ast.unparse(node).startswith(prefix):
                    node = copy.deepcopy(original)
                    break
        restored.append(node)
    b.body = restored
    class Restore(ast.NodeTransformer):
        def visit_Expr(self, node):
            if ast.unparse(node) == "root_layer.subLayerPaths.append(source)":
                return ast.parse("session.subLayerPaths.append(source)").body[0]
            return self.generic_visit(node)
        def visit_Return(self, node):
            if isinstance(node.value, ast.Dict):
                pairs = [(k, v) for k, v in zip(node.value.keys, node.value.values)
                         if not isinstance(k, ast.Constant) or k.value not in {"source_root_layer_occurrences", "SDK_Render_overlap_admission", "source_is_weak_root_sublayer_not_session_sublayer"}]
                node.value.keys, node.value.values = [k for k, _ in pairs], [v for _, v in pairs]
            return self.generic_visit(node)
    b = Restore().visit(b)
    check("complete compose AST differs only declared overlap ownership/sublayer/metadata", dump(a) == dump(b))
    source171 = json.loads((ROOT / "docs/style_improvement_20261002/native_runtime_epoch_v5/native_pedestrian_source_v5.json").read_text())
    for row in source171["files"]:
        if row["path"] in {"scripts/replay_paper_pedestrians.py", "CrowdSim/humanoid/runtime/recorded_scene.py"}:
            check("cuboid replay source unchanged " + row["path"], all(record(ROOT / row["path"])[k] == row[k] for k in ("bytes", "sha256")))
    module = load("CrowdSim.humanoid.runtime._independent_visual_candidate", CAND / "native_visual_scene_candidate.py")
    functions = load("visual_owner_proof_math", CAND / "verify_actual_warehouse.py")
    probe = json.loads((ROOT / "docs/style_improvement_20261002/native_visual_placeholder_sdk_probe_v1/actual_empty_stage.json").read_text())
    failure = json.loads((ROOT / "output/diffusion_amp/native_warehouse_recorded_replay_30s_v1/result.json").read_text())
    saved = failure["native_visual_scene"]
    physical_result = json.loads(Path(saved["recorded_source_result"]["path"]).read_text())
    contract = module.prepare_native_visual_scene(saved["evaluation_scene"], physical_result)
    for key in ("recorded_source_result", "recorded_states"):
        contract[key] = saved[key]
    check("actual entry prepared contract equals failed original serialized resource closure", json.loads(json.dumps(contract)) == saved)
    source = Usd.Stage.Open(contract["source_USD"]["path"])
    original_RAM = source.GetRootLayer().ExportToString()
    matrices = functions.matrices(source)
    layer_records = [record(r["path"]) for r in contract["source_layers"]]
    check("all53 original source layers fullSHA before", layer_records == contract["source_layers"] and len(layer_records) == 53)
    runtime = Usd.Stage.CreateInMemory()
    check("real captured SDK root text imports", runtime.GetRootLayer().ImportFromString(probe["root_layer_text"]))
    properties = functions.properties(runtime)
    result = module.compose_native_visual_scene(runtime, contract)
    actual = functions.matrices(runtime)
    check("all1442 source paths default+t0 exact", len(matrices) == 2884 and all(actual.get(k) == v for k, v in matrices.items()))
    check("all42 SDK authored render properties exactly preserved", len(properties) == 42 and properties == functions.properties(runtime))
    check("original source RAM unchanged", original_RAM == source.GetRootLayer().ExportToString())
    check("all53 source layer file bytes unchanged after", layer_records == [record(r["path"]) for r in contract["source_layers"]])
    check("original source occurs once weak root and never session", result["source_root_layer_occurrences"] == 1 and result["source_session_occurrences"] == 0)
    check("existing SDK seven paths admitted nonspatial", len(result["SDK_Render_overlap_admission"]["SDK_owned_paths"]) == 7)
    check("actual source physics scenes inactive", all(not runtime.GetPrimAtPath(p).IsActive() for p in result["disabled_authored_PhysicsScene_paths"]))
    check("actual source collision flags false", all(not UsdPhysics.CollisionAPI(runtime.GetPrimAtPath(p)).GetCollisionEnabledAttr().Get() for p in result["disabled_collision_paths"]))
    check("actual source rigidbody flags false", all(not UsdPhysics.RigidBodyAPI(runtime.GetPrimAtPath(p)).GetRigidBodyEnabledAttr().Get() for p in result["disabled_rigid_body_paths"]))
    check("finite reference floor purely visual", not runtime.GetPrimAtPath(module.FLOOR_PATH).HasAPI(UsdPhysics.CollisionAPI))
    conflict = Usd.Stage.CreateInMemory()
    conflict.GetRootLayer().ImportFromString(probe["root_layer_text"])
    UsdPhysics.Scene.Define(conflict, "/ExistingPhysics")
    try:
        module.compose_native_visual_scene(conflict, contract)
    except ValueError as e:
        check("actual existing PhysicsScene remains rejected", "existing active PhysicsScene" in str(e))
    else:
        raise AssertionError("existing PhysicsScene was admitted")
    check("production unchanged after review", all(record(production)[k] == freeze["production"][k] for k in ("bytes", "sha256")))
    check("noKit/Torch import", not any(n in sys.modules for n in ("omni", "isaacsim", "isaaclab", "torch")))
    report = {"schema": "native_visual_frozen_candidate_independent_review.v1", "status": "CPU_review_passed_actual_render_pending",
              "CPU_only": True, "production_or_candidate_modified": False, "SDK_started": False, "GPU_used": False,
              "checks_passed": len(checks), "checks": checks, "candidate": record(CAND / "native_visual_scene_candidate.py"),
              "patch": record(CAND / "production_patch.diff"), "author_freeze": record(CAND / "source_freeze.json"),
              "nonRender_source_Xformable_paths": len(matrices) // 2, "default_t0_exact_matrix_comparisons": len(matrices),
              "SDK_property_comparisons": len(properties), "source_layer_count": len(layer_records),
              "source_RAM_sha256": hashlib.sha256(original_RAM.encode()).hexdigest(), "composition_report": result,
              "original_live_session_text_reconstructed": False, "rendered_frames": 0, "actual_render_success_verified": False,
              "reader": record(__file__), "scope": "Actual USD and captured SDK root text reconstructed on standaloneUSD26.08; source/collision gates are real CPU composition observations, not actual native Kit render success."}
    (DOC / "visual_review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "checks_passed": len(checks), "matrices": len(matrices), "properties": len(properties)}))


if __name__ == "__main__":
    main()
