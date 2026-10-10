"""Read-only review of the staged v8 proposal; no Kit or GPU execution."""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from pxr import Usd, UsdGeom

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
DOC = Path(__file__).resolve().parent
PROPOSAL = ROOT / "docs/style_improvement_20261002/native_map_clock_v8_proposal"
CANDIDATE = ROOT / "docs/style_improvement_20261002/native_map_clock_v8_candidate"


def record(path):
    path = Path(path).resolve()
    payload = path.read_bytes()
    return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def tree(path):
    return ast.parse(Path(path).read_text())


def dump(node):
    return ast.dump(node, include_attributes=False)


class StripV8Dispatch(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        return None if node.module == "native_map_v8" else node

    def visit_If(self, node):
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Call) and isinstance(node.test.func, ast.Name) and node.test.func.id == "is_v8":
            return node.orelse
        return node


class NormalizeV8Verifier(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.name = node.name.replace("verify_v8", "verify_v7")
        return self.generic_visit(node)

    def visit_Name(self, node):
        if node.id == "is_v8":
            node.id = "is_v7"
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            node.value = node.value.replace("Native v8", "Native v7").replace("v8_exact_probe_SHA", "v7_exact_probe_SHA")
        return node

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "_session_range_evidence":
            return None
        return self.generic_visit(node)

    def visit_Dict(self, node):
        retained = [(key, value) for key, value in zip(node.keys, node.values)
                    if not isinstance(key, ast.Constant) or key.value != "runtime_time_range_override"]
        node.keys, node.values = map(list, zip(*retained))
        return self.generic_visit(node)


class TimelineContract:
    """USD-only setter double; it cannot establish native Timeline ownership."""
    def __init__(self, stage):
        self.stage = stage

    def set_start_time(self, seconds):
        self.stage.SetStartTimeCode(seconds * self.stage.GetTimeCodesPerSecond())

    def set_end_time(self, seconds):
        self.stage.SetEndTimeCode(seconds * self.stage.GetTimeCodesPerSecond())

    def set_auto_update(self, enabled):
        assert enabled is False

    def set_current_time(self, seconds):
        assert seconds == 0.

    def get_start_time(self):
        return self.stage.GetStartTimeCode() / self.stage.GetTimeCodesPerSecond()

    def get_end_time(self):
        return self.stage.GetEndTimeCode() / self.stage.GetTimeCodesPerSecond()


def transforms(stage):
    prims = [p for p in stage.Traverse() if p.IsA(UsdGeom.Xformable)]
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


def main():
    checks = []

    def check(name, value):
        checks.append({"name": name, "passed": bool(value)})
        if not value:
            raise AssertionError(name)

    proposed = PROPOSAL / "proposed"
    before = PROPOSAL / "before"
    source_rows = json.loads((PROPOSAL / "before_source.json").read_text())
    for row in source_rows:
        relative = Path(row["path"]).relative_to(ROOT)
        saved, current = record(before / relative), record(row["path"])
        check("saved_before_matches_identity:" + str(relative), all(saved[k] == row[k] for k in ("bytes", "sha256")))
        check("current_business_unchanged:" + str(relative), current == row)
    producer = proposed / "scripts/probe_paper_scene_maps.py"
    merge_manifest = json.loads((PROPOSAL / "merge_manifest.json").read_text())
    allowed = (
        "CrowdSim/humanoid/runtime/dynbench_scene.py",
        "CrowdSim/humanoid/runtime/native_grid_scene.py",
        "CrowdSim/humanoid/runtime/native_map_v8.py",
        "docs/style_improvement_20261002/native_map_admission_v7/audit_actual_four_scenes.py",
        "scripts/probe_paper_scene_maps.py",
        "tests/test_native_map_audit_reader.py",
        "tests/test_native_map_v7.py",
        "tests/test_native_map_v8.py",
        "tests/test_paper_scene_maps_clock_v8.py",
    )
    check("bounded_manifest_has_exact_nine_paths", tuple(r["relative_path"] for r in merge_manifest["files"]) == allowed)
    total = 0
    for row in merge_manifest["files"]:
        relative = Path(row["relative_path"])
        target, payload = ROOT / relative, proposed / relative
        check("bounded_paths_no_symlinks_or_traversal:" + str(relative),
              not relative.is_absolute() and ".." not in relative.parts and target.resolve() == target and payload.resolve() == payload)
        check("merge_before_and_after_bytes_verified:" + str(relative),
              (not target.exists() if row["before"] is None else record(target) == row["before"])
              and record(payload) == row["after"] and row["after"]["bytes"] < 250000)
        total += row["after"]["bytes"]
    check("total_merge_payload_bounded_and_exact", total == merge_manifest["total_proposed_bytes"] == 245852 and total < 500000)
    for row in merge_manifest["protected_unchanged_files"]:
        check("protected_source_unchanged:" + row["path"], record(row["path"]) == row)
    v8_path = proposed / "CrowdSim/humanoid/runtime/native_map_v8.py"
    v8_tree = tree(v8_path)
    constant = next(n.value.value for n in v8_tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "PROBE_SHA256" for t in n.targets))
    check("exact_v8_producer_SHA_matches_dispatch", record(producer)["sha256"] == constant)
    for relative in ("CrowdSim/humanoid/runtime/dynbench_scene.py", "CrowdSim/humanoid/runtime/native_grid_scene.py"):
        normalized = StripV8Dispatch().visit(tree(proposed / relative))
        check("old_v5_v7_AST_recovered_without_v8_dispatch:" + relative, dump(normalized) == dump(tree(before / relative)))
    old_verify = next(n for n in tree(before / "CrowdSim/humanoid/runtime/native_map_v7.py").body
                      if isinstance(n, ast.FunctionDef) and n.name == "verify_v7")
    new_verify = next(n for n in v8_tree.body if isinstance(n, ast.FunctionDef) and n.name == "verify_v8")
    normalized = NormalizeV8Verifier().visit(deepcopy(new_verify))
    check("all_v7_geometry_clock_math_exactly_preserved_in_v8", dump(normalized) == dump(old_verify))
    old_functions = {n.name: n for n in tree(before / "scripts/probe_paper_scene_maps.py").body if isinstance(n, ast.FunctionDef)}
    new_functions = {n.name: n for n in tree(producer).body if isinstance(n, ast.FunctionDef)}
    changed = [name for name in old_functions if dump(old_functions[name]) != dump(new_functions[name])]
    check("only_native_execution_function_changed", changed == ["run_native_scene"])
    source_exports = [ast.unparse(n) for n in ast.walk(tree(producer)) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Attribute) and n.func.attr == "Export"]
    check("all_USD_exports_are_owned_session_layers", all("GetSessionLayer().Export" in text for text in source_exports))
    check("producer_has_no_USD_Save_calls", not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                                                   and n.func.attr in ("Save", "SaveSessionLayers") for n in ast.walk(tree(producer))))
    spec = importlib.util.spec_from_file_location("independent_proposed_v8_CPU", producer)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    proof = json.loads((CANDIDATE / "actual_four_scene_USD_session_proof.json").read_text())
    check("original_candidate_proof_matches_candidate_bytes", record(proof["candidate"]["path"]) == proof["candidate"])
    rows = []
    for reference in proof["sources"]:
        source = reference["source"]
        check("source_root_file_matches_prior_real_proof:" + reference["scene"], record(source["path"]) == source)
        stage = Usd.Stage.Open(source["path"])
        root_text, before_transforms = stage.GetRootLayer().ExportToString(), transforms(stage)
        check("reference_default_t0_transform_hashes_match:" + reference["scene"], before_transforms == reference["transforms"])
        stage.SetEditTarget(stage.GetSessionLayer())
        metadata = module.freeze_query_timeline_range_in_session(stage, TimelineContract(stage))
        check("proposed_helper_source_RAM_unchanged:" + reference["scene"], stage.GetRootLayer().ExportToString() == root_text)
        check("proposed_helper_root_file_unchanged:" + reference["scene"], record(source["path"]) == source)
        after_transforms = transforms(stage)
        check("proposed_helper_default_t0_transforms_bytewise_equal:" + reference["scene"], before_transforms == after_transforms)
        check("proposed_helper_keeps_original_TCPS:" + reference["scene"], stage.GetTimeCodesPerSecond() == reference["source_original_composed_range_start_end_tcps"][2])
        rows.append({"scene": reference["scene"], "source": source, "proposed_session_override": metadata,
                     "transforms": after_transforms, "native_Timeline_instantiated": False})
    check("no_Kit_GPU_or_Torch_imported", not any(name in sys.modules for name in ("omni", "isaacsim", "isaaclab", "torch")))
    report = {"schema": "native_map_clock_v8_independent_read_only_review.v1", "status": "CPU_source_and_USD_review_passed_native_pending",
              "CPU_only": True, "Kit_started": False, "GPU_used": False, "production_or_proposal_modified": False,
              "checks": checks, "standalone_USD_version": list(Usd.GetVersion()), "sources": rows,
              "changed_existing_producer_functions": changed, "USD_export_calls": source_exports,
              "producer": record(producer), "new_consumer": record(v8_path),
              "merge_manifest": record(PROPOSAL / "merge_manifest.json"), "merge_script": record(PROPOSAL / "merge_v8_proposal.py"),
              "bounded_patch": record(PROPOSAL / "proposed.patch"), "merge_payload_bytes": total,
              "prior_real_USD_proof": record(CANDIDATE / "actual_four_scene_USD_session_proof.json"),
              "proposed_source_files": [record(p) for p in sorted(proposed.rglob("*.py"))],
              "reader": record(__file__), "native_v8_execution_observed": False,
              "native_clock_fix_verified": False}
    (DOC / "verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "checks": len(checks), "all_passed": all(c["passed"] for c in checks),
                      "scenes": len(rows), "native_v8_execution_observed": False}))


if __name__ == "__main__":
    main()
