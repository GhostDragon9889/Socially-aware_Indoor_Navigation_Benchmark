"""Independent CPU review of staged v9 admission and same-path cached exports."""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

from pxr import Sdf

ROOT = Path(__file__).resolve().parents[3]
DOC = Path(__file__).resolve().parent
PROPOSAL = ROOT / "docs/style_improvement_20261002/native_map_clock_v9_proposal_v2"
sys.path.insert(0, str(ROOT))


def record(path):
    path = Path(path).resolve()
    b = path.read_bytes()
    return {"path": str(path), "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


def dump(node):
    return ast.dump(node, include_attributes=False)


def main():
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    files = PROPOSAL / "proposed/CrowdSim/humanoid/runtime"
    package = "CrowdSim.humanoid.runtime."
    verifier = module(package + "native_map_v9", files / "native_map_v9.py")
    provider_module = module(package + "dynbench_scene", files / "dynbench_scene.py")
    class StripV9(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return None if node.module == "native_map_v9" else node
        def visit_If(self, node):
            if isinstance(node.test, ast.Call) and isinstance(node.test.func, ast.Name) and node.test.func.id == "is_v9":
                return self.visit(node.orelse[0]) if node.orelse else None
            return self.generic_visit(node)
    for name in ("dynbench_scene.py", "native_grid_scene.py"):
        before = ast.parse((PROPOSAL / "before/CrowdSim/humanoid/runtime" / name).read_text())
        after = StripV9().visit(ast.parse((files / name).read_text()))
        check("old V5/V7/V8 full dispatch AST exact " + name, dump(before) == dump(after))
    result = ROOT / "output/scene_maps/native_warehouse_anonymous_root_v9_candidate_v1/Warehouse/result.json"
    original = json.loads(result.read_text())
    provider = provider_module.DynBenchSceneProvider.from_preflight(original["preflight"]["path"], expected_sha256=original["preflight"]["sha256"])
    provider.admit_native_map(result, expected_sha256=record(result)["sha256"])
    admitted = verifier.verify_v9(provider, original, admitted=True)
    check("real Warehouse complete raw/crop/CSR/floor/height admission passes", admitted["canonical_known5_cells"] == 56239)
    cases = []
    def reject(name, data):
        try:
            verifier._session_range_evidence(provider, data)
        except ValueError as error:
            cases.append({"case": name, "rejected": True, "reason": str(error)})
            check("reject " + name, True)
        else:
            raise AssertionError("Corrupt export admitted: " + name)
    with tempfile.TemporaryDirectory(prefix="native_v9_independent_raw_", dir="/tmp") as tmp:
        folder = Path(tmp)
        source_text = Path(original["exports"]["runtime_session.usda"]["path"]).read_text()
        session = folder / "cached_session.usda"
        session.write_text(source_text)
        data = deepcopy(original)
        data["exports"]["runtime_session.usda"] = record(session)
        # Explicitly keep a Sdf cache reference across same-file replacement.
        cached = Sdf.Layer.FindOrOpen(str(session))
        verifier._session_range_evidence(provider, data)
        check("cached original real session initially valid", cached is not None)
        old_cached_text = cached.ExportToString()
        session.write_text(source_text.replace('over "Root"\n{', 'over "Root"\n{\n    double3 xformOp:translate = (100, 0, 0)\n    uniform token[] xformOpOrder = ["xformOp:translate"]', 1))
        data["exports"]["runtime_session.usda"] = record(session)
        check("Sdf cached layer actually stale while file bytes changed", cached.ExportToString() == old_cached_text and session.read_text() != source_text)
        reject("same filepath cached session rewrite and rehash source translation", data)
        root = folder / "root_spatial.usda"
        root.write_text(Path(original["exports"]["runtime_root.usda"]["path"]).read_text() + '\nover "Root"\n{\n    double3 xformOp:translate = (100, 0, 0)\n}\n')
        data = deepcopy(original)
        data["exports"]["runtime_root.usda"] = record(root)
        data["runtime_time_range_override"]["source_root_in_memory_sha256_at_query"] = record(root)["sha256"]
        reject("rehashed root spatial opinions even with matching claimed query SHA", data)
        data = deepcopy(original)
        data["runtime_time_range_override"]["source_root_in_memory_sha256_at_query"] = "f" * 64
        reject("query root RAM SHA differs from raw export", data)
        data = deepcopy(original)
        for values in (data["runtime_stage_composition"], data["runtime_time_range_override"]):
            for key in list(values):
                if key.startswith("original_source_root_in_memory_sha256"):
                    values[key] = "f" * 64
        reject("consistent forged original RAM claims contradict actual source", data)
        session.write_bytes(b"not USD\xff")
        data = deepcopy(original)
        data["exports"]["runtime_session.usda"] = record(session)
        reject("invalid export bytes remain hard rejection", data)
    check("actual result bytes preserved", record(result) == provider._native["record"])
    check("no Kit/Torch import", not any(name in sys.modules for name in ("omni", "isaacsim", "isaaclab", "torch")))
    report = {"schema": "native_v9_v2_independent_actual_raw_CPU_review.v1", "status": "CPU_complete_Warehouse_and_cache_fix_review_passed",
              "CPU_only": True, "production_or_candidate_modified": False, "SDK_started": False, "GPU_used": False,
              "checks_passed": len(checks), "checks": checks, "corruption_cases": cases,
              "actual_Warehouse_result": record(result), "actual_Warehouse_admission": admitted,
              "candidate_verifier": record(files / "native_map_v9.py"), "reader": record(__file__),
              "native_four_scenes_verified": False,
              "limits": "This reader recomputes actual completed Warehouse native artifacts only; cache failures mutate temporary exports only. Other three scenes await native v9 execution. No original output, candidate, stage/source or production bytes were modified."}
    (DOC / "v9_v2_review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "checks_passed": len(checks), "corruptions_rejected": len(cases), "canonical_known5_cells": admitted["canonical_known5_cells"]}))


if __name__ == "__main__":
    main()
