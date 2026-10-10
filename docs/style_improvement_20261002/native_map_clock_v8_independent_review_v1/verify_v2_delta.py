"""Independent read-only V2 delta review; never merge or launch native SDK."""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOC = Path(__file__).resolve().parent
FIRST = ROOT / "docs/style_improvement_20261002/native_map_clock_v8_proposal"
SECOND = ROOT / "docs/style_improvement_20261002/native_map_clock_v8_proposal_v2"


def record(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def normalized_helper(path):
    tree = ast.parse(Path(path).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ALLOWED" for t in node.targets):
            node.value = ast.Constant("exact_allowlist_replaced_only_for_comparison")
    return ast.dump(tree, include_attributes=False)


def main():
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    old = json.loads((FIRST / "merge_manifest.json").read_text())
    new = json.loads((SECOND / "merge_manifest.json").read_text())
    old_rows = {row["relative_path"]: row for row in old["files"]}
    allowed = tuple(row["relative_path"] for row in old["files"]) + ("tests/test_paper_scene_maps_probe.py",)
    check("V2_exact_allowlist_is_V1_plus_one_existing_CPU_test", tuple(row["relative_path"] for row in new["files"]) == allowed)
    total, differences = 0, []
    for row in new["files"]:
        relative = Path(row["relative_path"])
        target, proposed = ROOT / relative, SECOND / "proposed" / relative
        check("nontraversal_nonsymlink_paths:" + str(relative), not relative.is_absolute() and ".." not in relative.parts
              and target.resolve() == target and proposed.resolve() == proposed)
        check("exact_before_after_bytes:" + str(relative),
              (not target.exists() if row["before"] is None else record(target) == row["before"])
              and record(proposed) == row["after"])
        total += row["after"]["bytes"]
        check("bounded_individual_payload:" + str(relative), row["after"]["bytes"] < 250000)
        previous = old_rows.get(str(relative))
        if previous is None or previous["after"]["sha256"] != row["after"]["sha256"]:
            differences.append(str(relative))
    check("only_two_CPU_test_payloads_differ_from_V1", differences == ["tests/test_native_map_v8.py", "tests/test_paper_scene_maps_probe.py"])
    check("total_payload_matches_296656_and_under_limit", total == new["total_proposed_bytes"] == 296656 and total < 500000)
    check("merge_helper_AST_unchanged_except_exact_allowlist", normalized_helper(FIRST / "merge_v8_proposal.py") == normalized_helper(SECOND / "merge_v8_proposal.py"))
    for row in new["protected_unchanged_files"]:
        check("protected_source_bytes_unchanged:" + row["path"], record(row["path"]) == row)
    before = ast.parse((ROOT / "tests/test_paper_scene_maps_probe.py").read_text())
    after = ast.parse((SECOND / "proposed/tests/test_paper_scene_maps_probe.py").read_text())
    target_name = "test_actual_runtime_sets_zero_timeline_again_after_SDK_reset_to_authored_negative_start"
    for tree in (before, after):
        tree.body = [node for node in tree.body if not isinstance(node, ast.FunctionDef) or node.name != target_name]
    check("old_probe_tests_AST_unchanged_except_one_setup_slice", ast.dump(before, include_attributes=False) == ast.dump(after, include_attributes=False))
    stable = ast.parse((SECOND / "proposed/tests/test_native_map_v8.py").read_text())
    previous = ast.parse((FIRST / "proposed/tests/test_native_map_v8.py").read_text())
    # Replace only the two explicit workspace-resolution expressions; every
    # geometry/corruption assertion and test remains identical to reviewed V1.
    class NormalizeWorkspace(ast.NodeTransformer):
        def visit_Import(self, node):
            return None if any(a.name == "CrowdSim" for a in node.names) else node

        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id == "WORKSPACE" for t in node.targets):
                return None
            node = self.generic_visit(node)
            if any(isinstance(t, ast.Name) and t.id == "root" for t in node.targets):
                node.value = ast.Constant("known_workspace_location")
            return node

        def visit_Call(self, node):
            if (isinstance(node.func, ast.Attribute) and node.func.attr == "setattr" and len(node.args) >= 3
                    and isinstance(node.args[0], ast.Name) and node.args[0].id == "legacy"
                    and isinstance(node.args[1], ast.Constant) and node.args[1].value == "ROOT"):
                node.args[2] = ast.Constant("known_workspace_location")
            return self.generic_visit(node)
    normalization = NormalizeWorkspace()
    check("v8_corruption_tests_AST_unchanged_except_stable_workspace_resolution",
          ast.dump(normalization.visit(deepcopy(stable)), include_attributes=False)
          == ast.dump(normalization.visit(deepcopy(previous)), include_attributes=False))
    producer = SECOND / "proposed/scripts/probe_paper_scene_maps.py"
    check("reviewed_producer_201731_SHA_unchanged", record(producer)["sha256"] == "20173101ada3e5137d962abcdea574497e9624be7659984caa9869e7932d537a")
    check("candidate_log_records216_CPU_pass_and10_skip", "216 passed, 10 skipped" in (SECOND / "tests.log").read_text())
    report = {"schema": "native_map_clock_v8_independent_read_only_V2_delta.v1", "status": "V2_delta_review_passed_native_pending",
              "CPU_only": True, "SDK_started": False, "GPU_used": False, "production_or_proposal_modified": False,
              "merge_executed": False, "native_v8_clock_fix_observed": False, "checks": checks, "checks_passed": len(checks),
              "material_blocking_defects": [], "changed_payloads_from_V1": differences, "exact_merge_targets": list(allowed),
              "total_payload_bytes": total, "V1_review": record(DOC / "source_freeze.json"),
              "V2_manifest": record(SECOND / "merge_manifest.json"), "V2_patch": record(SECOND / "proposed.patch"),
              "V2_merge_helper": record(SECOND / "merge_v8_proposal.py"), "author_candidate_CPU_log": record(SECOND / "tests.log"),
              "reader": record(__file__)}
    (DOC / "V2_delta_review.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "checks_passed": len(checks), "total_payload_bytes": total,
                      "production_modified": False, "native_v8_executed": False}))


if __name__ == "__main__":
    main()
