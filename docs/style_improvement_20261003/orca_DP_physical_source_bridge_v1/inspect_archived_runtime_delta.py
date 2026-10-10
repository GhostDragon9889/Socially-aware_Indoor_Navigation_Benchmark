"""Finite readonly archive/function comparison; no branch-equivalence claim."""
import ast
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PLAN = ROOT / "docs/style_improvement_20261003/local_Office_baseline_probe_preparation_v1/plan.json"
CURRENT = ROOT / "docs/style_improvement_20261003/native_runtime_epoch_v12/native_pedestrian_source_v12.json"
CORE = (
    "CrowdSim/humanoid/amp/controller.py", "CrowdSim/humanoid/amp/network.py", "CrowdSim/humanoid/amp/observations.py",
    "CrowdSim/humanoid/amp_world.py", "CrowdSim/humanoid/physics_views.py", "CrowdSim/humanoid/evaluation.py",
    "CrowdSim/humanoid/diffusion/planner.py", "CrowdSim/humanoid/diffusion/paper.py",
    "CrowdSim/humanoid/runtime/evaluation_actor.py", "CrowdSim/humanoid/application/deployment.py",
    "CrowdSim/humanoid/application/native_deployment.py", "CrowdSim/humanoid/contact_monitor.py",
    "CrowdSim/humanoid/foot_capture.py", "CrowdSim/humanoid/foot_support.py",
    "CrowdSim/humanoid/ground_support_archive.py", "CrowdSim/humanoid/motion_reset.py",
    "protomotions/simulator/isaaclab/config.py", "protomotions/simulator/isaaclab/simulator.py",
    "protomotions/simulator/isaaclab/utils/scene.py", "protomotions/utils/rotations.py",
)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def record(path):
    b = path.read_bytes()
    return {"path": str(path), "bytes": len(b), "sha256": digest(b)}


def normalized_rows(manifest):
    return {str(Path(row["path"]).relative_to(ROOT)) if Path(row["path"]).is_absolute() else row["path"]: row
            for row in manifest["files"]}


def archive_payloads(path, declared):
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        assert len(members) == len(declared) and {m.name for m in members} == set(declared)
        payloads = {}
        for m in members:
            assert m.isfile()
            b = archive.extractfile(m).read()
            assert len(b) == declared[m.name]["bytes"] and digest(b) == declared[m.name]["sha256"]
            payloads[m.name] = b
    return payloads


def functions(source):
    result = {}

    def visit(body, prefix=""):
        for node in body:
            if isinstance(node, ast.ClassDef):
                visit(node.body, prefix + node.name + ".")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                result[prefix + node.name] = ast.dump(node, include_attributes=False)

    visit(ast.parse(source).body)
    return result


def main():
    plan = json.loads(PLAN.read_text())
    current = json.loads(CURRENT.read_text())
    rows = normalized_rows(current)
    assert len(rows) == 185
    current_archive = Path(plan["production_source"]["archive"]["path"])
    assert record(current_archive) == plan["production_source"]["archive"]
    payloads = archive_payloads(current_archive, rows)
    output = []
    for actors in ("4", "15"):
        historical = plan["historical_baseline_native_stability"][actors]
        epoch = historical["historical_source_epoch"]
        manifest_path, archive_path = Path(epoch["manifest"]["path"]), Path(epoch["archive"]["path"])
        assert record(manifest_path) == epoch["manifest"] and record(archive_path) == epoch["archive"]
        old = normalized_rows(json.loads(manifest_path.read_text()))
        old_payloads = archive_payloads(archive_path, old)
        changed = sorted(name for name in old.keys() & rows.keys() if old_payloads[name] != payloads[name])
        exact_core = []
        for name in CORE:
            assert old_payloads[name] == payloads[name]
            exact_core.append({"path": name, "bytes": len(payloads[name]), "sha256": digest(payloads[name])})
        function_delta = []
        for name in changed:
            if not name.endswith(".py") or name.startswith("tests/"):
                continue
            before, after = functions(old_payloads[name]), functions(payloads[name])
            function_delta.append({"file": name,
                                   "unchanged_existing_function_AST": sorted(k for k in before.keys() & after.keys() if before[k] == after[k]),
                                   "changed_existing_function_AST": sorted(k for k in before.keys() & after.keys() if before[k] != after[k]),
                                   "added_function_AST": sorted(after.keys() - before.keys()),
                                   "removed_function_AST": sorted(before.keys() - after.keys())})
        output.append({"historical_actors": int(actors), "historical_epoch": epoch,
                       "historical_complete_raw_audit": historical["audit"],
                       "common_files_exact": len(old) - len(changed), "changed_existing_files": changed,
                       "new_files": sorted(rows.keys() - old.keys()), "removed_files": sorted(old.keys() - rows.keys()),
                       "twenty_core_files_whole_bytes_exact": exact_core, "changed_python_function_AST_inventory": function_delta,
                       "all_archived_source_bytes_verified": True})
    report = {"schema": "orca_DP.historical_vs_current_runtime_source_inventory.v1", "status": "source_inventory_only",
              "current_source_manifest": record(CURRENT), "current_archive": record(current_archive), "current_source_count": 185,
              "comparisons": output,
              "does_not_prove_all_runtime_branches_equivalent": True,
              "historical_oldDP_vs_current_newDP_is_same_epoch_paired_experiment": False,
              "recommended_actual_comparison": "On one fixed current source/profile/controller/scene/protocol, run old63e and newb563 separately, retain all6000controls/4actors/200s and failed results. Native4 requires complete zero F/static/peer fixture; native15 requires complete native4 physicalzero.",
              "SDK_started": False, "GPU_used": False, "checkpoint_deserialized": False, "sealed34_outputs_read": False}
    with (HERE / "source_inventory.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "current_sources": 185,
                      "historical_changed_counts": [len(r["changed_existing_files"]) for r in output],
                      "core_whole_byte_equal_files_each": len(CORE), "runtime_equivalence_claimed": False}))


if __name__ == "__main__":
    main()
