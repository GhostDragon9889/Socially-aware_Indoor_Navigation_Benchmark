"""Reuse frozen native measurements for one pinned127-member navigation run.

Only prelaunch source admission is adapted. The full force/COM/clock/contact
checks and thresholds remain those of the frozen ground-support auditor.
No archived Python is imported or executed and no current-tree identity is
substituted for the exact prelaunch archive.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent.parent
MANIFEST = DOCS / "amass_root_step1000_navigation_source_v1.json"
MANIFEST_SHA = "c9be09f46086c5909b864806bf9242d9f422a64122f1da88888443b760d2b752"
ARCHIVE_SHA = "c5c8c4d26ffe6220405742ec842f683a708c84a8afcbd41489bf5e9ade5d7dd5"
TRAINING_MANIFEST_SHA = "044369bf015141c6483c8044f9c6a6c5f9a50655f1ea867f621d7fa7893cbb20"
HELPER = DOCS / "audit_ground_support_capture.py"
HELPER_SHA = "f851e7d5f90d48d830c564fdc2b2e446dfe35daa36ec636aba6f946ec7752c3f"
INDEPENDENT = DOCS / "amass_root_curriculum_independent_integration.json"
INDEPENDENT_SHA = "bb801d3727ddaf11286461782b0923f379f3fe7bcaf365912de6076c438efcfb"
CONFIGURATION = "CrowdSim/humanoid/application/configuration.py"
INPUT = ROOT / "output/diffusion_amp/shared_amass_root_curriculum_ground_support_200s_v1"
FLOOR_CONTROL_AST_SHA = "1adbfe51b09a2311b3e450925cface7b4f81a1fbb122dd945838d16a9272a74f"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_helper():
    assert sha(HELPER.read_bytes()) == HELPER_SHA
    spec = importlib.util.spec_from_file_location("amass_root_frozen_ground_auditor", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ground_cli_nodes(text):
    """Exact ground options and enclosing standalone gates, by function name."""
    selected = []
    for function in ast.parse(text).body:
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in function.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
                call = node.value
                if any(isinstance(arg, ast.Constant) and arg.value in {
                        "--measure-ground-support", "--ground-support-threshold-n"} for arg in call.args):
                    selected.append((function.name, ast.dump(node, include_attributes=False)))
            if isinstance(node, ast.If) and any(isinstance(child, ast.Attribute) and
                    child.attr in {"measure_ground_support", "ground_support_threshold_n"}
                    for child in ast.walk(node.test)):
                selected.append((function.name, ast.dump(node, include_attributes=False)))
    return selected


def verify_exact_archive(module, audit, path):
    audit.require(Path(path).resolve() == MANIFEST.resolve(), "one explicit127-member prelaunch manifest")
    manifest_record = module.fingerprint(path)
    audit.require(manifest_record["sha256"] == MANIFEST_SHA, "known127-member manifest full SHA256")
    manifest = json.loads(MANIFEST.read_text())
    archive_path = ROOT / manifest["archive"]["path"]
    archive_record = module.fingerprint(archive_path)
    audit.require(archive_record["sha256"] == ARCHIVE_SHA == manifest["archive"]["sha256"]
                  and archive_record["bytes"] == manifest["archive"]["bytes"],
                  "known source archive exact bytes and SHA256")
    declared = {entry["path"]: entry for entry in manifest["files"]}
    audit.require(len(declared) == len(manifest["files"]) == 127 and all(
        not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts
        and "\\" not in name for name in declared), "pinned127 unique relative nontraversal source paths")
    source = {}
    with tarfile.open(archive_path) as stream:
        members = stream.getmembers()
        audit.require(len(members) == 127 and {item.name for item in members} == set(declared)
                      and all(item.isfile() and item.size <= 16*1024*1024 for item in members),
                      "exact regular127 archive members; no links or extra paths")
        for item in members:
            raw = stream.extractfile(item).read()
            expected = declared[item.name]
            audit.require(item.name not in source and len(raw) == expected["bytes"]
                          and sha(raw) == expected["sha256"], "source member:"+item.name)
            source[item.name] = raw
    training_path = ROOT/manifest["training_source_manifest"]
    audit.require(sha(training_path.read_bytes()) == TRAINING_MANIFEST_SHA
                  == manifest["training_source_manifest_sha256"], "parent124 training manifest identity")
    training = {entry["path"]: entry for entry in json.loads(training_path.read_text())["files"]}
    changes = {name for name in training if declared[name]["sha256"] != training[name]["sha256"]
               or declared[name]["bytes"] != training[name]["bytes"]}
    added = set(declared)-set(training)
    audit.require(len(training) == 124 and set(training) <= set(declared)
                  and changes == {"docs/style_improvement_20261002/watch_amass_root_checkpoints.py"}
                  and added == {"docs/style_improvement_20261002/watch_amass_root_checkpoints_actual_v2.py",
                                "docs/style_improvement_20261002/amass_root_step1000_ground_support_200s_preparation_v1.json",
                                "docs/style_improvement_20261002/amass_root_step1000_candidate_selection_v1.json"},
                  "124 parent paths change only CPU watcher plus three exact CPU provenance documents")
    audit.require(sha(INDEPENDENT.read_bytes()) == INDEPENDENT_SHA, "original independent CPU gate full identity")
    independent = json.loads(INDEPENDENT.read_text())
    audit.require(independent["status"] == "passed" and all(declared[name]["sha256"] == digest
                  for name, digest in independent["sources"].items()),
                  "all independently reviewed new curriculum/core/config sources remain exact")
    old_configuration = DOCS/"ground_support_capture_source_freeze"/CONFIGURATION
    old_identity = json.loads((DOCS/"ground_support_integration_review.json").read_text())["source_files"][CONFIGURATION]
    audit.require(module.fingerprint(old_configuration)["sha256"] == old_identity["sha256"],
                  "old ground configuration reference retains reviewed identity")
    old_nodes = ground_cli_nodes(old_configuration.read_text())
    new_nodes = ground_cli_nodes(source[CONFIGURATION].decode())
    audit.require(len(old_nodes) == 6 and old_nodes == new_nodes,
                  "exact two ground options and four eval/power/threshold/overwrite gates unchanged")
    def loop(text):
        nodes = [node for node in ast.walk(ast.parse(text)) if isinstance(node, ast.For)
                 and isinstance(node.target, ast.Name) and node.target.id == "step"]
        audit.require(len(nodes) == 1, "one explicit outer physical control loop")
        return ast.dump(nodes[0], include_attributes=False)
    old_runner = DOCS/"ground_support_capture_source_freeze/scripts/train_paper_amp.py"
    new_loop = loop(source["scripts/train_paper_amp.py"].decode())
    audit.require(sha(new_loop.encode()) == FLOOR_CONTROL_AST_SHA and new_loop == loop(old_runner.read_text()),
                  "entire physical control loop equals original reviewed ground capture AST")
    return {"manifest": manifest_record, "archive": archive_record, "member_count": 127,
            "member_records": manifest["files"], "current_tree_unchanged_claim": False,
            "admission": "one known127-member archive, exact full hashes; independent new configuration proof",
            "independent_CPU_gate": module.fingerprint(INDEPENDENT)}, source


def install_configuration_identity_adapter(module):
    """One expected-source assignment; every measurement gate stays unchanged."""
    rows = json.loads(MANIFEST.read_text())["files"]
    identity = next(row for row in rows if row["path"] == CONFIGURATION)
    tree = ast.parse(inspect.getsource(module.verify_sources))
    original = copy.deepcopy(tree)
    function = tree.body[0]
    index = next(i for i, node in enumerate(function.body) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "frozen" for target in node.targets))
    function.body.insert(index+1, ast.parse(f"frozen[{CONFIGURATION!r}] = NAV_CONFIGURATION_IDENTITY").body[0])
    projection = copy.deepcopy(tree)
    del projection.body[0].body[index+1]
    assert ast.dump(projection, include_attributes=False) == ast.dump(original, include_attributes=False)
    module.NAV_CONFIGURATION_IDENTITY = identity
    exec(compile(ast.fix_missing_locations(tree), str(HELPER), "exec"), module.__dict__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preparation-only", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists() or args.output.with_suffix(".md").exists():
        parser.error("Empty CUDA visibility and fresh report paths required")
    module = load_helper()
    audit = module.Audit()
    try:
        preparation, _ = verify_exact_archive(module, audit, MANIFEST)
        if args.preparation_only:
            report = {"status": "passed", "preparation_only": True, "actual_native_capture_validated": False,
                      "source_preparation": preparation, "measurement_artifacts_not_read": True}
        else:
            install_configuration_identity_adapter(module)
            module.verify_archive = lambda audit, path: verify_exact_archive(module, audit, path)
            report = module.audit_capture(INPUT, audit, source_archive_manifest=MANIFEST)
            report.update(status="passed", preparation_only=False)
    except Exception as error:
        report = {"status": "failed", "input": str(INPUT.resolve()), "preparation_only": args.preparation_only,
                  "actual_native_capture_validated": False, "error": f"{type(error).__name__}: {error}"}
    report.update(CPU_only=True, SDK_started_by_auditor=False, GPU_used_by_auditor=False,
        heldout34_test_outputs_read=False, checks=audit.checks, audit_script=module.fingerprint(__file__),
        unchanged_measurement_auditor=module.fingerprint(HELPER), all_measurement_checks_unchanged=True,
        configuration_admission="exact independent CPU-gated source; ground CLI gates and full physical loop unchanged",
        command=sys.argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    if report["status"] == "passed" and not args.preparation_only:
        text = module.markdown(report)
        command = "CUDA_VISIBLE_DEVICES='' ../IsaacLab-Protomotions/.venv/bin/python"
        text = text[:text.index(command)] + command + (
            " docs/style_improvement_20261002/audit_amass_root_navigation_ground.py"
            " --output /tmp/amass_root_navigation_ground_audit.json\n```\n\n"
            "严格绑定已知127文件prelaunch快照；新增configuration通过原独立CPU课程审查，"
            "两ground CLI项及四门禁与旧冻结源码AST相等，完整物理控制循环逐位AST相同。"
            "全部测量公式和检查复用原审计器；不要求后续current tree等于历史源码。\n")
        with args.output.with_suffix(".md").open("x") as stream:
            stream.write(text)
    print(json.dumps({"status": report["status"], "checks": len(audit.checks),
                      "preparation_only": args.preparation_only, "error": report.get("error")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
