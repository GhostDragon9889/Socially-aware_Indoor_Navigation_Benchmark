"""Reuse frozen measurement checks with one exact, pinned112-file source archive.

The prior auditor supports an older source-prefix set. This wrapper changes only
archive admission, and accepts the known manifest SHA and its exact112 members.
It neither modifies that auditor nor executes/imports archived source members.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile

DOCS = Path(__file__).resolve().parent.parent
ROOT = DOCS.parent.parent
MANIFEST = DOCS / "near_horizon_eta025_source_v1.json"
MANIFEST_SHA = "5fd9bb697fb015102b110557a29c86bad5aa72a674625b51bf045a318400e016"
ARCHIVE_SHA = "2b6cbfcb67376ed7de7e7da5bcbe0b443e35d0558c5ad8f59fd0aa28c3fed6cf"
INPUT = ROOT / "output/diffusion_amp/shared_amass2848_near_horizon_eta025_ground_support_200s_v1"
HELPER = DOCS / "audit_ground_support_capture.py"
CONFIGURATION = "CrowdSim/humanoid/application/configuration.py"
ADDED_CONFIGURATION_STATEMENTS = (
    "from ..waypoint_weighting import validate_near_horizon_mass",
    '''parser.add_argument("--near-horizon-waypoint-mass", type=float, default=0.0,
        help="Evaluation-only engineering redistribution eta in [0,1] toward timed waypoints<=1s; zero preserves original mean")''',
    '''try:
    validate_near_horizon_mass(args.near_horizon_waypoint_mass)
except ValueError as error:
    parser.error(str(error))''',
    '''if args.near_horizon_waypoint_mass > 0 and (not args.evaluate_steps or args.route_guidance != "timed_path"
            or args.diffusion_guidance_mode == "legacy_posterior" or args.trajectory_source != "diffusion"):
        parser.error("Near-horizon waypoint priority requires diffusion evaluation, timed_path and clean guidance")''',
)


def load_helper():
    specification = importlib.util.spec_from_file_location("eta025_frozen_ground_auditor", HELPER)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def verify_exact_archive(module, audit, path):
    path = Path(path).resolve()
    audit.require(path == MANIFEST.resolve(), "ETA source manifest is the one explicit prelaunch document")
    manifest_record = module.fingerprint(path)
    audit.require(manifest_record["sha256"] == MANIFEST_SHA, "exact112-member manifest full SHA256")
    manifest = json.loads(path.read_text())
    archive_path = ROOT / manifest["archive"]["path"]
    archive_record = module.fingerprint(archive_path)
    audit.require(archive_record["sha256"] == ARCHIVE_SHA == manifest["archive"]["sha256"]
                  and archive_record["bytes"] == manifest["archive"]["bytes"],
                  "prelaunch source archive bytes and full SHA256")
    declared = {entry["path"]: entry for entry in manifest["files"]}
    # Manifest SHA binds this exact membership. No arbitrary new source paths
    # are admitted by broadening a shared whitelist or repacking the archive.
    audit.require(len(declared) == len(manifest["files"]) == 112 and
                  all(not PurePosixPath(name).is_absolute() and ".." not in PurePosixPath(name).parts
                      and "\\" not in name for name in declared),
                  "exact pinned112 public relative paths without traversal")
    source = {}
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        audit.require(len(members) == 112 and {m.name for m in members} == set(declared)
                      and all(m.isfile() and m.size <= 16 * 1024 * 1024 for m in members),
                      "exact regular source members; no links or unexpected files")
        for member in members:
            data = archive.extractfile(member).read()
            entry = declared[member.name]
            audit.require(member.name not in source and len(data) == entry["bytes"]
                          and hashlib.sha256(data).hexdigest() == entry["sha256"],
                          f"exact prelaunch source member: {member.name}")
            source[member.name] = data
    verify_configuration_delta(module, audit, source[CONFIGURATION])
    return {"manifest": manifest_record, "archive": archive_record, "member_count": len(source),
            "member_records": manifest["files"], "current_tree_unchanged_claim": False,
            "admission": "one known112-member prelaunch manifest and archive, pinned by complete SHA256"}, source


def verify_configuration_delta(module, audit, source):
    """Prove all existing frontend/ground checks unchanged aside from four additions."""
    old_path = DOCS / "ground_support_capture_source_freeze" / CONFIGURATION
    frozen_identity = json.loads((DOCS / "ground_support_integration_review.json").read_text())["source_files"][CONFIGURATION]
    actual_identity = module.fingerprint(old_path)
    audit.require(actual_identity["sha256"] == frozen_identity["sha256"]
                  and actual_identity["bytes"] == frozen_identity["bytes"],
                  "old configuration snapshot matches frozen ground-review identity")
    expected = {ast.dump(ast.parse(statement).body[0], include_attributes=False)
                for statement in ADDED_CONFIGURATION_STATEMENTS}
    removed = []

    class RemoveExactAdditions(ast.NodeTransformer):
        def visit(self, node):
            dump = ast.dump(node, include_attributes=False)
            if dump in expected:
                removed.append(dump)
                return None
            return super().visit(node)

    projected = RemoveExactAdditions().visit(ast.parse(source))
    audit.require(len(removed) == 4 and set(removed) == expected,
                  "only four exact declared near-horizon frontend additions removed")
    audit.require(ast.dump(projected, include_attributes=False)
                  == ast.dump(ast.parse(old_path.read_text()), include_attributes=False),
                  "entire prior configuration AST identical after exact near-horizon additions")


def install_configuration_identity_adapter(module):
    """One local expected-source assignment, after a strict independent AST proof.

    The trusted original verification function is compiled unchanged apart from
    this assignment. Its full measurement tests and every other source check are
    reused; neither the original file nor its historical review JSON is edited.
    """
    entries = json.loads(MANIFEST.read_text())["files"]
    identity = next(entry for entry in entries if entry["path"] == CONFIGURATION)
    tree = ast.parse(inspect.getsource(module.verify_sources))
    original = copy.deepcopy(tree)
    function = tree.body[0]
    index = next(i for i, node in enumerate(function.body) if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "frozen" for target in node.targets))
    insertion = ast.parse(f"frozen[{CONFIGURATION!r}] = ETA_CONFIGURATION_IDENTITY").body[0]
    function.body.insert(index + 1, insertion)
    audit_projection = copy.deepcopy(tree)
    del audit_projection.body[0].body[index + 1]
    assert ast.dump(audit_projection, include_attributes=False) == ast.dump(original, include_attributes=False)
    module.ETA_CONFIGURATION_IDENTITY = identity
    exec(compile(ast.fix_missing_locations(tree), str(HELPER), "exec"), module.__dict__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".md").exists():
        parser.error("Fresh report paths required; historical failures remain unchanged")
    module = load_helper()
    # Archive verification and exact4-statement proof run before this expected
    # configuration identity is used by the original source verification loop.
    install_configuration_identity_adapter(module)
    module.verify_archive = lambda audit, path: verify_exact_archive(module, audit, path)
    audit = module.Audit()
    try:
        report = module.audit_capture(INPUT, audit, source_archive_manifest=MANIFEST)
        report["status"] = "passed"
    except Exception as error:
        report = {"status": "failed", "input": str(INPUT.resolve()),
                  "actual_native_capture_validated": False, "error": f"{type(error).__name__}: {error}"}
    report["checks"] = audit.checks
    report["audit_script"] = module.fingerprint(__file__)
    report["unchanged_measurement_auditor"] = module.fingerprint(HELPER)
    report["archive_admission_change_only"] = True
    report["frontend_identity_delta"] = "exact4 near-horizon additions, whole remaining AST equals old frozen configuration"
    report["all_measurement_checks_unchanged"] = True
    report["historical205_check_failure_preserved"] = "ground_support_actual_verification.json"
    report["command"] = sys.argv
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    if report["status"] == "passed":
        text = module.markdown(report)
        command_start = "CUDA_VISIBLE_DEVICES='' ../IsaacLab-Protomotions/.venv/bin/python"
        text = text[:text.index(command_start)] + command_start + (
            " docs/style_improvement_20261002/navigation_near_horizon_eta025_comparison_v1/"
            "audit_eta025_ground_capture.py --output /tmp/eta025_ground_capture.json\n```\n\n"
            "复用原始完整测量审计，仅以固定manifest和archive SHA允许已知112个公开源码成员。"
            "configuration仅四个近段guidance新增节点，剥离后whole AST与旧冻结文件相同；其余足部源和完整控制循环严格一致。"
            "旧205/321项失效报告保留；不存在测量检查放宽、runtime/evaluator修改或current-tree全部相同声明。\n")
        with args.output.with_suffix(".md").open("x") as stream:
            stream.write(text)
    print(json.dumps({"status": report["status"], "checks": len(audit.checks),
                      "output": str(args.output), "error": report.get("error")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
