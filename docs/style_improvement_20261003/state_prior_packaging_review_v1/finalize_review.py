"""Preserve the already built archive and bind the actual CPU evidence."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def copy_once(source, target, expected):
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(source, target)
    actual = record(target)
    require(actual["bytes"] == expected["bytes"] and actual["sha256"] == expected["sha256"],
            "Existing/new preserved artifact differs: " + str(target))
    return actual


def main():
    build = json.loads((HERE / "sdist_verification.json").read_text())
    frontend = json.loads((HERE / "extracted_frontend_checks.json").read_text())
    require(build["status"] == frontend["status"] == "passed", "CPU gates did not pass")
    require(len(build["required_paths"]) == 28 and build["original_required_paths_count"] == 26,
            "Required distribution contract changed")
    require(len(build["nine_entrypoint_bytes"]) == 9 and len(build["new_helper_and_profile_bytes"]) == 2,
            "Missing entrypoint or new coupled initialization input")
    require(not build["forbidden_resource_members"] and not build["heavy_modules_imported"] and
            not frontend["heavy_modules_imported"], "External resource or runtime leaked into source gate")
    archive = copy_once(Path(build["artifact"]["path"]),
                        HERE / "artifacts/protomotions-3.1.0-v11-review.tar.gz", build["artifact"])
    guide_paths = ["docs/AMP_STYLE_PROGRESS_20261002_ZH.md", "FULL_PIPELINE.md",
                   "CrowdSim/humanoid/README.md", "docs/STRUCTURE_AND_EVALUATION_CURRENT_ZH.md"]
    guide_rows = []
    for path in guide_paths:
        snapshot = next(row for row in build["copied_inputs"] if row["path"] == path)
        preserved = copy_once(Path(build["copy_tree"]) / path,
                              HERE / "guide_snapshots" / path, snapshot)
        current = record(ROOT / path)
        guide_rows.append({"path": path, "build_copy": snapshot, "preserved_copy": preserved,
                           "current_at_review": current,
                           "current_matches_build_copy": snapshot["sha256"] == current["sha256"],
                           "selected_v11_member": False})
    frozen_path = Path(build["frozen_v11_manifest"]["path"])
    require(record(frozen_path)["sha256"] == build["frozen_v11_manifest"]["sha256"],
            "V11 manifest changed")
    frozen = json.loads(frozen_path.read_text())
    require(len(frozen["files"]) == frozen["source_count"] == 183, "V11 source count changed")
    source_rows = []
    for row in frozen["files"]:
        now = record(row["path"])
        require(now["bytes"] == row["bytes"] and now["sha256"] == row["sha256"],
                "Frozen V11 selected source changed: " + row["path"])
        source_rows.append({"path": row["path"], "bytes": row["bytes"], "sha256": row["sha256"],
                            "current_bytes_exact": True})
    frozen_names = {str(Path(row["path"]).relative_to(ROOT)) for row in frozen["files"]}
    require(not (set(guide_paths) & frozen_names), "Guide was unexpectedly selected into V11")
    egg = {str(path.relative_to(ROOT)): {key: value for key, value in record(path).items() if key != "path"}
           for path in sorted((ROOT / "protomotions.egg-info").rglob("*")) if path.is_file()}
    require(egg == build["workspace_all_egg_info_before"] == build["workspace_all_egg_info_after"],
            "Production egg-info changed")
    old_proof = ROOT / "docs/current_structure_review_20261003/packaging/sdist_verification.json"
    previous = json.loads(old_proof.read_text())
    old_archive = record(previous["artifact"]["path"])
    require(old_archive["bytes"] == previous["artifact"]["bytes"] and
            old_archive["sha256"] == previous["artifact"]["sha256"], "Old archive changed")
    evidence_names = ["MANIFEST.in.before", "build_isolated_v11_sdist.py", "build.log",
                      "sdist_verification.json", "check_extracted_v11_frontend.py", "extracted_frontend.log",
                      "extracted_frontend_failed_parser.log", "extracted_frontend_checks.json",
                      "simulate_paper_pedestrians_help.txt", "train_paper_amp_help.txt",
                      "packaged_manifest_absent.json", "from_scratch_rejected.json",
                      "explicit_external_resources.json", "finalize_review.py"]
    proof = {"schema": "public_state_prior.v11_packaging_review.v1", "status": "passed",
             "checked_at_utc": datetime.now(timezone.utc).isoformat(),
             "build_report_snapshot_complete_by_utc": build["checked_at_utc"],
             "archive": archive, "original_tmp_archive": build["artifact"],
             "frozen_v11_manifest": record(frozen_path), "selected_v11_source_count": 183,
             "selected_v11_current_source_exact": source_rows,
             "sdist_v11_source_members_exact_count": len(build["frozen_v11_source_members_exact"]),
             "v11_evidence_paths_not_declared_for_distribution": build["frozen_v11_evidence_paths_not_declared_for_distribution"],
             "nine_entrypoints_exact": build["nine_entrypoint_bytes"],
             "original26_and_new2_required_exact": build["required_paths"],
             "new_helper_and_profile_exact": build["new_helper_and_profile_bytes"],
             "forbidden_resource_member_count": 0, "production_egg_info_unchanged": True,
             "production_egg_info_current": egg, "guide_snapshot_rows": guide_rows,
             "guide_snapshot_time_scope": "The preserved actual package copied inputs were verified against production when the build report completed at 02:40:13 UTC. Four current guide prose updates occurred afterward; they are not members of the frozen selected V11 source183. No rebuild of this archive was performed.",
             "old_distribution_preserved": {"proof": record(old_proof), "archive": old_archive},
             "cpu_evidence": [record(HERE / name) for name in evidence_names],
             "scope": "Existing installed PEP517 backend, isolated /tmp build and extraction; no dependency installation, GPU/Kit/training, production edits or model deserialization. Positive resource preflight reads explicit external original resources; source archive contains none.",
             "historical_harness_correction": "The first extracted resource call intentionally omitted external resources but CLI parsing rejects a missing diffusion file before resource inspection. Its failure log is retained. The final gate supplies the immutable external diffusion path explicitly and tests absent packaged AMASS manifest, forbidden from-scratch, and successful explicit external-resource preflight.",
             "historical_diagnostic_correction": "An earlier agent message incorrectly conflated package copied_inputs with selected V11 files for humanoid README. Actual manifest has no README basename, all183 current source bytes are exact, and no V12 was created.",
             "limitations": ["Source distribution snapshot, not dependency installation or wheel execution.",
                             "Nine requested entrypoints are byte-verified; two original cold help frontends are executed.",
                             "Local presence/header/public checksum checks do not prove physics, tensor compatibility or style quality.",
                             "Thirteen frozen V11 evidence-only docs paths remain outside the declared source distribution."]}
    (HERE / "packaging_review.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps({key: proof[key] for key in ("status", "archive", "selected_v11_source_count",
                      "sdist_v11_source_members_exact_count", "forbidden_resource_member_count",
                      "production_egg_info_unchanged")}, indent=2))


if __name__ == "__main__":
    main()
