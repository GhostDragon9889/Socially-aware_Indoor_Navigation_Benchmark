"""CPU verification of the actual source artifact; no models or SDK imports."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import xml.etree.ElementTree as ET


REPO = Path(__file__).resolve().parents[2]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=Path(
        "/tmp/protomotions-final-packaging-20261001/protomotions-3.1.0.tar.gz"
    ))
    parser.add_argument("--output", type=Path, default=REPO / "docs/amp_improvement_20261001/packaging_structure_verification.json")
    parser.add_argument("--junit", type=Path, default=REPO / "docs/amp_improvement_20261001/packaging_tests.xml")
    parser.add_argument("--allow-current-guide-updates", action="store_true",
                        help="Record later edits to the two root-owned Chinese guides; code and all other files must still match")
    args = parser.parse_args()
    required = sorted(str(path.relative_to(REPO)) for path in (
        REPO / "CrowdSim/humanoid/application"
    ).glob("*.py")) + [
        "CrowdSim/humanoid/training/trajectories.py",
        "CrowdSim/humanoid/training/reference.py",
        "CrowdSim/humanoid/contact_monitor.py",
        "CrowdSim/tools/smpl_mesh_visualizer.py",
        "scripts/train_paper_amp.py",
        "scripts/benchmark_paper_amp_controller.py",
        "scripts/benchmark_paper_clean_guidance.py",
        "scripts/fetch_pacer_reference.py",
        "scripts/evaluate_paper_pedestrians.py",
        "scripts/probe_paper_pedestrian_contacts.py",
        "tests/test_reference_adapter.py",
        "tests/test_amp_application.py",
        "tests/test_pedestrian_contact_monitor.py",
        "tests/test_amp_deterministic_evaluation.py",
        "FULL_PIPELINE.md",
        "docs/STRUCTURE_AND_EVALUATION_CURRENT_ZH.md",
        "docs/PEDESTRIAN_AMP_DIFFUSION_PROGRESS_20261001_ZH.md",
    ]
    replay = "scripts/replay_paper_pedestrians.py"
    if (REPO / replay).is_file():
        required.extend([replay, "tests/test_paper_recorded_replay.py"])
    with tarfile.open(args.artifact) as archive:
        members = {
            member.name.split("/", 1)[-1]: member
            for member in archive.getmembers() if member.isfile()
        }
        missing = sorted(set(required) - members.keys())
        content_match = {
            path: sha256(archive.extractfile(members[path]).read())
            == sha256((REPO / path).read_bytes())
            for path in required if path in members
        }
        required_hashes = {
            path: sha256(archive.extractfile(members[path]).read())
            for path in required if path in members
        }
        forbidden = [
            path for path in members
            if path.startswith(("output/", "Assets/", "data/pretrained_models/", "data/scene_resources/"))
            or path.startswith(("protomotions/data/assets/mjcf/smpl", "protomotions/data/assets/mesh/smpl/",
                                "protomotions/data/assets/usd/smpl"))
            or path.lower().endswith((".pt", ".pth", ".ckpt", ".pkl", ".hdr", ".npz", ".mp4"))
        ]
        extraction = Path(tempfile.mkdtemp(prefix="protomotions-source-import-20261001-", dir="/tmp"))
        archive.extractall(extraction, filter="data")
    source = next(extraction.glob("protomotions-*"))
    probe = '''import contextlib, io, json, os, sys
from pathlib import Path
import CrowdSim.humanoid.application.configuration as configuration
import CrowdSim.humanoid.application.settings as settings
import CrowdSim.humanoid.application.resources as resources
import CrowdSim.humanoid.application.preflight as preflight
settings.TrainingConfig().validate()
profile = configuration.load_amp_profile("configs/diffusion_amp/method_reconstruction.yaml")
os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
workspace = configuration.validate_deterministic_evaluation_environment(True)
workspace_readonly = "CUBLAS_WORKSPACE_CONFIG" not in os.environ
os.environ["CUBLAS_WORKSPACE_CONFIG"] = "package-probe-invalid"
try: configuration.validate_deterministic_evaluation_environment(True)
except ValueError: invalid_workspace_rejected = True
else: invalid_workspace_rejected = False
invalid_workspace_unchanged = os.environ["CUBLAS_WORKSPACE_CONFIG"] == "package-probe-invalid"
os.environ.pop("CUBLAS_WORKSPACE_CONFIG")
with contextlib.redirect_stdout(io.StringIO()) as help_text:
    try: configuration.parse_amp_args(["--help"])
    except SystemExit as exc: help_exit = exc.code
with contextlib.redirect_stderr(io.StringIO()) as error_text:
    try: configuration.parse_amp_args([])
    except SystemExit as exc: no_weights_exit = exc.code
for path in ("CrowdSim/humanoid/training/trajectories.py", "CrowdSim/humanoid/training/reference.py", "CrowdSim/humanoid/contact_monitor.py"):
    compile(Path(path).read_text(), path, "exec")
print(json.dumps({"application_source": configuration.__file__,
    "torch_or_kit_imports": [name for name in sys.modules if name == "torch" or name.startswith(("isaaclab", "isaacsim", "omni"))],
    "help_exit": help_exit, "help_has_diffusion_samples": "--diffusion-samples" in help_text.getvalue(),
    "help_has_deterministic_evaluation": "--deterministic-evaluation" in help_text.getvalue(),
    "cublas_preflight": {"recommended_workspace": workspace, "does_not_mutate_environment": workspace_readonly,
        "invalid_workspace_rejected": invalid_workspace_rejected, "invalid_workspace_unchanged": invalid_workspace_unchanged},
    "default_missing_weights_exit": no_weights_exit,
    "default_missing_weights_error": error_text.getvalue().strip().splitlines()[-1],
    "profile_mode": profile["mode"]}))
'''
    env = dict(os.environ, PYTHONPATH=str(source), CUDA_VISIBLE_DEVICES="")
    process = subprocess.run(
        [sys.executable, "-c", probe], cwd=source, env=env,
        capture_output=True, text=True, timeout=30,
    )
    if process.returncode:
        raise RuntimeError(process.stderr)
    probe_result = json.loads(process.stdout)
    mutable_guides = {"docs/STRUCTURE_AND_EVALUATION_CURRENT_ZH.md",
                      "docs/PEDESTRIAN_AMP_DIFFUSION_PROGRESS_20261001_ZH.md"}
    changed = [path for path, match in content_match.items() if not match]
    allowed_document_changes = {
        path: {"archive_sha256": required_hashes[path], "current_sha256": sha256((REPO / path).read_bytes())}
        for path in changed if args.allow_current_guide_updates and path in mutable_guides
    }
    unexpected_changed = sorted(set(changed) - allowed_document_changes.keys())
    before_path = REPO / "docs/amp_improvement_20261001/before/MANIFEST.in"
    manifest = REPO / "MANIFEST.in"
    before_lines = set(before_path.read_text().splitlines())
    added = [line.removeprefix("include ") for line in manifest.read_text().splitlines()
             if line.startswith("include ") and line not in before_lines]
    junit = args.junit.resolve()
    suites = list(ET.parse(junit).getroot().iter("testsuite"))
    tests = sum(int(suite.attrib["tests"]) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", 0)) + int(suite.attrib.get("errors", 0)) for suite in suites)
    report = {
        "schema": "navisaaclab.package_structure_verification.v1",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_snapshot": "Current frozen deterministic-evaluation, replay-v2 and mesh code; guide contents are the build-time snapshot and may be updated afterwards.",
        "scope": "Actual setuptools sdist built without isolation/network; only MANIFEST.in changed. Wheel package discovery and business code unchanged.",
        "build_backend": "setuptools.build_meta.build_sdist",
        "build_command_note": "Repository-local build namespace shadows python -m build. The configured PEP517 backend was called directly; no dependencies installed.",
        "source_distribution": {"path": str(args.artifact), "bytes": args.artifact.stat().st_size,
                                "sha256": sha256(args.artifact.read_bytes()), "file_count": len(members),
                                "archive_mtime_utc": datetime.fromtimestamp(args.artifact.stat().st_mtime, timezone.utc).isoformat()},
        "manifest": {"path": "MANIFEST.in", "before_path": str(before_path.relative_to(REPO)),
                     "before_sha256": sha256(before_path.read_bytes()), "sha256": sha256(manifest.read_bytes()),
                     "added_files": added},
        "required_source_files": required,
        "missing": missing,
        "required_bytes_match_current": content_match,
        "required_source_sha256": required_hashes,
        "current_guide_updates_explicitly_allowed": args.allow_current_guide_updates,
        "current_guide_changes_after_snapshot": allowed_document_changes,
        "unexpected_changed_files": unexpected_changed,
        "forbidden_binaries_or_licensed_smpl": forbidden,
        "extracted_source_frontend_probe": probe_result,
        "packaging_tests": {"path": str(junit.relative_to(REPO)), "tests": tests, "failures_or_errors": failures},
        "boundary": {
            "core_wheel": "Only protomotions code and approved non-SMPL robot assets; CrowdSim remains a source-distribution/source-checkout workflow.",
            "source_application": "Application, trajectories, dense reference, contact helper, training/evaluation scripts, YAML/JSON configs and small maps are included.",
            "external_resources": "SMPL parameters, author data, weights and large scenes/textures/HDR/video/NPZ remain external. Extracted source rejects absent default weights before Torch/SDK import.",
            "documentation": "Current entry guides included. Historical evidence links refer to the full source checkout and are not all embedded in sdist.",
        },
    }
    report["status"] = "passed" if (
        not missing and not unexpected_changed and not forbidden and failures == 0
        and not probe_result["torch_or_kit_imports"] and probe_result["help_exit"] == 0
        and probe_result["default_missing_weights_exit"] == 2
        and probe_result["help_has_deterministic_evaluation"]
        and all(value for key, value in probe_result["cublas_preflight"].items() if key != "recommended_workspace")
    ) else "failed"
    if report["status"] == "passed" and allowed_document_changes:
        report["status"] = "passed_code_snapshot_with_later_guide_updates"
    output = args.output
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "files": len(members), "missing": missing,
                      "changed_after_build": [path for path, match in content_match.items() if not match],
                      "heavy_imports": probe_result["torch_or_kit_imports"]}))
    return 0 if report["status"].startswith("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
