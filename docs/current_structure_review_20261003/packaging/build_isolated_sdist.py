"""Copy packaging inputs to /tmp and build via installed PEP 517 backend.

No project module imports, environment installation or in-workspace egg-info
write. Small documentation/configuration files are source-checkout examples;
external weights, motion banks, body parameters and DynBench scenes are absent.
"""
from __future__ import annotations

from datetime import datetime, timezone
import fnmatch
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_paths():
    # The backend reads this template itself; it must exist in the copied tree.
    result = {ROOT / "MANIFEST.in"}
    for line in (ROOT / "MANIFEST.in").read_text().splitlines():
        words = line.split()
        if words and words[0] == "include":
            for pattern in words[1:]:
                result.update(path for path in ROOT.glob(pattern) if path.is_file())
    for directory, suffixes in (
        ("protomotions", {".py"}), ("examples/experiments", {".py"}),
        ("CrowdSim", {".py", ".yaml", ".md", ".sh"}),
        ("CrowdSim/maps", {".png"}), ("src", {".py", ".toml", ".md", ".json"}),
        ("configs", {".json", ".yaml"}), ("tests", {".py"}),
        ("data/scripts", {".py", ".sh", ".yaml"}),
        ("legal", {".txt"}), ("env/isaaclab3_sim6", {".toml", ".lock", ".json", ".md"}),
    ):
        for path in (ROOT / directory).rglob("*"):
            parts = path.relative_to(ROOT).parts
            if any(value in {"__pycache__", "build", "dist", ".git"} or value.endswith(".egg-info") for value in parts):
                continue
            if path.is_file() and path.suffix in suffixes:
                result.add(path)
    result.add(ROOT / "src/LICENSE")
    for path in (ROOT / "protomotions/data/assets").rglob("*"):
        relative = path.relative_to(ROOT).as_posix()
        if (relative.startswith("protomotions/data/assets/mesh/smpl/")
                or fnmatch.fnmatch(relative, "protomotions/data/assets/mjcf/smpl*")
                or fnmatch.fnmatch(relative, "protomotions/data/assets/usd/smpl*")):
            continue
        if path.is_file():
            result.add(path)
    return sorted(result)


def main():
    workspace_sources = ROOT / "protomotions.egg-info/SOURCES.txt"
    sources_before = sha256(workspace_sources)
    temp = Path(tempfile.mkdtemp(prefix="protomotions-current-sdist-20261003-"))
    tree = temp / "source"
    tree.mkdir()
    copied = []
    for source in selected_paths():
        relative = source.relative_to(ROOT)
        target = tree / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        copied.append({"path": relative.as_posix(), "bytes": target.stat().st_size,
                       "sha256": sha256(target)})
    artifacts = temp / "dist"
    artifacts.mkdir()
    os.chdir(tree)
    # python-build is absent, but the declared installed build backend is usable.
    from setuptools.build_meta import build_sdist
    filename = build_sdist(str(artifacts))
    artifact = artifacts / filename
    required = [
        "scripts/simulate_paper_pedestrians.py", "scripts/curate_amass_pedestrian_data.py",
        "scripts/convert_amass_joint_velocities.py", "scripts/evaluate_pedestrian_style.py",
        "scripts/benchmark_paper_motion_tracking.py", "scripts/probe_paper_foot_support.py",
        "scripts/probe_paper_scene_maps.py", "scripts/probe_paper_pre_action.py",
        "scripts/prepare_native_pedestrian_scene.py", "docs/AMP_STYLE_PROGRESS_20261002_ZH.md",
        "FULL_PIPELINE.md", "CrowdSim/humanoid/README.md", "docs/STRUCTURE_AND_EVALUATION_CURRENT_ZH.md",
        "CrowdSim/humanoid/application/deployment.py", "CrowdSim/humanoid/application/native_deployment.py",
        "CrowdSim/humanoid/runtime/amp_inference.py", "CrowdSim/humanoid/runtime/evaluation_actor.py",
        "CrowdSim/humanoid/training/amass_curation.py", "CrowdSim/humanoid/training/terrain_statistics.py",
        "configs/diffusion_amp/amp_style_amass_backward_penalty.yaml",
        "configs/diffusion_amp/amp_style_mixed_state_action_terrain_protected.yaml",
        "configs/diffusion_amp/native_warehouse_pedestrians_clearance08_v1.yaml",
        "configs/diffusion_amp/native_warehouse_pedestrians_15_clearance08_v1.yaml",
        "env/isaaclab3_sim6/profile.json", "env/isaaclab3_sim6/uv.lock", "MANIFEST.in",
    ]
    member_files, violations, checked = [], [], []
    with tarfile.open(artifact) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            relative = member.name.split("/", 1)[1]
            member_files.append(relative)
            if (relative.startswith(("output/", "data/pretrained_models/", "data/diffusion_amp/", "data/scene_resources/"))
                    or relative.endswith((".pt", ".ckpt", ".pth", ".pkl", ".hdr"))
                    or relative.startswith("protomotions/data/assets/mesh/smpl/")
                    or fnmatch.fnmatch(relative, "protomotions/data/assets/mjcf/smpl*")
                    or fnmatch.fnmatch(relative, "protomotions/data/assets/usd/smpl*")):
                violations.append(relative)
        for relative in required:
            member = archive.getmember(filename.removesuffix(".tar.gz") + "/" + relative)
            digest = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
            source = next(item for item in copied if item["path"] == relative)
            checked.append({"path": relative, "bytes": member.size,
                            "sha256": digest, "copied_bytes_exact": digest == source["sha256"]})
    assert not violations
    assert all(item["copied_bytes_exact"] for item in checked)
    assert sha256(workspace_sources) == sources_before
    heavy = sorted(name for name in sys.modules if name.split(".")[0] in {
        "torch", "numpy", "isaacsim", "isaaclab", "omni", "warp", "pxr"})
    assert not heavy
    report = {
        "schema": "navisaaclab.current_source_distribution_gate.v1", "status": "passed",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "python_lexical_path": sys.executable,
        "method": "setuptools.build_meta.build_sdist in independent /tmp copy; no-isolation existing backend",
        "build_tool_versions": {name: metadata.version(name) for name in ("setuptools", "wheel")},
        "python_build_module": "not installed; direct declared PEP517 backend used without download",
        "copy_tree": str(tree), "artifact": {"path": str(artifact), "bytes": artifact.stat().st_size,
            "sha256": sha256(artifact)}, "manifest_before_sha256": sha256(OUTPUT / "MANIFEST.in.before"),
        "manifest_after_sha256": sha256(ROOT / "MANIFEST.in"), "copied_input_count": len(copied),
        "copied_inputs": copied, "sdist_file_count": len(member_files), "sdist_files": sorted(member_files),
        "required_paths": checked, "forbidden_resource_members": violations,
        "workspace_egg_info_sources_before_sha256": sources_before,
        "workspace_egg_info_sources_after_sha256": sha256(workspace_sources),
        "workspace_egg_info_unchanged": True, "heavy_modules_imported": heavy,
        "limitations": ["Source distribution snapshot; no wheel dependency installation or runtime execution.",
            "Source application and config examples require explicit external AMASS/SMPL/models/scenes.",
            "Only explicitly selected current guides are included; evidence reports/videos are source-checkout artifacts.",
            "CPU header/resource preflight does not prove model tensors, physics or navigation quality."]}
    (OUTPUT / "sdist_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "artifact", "copied_input_count", "sdist_file_count", "manifest_before_sha256", "manifest_after_sha256", "workspace_egg_info_unchanged", "heavy_modules_imported")}, indent=2))


if __name__ == "__main__":
    main()
