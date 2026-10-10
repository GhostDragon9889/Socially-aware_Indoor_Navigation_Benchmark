"""Read only, SDK-free current entrypoint/resource/split checks.

Only this tool's JSON output is written. No tensor deserialization, corpus
sampling, data hashing, execution of documented installers, or SDK startup.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.application.configuration import load_amp_profile, parse_amp_args  # noqa: E402 -- source checkout bootstrap
from CrowdSim.humanoid.application.deployment import parse_deployment_args, preflight_deployment  # noqa: E402 -- source checkout bootstrap
from CrowdSim.humanoid.application.resources import preflight_amp_resources  # noqa: E402 -- source checkout bootstrap


def small_record(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    documents = [ROOT / value for value in (
        "FULL_PIPELINE.md", "CrowdSim/humanoid/README.md",
        "docs/STRUCTURE_AND_EVALUATION_CURRENT_ZH.md", "pyproject.toml")]
    links, commands = [], []
    for doc in documents:
        text = doc.read_text()
        for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", text):
            target = match[1].strip("<>")
            if "://" in target or target.startswith("#"):
                continue
            target = target.split("#", 1)[0]
            resolved = (doc.parent / target).resolve()
            links.append({"document": str(doc.relative_to(ROOT)),
                          "line": text.count("\n", 0, match.start()) + 1,
                          "target": target, "resolved": str(resolved),
                          "exists": resolved.exists()})
        for block in re.finditer(r"```bash\n(.*?)```", text, re.S):
            body = block[1].replace("\\\n", " ")
            for command in body.splitlines():
                if command.strip():
                    tokens = shlex.split(command)
                    commands.append({"document": str(doc.relative_to(ROOT)),
                        "block_line": text.count("\n", 0, block.start()) + 1,
                        "command": command.strip(), "tokens": tokens,
                        "executed": False})
    manifest_path = ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json"
    manifest = json.loads(manifest_path.read_text())
    base = json.loads((ROOT / "data/diffusion_amp/amass_manifest.json").read_text())
    split_records, parent_sets = {}, {}
    for split, bundle in manifest["splits"].items():
        clips = bundle["clips"]
        parents = {clip.get("parent_source_path", clip["source_path"]).split("#frames=", 1)[0]
                   for clip in clips}
        parent_sets[split] = parents
        bank = Path(bundle["path"])
        if not bank.is_absolute():
            bank = manifest_path.parent / bank
        bank = bank.resolve()
        split_records[split] = {"clips": len(clips), "parents": len(parents),
            "path": str(bank), "stat_bytes": bank.stat().st_size,
            "declared_bytes": bundle["bytes"], "declared_sha256": bundle["sha256"],
            "bank_hash_recomputed": False,
            "original_heldout_clips_exact": clips == base["splits"][split]["clips"] if split != "train" else None,
            "original_heldout_bundle_exact": bundle == base["splits"][split] if split != "train" else None}
    parent_intersections = {first + ":" + second: len(parent_sets[first] & parent_sets[second])
        for first, second in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    bounds = {}
    for clip in manifest["splits"]["train"]["clips"]:
        parent = clip["parent_source_path"]
        bounds.setdefault(parent, []).append((clip["parent_control_frame_start"], clip["parent_control_frame_stop"]))
    overlaps = []
    for parent, intervals in bounds.items():
        intervals.sort()
        for previous, current in zip(intervals, intervals[1:]):
            if previous[1] > current[0]:
                overlaps.append({"parent": parent, "previous": previous, "current": current})
    selected = ["--inference-only", "--amp-checkpoint",
        "output/diffusion_amp/amp_style_amass2848_backward_penalty_v1/step_1000.pt",
        "--config", "configs/diffusion_amp/amp_style_amass_backward_penalty.yaml",
        "--diffusion-checkpoint", "output/diffusion_amp/social_diffusion_training_extended_v2/step_5000.ckpt",
        "--manifest", "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json",
        "--duration-seconds", "200", "--output", "output/diffusion_amp/STRUCTURE_AUDIT_NO_CREATE"]
    preflights = {}
    for name, extra in (("selected_cuboid", []),
        ("selected_native4", ["--native-dynbench", "--scene-config",
            "configs/diffusion_amp/native_warehouse_pedestrians_clearance08_v1.yaml"]),
        ("selected_native15", ["--native-dynbench", "--actors", "15", "--scene-config",
            "configs/diffusion_amp/native_warehouse_pedestrians_15_clearance08_v1.yaml"])):
        try:
            result = preflight_deployment(parse_deployment_args(selected + extra))
            preflights[name] = {"status": "passed", "result": result}
        except Exception as error:
            preflights[name] = {"status": "failed", "error": repr(error)}
    for name, argv in (
        ("documented_method", ["--updates", "1000", "--num-envs", "64", "--physics-only", "--synchronized-replanning", "--output", "output/diffusion_amp/STRUCTURE_AUDIT_NO_CREATE"]),
        ("documented_pure_preflight", ["--config", "configs/diffusion_amp/method_reconstruction.yaml", "--diffusion-checkpoint", "output/diffusion_amp/diffusion_training/step_1000.ckpt", "--output", "output/diffusion_amp/STRUCTURE_AUDIT_NO_CREATE"]),
        ("readme_train", ["--config", "configs/diffusion_amp/amp_style_amass_backward_penalty.yaml", "--manifest", "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json", "--diffusion-checkpoint", "output/diffusion_amp/social_diffusion_training_extended_v2/step_5000.ckpt", "--trajectory-source", "pacer_curriculum", "--curriculum-mode", "smooth", "--curriculum-speed-max-mps", "2", "--curriculum-accel-max-mps2", "1.5", "--num-envs", "256", "--updates", "1000", "--rollout-steps", "32", "--episode-steps", "300", "--terrain-extent-m", "192", "--spawn-margin-m", "48", "--checkpoint-every", "100", "--synchronized-replanning", "--physics-only", "--output", "output/diffusion_amp/STRUCTURE_AUDIT_NO_CREATE"])):
        try:
            args = parse_amp_args(argv)
            result = preflight_amp_resources(args, load_amp_profile(args.config))
            preflights[name] = {"status": "passed", "argv": argv, "result": result.as_dict()}
        except Exception as error:
            preflights[name] = {"status": "failed", "argv": argv, "error": repr(error)}
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    heavy = sorted(name for name in sys.modules if name.split(".")[0] in {
        "torch", "numpy", "isaacsim", "isaaclab", "omni", "warp", "pxr"})
    result = {"schema": "navisaaclab.current_structure_read_only_review.v1",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": str(ROOT), "python_lexical_path": sys.executable,
        "scope": "read-only CPU; no tensor deserialization, bank hash, SDK, GPU, runtime or training",
        "documents": [small_record(doc) for doc in documents], "local_links": links,
        "missing_links": [item for item in links if not item["exists"]],
        "documented_commands_not_executed": commands,
        "pure_actual_preflights": preflights, "heavy_modules_imported": heavy,
        "preflight_output_created": (ROOT / "output/diffusion_amp/STRUCTURE_AUDIT_NO_CREATE").exists(),
        "manifest": small_record(manifest_path), "manifest_selection": manifest["selection"],
        "splits": split_records, "parent_intersections": parent_intersections,
        "overlapping_selected_parent_intervals": overlaps,
        "train_control_frames": sum(clip["frames"] for clip in manifest["splits"]["train"]["clips"]),
        "train_max_segments_per_parent": max(len(intervals) for intervals in bounds.values()),
        "project": {"name": pyproject["project"]["name"], "scripts": pyproject["project"]["scripts"],
            "wheel_package_find": pyproject["tool"]["setuptools"]["packages"]["find"],
            "crowd_optional_dependencies": pyproject["project"]["optional-dependencies"]["crowd"]}}
    destination = Path(__file__).with_name("actual_read_only_checks.json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"artifact": str(destination), "links": len(links),
        "missing_links": result["missing_links"], "preflights": {key: item["status"] for key, item in preflights.items()},
        "heavy_modules": heavy, "splits": {key: {field: item[field] for field in ("clips", "parents", "original_heldout_clips_exact", "original_heldout_bundle_exact")} for key, item in split_records.items()},
        "parent_intersections": parent_intersections, "overlap_count": len(overlaps)}, indent=2))


if __name__ == "__main__":
    main()
