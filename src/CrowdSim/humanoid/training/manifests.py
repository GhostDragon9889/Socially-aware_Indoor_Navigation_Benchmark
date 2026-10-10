"""Integrity and provenance gates for real supplied AMASS training corpora."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch

from protomotions.simulator.base_simulator.simulator_state import StateConversion


class ManifestError(ValueError):
    pass


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resource_path(value, manifest_path):
    path = Path(value).expanduser()
    return path if path.is_absolute() else manifest_path.parent / path


def validate_training_manifest(path, *, mode="paper_method_reconstruction",
                               evaluation_manifest=None, verify_package_hashes=False):
    """Validate actual tensor packages, IDs, names, split disjointness and files.

    Existing multi-GB packages use weights_only/mmap, reading metadata and one
    selected pose per clip. Hash recomputation is opt-in and explicitly reported;
    a declared checksum is not misreported as a new content verification.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise ManifestError(f"Training manifest is missing: {path}")
    document = json.loads(path.read_text())
    if mode not in {"paper_method_reconstruction", "strict_author_experiment"}:
        raise ManifestError("Unknown provenance mode")
    if document.get("schema") != "navisaaclab.amass_reconstruction.v1":
        raise ManifestError("Training manifest must declare navisaaclab.amass_reconstruction.v1")
    if mode == "strict_author_experiment" and document.get("author_clip_list") is not True:
        raise ManifestError("Strict author experiment requires the actual author 2,848 clip manifest; derived locomotion clips are not a substitute")
    splits = document.get("splits", {})
    if not {"train", "validation", "test"}.issubset(splits):
        raise ManifestError("Train, validation and test splits must be declared explicitly")
    resources = []
    identifiers = {}
    counts = {}
    for split, bundle in splits.items():
        if split not in {"train", "validation", "test"}:
            raise ManifestError(f"Unknown training split: {split}")
        package = _resource_path(bundle["path"], path).resolve()
        if not package.is_file() or package.stat().st_size != bundle.get("bytes"):
            raise ManifestError(f"Motion package missing or byte size changed: {package}")
        declared = bundle.get("sha256")
        if not isinstance(declared, str) or len(declared) != 64 or any(char not in "0123456789abcdef" for char in declared):
            raise ManifestError(f"Motion package requires a SHA256 provenance value: {package}")
        with package.open("rb") as stream:
            if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ManifestError(f"Motion package is an unresolved LFS pointer: {package}")
        if verify_package_hashes and _sha256(package) != declared:
            raise ManifestError(f"Motion package checksum differs from manifest: {package}")
        with torch.serialization.safe_globals([StateConversion]):
            data = torch.load(package, map_location="cpu", weights_only=True, mmap=True)
        if data["gts"].ndim != 3 or data["gts"].shape[1:] != (24, 3) or data["dps"].shape != (len(data["gts"]), 69):
            raise ManifestError(f"Motion package violates 24body/69DOF SMPL contract: {package}")
        clips = bundle.get("clips", [])
        if not clips or bundle.get("clip_count", len(clips)) != len(clips):
            raise ManifestError(f"{split} needs nonempty clips and a consistent count")
        seen_ids = set()
        seen_sources = set()
        for clip in clips:
            motion_id = clip.get("motion_id")
            if type(motion_id) is not int or not 0 <= motion_id < len(data["motion_files"]):
                raise ManifestError(f"Invalid actual MotionLib ID in {split}: {motion_id}")
            if motion_id in seen_ids:
                raise ManifestError(f"Duplicate motion ID in {split}: {motion_id}")
            seen_ids.add(motion_id)
            source = str(clip.get("source_path", "")).replace("\\", "/")
            if source != str(data["motion_files"][motion_id]).replace("\\", "/"):
                raise ManifestError(f"Motion source identifier does not match package at {split}/{motion_id}")
            if source in seen_sources:
                raise ManifestError(f"Duplicate original source in {split}: {source}")
            seen_sources.add(source)
            if clip.get("frames") != int(data["motion_num_frames"][motion_id]):
                raise ManifestError(f"Motion frame count changed at {split}/{motion_id}")
            dt = clip.get("dt")
            if not isinstance(dt, (int, float)) or not math.isfinite(dt) or dt <= 0 or abs(dt - float(data["motion_dt"][motion_id])) > 1e-7:
                raise ManifestError(f"Motion dt changed or invalid at {split}/{motion_id}")
            start = int(data["length_starts"][motion_id])
            for key in ("gts", "grs", "dps", "dvs"):
                if not torch.isfinite(data[key][start]).all():
                    raise ManifestError(f"Nonfinite actual pose in {split}/{motion_id}/{key}")
        identifiers[split] = seen_sources
        counts[split] = len(clips)
        resources.append({"split": split, "path": str(package), "bytes": package.stat().st_size,
                          "declared_sha256": declared, "sha256_recomputed": verify_package_hashes,
                          "actual_tensor_metadata_verified": True})
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = identifiers[first] & identifiers[second]
        if overlap:
            raise ManifestError(f"Original source leakage between {first} and {second}: {sorted(overlap)[:3]}")
    scene_report = {"paper_eight_scene_evaluation_available": False, "scene_count": 0}
    if evaluation_manifest is not None:
        evaluation_path = Path(evaluation_manifest).resolve()
        if not evaluation_path.is_file():
            raise ManifestError(f"Evaluation manifest is missing: {evaluation_path}")
        evaluation = json.loads(evaluation_path.read_text())
        scenes = evaluation.get("scenes", [])
        scene_ids = set()
        for scene in scenes:
            scene_id = scene.get("id")
            if not isinstance(scene_id, str) or not scene_id or scene_id in scene_ids:
                raise ManifestError("Evaluation scenes need unique explicit IDs")
            scene_ids.add(scene_id)
            config = _resource_path(scene["config"], evaluation_path)
            if not config.is_file():
                raise ManifestError(f"Evaluation scene configuration is missing: {config}")
            people = scene.get("pedestrians")
            if type(people) is not int or not 3 <= people <= 15 or scene.get("duration_seconds") != 200:
                raise ManifestError("Paper evaluation requires 3–15 pedestrians and 200 seconds per scene")
        scene_report = {"paper_eight_scene_evaluation_available": len(scenes) == 8,
                        "scene_count": len(scenes), "manifest": str(evaluation_path),
                        "manifest_sha256": _sha256(evaluation_path)}
        if mode == "strict_author_experiment" and evaluation.get("author_scene_list") is not True:
            raise ManifestError("Strict author mode requires the actual author scene identities/protocol")
    if mode == "strict_author_experiment":
        author = document.get("author_provenance", {})
        if counts["train"] != 2848:
            raise ManifestError("Strict author training requires exactly 2,848 actual author training clips")
        if not scene_report["paper_eight_scene_evaluation_available"]:
            raise ManifestError("Strict author experiment lacks the actual eight-scene evaluation manifest")
        if not author.get("source_url") or not author.get("release_revision") or not author.get("independently_verified"):
            raise ManifestError("Strict author provenance needs source URL/revision and independent verification")
        if not verify_package_hashes:
            raise ManifestError("Strict author resource integrity requires complete package checksum verification")
    return {"mode": mode, "status": "verified_supplied_method_reconstruction" if mode == "paper_method_reconstruction" else "verified_author_declared_manifest_contract",
        "strict_manifest_gate_passed": mode == "strict_author_experiment", "manifest": str(path),
        "manifest_sha256": _sha256(path), "author_clip_list": document.get("author_clip_list", False),
        "counts": counts, "heldout_count": counts["validation"] + counts["test"],
        "resources": resources, "evaluation": scene_report, "selection": document.get("selection", {}),
        "actual_author_experiment_reproduced": False,
        "qualification": "Resource/metadata integrity alone is not training convergence or a TableII replication."}
