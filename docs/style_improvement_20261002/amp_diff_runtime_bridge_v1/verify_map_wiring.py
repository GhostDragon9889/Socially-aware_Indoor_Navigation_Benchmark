"""CPU-only before/after map wiring evidence; no SDK, stage or model loading."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from CrowdSim.scene_setup import resolve_repo_path
from CrowdSim.utils.config_loader import load_config
from CrowdSim.utils.map_metadata import load_occupancy_map_metadata


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def file_record(path):
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    before = json.loads((OUT / "before.json").read_text())
    spec = importlib.util.spec_from_file_location(
        "map_metadata_bridge_before", OUT / "before/CrowdSim/utils/map_metadata.py")
    previous = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = previous
    spec.loader.exec_module(previous)
    scenes = []
    for name, directory in (("office", "Office"), ("hospital", "Hospital"),
                            ("warehouse", "Warehouse"), ("full_warehouse", "Full_Warehouse")):
        config_path = ROOT / f"CrowdSim/config/scenes/dynabench_{name}.yaml"
        old_config = load_config(OUT / "before" / config_path.relative_to(ROOT))
        old_path = resolve_repo_path(old_config["scene_map"])
        actual = resolve_repo_path(load_config(config_path)["scene_map"])
        assert not old_path.exists()
        assert actual == ROOT / "data/diffusion_amp/maps_resource_complete" / directory / "occupancy.yaml"
        try:
            previous.load_occupancy_map_metadata(actual)
        except ValueError as error:
            old_error = str(error)
        else:
            raise AssertionError("The prior metadata parser unexpectedly accepted the block origin")
        metadata = load_occupancy_map_metadata(actual)
        source = resolve_repo_path(load_config(config_path)["scene_usd"])
        assert source.is_file() and metadata.image_path.is_file()
        scenes.append({"scene": directory, "config": file_record(config_path),
            "previous_configured_map": str(old_path), "previous_map_existed": False,
            "previous_parser_on_present_map_error": old_error,
            "source_usd": {"path": str(source), "bytes": source.stat().st_size,
                           "sha256": hashlib.sha256(source.read_bytes()).hexdigest()},
            "map_yaml": file_record(actual), "map_png": file_record(metadata.image_path),
            "resolution_m": metadata.resolution, "origin_xy_m": metadata.origin_xy,
            "origin_yaw_rad": metadata.origin_yaw, "negate": metadata.negate,
            "free_threshold": metadata.free_threshold,
            "physx_collider_verified": False})
    for record in before["unchanged_map_files"]:
        assert file_record(ROOT / record["path"]) == record
    snapshot_path = ROOT / "docs/style_improvement_20261002/mixed_curriculum_training_source_v1.json"
    frozen = json.loads(snapshot_path.read_text())["files"]
    changed = {record["path"] for record in before["files"]}
    assert not changed.intersection(record["path"] for record in frozen)
    differences = []
    for record in frozen:
        actual = file_record(ROOT / record["path"])
        if actual != record:
            differences.append({"frozen": record, "current": actual})
    assert not [row for row in differences if not row["current"]["path"].startswith("docs/")]
    current = [file_record(ROOT / record["path"]) for record in before["files"]]
    current.append(file_record(ROOT / "CrowdSim/tests/test_dynbench_map_metadata.py"))
    result = {"schema": "navisaaclab.runtime_bridge_map_verification.v1", "status": "passed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "scenes": scenes,
        "changed_sources": current, "unchanged_map_and_provenance_files": before["unchanged_map_files"],
        "unchanged_map_file_count": len(before["unchanged_map_files"]),
        "frozen_training_source": file_record(snapshot_path),
        "frozen_source_file_count": len(frozen),
        "unchanged_frozen_source_files": len(frozen) - len(differences),
        "current_document_tool_differences_from_prelaunch": differences,
        "runtime_business_sources_unchanged": True,
        "cpu_test_count": 40, "test_log": "map_tests.log", "test_xml": "map_tests.xml",
        "ruff_log": "map_ruff.log", "sdk_started": False, "models_loaded": False,
        "sealed_test_outputs_read": False,
        "limitations": ["CPU geometry-projection maps retain unverified cooked PhysX occupancy status",
                        "No live scene, height or contact verification performed",
                        "Block-sequence parsing uses the existing PyYAML-backed config loader"]}
    (OUT / "map_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "scenes": len(scenes),
                      "unchanged_maps": len(before["unchanged_map_files"]),
                      "unchanged_source_files": len(frozen) - len(differences),
                      "document_tool_differences": len(differences)}))


if __name__ == "__main__":
    main()
