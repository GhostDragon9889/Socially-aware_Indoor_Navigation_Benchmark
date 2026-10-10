"""CPU-only real local resource refusal/source freeze proof, no SDK import."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from CrowdSim.humanoid.runtime.dynbench_scene import DynBenchSceneProvider, file_record, reference_ground_geometry


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def main():
    rows = []
    for scene in ("Office", "Warehouse", "Full_Warehouse", "Hospital"):
        path = ROOT / "docs/style_improvement_20261002/paper_scene_maps_reference_ground_preflight_v6" / scene / "preflight.json"
        data = json.loads(path.read_text())
        immutable = [data["source"], *data["layers"], *data["input_projection"].values(),
                     data["reference_ground_recipe"], data["reference_ground"]["overlay_source"]]
        assert all(file_record(row["path"]) == row for row in immutable)
        points, counts, indices = reference_ground_geometry(data["projection_metadata"]["bounds"],
            data["projection_metadata"]["ground"], data["reference_ground"]["margin_m"])
        assert points == data["reference_ground"]["points"]
        assert counts == data["reference_ground"]["face_counts"] and indices == data["reference_ground"]["face_indices"]
        current_script = file_record(data["script"]["path"])
        error = None
        try:
            provider = DynBenchSceneProvider.from_preflight(path, expected_sha256=file_record(path)["sha256"])
            provider.require_navigation_map()
        except (ValueError, RuntimeError) as failure:
            error = str(failure)
        assert error is not None
        rows.append({"scene": scene, "preflight": file_record(path), "source": data["source"],
            "recorded_static_projection_and_layer_records_checked": len(immutable),
            "source_and_floor_projection_records_byte_identical": True,
            "reference_floor_geometry_equals_recorded_reviewed_recipe": True,
            "negative_map_minimum_xy": data["projection_metadata"]["bounds"][0],
            "recorded_probe_script": data["script"], "current_probe_script": current_script,
            "probe_script_bytes_changed": data["script"] != current_script,
            "navigation_admission_refused": True, "refusal": error})
    manifest_path = ROOT / "docs/style_improvement_20261002/mixed_curriculum_training_source_v1.json"
    manifest = json.loads(manifest_path.read_text())
    matching, differing = [], []
    for old in manifest["files"]:
        now = file_record(ROOT / old["path"])
        (matching if now["bytes"] == old["bytes"] and now["sha256"] == old["sha256"] else differing).append(old["path"])
    assert differing == ["docs/style_improvement_20261002/audit_mixed_actual_checkpoint.py"]
    entry_path = ROOT / "docs/paper_pedestrian_deployment/source_freeze_v1.json"
    entry = json.loads(entry_path.read_text())
    for record in entry["source_files"]:
        current = file_record(ROOT / record["path"])
        assert current["bytes"] == record["bytes"] and current["sha256"] == record["sha256"]
    assert not any(name.split(".")[0] in {"torch", "numpy", "pxr", "omni", "isaaclab", "isaacsim"} for name in sys.modules)
    report = {"schema": "dynbench_scene_provider_prepared_CPU_gate.v1", "status": "passed_refusal_and_source_contracts_only",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "CPU_only": True, "SDK_started": False,
        "GPU_used": False, "native_map_admitted": False, "new_provider_physical_execution_verified": False,
        "actual_local_preflight_checks": rows, "frozen_mixed_source_manifest": file_record(manifest_path),
        "frozen_source135_current_matching_count": len(matching), "registered_docs_only_old_difference": differing,
        "all_frozen_non_docs_source135_paths_unchanged": True,
        "frozen_thin_entry_manifest": file_record(entry_path), "all_five_thin_entry_files_byte_unchanged": True,
        "provider_source": file_record(ROOT / "CrowdSim/humanoid/runtime/dynbench_scene.py"),
        "focused_tests": file_record(ROOT / "tests/test_dynbench_scene_provider.py"),
        "limits": ["CPU projection and SDK-free preflight are refused; no real native map has passed.",
            "Synthetic native JSON/query fixtures establish only API/error semantics.",
            "Existing AMPWorld has no pre-create hook; provider is prepared_not_runtime_integrated.",
            "No 34test capture read, weights changed, or old source archive edited."]}
    (OUT / "actual_prepared_gate.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print({"status": report["status"], "real_local_preflight_scenes": len(rows), "native_admitted": False,
           "source135_non_docs_unchanged": True, "thin_entry_files_unchanged": True})


if __name__ == "__main__":
    main()
