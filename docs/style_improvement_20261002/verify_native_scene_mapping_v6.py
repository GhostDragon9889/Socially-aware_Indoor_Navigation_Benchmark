#!/usr/bin/env python3
"""CPU verification of frozen native raw layouts; performs no new physics queries."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import probe_paper_scene_maps as probe  # noqa: E402

FIXTURES = {
    "Warehouse": ("native_reference_warehouse_floor_owner_v5", "", 0),
    "Office": ("native_reference_remaining_three_floor_owner_v5", "6a115898dd2330a9ba074e681607bd7e01ddf2daf56a65cda8a4c131d8286a64", 3532),
    "Hospital": ("native_reference_remaining_three_floor_owner_v5", "e0954ef725688b3fc3009204ff5586e9aff28aaba15bcbbcfbeea44f26eecbd1", 0),
    "Full_Warehouse": ("native_reference_remaining_three_floor_owner_v5", "4239c9b47b7b0853ca8500d4cda0adb659f70efc2ac115a0d98f9953db0d60c7", 0),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = []
    for scene, (folder, sha, missing_count) in FIXTURES.items():
        path = ROOT / "output/scene_maps" / folder / scene / "native_omap.npz"
        if sha and probe.digest(path) != sha:
            raise ValueError(f"{scene}: preserved raw SHA mismatch")
        projection_path = ROOT / "data/diffusion_amp/maps_resource_complete" / scene / "provenance.json"
        projection = json.loads(projection_path.read_text())
        with np.load(path, allow_pickle=False) as a:
            values = [a[k] for k in ("buffer", "occupied_positions", "free_positions", "min_bound", "max_bound", "dimensions")]
            image, proof = probe.reconstruct_native_map(*values, known5_position_source="official_buffer_centers")
            canonical = probe.canonical_known5_positions(values[0], values[2], *values[3:], source="official_buffer_centers")
            window = probe.projection_grid_window(*values[3:], projection["bounds_xy_m"], projection["image_dimensions_xy"])
            exported = probe.crop_to_projection_grid(image, window)
            rows_grid, cols_grid = np.indices(exported.shape)
            actual_xy = values[3][:2] + np.stack((cols_grid+window["native_window_columns"][0]+.5,
                values[5][1]-rows_grid-.5), axis=-1)*probe.RESOLUTION
            declared_xy = np.asarray(projection["bounds_xy_m"])[0] + np.stack((cols_grid+.5,
                exported.shape[0]-rows_grid-.5), axis=-1)*probe.RESOLUTION
            max_error = float(np.abs(actual_xy-declared_xy).max())
            if max_error > 1e-5 or len(canonical["missing_known5_buffer_indices"]) != missing_count:
                raise ValueError(f"{scene}: world crop/known5 count differs from measured contract")
            absent = canonical["missing_known5_buffer_indices"]
            np.testing.assert_array_equal(np.sort(canonical["native_buffer_indices"]), np.flatnonzero(values[0] == 5))
            if int((image == 205).sum()) != int((values[0] == 6).sum()):
                raise ValueError(f"{scene}: unknown upgraded")
            rows.append({"scene": scene, "raw": probe.file_record(path), "projection": probe.file_record(projection_path),
                "raw_native_dimensions": values[5].tolist(), "raw_native_bounds": [values[3].tolist(), values[4].tolist()],
                "proof": proof, "export_window": window, "all_export_world_centers_checked": exported.size,
                "max_world_center_error_m": max_error, "missing_known5_native_columns": np.unique(absent % int(values[5][0])).tolist(),
                "missing_known5_native_rows_count": len(np.unique(absent // int(values[5][0]))),
                "new_native_floor_or_height_queries_performed": False, "scene_resource_gate_passed": False})
    result = {"schema": "dynbench.native_mapping_CPU_proof.v6", "status": "passed_CPU_only",
        "CPU_only": True, "SDK_started": False, "GPU_used": False,
        "script": probe.file_record(ROOT / "scripts/probe_paper_scene_maps.py"),
        "tool": probe.file_record(__file__), "fixtures": rows,
        "native_v6_pending": True, "author_map_claim": False,
        "required_actual_gate": "Every native5 including missing API centers/padding needs direct full-height and floor queries; no foreign/coplandar owner; no query clock advance."}
    with Path(args.output).open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print("Four frozen raw fixtures verified; no new native physics validation claimed.")


if __name__ == "__main__":
    main()
