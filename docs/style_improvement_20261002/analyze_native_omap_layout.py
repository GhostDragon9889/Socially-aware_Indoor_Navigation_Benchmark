"""Read-only exact-layout diagnosis for a failed native OMAP capture; no SDK."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def identity(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def analyze(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    raw, low, high, dims = (data[k] for k in ("buffer", "min_bound", "max_bound", "dimensions"))
    width, height = map(int, dims[:2])
    indices, errors = {}, {}
    for kind in ("occupied", "free"):
        positions = data[kind+"_positions"]
        ij = np.rint((positions[:, :2]-low[:2])/.1-.5).astype(np.int64)
        assert ((ij >= 0) & (ij < dims[:2])).all()
        errors[kind] = float(np.max(np.abs(low[:2]+(ij+.5)*.1-positions[:, :2])))
        assert errors[kind] <= .0002 and len(np.unique(ij, axis=0)) == len(ij)
        indices[kind] = ij
    layouts = []
    for order in ("x_fastest", "y_fastest"):
        for reverse_x in (False, True):
            for reverse_y in (False, True):
                def linear(ij):
                    x = width-1-ij[:, 0] if reverse_x else ij[:, 0]
                    y = height-1-ij[:, 1] if reverse_y else ij[:, 1]
                    return y*width+x if order == "x_fastest" else x*height+y
                oi, fi = linear(indices["occupied"]), linear(indices["free"])
                ov, fv = raw[oi], raw[fi]
                missing = np.setdiff1d(np.arange(width*height), np.r_[oi, fi])
                ok = (np.array_equal(np.sort(oi), np.flatnonzero(raw == 4))
                      and np.isin(np.flatnonzero(raw == 5), fi).all()
                      and not np.isin(fi, oi).any() and not (fv == 4).any()
                      and (raw[missing] == 6).all())
                layouts.append({"order": order, "reverse_x": reverse_x, "reverse_y": reverse_y,
                    "occupied_position_labels": {str(int(v)): int((ov == v).sum()) for v in (4, 5, 6)},
                    "reported_free_position_labels": {str(int(v)): int((fv == v).sum()) for v in (4, 5, 6)},
                    "missing_position_count": len(missing), "missing_position_labels": {str(int(v)): int((raw[missing] == v).sum()) for v in (4, 5, 6)},
                    "exact_known_occupied_and_free_world_roundtrip": bool(ok)})
    admitted = [layout for layout in layouts if layout["exact_known_occupied_and_free_world_roundtrip"]]
    assert len(admitted) == 1
    chosen = admitted[0]
    assert chosen["order"] == "x_fastest" and chosen["reverse_x"] and not chosen["reverse_y"]
    fi = indices["free"]
    oi = indices["occupied"]
    present = np.r_[oi[:, 1]*width+oi[:, 0], fi[:, 1]*width+fi[:, 0]]
    missing = np.setdiff1d(np.arange(width*height), present)
    return {"status": "layout_diagnosed_navigation_not_validated", "CPU_only": True, "SDK_started": False,
        "input": identity(path), "script": identity(__file__), "dimensions": dims.tolist(),
        "min_bound_world": low.tolist(), "max_bound_world": high.tolist(),
        "raw_counts": {str(int(v)): int((raw == v).sum()) for v in (4, 5, 6)},
        "position_counts": {k: len(v) for k, v in indices.items()}, "world_center_max_error_m": errors,
        "all_eight_axis_order_candidates": layouts, "unique_consistent_layout": chosen,
        "world_position_missing_y_indices": np.unique(missing//width, return_counts=True)[0].tolist(),
        "missing_y_counts": np.unique(missing//width, return_counts=True)[1].tolist(),
        "installed_primary_layout_source": "isaacsim.asset.gen.omap/utils/utils.py:compute_coordinates top_left=(maxX-cell/2,minY+cell/2), top_right=(minX+cell/2,minY+cell/2); MapGenerator.h:getBuffer row-major x then y",
        "conclusion": "get_free_positions includes raw unknown cells. Preserve label6; only raw5 verified by listed world position can be navigation-free. Raw4 must exactly match all occupied world positions. No unverified layout fallback.",
        "height_band_boundary": "Native2D position z=.75, returned cell bounds z=.7..8. These do not establish full .15..1.8m band geometry sampling. Native full-height overlap comparison remains necessary; no height-volume implementation claim.",
        "navigation_ready": False, "scene_resource_gate_passed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.input)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "raw_counts": result["raw_counts"], "layout": result["unique_consistent_layout"]}))


if __name__ == "__main__":
    main()
