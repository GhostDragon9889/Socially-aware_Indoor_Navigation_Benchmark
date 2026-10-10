"""CPU-only frozen curated-bank metadata audit; does not edit any data bank."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from CrowdSim.humanoid.style_evaluation import parent_source_identity  # noqa: E402


def check(manifest_path):
    document = json.loads(manifest_path.read_text())
    entry = document["splits"]["train"]
    bank_path = Path(entry["path"])
    if not bank_path.is_absolute():
        bank_path = manifest_path.parent/bank_path
    bank = torch.load(bank_path, map_location="cpu", weights_only=True, mmap=True)
    original_path = Path(document["curation"]["source_stats"]["train"]["path"])
    original = torch.load(original_path, map_location="cpu", weights_only=True, mmap=True)
    clips = entry["clips"]
    integer_fields = (
        "motion_id", "original_motion_id", "frames", "parent_control_frame_start",
        "parent_control_frame_stop", "parent_source_frame_start", "parent_source_frame_stop",
        "parent_source_frames", "source_frames",
    )
    float_fields = ("dt", "parent_start_time_s", "parent_end_time_s", "parent_source_dt", "source_dt")
    intervals = defaultdict(list)
    rates = Counter()
    dtype_counts = {key: Counter() for key in (*integer_fields, *float_fields)}
    examples = {}
    total_frames = 0
    maximum_clock_rounding = 0.0
    assert document["selection"]["dof_velocity_frame"] == entry["dof_velocity_frame"] == bank["dof_velocity_frame"] == "child_joint"
    assert len(clips) == entry["clip_count"] == len(bank["motion_files"])
    for index, row in enumerate(clips):
        for field in integer_fields:
            assert type(row[field]) is int, (index, field, type(row[field]))
            dtype_counts[field][type(row[field]).__name__] += 1
        for field in float_fields:
            assert type(row[field]) is float and math.isfinite(row[field]), (index, field)
            dtype_counts[field][type(row[field]).__name__] += 1
        resampling = row["resampling"]
        assert row["motion_id"] == index
        old_index = row["original_motion_id"]
        parent = parent_source_identity(row)
        assert parent == original["motion_files"][old_index]
        assert row["source_path"] == bank["motion_files"][index]
        old_frames = int(original["motion_num_frames"][old_index])
        old_dt = float(original["motion_dt"][old_index])
        assert row["parent_source_frames"] == row["source_frames"] == resampling["source_frames"] == old_frames
        assert row["parent_source_dt"] == row["source_dt"] == resampling["source_dt_metadata"] == old_dt
        start, stop = row["parent_control_frame_start"], row["parent_control_frame_stop"]
        source_start, source_stop = row["parent_source_frame_start"], row["parent_source_frame_stop"]
        assert 0 <= start < stop <= resampling["target_frames"]
        assert 0 <= source_start < source_stop <= old_frames
        assert row["frames"] == stop-start == int(bank["motion_num_frames"][index])
        assert int(bank["length_starts"][index]) == total_frames
        total_frames += row["frames"]
        assert abs(float(bank["motion_dt"][index])-1/30) < 2e-9
        assert row["dt"] == resampling["target_dt"] == 1/30
        assert row["parent_start_time_s"] == start/30
        assert row["parent_end_time_s"] == (stop-1)/30
        nominal_dt = resampling["source_dt"]
        rate = round(1/nominal_dt)
        assert rate in {30, 50} and nominal_dt == 1/rate
        assert type(resampling["source_rate_changed"]) is bool
        assert resampling["source_rate_changed"] == (rate == 50)
        assert resampling["dof_velocity_frame"] == row["dof_velocity_frame"] == "child_joint"
        assert resampling["target_frames"] == math.floor((old_frames-1)*nominal_dt*30+1e-6)+1
        assert source_start == math.floor(start/(30*nominal_dt)+1e-5)
        assert source_stop == math.ceil((stop-1)/(30*nominal_dt)-1e-5)+1
        # Bounds describe interpolation dependencies in the original sample
        # grid. Their count must not be equated with the30Hz control count.
        assert abs(old_dt-nominal_dt) < 2e-9
        maximum_clock_rounding = max(maximum_clock_rounding, abs(old_dt-nominal_dt))
        intervals[parent].append((start, stop))
        rates[str(rate)] += 1
        examples.setdefault(str(rate), row)
    for parent, bounds in intervals.items():
        ordered = sorted(bounds)
        assert len(ordered) <= 4
        assert all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:])), parent
    assert total_frames == len(bank["dvs"])
    tensor_metadata = {key: {"dtype": str(value.dtype), "shape": list(value.shape)}
                       for key, value in bank.items() if isinstance(value, torch.Tensor)}
    return {
        "status": "passed", "CPU_only": True, "kit_started": False,
        "torch_cuda_initialized": torch.cuda.is_initialized(),
        "manifest": str(manifest_path), "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "rows_checked": len(clips), "parent_sources": len(intervals), "total_control_frames": total_frames,
        "row_field_python_types": {key: dict(value) for key, value in dtype_counts.items()},
        "source_hz_segment_counts": dict(rates), "source_grid_metadata_rounding_max_s": maximum_clock_rounding,
        "source_metadata_compared_to_actual_original_bank": True,
        "all_motion_paths_match_actual_bank": True, "all_frame_boundaries_valid": True,
        "all_within_parent_segments_nonoverlapping": True, "all_frame_declarations_child_joint": True,
        "source_dependency_bounds_are_original_30_or_50hz_half_open": True,
        "control_bounds_are_reconstructed_30hz_half_open": True,
        "tensor_metadata": tensor_metadata,
        "bank_sha256_declared": entry["sha256"], "full_bank_hash_recomputed_by_this_audit": False,
        "examples_by_original_hz": {key: {field: value[field] for field in (
            "motion_id", "original_motion_id", "source_path", "parent_source_path", "frames", "dt",
            "parent_control_frame_start", "parent_control_frame_stop", "parent_source_frame_start",
            "parent_source_frame_stop", "parent_source_frames", "parent_source_dt",
            "parent_start_time_s", "parent_end_time_s", "resampling", "dof_velocity_frame")}
            for key, value in examples.items()},
        "scope": "Metadata/index/clock verification only; no full-payload numerical quality or author2848-ID equivalence claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = check(args.manifest.resolve())
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "rows_checked", "parent_sources", "source_hz_segment_counts", "total_control_frames")}))


if __name__ == "__main__":
    main()
