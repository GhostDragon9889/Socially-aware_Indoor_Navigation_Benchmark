"""Independent train-only exhaustive floor bound for the declared linear tail.

No runtime curriculum code, simulation, or heldout outputs are imported. Every
legal reset phase is checked with suffix extrema and the five-second linear
endpoint. Translation is affine, so spawn-grid extrema prove all256 origins.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def fingerprint(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Fresh report required")
    torch.set_num_threads(4)
    manifest = json.loads(args.manifest.read_text())
    entry = manifest["splits"]["train"]
    path = Path(entry["path"])
    if not path.is_absolute():
        path = args.manifest.resolve().parent / path
    bank_record = fingerprint(path)
    assert bank_record["sha256"] == entry["sha256"] and bank_record["bytes"] == entry["bytes"]
    data = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    assert len(entry["clips"]) == 2848 and len({row["motion_id"] for row in entry["clips"]}) == 2848
    ids = np.arange(256)
    origins = np.stack((48 + (ids // 16) * 4, 48 + (ids % 16) * 4), axis=-1).astype(np.float64)
    assert len(np.unique(origins, axis=0)) == 256
    low_origin, high_origin = origins.min(0), origins.max(0)
    global_low, global_high = np.full(2, np.inf), np.full(2, -np.inf)
    witnesses, per_clip = {}, []
    total_candidates = 0
    for row in entry["clips"]:
        mid = row["motion_id"]
        start, count = int(data["length_starts"][mid]), int(data["motion_num_frames"][mid])
        assert count == row["frames"] and count > 40
        assert abs(float(data["motion_dt"][mid]) - 1 / 30) < 1e-7
        roots = data["gts"][start:start + count, 0, :2].numpy().astype(np.float64)
        assert np.isfinite(roots).all()
        # Eleven positions [-11..-1] cover ten actual 30Hz intervals.
        end_velocity = (roots[-1] - roots[-11]) / (10 / 30)
        linear_endpoint = roots[-1] + 5 * end_velocity
        suffix_low = np.minimum.accumulate(roots[::-1], axis=0)[::-1]
        suffix_high = np.maximum.accumulate(roots[::-1], axis=0)[::-1]
        phases = np.arange(9, count - 31)  # Inclusive9..T-32:history10 plus31 real transitions.
        assert len(phases) == count - 40 and phases[-1] + 31 == count - 1
        relative_low = np.minimum(suffix_low[phases], linear_endpoint) - roots[phases]
        relative_high = np.maximum(suffix_high[phases], linear_endpoint) - roots[phases]
        world_low, world_high = relative_low + low_origin, relative_high + high_origin
        total_candidates += len(phases)
        lower, upper = world_low.min(0), world_high.max(0)
        for axis in range(2):
            if lower[axis] < global_low[axis]:
                global_low[axis] = lower[axis]
                i = int(world_low[:, axis].argmin())
                witnesses[f"lower_{axis}"] = {"motion_id": mid, "source_path": row["source_path"],
                    "reset_frame": int(phases[i]), "spawn_coordinate_m": float(low_origin[axis]),
                    "coordinate_m": float(lower[axis]), "end_velocity_xy_m_s": end_velocity.tolist()}
            if upper[axis] > global_high[axis]:
                global_high[axis] = upper[axis]
                i = int(world_high[:, axis].argmax())
                witnesses[f"upper_{axis}"] = {"motion_id": mid, "source_path": row["source_path"],
                    "reset_frame": int(phases[i]), "spawn_coordinate_m": float(high_origin[axis]),
                    "coordinate_m": float(upper[axis]), "end_velocity_xy_m_s": end_velocity.tolist()}
        per_clip.append({"motion_id": mid, "source_path": row["source_path"], "frames": count,
                         "legal_reset_phase_count": len(phases), "world_xy_lower_bound_m": lower.tolist(),
                         "world_xy_upper_bound_m": upper.tolist(), "end_velocity_xy_m_s": end_velocity.tolist()})
    lower_clearance, upper_clearance = global_low - 1, 191 - global_high
    passed = bool((lower_clearance >= 0).all() and (upper_clearance >= 0).all())
    report = {"schema": "navisaaclab.amass_root_linear_tail_floor_bounds.v1", "status": "passed" if passed else "failed",
              "CPU_only": True, "Kit_started": False, "GPU_used": False, "heldout_outputs_read": False,
              "runtime_curriculum_imported": False, "runtime_curriculum_correctness_claim": False,
              "script": fingerprint(__file__), "manifest": fingerprint(args.manifest), "train_bank": bank_record,
              "clips": len(per_clip), "legal_reset_phases": total_candidates, "spawn_origins": origins.tolist(),
              "analytically_covered_phase_origin_pairs": total_candidates * len(origins),
              "control_hz": 30, "history_frames": 10, "required_real_future_transitions": 31,
              "phase_rule": "9<=f0<=T-32 inclusive, no eligibility filtering or resampling fallback",
              "fixed_translation": "world trajectory=spawn_origin+source_root(frame)-source_root(f0)",
              "linear_tail_rule": "only beyond real end: R_end+((R_end-R_end_minus10)/(10/30))*future_time_s",
              "linear_future_horizon_s": 5.0, "terrain_bounds_xy_m": [0, 0, 192, 192], "root_margin_m": 1.0,
              "world_xy_lower_bound_m": global_low.tolist(), "world_xy_upper_bound_m": global_high.tolist(),
              "remaining_lower_margin_m": lower_clearance.tolist(), "remaining_upper_margin_m": upper_clearance.tolist(),
              "proof": "For each reset f0, exact coordinate extrema of all real suffix points plus linear-end5sec endpoint. Linear segments attain extrema at endpoints; affine spawn translation extrema cover all256 Cartesian origins.",
              "extreme_witnesses": witnesses, "per_clip": per_clip,
              "limitations": ["Bounds certify the declared root command path, not actual physical tracking/body-collider support/falls.",
                              "The suffix includes even frames beyond episode300cap, yielding a conservative all-source proof.",
                              "Linear tail is engineering extrapolation, not observed AMASS future or author training metadata.",
                              "Source bank/root coordinates are measured inputs. Float64 analysis is independent of runtime float32 implementation."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "clips", "legal_reset_phases",
                     "analytically_covered_phase_origin_pairs", "world_xy_lower_bound_m", "world_xy_upper_bound_m")}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
