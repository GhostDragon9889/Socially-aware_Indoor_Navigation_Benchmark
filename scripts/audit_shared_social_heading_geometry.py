#!/usr/bin/env python3
"""Audit all fixed AMASS2848 frames and completed S1/S2 body-axis/reset captures."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.analysis import heading_geometry as geometry  # noqa: E402
from CrowdSim.humanoid.analysis import tracking_error_decomposition as original  # noqa: E402
from scripts import verify_shared_social_evidence_admission as gate  # noqa: E402
from scripts.diagnose_shared_social_training import identity  # noqa: E402

MANIFEST = ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json"
MANIFEST_SHA = "68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca"
BANK_SHA = "46e0241108e200409c08d1115a270205fe0b6ce827e366a06e8a09f6a2cc82c0"
POSTS = {"S1": ROOT / "docs/style_improvement_20261005/shared_social_evaluation_v1/popart_post_training_execution.json",
    "S2": ROOT / "docs/style_improvement_20261005/shared_social_full_epochs_v1/post_training_execution.json"}


def concatenate(rows):
    return {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}


def captured_axes(receipt):
    bundle, measurements = gate.admit_post(receipt)
    report = bundle["tracking"]
    geometry.require(report["physics"]["body_names"] == list(geometry.SMPL_BODY_NAMES), "Capture body-name order changed")
    inputs = [identity(receipt), *measurements.values(), *gate.bundle_sources(bundle, measurements)]
    rows, reset, all_geometry = [], [], {"actual": [], "reference": []}
    for row in report["sequences"]:
        source = identity(row["capture"])
        geometry.require(source["sha256"] == row["capture_sha256"], "Completed capture changed")
        inputs.append(source)
        with np.load(source["path"], allow_pickle=False) as arrays:
            original.verify_capture(arrays, row)
            measured = {}
            for name, prefix in (("actual", ""), ("reference", "reference_")):
                values = geometry.body_axes(arrays[prefix+"rigid_body_pos"][:, 0], arrays[prefix+"rigid_body_rot"][:, 0])
                metric = {k: v[arrays["metric_frame"]] for k, v in values.items()}
                all_geometry[name].append(metric)
                measured[name] = geometry.summarize(metric)
            reset.append({"motion_id": row["motion_id"],
                **{key: [float(arrays[key][0]), float(arrays[key][1])] for key in
                    ("mean_body_position_error_m", "mean_body_rotation_error_rad")}})
        rows.append({"motion_id": row["motion_id"], "capture": source, "geometry": measured})
    complete = {key: concatenate(value) for key, value in all_geometry.items()}
    geometry.require(len(rows) == 22 and all(len(v["hip_width_m"]) == 11982 for v in complete.values()), "Retain complete22/11982 metric frames")
    return report, inputs, complete, {"original_world_tracking_aggregate": report["aggregate"],
        "geometry": {k: geometry.summarize(v) for k, v in complete.items()}, "sequences": rows,
        "reset_and_first_POST_errors": reset, "failed_and_fallen_metric_frames_filtered": False}


def bank_axes():
    import torch
    from CrowdSim.humanoid.amass_corpus import AMASSCorpus

    geometry.require(not torch.cuda.is_initialized(), "CPU corpus reader initialized CUDA")
    manifest = identity(MANIFEST)
    geometry.require(manifest["sha256"] == MANIFEST_SHA, "Fixed2848 manifest changed")
    corpus = AMASSCorpus(MANIFEST, device="cpu", seed=7, dof_velocity_frame="child_joint")
    bank = identity(corpus.entry["path"])
    geometry.require(bank["sha256"] == BANK_SHA and corpus.ids.tolist() == list(range(2848)), "Require exact fixed2848 bank and order")
    data, records = corpus.data, corpus.entry["clips"]
    geometry.require(data["gts"].shape == (523272, 24, 3) and data["grs"].shape == (523272, 24, 4), "Require the complete source bank")
    all_values, rows, expected_start = [], [], 0
    for row in records:
        mid, count = row["motion_id"], row["frames"]
        start = int(data["length_starts"][mid])
        geometry.require(start == expected_start and int(data["motion_num_frames"][mid]) == count
            and data["motion_files"][mid] == row["source_path"] and abs(float(data["motion_dt"][mid])-1/30) < 1e-7,
            "Actual source ID/path/count/30Hz frame partition changed")
        values = geometry.body_axes(data["gts"][start:start+count].numpy(), data["grs"][start:start+count].numpy())
        all_values.append(values)
        rows.append({"motion_id": mid, "source_path": row["source_path"], "frames": count, "geometry": geometry.summarize(values)})
        expected_start += count
    geometry.require(expected_start == 523272 and not torch.cuda.is_initialized(), "Incomplete bank or CUDA initialization")
    complete = concatenate(all_values)
    return [manifest, bank], complete, {"clips": 2848, "frames": 523272, "geometry": geometry.summarize(complete),
        "sequences": rows, "body_order": "COMMON24 declared by original corpus/converter; bank has no stored body-name field"}, identity(Path(sys.modules[AMASSCorpus.__module__].__file__))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    geometry.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not args.output_directory.exists(), "Require fresh CPU-only output")
    sources = [identity(p) for p in (Path(__file__), Path(geometry.__file__), Path(original.__file__),
        ROOT / "CrowdSim/humanoid/motion_tracking.py", Path(gate.__file__), Path(gate.original.__file__),
        ROOT / "scripts/diagnose_shared_social_training.py")]
    geometry.require(sources[3]["sha256"] == "da482dcd565d42ef9f6165606428ad510f077bdd68c3fab09c73a840149eac56",
        "Frozen world tracking metric changed")
    reports, inputs, runs, archives = [], [], {}, {}
    for name, receipt in POSTS.items():
        tracking, bound, values, result = captured_axes(receipt)
        reports.append(tracking)
        inputs.extend(bound)
        runs[name] = result
        archives.update({name+"_"+k: v for k, v in values.items()})
    gate.tracking_protocol(*reports)
    bound, values, result, source = bank_axes()
    inputs.extend(bound)
    sources.append(source)
    runs["train2848"] = result
    archives["train2848"] = values
    inputs = gate.unique_artifacts(inputs)
    geometry.require(not any(n.split(".")[0] in {"isaacsim", "isaaclab", "omni", "pxr", "warp"} for n in sys.modules), "SDK imported")
    geometry.require(all(identity(row["path"]) == row for row in inputs+sources), "Input/source changed during audit")
    args.output_directory.mkdir(parents=True, exist_ok=False)
    for name, arrays in archives.items():
        p = args.output_directory / (name+"_all_frames.npz")
        with p.open("xb") as stream:
            np.savez(stream, **arrays)
        runs["train2848" if name == "train2848" else name.split("_")[0]].setdefault("all_frame_archives", {})[name] = identity(p)
    report = {"schema": "shared_social.fixed2848_and_complete22_heading_geometry_CPU_audit.v1",
        "status": "completed_CPU_geometry_observation", "sources": sources, "inputs": inputs, "runs": runs,
        "definition": "Normalized measured XYZW root geometry; local L_Hip-R_Hip axis, (L_Hip-R_Hip) cross root+Z anatomical forward; heading angle only where both XY projections exist",
        "scope": "Anatomical-axis and native-reset correspondence diagnostics; no velocity-based forward assumption or paper metric replacement",
        "SDK_started": False, "training_changed": False, "causal_training_failure_proven": False,
        "paper_effect_achieved": False, "default_model_promoted": False}
    p = args.output_directory / "result.json"
    with p.open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"result": identity(p), "all_train_frames": 523272, "per_model_metric_frames": 11982,
        "hip_local_Y_angles_deg": {k: v["geometry"]["hip_to_local_Y_deg"]["mean"] if k == "train2848"
            else v["geometry"]["actual"]["hip_to_local_Y_deg"]["mean"] for k, v in runs.items()}, "SDK_started": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
