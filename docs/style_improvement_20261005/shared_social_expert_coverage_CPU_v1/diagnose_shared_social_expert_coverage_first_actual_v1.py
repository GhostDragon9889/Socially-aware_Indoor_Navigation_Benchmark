#!/usr/bin/env python3
"""Train-only window prior and actual terminal expert exposure; no resampling.

Uses the fixed v7 train bank and the unmodified short-window classifier.
Clip proxy classes and BABEL annotation coverage are distinct from action
ground truth. A terminal batch does not prove whole-training exposure or cause
any observed motion-quality result. No policy or SDK is constructed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import stat
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json"
MANIFEST_SHA = "68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca"
OVERLAY = ROOT / "docs/style_improvement_20261004/babel_train_semantics_v1/actual_train_semantic_overlay.json"
OVERLAY_SHA = "09e16f40f6cefa283c6366589577ad0af1b2aa8123dc032af3f65893ab7889b1"
CLASSIFIER = ROOT / "CrowdSim/humanoid/training/expert_sampling.py"
CLASSIFIER_SHA = "b16984a920f21ca92c3fe01976d9c0453a75bf54d3f46df369984ff27453f9cd"
ROOT_FEATURE_SHA = "2b80a0260e5c279228971df384fb92cdf80b709933f5519b3d2d1e3b86ef7996"
sys.path.insert(0, str(ROOT))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(path):
    path = Path(path)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "Require ordinary independent input: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    after = path.lstat()
    keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(all(getattr(before, key) == getattr(after, key) for key in keys), "Input changed while hashing")
    return {"path": str(path.resolve()), "bytes": before.st_size, "sha256": digest.hexdigest()}


def bound(expected):
    require(identity(expected["path"]) == expected, "Bound artifact changed: " + expected["path"])
    return json.loads(Path(expected["path"]).read_text())


def semantic_status(row):
    if row["no_train_annotation_match"]:
        return "no_train_annotation_match"
    if row["raw_time_alignment"]["full_sample_time_range_union_covered"]:
        return "full_clip_timed_union_covered_not_single_action"
    return "matched_without_full_clip_timed_union"


def prior_and_exposure(clips, labels, motion_ids, frame_indices, strata, semantics):
    """Sum actual clip-uniform/legal-center probabilities, with all rows kept."""
    require(len(clips) == len(labels) == len(semantics) and len(clips) > 0, "Incomplete train registry")
    require(motion_ids.ndim == frame_indices.ndim == 1 and motion_ids.dtype == frame_indices.dtype == np.int64
            and len(motion_ids) == len(frame_indices) > 0, "Require complete integer sampled windows")
    by_id, prior, counts, joint = {}, defaultdict(float), Counter(), defaultdict(lambda: defaultdict(float))
    rows, eligible_total = [], 0
    for clip, values, semantic in zip(clips, labels, semantics, strict=True):
        motion, room = clip["motion_id"], clip["frames"] - 10
        require(type(motion) is int and motion not in by_id and type(clip["frames"]) is int and room > 0,
                "Duplicate IDs or no legal history10/future1 window")
        require(values.shape == (room,) and values.dtype == np.int64 and np.isin(values, np.arange(len(strata))).all(),
                "Classifier registry does not contain every legal center")
        require(semantic["original_v7_train_row"] == clip, "BABEL overlay changed original train row")
        label_counts = np.bincount(values, minlength=len(strata))
        mass = label_counts.astype(np.float64) / room / len(clips)
        for name, number, probability in zip(strata, label_counts.tolist(), mass.tolist(), strict=True):
            counts[name] += number
            prior[name] += probability
            joint[clip["style_class"]][name] += probability
        status = semantic_status(semantic)
        by_id[motion] = (clip, values, status)
        rows.append({"motion_id": motion, "parent": clip["parent_source_path"], "dataset": clip["dataset"],
                     "subject": clip["subject_namespace"], "whole_clip_kinematic_proxy": clip["style_class"],
                     "legal_centers": room, "stratum_center_counts": dict(zip(strata, label_counts.tolist(), strict=True)),
                     "BABEL_train_coverage_status": status})
        eligible_total += room
    require(abs(sum(prior.values()) - 1) < 1e-10, "Original clip-uniform prior did not normalize")
    observed = {key: Counter() for key in ("window_stratum", "whole_clip_proxy", "dataset", "BABEL_train_coverage")}
    selected, unique_windows, parents, subjects = [], set(), set(), set()
    for motion, frame in zip(motion_ids.tolist(), frame_indices.tolist(), strict=True):
        require(motion in by_id, "Sampled motion is outside fixed train pool")
        clip, values, status = by_id[motion]
        require(9 <= frame < clip["frames"] - 1, "Sampled center crosses history/future boundary")
        label = int(values[frame - 9])
        for key, value in (("window_stratum", strata[label]), ("whole_clip_proxy", clip["style_class"]),
                           ("dataset", clip["dataset"]), ("BABEL_train_coverage", status)):
            observed[key][value] += 1
        selected.append(label)
        unique_windows.add((motion, frame))
        parents.add(clip["parent_source_path"])
        subjects.add(clip["subject_namespace"])
    return {"all_train_clips": len(clips), "all_legal_centers": eligible_total,
            "strata": {name: {"legal_center_count": counts[name], "window_uniform_fraction": counts[name] / eligible_total,
                              "actual_clip_uniform_legal_center_probability": prior[name],
                              "terminal_sample_count": observed["window_stratum"][name],
                              "terminal_sample_fraction": observed["window_stratum"][name] / len(motion_ids),
                              "expected_terminal_count_under_original_prior": len(motion_ids) * prior[name]} for name in strata},
            "whole_clip_proxy_by_window_stratum_prior": dict(joint),
            "terminal": {"rows": len(motion_ids), "unique_clips": len(set(motion_ids.tolist())),
                         "unique_centers": len(unique_windows), "unique_parents": len(parents), "unique_subjects": len(subjects),
                         "counts": {key: dict(value) for key, value in observed.items()},
                         "entire_training_exposure_proven": False}, "train_registry": rows}, np.array(selected, np.int64)


def admit_diagnostic(receipt_path):
    receipt_identity = identity(receipt_path)
    receipt = bound(receipt_identity)
    require(receipt["schema"] == "shared_social.terminal_D_input_sensitivity_execution.v1"
            and type(receipt["CPU_execution"]["actual_exit_code"]) is int and receipt["CPU_execution"]["actual_exit_code"] == 0
            and bool(receipt["CPU_execution"]["terminal_chunk_id"]), "Require actual observed terminal diagnostic exit0")
    artifacts = receipt["artifacts"]
    diagnostic = bound(artifacts["diagnostic"])
    require(diagnostic["per_sample"] == artifacts["per_sample"] and diagnostic["reader"] == artifacts["reader"],
            "CPU diagnostic/per-sample source ownership differs")
    admission = diagnostic["training_admission"]
    training = bound(admission["receipt"])
    observed = bound(training["external_exit_metadata"])
    require(type(training["terminal_exit_code"]) is int and training["terminal_exit_code"] == 0
            and type(training["CPU_audit_exit_code"]) is int and training["CPU_audit_exit_code"] == 0,
            "Training receipt must retain actual integer successful exits")
    for name, marker in (("training", training["terminal_tool_chunk"]), ("CPU_audit", training["CPU_audit_terminal_tool_chunk"])):
        require(type(observed[name]["exit_code"]) is int and observed[name]["exit_code"] == 0
                and observed[name]["chunk_id"] == marker, "Original observed training/audit exits differ")
    require(training["result_identity"] == admission["result"] and training["audit_identity"] == admission["audit"]
            and training["checkpoint_identity"] == admission["checkpoint"], "Original training admission ownership differs")
    result, audit = bound(admission["result"]), bound(admission["audit"])
    rollout = receipt["training_inputs"]["rollout"]
    rollout_identity = {key: rollout[key] for key in ("path", "bytes", "sha256")}
    require(audit["rollout"] == rollout_identity and audit["expert_windows_drawn"] == training["actual_expert_windows_drawn"]
            and result["expert_contract"]["manifest"]["sha256"] == MANIFEST_SHA
            and result["expert_contract"]["sampler"] == "uniform_clip_then_legal_center"
            and result["expert_contract"]["sampling_identity"] is None
            and result["expert_contract"]["manifest"] == diagnostic["population_contract"]["train_manifest"]
            and diagnostic["population_contract"]["expert_rows"] == receipt["samples"]["expert"] == 4096,
            "Fixed pool, actual expert packet or total draw count differs")
    records = [receipt_identity, artifacts["diagnostic"], artifacts["per_sample"], artifacts["reader"], admission["receipt"],
               training["external_exit_metadata"], admission["result"], admission["audit"], rollout_identity]
    return diagnostic, rollout_identity, records


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--diagnostic-exit-receipt", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Explicit CUDA-hidden CPU process required")
    require(not args.output_directory.exists(), "Preserve previous outputs; choose new directory")
    originals = [identity(path) for path in (Path(__file__), MANIFEST, OVERLAY, CLASSIFIER)]
    require([item["sha256"] for item in originals[1:]] == [MANIFEST_SHA, OVERLAY_SHA, CLASSIFIER_SHA], "Pinned data/source changed")
    manifest, overlay = (json.loads(path.read_text()) for path in (MANIFEST, OVERLAY))
    entry = manifest["splits"]["train"]
    require(entry["clip_count"] == len(entry["clips"]) == len(overlay["train"]) == 2848, "Require fixed2848 train registry")
    bank_path = Path(entry["path"])
    if not bank_path.is_absolute():
        bank_path = MANIFEST.parent / bank_path
    bank_record = identity(bank_path)
    require(bank_record["sha256"] == entry["sha256"] and bank_record["bytes"] == entry["bytes"], "Actual train bank differs")
    originals.append(bank_record)
    diagnostic, rollout_record, records = admit_diagnostic(args.diagnostic_exit_receipt)
    require(identity(rollout_record["path"]) == rollout_record, "Actual sealed terminal rollout changed")
    import torch
    from CrowdSim.humanoid.training.expert_sampling import classify_windows, STRATA, WindowThresholds
    require(not torch.cuda.is_initialized(), "CPU classifier must not initialize CUDA")
    rng_before = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())
    data = torch.load(bank_path, weights_only=True, map_location="cpu", mmap=True)
    packet = torch.load(rollout_record["path"], weights_only=True, map_location="cpu", mmap=True)["expert"]
    saved = torch.load(diagnostic["per_sample"]["path"], weights_only=True, map_location="cpu", mmap=True)["arrays"]
    require(torch.equal(packet["motion_ids"], saved["expert_motion_id"])
            and torch.equal(packet["frame_indices"], saved["expert_frame_index"]), "Diagnostic IDs differ from actual terminal packet")
    labels, digest, ranges = [], hashlib.sha256(), []
    for row in entry["clips"]:
        motion, count = row["motion_id"], row["frames"]
        start = int(data["length_starts"][motion])
        require(int(data["motion_num_frames"][motion]) == count and data["motion_files"][motion] == row["source_path"]
                and abs(float(data["motion_dt"][motion]) - 1 / 30) < 1e-7, "Bank clip source/bounds/timing differs")
        ranges.append((start, start + count))
        velocity, rotation = data["gvs"][start:start + count, 0, :2], data["grs"][start:start + count, 0]
        require(velocity.dtype == rotation.dtype == torch.float32, "Require original FP32 root features")
        digest.update(velocity.contiguous().numpy().tobytes())
        digest.update(rotation.contiguous().numpy().tobytes())
        labels.append(classify_windows(velocity, rotation).numpy())
    require(all(left[1] <= right[0] for left, right in zip(sorted(ranges), sorted(ranges)[1:]))
            and digest.hexdigest() == ROOT_FEATURE_SHA, "Overlapping or changed original train root feature registry")
    measurement, sampled_labels = prior_and_exposure(entry["clips"], labels, saved["expert_motion_id"].numpy(),
        saved["expert_frame_index"].numpy(), STRATA, overlay["train"])
    require(measurement["all_legal_centers"] == 494792, "Incomplete registered train center inventory")
    numpy_after = np.random.get_state()
    require(random.getstate() == rng_before[0] and rng_before[1][0] == numpy_after[0]
            and np.array_equal(rng_before[1][1], numpy_after[1]) and rng_before[1][2:] == numpy_after[2:]
            and torch.equal(torch.random.get_rng_state(), rng_before[2]) and not torch.cuda.is_initialized(), "RNG/CUDA state changed")
    require(not any(name.split('.')[0] in {"isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules), "CPU reader imported SDK")
    require(all(identity(item["path"]) == item for item in originals + records), "Input/source changed during observation")
    args.output_directory.mkdir(parents=True, exist_ok=False)
    arrays = args.output_directory / "actual_terminal_centers.npz"
    with arrays.open("xb") as stream:
        np.savez(stream, motion_ids=saved["expert_motion_id"].numpy(), frame_indices=saved["expert_frame_index"].numpy(), window_strata=sampled_labels)
    report = {"schema": "shared_social.train_prior_and_actual_terminal_exposure.v1", "status": "completed_CPU_observation",
              "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "sources": originals, "actual_completed_inputs": records,
              "unchanged_original_classifier": {"strata": list(STRATA), "thresholds": vars(WindowThresholds()), "root_feature_sha256": digest.hexdigest()},
              "measurement": measurement, "actual_terminal_centers": identity(arrays), "CPU_rng_unchanged": True,
              "CUDA_initialized": False, "SDK_imported": False, "policy_or_checkpoint_loaded": False,
              "sampling_performed": False, "expert_prior_modified": False, "training_effect_or_cause_proven": False,
              "test34_motion_arrays_read": False, "paper_effect_achieved": False,
              "limits": ["Original prior is uniform clip then legal center; window-uniform fractions are a separate denominator.",
                         "Five short-window kinematic strata are not whole-clip labels, action truth or evaluation speed bins.",
                         "BABEL union coverage does not establish one action or annotator agreement; unknowns retained.",
                         "Actual terminal batch contains4096 rows; whole8.192M expert exposure was not recorded per ID.",
                         "No causal explanation, new training, sampled replacement experts or model promotion."]}
    with (args.output_directory / "diagnostic.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "all_train_clips": measurement["all_train_clips"],
                      "all_legal_centers": measurement["all_legal_centers"], "strata": measurement["strata"], "terminal": measurement["terminal"]}))


if __name__ == "__main__":
    main()
