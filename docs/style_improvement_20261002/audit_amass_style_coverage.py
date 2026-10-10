"""CPU-only, read-only coverage diagnostics for the frozen v7 AMASS bank.

No re-curation, model loading, heldout metrics, FK rebuild or simulation occurs.
New thresholds describe motion proxies; they never manufacture semantic labels.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json"
MANIFEST_SHA = "68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca"
FEET = (3, 4, 7, 8)
FAST = re.compile(r"\b(?:run(?:ning)?|jog(?:ging)?|sprint(?:ing)?|scamper(?:ing)?)\b", re.I)
LOCOMOTION = re.compile(r"\b(?:walk(?:ing)?|run(?:ning)?|jog(?:ging)?|sprint(?:ing)?|locomotion)\b", re.I)
STATIC = re.compile(r"\b(?:stand(?:ing)?|idle|static|look(?:ing)?)\b", re.I)
INPLACE = re.compile(r"\bin\s*place\b|\bon\s*(?:the\s*)?spot\b|treadmill", re.I)
THRESHOLDS = {
    "root_low_speed_mps": .12,
    "root_low_speed_fraction_minimum": .8,
    "foot_heading_relative_active_speed_mps": .3,
    "active_foot_interval_fraction_minimum": .25,
    "quiet_foot_interval_fraction_maximum": .1,
    "quiet_foot_world_height_span_p90_minus_p10_max_m": .05,
    "active_foot_world_height_span_p90_minus_p10_min_m": .08,
}


def fingerprint(path):
    path = Path(path).resolve()
    before = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Input changed while hashing: " + str(path))
    return {"path": str(path), "bytes": after.st_size, "sha256": h.hexdigest(),
            "mtime_ns": after.st_mtime_ns, "sha256_recomputed": True}


def summary(values):
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return {"count": 0}
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite diagnostic")
    return {"count": len(a), "mean": float(a.mean()),
            **{f"p{p}": float(np.percentile(a, p)) for p in (0, 10, 50, 90, 95, 99, 100)}}


def text_tags(text, *, filename=False):
    text = str(text).replace("_", " ").replace("-", " ")
    # KIT's walking_run08 is a named walking take; the suffix is not evidence
    # that its action class is running. Numeric runNN alone remains a proxy.
    suffix = filename and bool(re.search(r"walking\s+run\d+", text, re.I))
    return {"fast_locomotion_keyword": bool(FAST.search(text)) or
            (filename and not suffix and bool(re.search(r"(?:^|\s)(?:run|jog|sprint)\d+(?:\s|$)", text, re.I))),
            "locomotion_keyword": bool(LOCOMOTION.search(text)),
            "standing_looking_keyword": bool(STATIC.search(text)),
            "in_place_keyword": bool(INPLACE.search(text)),
            "walking_run_take_suffix_not_running_semantics": suffix}


def yaw(quat):
    x, y, z, w = np.moveaxis(np.asarray(quat, dtype=np.float64), -1, 0)
    return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def motion_features(bank, index, dt):
    start = int(bank["length_starts"][index])
    count = int(bank["motion_num_frames"][index])
    if count < 2 or start < 0 or start + count > len(bank["gts"]):
        raise ValueError("Invalid physical motion range")
    position = bank["gts"][start:start + count].numpy()
    quat = bank["grs"][start:start + count, 0].numpy()
    root = position[:, 0]
    if not np.isfinite(position).all() or not np.isfinite(quat).all():
        raise ValueError("Nonfinite input pose")
    velocity = np.diff(root[:, :2].astype(np.float64), axis=0) / dt
    speed = np.linalg.norm(velocity, axis=-1)
    angle = yaw(quat)
    turn_rate = np.diff(angle) / dt
    local = position[:, FEET].astype(np.float64) - root[:, None]
    c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
    x, y = local[..., 0].copy(), local[..., 1].copy()
    local[..., 0], local[..., 1] = c * x + s * y, -s * x + c * y
    foot_speed = np.linalg.norm(np.diff(local, axis=0) / dt, axis=-1)
    feet_active = (foot_speed.max(-1) > THRESHOLDS["foot_heading_relative_active_speed_mps"])
    height_span = np.percentile(position[:, FEET, 2], 90, axis=0) - np.percentile(position[:, FEET, 2], 10, axis=0)
    root_low = float((speed < THRESHOLDS["root_low_speed_mps"]).mean())
    active_fraction = float(feet_active.mean())
    span = float(height_span.max())
    quiet = (root_low >= .8 and active_fraction <= .1 and span <= .05)
    active = (root_low >= .8 and active_fraction >= .25 and span >= .08)
    return {"frames": count, "duration_s": (count - 1) * dt,
            "median_root_speed_from_pose_mps": float(np.median(speed)),
            "root_speed_p95_mps": float(np.percentile(speed, 95)),
            "root_low_speed_interval_fraction": root_low,
            "root_fast_interval_fraction": float((speed >= 1.8).mean()),
            "root_path_length_m": float(np.linalg.norm(np.diff(root[:, :2], axis=0), axis=-1).sum()),
            "root_extent_m": float(np.linalg.norm(root[:, :2] - root[0, :2], axis=-1).max()),
            "root_yaw_span_rad": float(np.ptp(angle)),
            "abs_yaw_rate_p95_rad_s": float(np.percentile(np.abs(turn_rate), 95)),
            "near_straight_interval_fraction": float((np.abs(turn_rate) < .2).mean()),
            "heading_relative_foot_speed_median_mps": float(np.median(foot_speed)),
            "heading_relative_foot_speed_p95_mps": float(np.percentile(foot_speed, 95)),
            "active_feet_interval_fraction": active_fraction,
            "foot_world_height_spans_p90_minus_p10_m": height_span.tolist(),
            "root_stationary_active_feet_proxy": bool(active),
            "kinematically_quiet_proxy": bool(quiet),
            "true_idle_or_in_place_locomotion_ground_truth": None}


def evidence(row):
    description = (row.get("semantic_annotation") or {}).get("description")
    return {"kind": "official_motion_description" if description else "filename_keyword_proxy",
            "text": description or Path(row.get("parent_source_path", row["source_path"])).name,
            "source_url": (row.get("semantic_annotation") or {}).get("source_url"),
            "tags": text_tags(description or Path(row.get("parent_source_path", row["source_path"])).name,
                              filename=not description)}


def grouped(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    out = {}
    for name, items in sorted(groups.items()):
        out[name] = {"clips": len(items), "clip_fraction": len(items) / len(rows),
                     "parents": len({r["parent_source_path"] for r in items}),
                     "duration_s": sum(r["duration_s"] for r in items),
                     "legal_expert_centers_history10_future1": sum(r["frames"] - 10 for r in items),
                     "median_root_speed_mps": summary([r["median_root_speed_from_pose_mps"] for r in items]),
                     "root_stationary_active_feet_proxy_clips": sum(r["root_stationary_active_feet_proxy"] for r in items),
                     "kinematically_quiet_proxy_clips": sum(r["kinematically_quiet_proxy"] for r in items)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing audit output")
    begun = time.monotonic()
    torch.set_num_threads(4)
    if torch.cuda.is_initialized():
        raise ValueError("Unexpected CUDA initialization")
    manifest_record = fingerprint(MANIFEST)
    if manifest_record["sha256"] != MANIFEST_SHA:
        raise ValueError("Frozen v7 manifest identity differs")
    document = json.loads(MANIFEST.read_text())
    entry = document["splits"]["train"]
    bank_record = fingerprint(entry["path"])
    if any(bank_record[k] != entry[k] for k in ("bytes", "sha256")):
        raise ValueError("Frozen v7 bank identity differs")
    bank = torch.load(entry["path"], weights_only=True, mmap=True, map_location="cpu")
    rows = []
    for row in entry["clips"]:
        index = row["motion_id"]
        if bank["motion_files"][index] != row["source_path"] or int(bank["motion_num_frames"][index]) != row["frames"]:
            raise ValueError("Manifest/bank frame or source mapping differs")
        parent = row["source_path"].split("#frames=", 1)[0]
        if row["parent_source_path"] != parent:
            raise ValueError("Parent source declaration differs")
        rows.append({"motion_id": index, "original_motion_id": row["original_motion_id"],
                     "parent_source_path": parent, "source_path": row["source_path"],
                     "dataset": row["dataset"], "subject_namespace": row["subject_namespace"],
                     "style_class": row["style_class"], "label_evidence": row["label_evidence"],
                     "semantic": evidence(row), **motion_features(bank, index, 1 / 30)})
    del bank
    styles = grouped(rows, "style_class")
    if {k: v["clips"] for k, v in styles.items()} != document["curation"]["selected_counts_by_style"]:
        raise ValueError("Stored class inventory differs")
    parents = Counter(r["parent_source_path"] for r in rows)
    namespaces = Counter(r["subject_namespace"] for r in rows)
    standing = [r for r in rows if r["style_class"] == "standing_looking_like"]
    fast_semantics = [r for r in rows if r["semantic"]["tags"]["fast_locomotion_keyword"]]
    labels_path = ROOT / "docs/style_improvement_20261002/cmu_motion_labels_join.json"
    labels = json.loads(labels_path.read_text())
    base_path = ROOT / "data/diffusion_amp/amass_manifest.json"
    base = json.loads(base_path.read_text())
    original = base["splits"]["train"]
    original_record = fingerprint(original["path"])
    if any(original_record[k] != original[k] for k in ("bytes", "sha256")):
        raise ValueError("Original train bank identity differs")
    raw = torch.load(original["path"], weights_only=True, mmap=True, map_location="cpu")
    allowed_positive = re.compile(document["selection"]["positive_filename_regex"], re.I)
    negative = re.compile(document["selection"]["negative_filename_regex"], re.I)
    candidates = []
    for index, source in enumerate(raw["motion_files"]):
        annotation = labels.get(source)
        text = annotation["description"] if annotation else Path(source).name
        tags = text_tags(text, filename=not annotation)
        if not (tags["fast_locomotion_keyword"] or tags["in_place_keyword"]):
            continue
        dataset = source.replace("\\", "/").split("/smpl/", 1)[1].split("/", 1)[0]
        dt = float(raw["motion_dt"][index])
        # Read positions at their genuine source rate; no invented 30Hz bank.
        features = motion_features(raw, index, dt)
        rejection = []
        if dataset in document["selection"]["excluded_datasets"]:
            rejection.append("excluded_dataset")
        if negative.search(text):
            rejection.append("negative_description_or_filename")
        if not allowed_positive.search(text):
            rejection.append("missing_navigation_keyword")
        if features["duration_s"] < 3 - 1e-6:
            rejection.append("shorter_than_current3s_minimum")
        candidates.append({"original_motion_id": index, "source_path": source, "dataset": dataset,
                           "selected_parent": source in parents, "selected_segment_count": parents[source],
                           "semantic": {"kind": "official_motion_description" if annotation else "filename_keyword_proxy",
                                        "text": text, "source_url": annotation.get("source_url") if annotation else None,
                                        "tags": tags}, "nominal_source_rate_hz": round(1 / dt),
                           "known_selection_rejection_reasons": rejection,
                           "future_eligibility_certified": False, **features})
    original_motion_count = len(raw["motion_files"])
    del raw
    heldout = {split: {r.get("parent_source_path", r["source_path"].split("#frames=", 1)[0])
                       for r in document["splits"][split]["clips"]}
               for split in ("validation", "test")}
    if set(parents) & (heldout["validation"] | heldout["test"]):
        raise ValueError("Selected parent leaks heldout")
    if any(c["source_path"] in heldout["validation"] | heldout["test"] for c in candidates):
        raise ValueError("Candidate leaks heldout")
    sampler_path = ROOT / "docs/style_improvement_20261002/expert_window_sampling_v7_verification.json"
    sampler = json.loads(sampler_path.read_text())
    if sampler["sampler"]["source"]["manifest_sha256"] != MANIFEST_SHA:
        raise ValueError("Existing short-window proof uses different manifest")
    root_med_error = max(abs(r["median_root_speed_from_pose_mps"] - source["median_speed_m_s"])
                         for r, source in zip(rows, entry["clips"], strict=True))
    report = {
        "schema": "navisaaclab.frozen_amass_v7_coverage_audit.v1", "status": "read_only_coverage_diagnostics_completed",
        "CPU_only": True, "GPU_used": False, "SDK_started": False, "Torch_CUDA_initialized": torch.cuda.is_initialized(),
        "model_loaded": False, "new_data_or_weights_created": False, "heldout_evaluation_outputs_read": False,
        "artifacts": {"audit_script": fingerprint(__file__), "manifest": manifest_record, "bank": bank_record,
                      "original_train_bank": original_record, "base_manifest": fingerprint(base_path),
                      "official_CMU_labels": fingerprint(labels_path), "expert_window_proof": fingerprint(sampler_path),
                      "curation_code": fingerprint(ROOT / "CrowdSim/humanoid/training/amass_curation.py")},
        "counts": {"clips": len(rows), "parents": len(parents), "subject_namespaces": len(namespaces),
                   "duration_s": sum(r["duration_s"] for r in rows), "frames": sum(r["frames"] for r in rows)},
        "class_and_dataset_coverage": {"by_class": styles, "by_dataset": grouped(rows, "dataset"),
                                        "by_label_evidence": grouped(rows, "label_evidence")},
        "candidate_inventory_metadata": {k: document["curation"][k] for k in
            ("train_raw_clips", "eligible_unique_train_clips", "candidate_counts_by_style", "selected_counts_by_style",
             "rejection_reason_counts_not_mutually_exclusive")},
        "proxy_thresholds": THRESHOLDS,
        "proxy_boundaries": "Root/foot motion only. Quiet is not proven true idle; active feet are not proven locomotion or foot contact/slip. Semantic evidence source is retained separately.",
        "standing_diagnostics": {"clips": len(standing), "quiet_proxy": sum(r["kinematically_quiet_proxy"] for r in standing),
                                  "stationary_root_active_feet_proxy": sum(r["root_stationary_active_feet_proxy"] for r in standing),
                                  "other_or_mixed_kinematics": sum(not r["kinematically_quiet_proxy"] and not r["root_stationary_active_feet_proxy"] for r in standing),
                                  "semantic_fast_keyword_clips": sum(r["semantic"]["tags"]["fast_locomotion_keyword"] for r in standing),
                                  "semantic_locomotion_keyword_clips": sum(r["semantic"]["tags"]["locomotion_keyword"] for r in standing),
                                  "rows": standing},
        "selected_fast_semantic_evidence": {"count": len(fast_semantics), "class_counts": dict(Counter(r["style_class"] for r in fast_semantics)),
                                            "evidence_counts": dict(Counter(r["label_evidence"] for r in fast_semantics)), "rows": fast_semantics},
        "short_window_evidence_reused_unchanged": {"reference": fingerprint(sampler_path), "legal_centers": sampler["sampler"]["legal_centers"],
                                                  "support": sampler["sampler"]["support"], "recomputed_here": False},
        "source_multiplicity": {"max_selected_segments_per_parent": max(parents.values()),
                                "parents_with_multiple_segments": sum(c > 1 for c in parents.values()),
                                "effective_parent_count_under_uniform_clip_prior": 1 / sum((n / len(rows)) ** 2 for n in parents.values()),
                                "largest_subject_namespace_clip_mass": max(namespaces.values()) / len(rows),
                                "top_namespaces": namespaces.most_common(10)},
        "distribution_summary": {name: summary([r[name] for r in rows]) for name in
                                 ("median_root_speed_from_pose_mps", "root_speed_p95_mps", "root_yaw_span_rad",
                                  "abs_yaw_rate_p95_rad_s", "root_low_speed_interval_fraction", "active_feet_interval_fraction")},
        "raw_existing_train_candidates_for_future_review": {"scanned_original_motion_count": original_motion_count,
                                                            "rows": candidates, "count": len(candidates),
                                                            "candidate_admission_scope": "Running/in-place evidence first. Raw rate physical features only; no fresh SLERP/FK, all-fields velocity gate, canonical duplication, semantic verification or selection certification."},
        "validation": {"selected_parent_overlap_with_heldout": [], "candidate_parent_overlap_with_heldout": [],
                       "v7_median_root_speed_pose_derivative_max_error_mps": root_med_error,
                       "v7_manifest_unchanged_after_read": fingerprint(MANIFEST) == manifest_record,
                       "v7_bank_stat_unchanged_after_read": bank_record["bytes"] == Path(entry["path"]).stat().st_size
                           and bank_record["mtime_ns"] == Path(entry["path"]).stat().st_mtime_ns},
        "selected_rows": rows, "elapsed_s": time.monotonic() - begun,
        "limitations": ["All five clip classes are exclusive kinematic proxies, not official action labels.",
                        "CMU page descriptions are original primary metadata; other dataset filenames remain weaker evidence.",
                        "Root speed alone cannot distinguish running, fast walking or stationary-root limb activity.",
                        "Heldout IDs and frozen style reference were not changed; heldout tracking/style outputs were not inspected.",
                        "Future candidate lists are not new approved training data and do not prove improved model quality."]}
    if not all(report["validation"][k] for k in ("v7_manifest_unchanged_after_read", "v7_bank_stat_unchanged_after_read")):
        raise ValueError("Frozen v7 resource changed during read")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "counts": report["counts"],
                      "standing": {k: v for k, v in report["standing_diagnostics"].items() if k != "rows"},
                      "raw_candidates": len(candidates), "elapsed_s": report["elapsed_s"]}))


if __name__ == "__main__":
    main()
