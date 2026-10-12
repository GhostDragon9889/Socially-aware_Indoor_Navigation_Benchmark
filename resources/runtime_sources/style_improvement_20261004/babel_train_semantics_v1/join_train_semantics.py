"""Read-only BABEL train metadata -> separate, complete AMASS v7 train overlay."""
from __future__ import annotations

from protomotions.repository import repository_root

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
REPO = repository_root(__file__)
ASSETS = REPO.parent / "Assets"
ZIP = ASSETS / "babel_v1-0_release.zip"
MANIFEST = REPO / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json"
ZIP_SHA = "863d79b78f2d0225df74140b94cd61e06e04217726ae4ba1d2f50bad0d902428"
MANIFEST_SHA = "68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca"
MEMBERS = {
    "train.json": (15101932, "5da8185d4374670682cd6e3391d8635e4297ecebf5fe19394d28c2ca08030c6d"),
    "extra_train.json": (7302453, "1643abb3e8313f4828d706373563e82db4a713f44bba1103df80f28f7ecc5f30"),
}
FAMILIES = ("seq_ann", "seq_anns", "frame_ann", "frame_anns")
SCHEMA = "babel.train_only.v7_semantic_metadata_overlay.v1"


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def file_record(path, raw=None):
    path = Path(path)
    raw = path.read_bytes() if raw is None else raw
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": sha256(raw)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def feature_key(feat_p):
    """Remove exactly the BABEL packaging prefix; never normalize a basename."""
    require(isinstance(feat_p, str), "feat_p must be a string")
    p = PurePosixPath(feat_p)
    require(not p.is_absolute() and len(p.parts) >= 3 and ".." not in p.parts
            and p.suffix == ".npz", f"Unsupported feat_p: {feat_p!r}")
    require(str(p) == feat_p, f"Noncanonical feat_p: {feat_p!r}")
    return "/".join(p.parts[1:])


def source_key(source, raw_root):
    """Use the conversion sidecar's original source, not the output filename."""
    p, root = PurePosixPath(source), PurePosixPath(str(raw_root))
    require(p.is_absolute() and ".." not in p.parts and p.suffix == ".npz"
            and str(p) == source, f"Unsupported sidecar source: {source!r}")
    try:
        return str(p.relative_to(root))
    except ValueError as exc:
        raise ValueError(f"Source outside registered raw root: {source}") from exc


def merge_intervals(intervals):
    """Union of continuous intervals; endpoint equality closes, not fills, a gap."""
    merged = []
    for start, end in sorted(intervals):
        require(math.isfinite(start) and math.isfinite(end) and end > start,
                "Only valid positive intervals can enter the union")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged


def timed_status(label, declared_duration):
    start, end = label.get("start_t"), label.get("end_t")
    if type(start) not in (int, float) or type(end) not in (int, float):
        return "missing_or_non_numeric_time"
    if not (math.isfinite(start) and math.isfinite(end)):
        return "nonfinite_time"
    if end < start:
        return "reversed_interval"
    if start < 0 or end == start:
        return "negative_or_empty_interval"
    # This is a declared-duration consistency diagnostic from the prior join,
    # not evidence that raw poses or an annotation's sample clock were verified.
    if end > declared_duration + 0.05:
        return "outside_declared_duration_by_more_than_0.05s"
    if not (label.get("raw_label") or label.get("proc_label")):
        return "missing_text_label"
    return "valid_timed_metadata"


def label_evidence(record_key, record, member):
    """Retain every family/annotator/label, including unknown and invalid ones."""
    duration = record.get("dur")
    require(type(duration) in (int, float) and math.isfinite(duration) and duration > 0,
            f"Invalid BABEL declared duration: {member}/{record_key}")
    evidence, diagnostics = [], []
    for family in FAMILIES:
        value = record.get(family)
        state = "absent" if family not in record else "null" if value is None else "present"
        if value is None:
            diagnostics.append({"family": family, "state": state})
            continue
        annotations = value if family.endswith("anns") and isinstance(value, list) else [value]
        if ((family.endswith("anns") and not isinstance(value, list))
                or (family.endswith("ann") and not isinstance(value, dict))):
            diagnostics.append({"family": family, "state": "unsupported_type"})
            continue
        if not annotations:
            diagnostics.append({"family": family, "state": "empty"})
        for annotation_index, annotation in enumerate(annotations):
            if not isinstance(annotation, dict) or not isinstance(annotation.get("labels"), list):
                diagnostics.append({"family": family, "annotation_index": annotation_index,
                                    "state": "unknown_labels_structure"})
                continue
            if not annotation["labels"]:
                diagnostics.append({"family": family, "annotation_index": annotation_index,
                                    "state": "empty_labels"})
            for label_index, label in enumerate(annotation["labels"]):
                if not isinstance(label, dict):
                    diagnostics.append({"family": family, "annotation_index": annotation_index,
                                        "label_index": label_index, "state": "unknown_label_structure"})
                    continue
                ref = {"member": member, "record_key": record_key, "babel_sid": record.get("babel_sid"),
                       "family": family, "annotation_index": annotation_index,
                       "label_index": label_index, "babel_lid": annotation.get("babel_lid"),
                       "anntr_id": annotation.get("anntr_id"), "seg_id": label.get("seg_id")}
                evidence.append({"ref": ref, "original_label": label,
                                 "null_or_empty_fields": [k for k in ("raw_label", "proc_label", "act_cat")
                                                          if not label.get(k)],
                                 "timed_status": timed_status(label, duration) if family.startswith("frame")
                                 else "sequence_level_not_time_assigned"})
    return evidence, diagnostics


def clip_time_and_coverage(row, sidecar, evidence):
    source_fps = sidecar["source_fps"]
    require(type(source_fps) in (int, float) and math.isfinite(source_fps) and source_fps > 0,
            "Invalid conversion source FPS")
    origin = sidecar["source_frame_slice"][0] / source_fps
    first, last = origin + row["parent_start_time_s"], origin + row["parent_end_time_s"]
    require(math.isfinite(first) and math.isfinite(last) and last >= first, "Invalid v7 time range")
    valid = [item for item in evidence if item["timed_status"] == "valid_timed_metadata"]
    union = merge_intervals([(item["original_label"]["start_t"], item["original_label"]["end_t"])
                             for item in valid])
    overlap = [item["ref"] for item in valid
               if min(last, item["original_label"]["end_t"]) > max(first, item["original_label"]["start_t"])]
    # A single merged interval must contain the entire first-to-last sample
    # range. Separate intervals covering the two endpoints are insufficient.
    full = any(start <= first + 1e-8 and end >= last - 1e-8 for start, end in union)
    return {"raw_slice_origin_s": origin, "raw_source_fps": source_fps,
            "conversion_slicing_fps": int(round(source_fps)),
            "conversion_fps_rounding_delta": int(round(source_fps)) - source_fps,
            "first_sample_time_s": first, "last_sample_time_s": last,
            "range_convention": "first_to_last_sample_inclusive; not the half-open frame stop",
            "valid_label_interval_union_s": union,
            "positive_interval_overlap_label_refs": overlap,
            "full_sample_time_range_union_covered": full,
            "coverage_is_not_single_action_or_annotator_agreement": True}


def read_annotations(archive):
    indexes, records = {}, []
    with ZipFile(archive) as z:
        for name, (expected_bytes, expected_sha) in MEMBERS.items():
            member = "babel_v1.0_release/" + name
            matches = [info for info in z.infolist() if info.filename == member]
            require(len(matches) == 1 and not (matches[0].flag_bits & 1), f"Missing/duplicate/encrypted {member}")
            raw = z.read(matches[0])  # Only these two train members are decoded; ZIP verifies their CRC.
            require(len(raw) == expected_bytes and sha256(raw) == expected_sha, f"Member identity changed: {member}")
            data = json.loads(raw)
            require(isinstance(data, dict), f"Expected BABEL record dictionary: {member}")
            index = {}
            for record_key, row in data.items():
                require(isinstance(row, dict), f"Invalid BABEL row: {record_key}")
                key = feature_key(row["feat_p"])
                require(key not in index, f"Ambiguous duplicate normalized feat_p within {member}: {key}")
                evidence, diagnostics = label_evidence(record_key, row, name)
                index[key] = {"record_key": record_key, "original_record": row,
                              "label_evidence": evidence, "family_diagnostics": diagnostics}
            indexes[name] = index
            records.append({"member": member, "bytes": len(raw), "sha256": sha256(raw),
                            "CRC32": format(matches[0].CRC, "08x"), "unique_feat_p_records": len(index)})
    return indexes, records


def disagreements(evidence):
    """Record distinctions, not a vote or proof that concurrent labels conflict."""
    timed = [item for item in evidence if item["timed_status"] == "valid_timed_metadata"]
    exact, pairs = defaultdict(set), []
    for i, item in enumerate(timed):
        label = item["original_label"]
        signature = json.dumps([label.get("proc_label"), sorted(label.get("act_cat") or [])])
        exact[(label["start_t"], label["end_t"])].add(signature)
        pairs.append((i, label["start_t"], label["end_t"], signature))
    distinct_overlap = []
    for n, (i, start, end, signature) in enumerate(pairs):
        for j, other_start, other_end, other_signature in pairs[n + 1:]:
            if signature != other_signature and min(end, other_end) > max(start, other_start):
                distinct_overlap.append([timed[i]["ref"], timed[j]["ref"]])
    return {"same_exact_span_distinct_label_sets": [list(span) for span, values in sorted(exact.items()) if len(values) > 1],
            "overlapping_distinct_label_pairs": distinct_overlap,
            "comparison_signature": "original proc_label and sorted act_cat; all original raw labels remain preserved",
            "interpretation": "possible multi-action/concurrent labels or annotator disagreement; no voting or truth conflict decision"}


def build_overlay():
    zip_record = file_record(ZIP)
    require(zip_record["sha256"] == ZIP_SHA, "Pinned BABEL ZIP identity changed")
    raw_manifest = MANIFEST.read_bytes()
    manifest_record = file_record(MANIFEST, raw_manifest)
    require(manifest_record["sha256"] == MANIFEST_SHA, "Pinned v7 manifest identity changed")
    clips = json.loads(raw_manifest)["splits"]["train"]["clips"]
    require(len(clips) == 2848, "Expected all 2848 v7 train rows")
    indexes, member_records = read_annotations(ZIP)
    parent_rows = defaultdict(list)
    for row in clips:
        parent_rows[row["parent_source_path"]].append(row)
    require(len(parent_rows) == 1853, "Expected 1853 v7 train parents")
    parents, parent_ids, summary, checks = [], {}, Counter(), []
    member_counts = {name: Counter() for name in MEMBERS}
    for parent_path, rows in parent_rows.items():
        parent_id = len(parents)
        parent_ids[parent_path] = parent_id
        sidecar_path = Path(parent_path).with_suffix(".json")
        require(PurePosixPath(parent_path).is_relative_to(PurePosixPath(str(ASSETS / "motion/processed/smpl"))),
                "v7 parent outside registered processed SMPL root")
        raw = sidecar_path.read_bytes()
        sidecar = json.loads(raw)
        sidecar_record = file_record(sidecar_path, raw)
        checks.append(sidecar_record)
        require(sidecar["output"] == parent_path and all(sidecar["output_frames"] == r["parent_source_frames"] for r in rows),
                f"Conversion output identity/frames differ: {parent_path}")
        require(type(sidecar["source_frames"]) is int and sidecar["source_frames"] > 0
                and type(sidecar["source_fps"]) in (int, float) and math.isfinite(sidecar["source_fps"])
                and sidecar["source_fps"] > 0 and type(sidecar["output_fps"]) in (int, float)
                and math.isfinite(sidecar["output_fps"]) and sidecar["output_fps"] > 0,
                "Invalid conversion frame count/FPS")
        frame_slice = sidecar["source_frame_slice"]
        require(isinstance(frame_slice, list) and len(frame_slice) == 2 and all(type(v) is int for v in frame_slice)
                and 0 <= frame_slice[0] < frame_slice[1] <= sidecar["source_frames"], "Invalid raw conversion slice")
        require(all(abs(r["parent_source_dt"] - 1 / sidecar["output_fps"]) <= 1e-7 for r in rows),
                "v7 processed source clock differs from conversion output FPS")
        key = source_key(sidecar["source"], ASSETS / "motion/raw")
        by_member = {name: index[key] for name, index in indexes.items() if key in index}
        evidence = [item for match in by_member.values() for item in match["label_evidence"]]
        for name, match in by_member.items():
            member_counts[name]["matched_parents"] += 1
            member_counts[name]["matched_clips"] += len(rows)
            for item in match["label_evidence"]:
                kind = "frame" if item["ref"]["family"].startswith("frame") else "seq"
                member_counts[name][kind + "_labels"] += 1
                member_counts[name].update(kind + "_null_or_empty_" + field for field in item["null_or_empty_fields"])
        matched = bool(by_member)
        summary["matched_parents" if matched else "unmatched_parents"] += 1
        summary["matched_clips" if matched else "unmatched_clips"] += len(rows)
        summary["parents_with_valid_timed_labels"] += int(any(item["timed_status"] == "valid_timed_metadata" for item in evidence))
        summary["invalid_timed_labels"] += sum(item["ref"]["family"].startswith("frame")
                                                and item["timed_status"] != "valid_timed_metadata" for item in evidence)
        distinctions = disagreements(evidence)
        for field in ("same_exact_span_distinct_label_sets", "overlapping_distinct_label_pairs"):
            summary[field] += len(distinctions[field])
            summary["parents_with_" + field] += int(bool(distinctions[field]))
        raw_duration = sidecar["source_frames"] / sidecar["source_fps"]
        parents.append({"parent_index": parent_id, "parent_source_path": parent_path, "exact_raw_source_key": key,
                        "conversion_sidecar_identity": sidecar_record, "original_conversion_metadata": sidecar,
                        "matched_train_member_records": by_member, "label_evidence": evidence,
                        "raw_duration_frames_over_fps_s": raw_duration,
                        "BABEL_minus_raw_duration_s": {name: match["original_record"]["dur"] - raw_duration
                                                       for name, match in by_member.items()},
                        "label_distinctions": distinctions})
    overlay = []
    for train_index, row in enumerate(clips):
        parent = parents[parent_ids[row["parent_source_path"]]]
        alignment = clip_time_and_coverage(row, parent["original_conversion_metadata"], parent["label_evidence"])
        summary["clips_with_positive_timed_overlap"] += int(bool(alignment["positive_interval_overlap_label_refs"]))
        summary["clips_full_sample_time_range_union_covered"] += int(alignment["full_sample_time_range_union_covered"])
        overlay.append({"train_index": train_index, "original_v7_train_row": row, "parent_index": parent["parent_index"],
                        "sequence_evidence_refs_not_time_assigned": [item["ref"] for item in parent["label_evidence"]
                                                                    if item["ref"]["family"].startswith("seq")],
                        "raw_time_alignment": alignment,
                        "no_train_annotation_match": not bool(parent["matched_train_member_records"])})
    # Verify the original metadata bytes remain unchanged before publishing.
    for identity in [zip_record, manifest_record, *checks]:
        require(file_record(identity["path"]) == identity, f"Input changed during join: {identity['path']}")
    return {"schema": SCHEMA, "status": "semantic_metadata_overlay_only",
            "sources": {"ZIP": zip_record, "v7_manifest": manifest_record, "train_annotation_members": member_records,
                        "executed_CLI_source": file_record(__file__)},
            "scope": {"train_clips": 2848, "train_parents": 1853, "BABEL_members_decoded": list(MEMBERS),
                      "heldout_annotations_or_outputs_used": False, "raw_motion_or_bank_loaded": False,
                      "NN_SDK_GPU_used": False, "original_v7_or_style5_or_sampling_changed": False,
                      "author_2848_list_recovered": False, "raw_pose_clock_independently_verified": False},
            "summary": {**dict(summary), "per_member": {name: dict(counts) for name, counts in member_counts.items()}},
            "parents": parents, "train": overlay}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "actual_train_semantic_overlay.json")
    args = parser.parse_args()
    output = args.output.resolve()
    require(output.is_relative_to(HERE) and output != HERE, "Output must be a new file inside this independent stage")
    require(not output.exists(), f"Preserve existing output: {output}")
    document = build_overlay()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": file_record(output), "summary": document["summary"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
