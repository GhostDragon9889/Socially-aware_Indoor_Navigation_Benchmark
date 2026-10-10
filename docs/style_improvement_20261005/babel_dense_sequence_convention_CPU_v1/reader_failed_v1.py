#!/usr/bin/env python3
"""Separate BABEL dense sequence-only whole-span assumptions from timed data.

Conservative extension: train.json, explicit null frame_ann, one seq label,
mul_act=False, positive duration. Extra annotations are never extended.
This adds provenance, not measured action times or replacement motion labels.
Source: https://babel.is.tue.mpg.de/data.html#
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.diagnose_shared_social_expert_coverage import identity, require  # noqa: E402

OVERLAY = ROOT / "docs/style_improvement_20261004/babel_train_semantics_v1/actual_train_semantic_overlay.json"
OVERLAY_SHA = "09e16f40f6cefa283c6366589577ad0af1b2aa8123dc032af3f65893ab7889b1"
RECEIPT = OVERLAY.parent / "actual_CPU_tool_receipt.json"
HELPER = ROOT / "scripts/diagnose_shared_social_expert_coverage.py"
HELPER_SHA = "53d6e82b41cb0f75c7683b83e42253e2fdd6d8da5f80e3d22b381179d2bdd514"


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def dense_parent_assumption(parent):
    """No category votes or extra-family fallback; exclusions remain explicit."""
    dense = parent["matched_train_member_records"].get("train.json")
    outcome = {"parent_index": parent["parent_index"], "parent_source_path": parent["parent_source_path"],
               "exact_raw_source_key": parent["exact_raw_source_key"], "eligible": False,
               "extra_annotation_used_for_extension": False, "measured_frame_interval": False}
    if dense is None:
        return {**outcome, "reason": "no_dense_train_match"}
    record = dense["original_record"]
    outcome["dense_record_key"] = dense["record_key"]
    outcome["original_sequence_annotation"] = record.get("seq_ann")
    duration = record.get("dur")
    outcome["original_BABEL_duration_s"] = duration
    if "frame_ann" not in record:
        reason = "frame_annotation_field_absent_unknown"
    elif record["frame_ann"] is not None:
        reason = "explicit_frame_annotation_present"
    elif not isinstance(record.get("seq_ann"), dict):
        reason = "sequence_annotation_absent_or_unsupported"
    elif record["seq_ann"].get("mul_act") is not False:
        reason = "not_explicit_single_action"
    elif not isinstance(record["seq_ann"].get("labels"), list) or len(record["seq_ann"]["labels"]) != 1:
        reason = "not_exactly_one_sequence_label"
    elif not finite_number(duration) or duration <= 0:
        reason = "invalid_BABEL_duration"
    else:
        label = record["seq_ann"]["labels"][0]
        if not isinstance(label, dict) or not all(isinstance(label.get(key), str) and label[key].strip()
                                                for key in ("raw_label", "proc_label")):
            reason = "missing_label_text"
        elif any(key in label and (not finite_number(label[key]) or label[key] != value)
                 for key, value in (("start_t", 0), ("end_t", duration))):
            reason = "sequence_time_fields_conflict_with_whole_span"
        else:
            evidence = [item for item in dense["label_evidence"] if item["ref"]["family"] == "seq_ann"
                        and item["ref"]["label_index"] == 0]
            require(len(evidence) == 1 and evidence[0]["original_label"] == label
                    and evidence[0]["ref"]["member"] == "train.json"
                    and evidence[0]["ref"]["record_key"] == dense["record_key"], "Original sequence evidence ownership differs")
            return {**outcome, "eligible": True, "reason": "dense_single_sequence_whole_span_assumption",
                    "assumed_interval_s": [0, duration], "original_label": label, "original_evidence_ref": evidence[0]["ref"],
                    "assumption_is_not_new_frame_annotation": True,
                    "null_category_retained": label.get("act_cat") is None,
                    "raw_duration_frames_over_fps_s": parent["raw_duration_frames_over_fps_s"],
                    "BABEL_minus_raw_duration_s": parent["BABEL_minus_raw_duration_s"]}
    return {**outcome, "reason": reason}


def project_to_clip(row, parent_assumption):
    require(row["parent_index"] == parent_assumption["parent_index"], "Clip/parent ownership differs")
    alignment = row["raw_time_alignment"]
    first, last = alignment["first_sample_time_s"], alignment["last_sample_time_s"]
    require(finite_number(first) and finite_number(last) and 0 <= first <= last, "Invalid original clip sample-time range")
    result = {"train_index": row["train_index"], "original_v7_train_row": row["original_v7_train_row"],
              "parent_index": row["parent_index"], "original_clip_first_last_sample_s": [first, last],
              "original_explicit_timed_union_covers_full_clip": alignment["full_sample_time_range_union_covered"],
              "original_no_train_annotation_match": row["no_train_annotation_match"],
              "dense_parent_assumption_eligible": parent_assumption["eligible"],
              "assumption_covers_full_clip": False, "assumed_positive_duration_overlap_s": None,
              "assumption_is_not_measured_frame_label": True}
    if parent_assumption["eligible"]:
        start, end = parent_assumption["assumed_interval_s"]
        lower, upper = max(first, start), min(last, end)
        result["assumption_covers_full_clip"] = start <= first and last <= end
        result["clip_relation_to_assumed_span"] = "inside" if result["assumption_covers_full_clip"] else "not_fully_inside"
        if upper > lower:
            result["assumed_positive_duration_overlap_s"] = [lower, upper]
        result["dense_original_label_ref"] = parent_assumption["original_evidence_ref"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not args.output.exists(), "Require CUDA-hidden CPU and new output")
    sources = [identity(Path(__file__)), identity(HELPER), identity(OVERLAY), identity(RECEIPT)]
    require(sources[1]["sha256"] == HELPER_SHA and sources[2]["sha256"] == OVERLAY_SHA, "Pinned original data/source changed")
    overlay, receipt = (json.loads(path.read_text()) for path in (OVERLAY, RECEIPT))
    require(receipt["schema"] == "babel.train_semantic_overlay.actual_root_CPU_receipt.v1"
            and type(receipt["actual_exit_code"]) is int and receipt["actual_exit_code"] == 0
            and isinstance(receipt["actual_completion_chunk"], str) and bool(receipt["actual_completion_chunk"])
            and {key: receipt["overlay"][key] for key in ("path", "bytes", "sha256")} == sources[2],
            "Require actual original overlay exit0 and bound artifact")
    manifest_record, zip_record = (overlay["sources"][key] for key in ("v7_manifest", "ZIP"))
    require(identity(manifest_record["path"]) == manifest_record and identity(zip_record["path"]) == zip_record,
            "Original manifest/ZIP changed")
    manifest = json.loads(Path(manifest_record["path"]).read_text())
    require(manifest["splits"]["train"]["clip_count"] == len(overlay["train"]) == 2848
            and manifest["splits"]["train"]["clips"] == [row["original_v7_train_row"] for row in overlay["train"]],
            "Original fixed2848 rows or order changed")
    members = [item for item in overlay["sources"]["train_annotation_members"] if item["member"].endswith('/train.json')]
    require(len(members) == 1, "Ambiguous original dense train member")
    member = members[0]
    with ZipFile(zip_record["path"]) as archive:
        require(archive.namelist().count(member["member"]) == 1, "Duplicate dense member")
        raw = archive.read(member["member"])
    require(len(raw) == member["bytes"] and hashlib.sha256(raw).hexdigest() == member["sha256"], "Dense member changed")
    dense_data = json.loads(raw)
    parents = overlay["parents"]
    require(len({parent["parent_index"] for parent in parents}) == len(parents), "Duplicate parent IDs")
    for parent in parents:
        matched = parent["matched_train_member_records"].get("train.json")
        if matched is not None:
            require(dense_data[matched["record_key"]] == matched["original_record"]
                    and matched["original_record"]["feat_p"] == parent["exact_raw_source_key"], "Dense original record/raw parent join differs")
    assumptions = [dense_parent_assumption(parent) for parent in parents]
    by_parent = {item["parent_index"]: item for item in assumptions}
    clips = []
    for index, row in enumerate(overlay["train"]):
        require(row["train_index"] == index, "Original train order changed")
        clips.append(project_to_clip(row, by_parent[row["parent_index"]]))
    sources += [manifest_record, zip_record]
    require(all(identity(item["path"]) == item for item in sources), "Input/source changed during analysis")
    require(not any(name.split('.')[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules),
            "Semantic-only builder imported model/SDK framework")
    statuses = Counter(item["reason"] for item in assumptions)
    category_counts = Counter()
    for item in assumptions:
        if item["eligible"]:
            categories = item["original_label"].get("act_cat")
            if categories:
                require(isinstance(categories, list) and all(isinstance(cat, str) for cat in categories), "Unsupported original action categories")
                category_counts.update(categories)
            else:
                category_counts["<null_or_empty_original_category>"] += 1
    statistics = {"all_train_clips": len(clips), "all_parent_sources": len(parents), "parent_status_counts": dict(statuses),
                  "eligible_dense_single_action_parents": sum(item["eligible"] for item in assumptions),
                  "eligible_parent_original_category_counts_overlapping": dict(category_counts),
                  "clips_with_eligible_parent": sum(item["dense_parent_assumption_eligible"] for item in clips),
                  "clips_fully_inside_assumed_span": sum(item["assumption_covers_full_clip"] for item in clips),
                  "clips_newly_full_span_by_assumption_only": sum(item["assumption_covers_full_clip"]
                       and not item["original_explicit_timed_union_covers_full_clip"] for item in clips),
                  "original_full_clip_explicit_timed_union": sum(item["original_explicit_timed_union_covers_full_clip"] for item in clips)}
    document = {"schema": "babel.fixed2848_dense_sequence_convention.v1", "status": "completed_CPU_provenance_extension",
                "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "sources": sources, "dense_member_read": member,
                "statistics": statistics, "parents": assumptions, "train": clips,
                "assumption_source": "https://babel.is.tue.mpg.de/data.html",
                "extra_train_whole_span_extension_performed": False, "original_overlay_modified": False,
                "fixed_train_registry_modified": False, "models_or_motion_arrays_loaded": False,
                "validation_or_test_annotations_read": False, "author_2848_IDs_recovered": False,
                "training_or_sampling_changed": False, "paper_effect_proven": False,
                "limits": ["Only explicit single-action dense train sequences with null frame_ann are extended; ambiguous/multi-action rows retained.",
                           "Whole-span timing is the dataset convention, not measured frame timing or annotator agreement.",
                           "Reported BABEL duration is not silently stretched to actual raw or converted clip duration.",
                           "Original labels/categories/nulls remain unchanged; no action vote, proxy relabeling, reweighting or new motion selection.",
                           "Original overlay may include extra_train evidence; this separate assumption layer only reads dense train.json."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(statistics))


if __name__ == "__main__":
    main()
