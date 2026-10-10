"""Admit full native captures for the existing frozen CPU style calculations.

The new boundary handles explicit N3..15, native scene and raw-audit schemas.
Pose extraction, reference selection, clusters and numerical scores are reused
unchanged. This module does not establish a process exit or paper equivalence.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from CrowdSim.humanoid.analysis import native_navigation_audit as raw
from scripts import evaluate_shared_social_navigation_style as metric

ROOT = raw.ROOT
OWNER_SCHEMA = "shared_social.actual_native_raw_CPU_audit_execution.v1"
STYLE_SCHEMA = "shared_social.paired_native_navigation_inclusive_style.v1"
SOURCES = (
    ("scripts/evaluate_shared_social_navigation_style.py", "394ce99e1eae7d89c886adc5cbcd6cb292b58bee3758609522a7b72270e6c10a"),
    ("scripts/evaluate_shared_social_style.py", "d3296011b0e08286ae109daea6a518c95f5f9fa9f24a474a20555cd002baa47c"),
    ("scripts/audit_shared_social_native.py", "1bdb0b784ceb888b205f87c5ec4352d81afa4faec60b05bca650f49d2026f208"),
    ("CrowdSim/humanoid/analysis/native_navigation_audit.py", "f80f005bb2796bf37168281f243ab462ee207223609b735656a41b7febe9fe8b"),
    ("CrowdSim/humanoid/style_evaluation.py", metric.METRIC_SHA),
)


def source_bindings():
    records = [metric.record(ROOT / name) for name, _ in SOURCES]
    metric.require(all(row["sha256"] == sha for row, (_, sha) in zip(records, SOURCES, strict=True)),
                   "Original native raw or frozen style source changed")
    frozen = metric.frozen
    reference_records = [metric.record(path) for path in
        (frozen.TOOL, frozen.REFERENCE, frozen.POSE_TOOL, frozen.BASIS, frozen.BASIS_EVALUATION, raw.MANIFEST)]
    expected = (frozen.TOOL_SHA, frozen.REFERENCE_SHA, frozen.POSE_TOOL_SHA, frozen.BASIS_SHA,
                frozen.BASIS_EVALUATION_SHA, raw.MANIFEST_SHA)
    metric.require(all(row["sha256"] == sha for row, sha in zip(reference_records, expected, strict=True)),
                   "Frozen held-out style resources or fixed2848 manifest changed")
    return {"native_style_reader": metric.record(__file__), "unchanged_style_and_raw_sources": records,
            "frozen_reference_resources": reference_records, "native_raw_kernels": raw.source_bindings()}


def validate_audit_contract(owner, audit, kernels, *, planner):
    """Check exposure/source declarations; actual files are admitted separately."""
    require = metric.require
    actors, controls = kernels.actors, kernels.controls
    raw._protocol(actors, controls)
    require(controls == 6000 and planner in {"diffusion", "sfm"}, "Native style requires a complete200s planner arm")
    require(owner.get("schema") == OWNER_SCHEMA
            and type(owner.get("actual_exit_code")) is not int and owner["actual_exit_code"] == 0
            and isinstance(owner.get("terminal_tool_chunk"), str) and bool(owner["terminal_tool_chunk"])
            and type(owner.get("actors")) is int and owner["actors"] == actors
            and type(owner.get("control_steps")) is int and owner["control_steps"] == controls
            and owner.get("planner") == planner,
            "Require externally observed successful native raw CPU audit for this exact cohort")
    exposures = {"actors": actors, "actual_controls": controls, "actor_controls": controls * actors,
        "physics_substeps": controls * 4, "native_actor_substep_samples": controls * 4 * actors,
        "unordered_pair_controls_exposure": controls * actors * (actors - 1) // 2}
    require(audit.get("schema") == raw.AUDIT_SCHEMA and audit.get("status") == "passed"
            and all(type(audit.get(key)) is int and audit[key] == value for key, value in exposures.items())
            and audit.get("planner") == planner and audit.get("scope") == "native_local_200s"
            and audit.get("duration_s") == 200 and audit.get("all_failure_contact_and_reset_frames_retained") is True
            and audit["recorded_body_pose"]["all24_same_control_native_body_positions_equal"] is True
            and audit["initial_capture_link"]["status"] == "live_initial_capture_linked"
            and audit["initial_capture_link"]["all_registered_first_raw_reference_position_velocity_anchors_equal"] is True,
            "Require the complete200s native body/initial audit with exact exposure units and retained failures")
    require(audit["reader"] == metric.record(ROOT / SOURCES[2][0])
            and audit["kernel_sources"] == kernels.sources and audit["kernel_derivation"] == kernels.derivation
            and audit.get("metric_formulae_changed") is False and audit.get("Torch_imported") is False
            and audit.get("SDK_executed_by_reader") is False
            and audit["execution_admission"]["declared_native_stage_floor_source_and_exposure_admitted"] is True
            and audit["execution_admission"]["actual_process_exit_independently_observed_by_reader"] is False,
            "Native style requires the unchanged independently audited raw reader and kernels")


def capture_samples(physical, ground, result, base, *, actors):
    """Read named full captures; reuse the original interval/normalization math."""
    raw._protocol(actors, 6000)
    require = metric.require
    controls = 6000
    data = {name: physical[name] for name in ("body_pos", "root_rot", "episode_ids", "reset_mask", "terminated")}
    names = ground["common_body_names"].tolist()
    require(names == result["state_capture"]["body_names"] and len(names) == 24 and set(names) == set(metric.BODY_NAMES),
            "Native style requires the exact named COMMON 24-body order")
    shapes = {"body_pos": (controls, actors, 24, 3), "root_rot": (controls, actors, 4),
              "episode_ids": (controls, actors), "reset_mask": (controls, actors), "terminated": (controls, actors)}
    require(all(data[name].shape == shape for name, shape in shapes.items())
            and data["terminated"].dtype == np.bool_, "Native style capture must contain every registered actor/control")
    native_body, native_quat = ground["body_link_pos_w"], ground["body_link_quat_w"]
    require(native_body.shape == (controls, 4, actors, 24, 3)
            and native_quat.shape == (controls, 4, actors, 24, 4)
            and np.array_equal(data["body_pos"], native_body[:, -1])
            and np.array_equal(data["root_rot"], native_quat[:, -1, :, names.index("Pelvis")])
            and np.array_equal(data["episode_ids"], ground["episode_ids"])
            and np.array_equal(data["reset_mask"], ground["reset_mask"]),
            "Native style body/root quaternion/episodes differ from the fourth physical substep")
    samples = metric.sequence_style_samples(data["body_pos"], data["root_rot"], body_names=names,
        episode_ids=data["episode_ids"], reset_mask=data["reset_mask"], spec=metric.StyleSpec())
    require(samples["summary"]["same_episode_transition_count"] == base["same_episode_position_intervals"]
            and int(data["terminated"].sum()) == base["fall_termination_actor_controls"],
            "Native style transition/failure denominators differ from the full raw audit")
    kept = int(data["terminated"][samples["end_frame_indices"], samples["actor_indices"]].sum())
    failures = int(data["terminated"].sum())
    return samples, {"all_capture_frames": controls * actors, "fall_termination_frames": failures,
        "fall_termination_interval_endpoints_kept": kept,
        "fall_termination_frames_without_legal_predecessor": failures - kept,
        "all_failure_frames_retained_in_capture": True, "failed_episodes_filtered": False,
        "initial_pose_rows_without_predecessor": actors,
        "reset_boundary_intervals_excluded": samples["summary"]["excluded_reset_transitions"],
        "pose_and_velocity_samples_use_same_episode_interval_endpoints": True,
        "scope": "Original inclusive interval metric; no success/fall/contact episode filtering"}


def _identities(value):
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= value.keys():
            yield {key: value[key] for key in ("path", "bytes", "sha256")}
        else:
            for child in value.values():
                yield from _identities(child)
    elif isinstance(value, list):
        for child in value:
            yield from _identities(child)


def read_completed_run(owner_path, *, actors, planner, scene_config, scene_sha256):
    """Require external raw CPU exit and re-admit actual native SDK resources."""
    bindings = source_bindings()
    kernels = raw.prepare_kernels(actors=actors, controls=6000)
    owner_identity = metric.record(owner_path)
    owner = metric.bound(owner_identity)
    # Fail an incomplete/old receipt before following any of its resource paths.
    metric.require(owner.get("schema") == OWNER_SCHEMA
        and type(owner.get("actual_exit_code")) is int and owner["actual_exit_code"] == 0
        and isinstance(owner.get("terminal_tool_chunk"), str) and bool(owner["terminal_tool_chunk"]),
        "Require externally observed successful native raw CPU audit")
    audit = metric.bound(owner["audit"])
    validate_audit_contract(owner, audit, kernels, planner=planner)
    selected = raw.admit_execution(Path(audit["execution_receipt"]["path"]), actors=actors, controls=6000,
                                  planner=planner, scene_config=scene_config, scene_sha256=scene_sha256)
    base = metric.bound(audit["base_audit"])
    result = selected.result
    metric.require(owner["scene_configuration"] == selected.request["configuration_file"]
        and audit["native_scene_request"] == selected.request and audit["result"] == selected.receipt["result"]
        and audit["execution_receipt"] == selected.identities[0]
        and audit["execution_admission"] == selected.admission
        and base["schema"] == "shared_social.source_bound_raw_array_kernel.v1" and base["status"] == "passed"
        and base["result"] == audit["result"] and base["execution_receipt"] == audit["execution_receipt"]
        and type(base["controls"]) is int and base["controls"] == 6000
        and type(base["actor_controls"]) is int and base["actor_controls"] == 6000 * actors
        and base["arrays"] == selected.arrays
        and result["physics"]["control_hz"] == 30 and result["physics"]["physics_hz"] == 120
        and audit["initial_capture_link"]["document"] == result["paired_initial_condition"]["document"]
        and audit["initial_capture_link"]["arrays"] == result["paired_initial_condition"]["arrays"],
        "Native style audit, actual exit, scene, initial condition or selected arrays disagree")
    for row in selected.arrays:
        path = Path(row["path"])
        metric.require(metric.record(path) == row and not path.is_symlink() and path.stat().st_nlink == 1
                       and not path.stat().st_mode & 0o222, "Actual native style raw arrays must be sealed and unchanged")
    with np.load(selected.arrays[0]["path"], allow_pickle=False) as physical, \
            np.load(selected.arrays[1]["path"], allow_pickle=False) as ground:
        samples, capture = capture_samples(physical, ground, result, base, actors=actors)
    identities = [owner_identity, owner["audit"], audit["base_audit"], *selected.identities,
                  *_identities(kernels.sources), *_identities(bindings)]
    metric.require(all(metric.record(row["path"]) == row for row in identities), "Native style admission inputs changed")
    return SimpleNamespace(samples=samples, result=result, base=base, identities=identities,
        admission={"owner": owner_identity, "audit": owner["audit"], "arrays": selected.arrays,
                   "actual_process_exit_independently_observed_by_reader": False, **capture})
