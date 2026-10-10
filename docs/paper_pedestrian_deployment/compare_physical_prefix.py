"""CPU-only actual30s entry vs actual200s guide20 prefix; no synthetic physics."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STYLE = ROOT / "docs/style_improvement_20261002"
ENTRY = Path(__file__).resolve().parent
BASELINE = ROOT / "output/diffusion_amp/shared_amass2848_backward_penalty_ground_support_200s_v1"
HELPER = STYLE / "audit_ground_support_capture.py"
HELPER_SHA = "f851e7d5f90d48d830c564fdc2b2e446dfe35daa36ec636aba6f946ec7752c3f"
BASELINE_RESULT_SHA = "7d7cddd22e1ce3c072825eee7f73661fbeb4fe4349a67f033c12dc411dbbf81e"
BASELINE_AUDIT_SHA = "fb32c5fa56dae76478919e4d8c91748f6bd78a57a7b8d1bb9fc5741919410307"
SOURCE135_SHA = "8cfb987d4dbd569b2d205c4ec015f29b49c8cb4cb8a6e7a03ed4a92becd18f8e"
GUIDE40_SOURCE70_SHA = "45ae45a6a7d9d72fe5d54a70cbab8d3aed41fdeb4240e0cf8832863940a1bce6"
ENTRY_FREEZE_SHA = "9b120a0c9c9e2a1570ee4f7b551469f5992064db33d28e625897dbed7029ec3b"
CONTROL_LOOP_SHA = "1adbfe51b09a2311b3e450925cface7b4f81a1fbb122dd945838d16a9272a74f"
RUNTIME_SOURCES = (
    "scripts/train_paper_amp.py", "CrowdSim/humanoid/amp_world.py",
    "CrowdSim/humanoid/training/trainer.py", "CrowdSim/humanoid/amp/observations.py",
    "CrowdSim/humanoid/diffusion/paper.py", "protomotions/simulator/isaaclab/simulator.py",
)
STATIC_GROUND_FIELDS = {"common_body_names", "foot_names"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def record(path):
    path = Path(path).resolve()
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1048576), b""):
            hasher.update(part)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": hasher.hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def load_helper():
    require(record(HELPER)["sha256"] == HELPER_SHA, "Frozen native ground auditor changed")
    spec = importlib.util.spec_from_file_location("entry_frozen_native_auditor", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_archive(path, expected_sha, count, *, entry=False):
    require(record(path)["sha256"] == expected_sha, "Source manifest hash changed")
    document = read(path)
    rows = document["source_files" if entry else "files"]
    archive_record = document["source_archive" if entry else "archive"]
    archive_path = ROOT / archive_record["path"]
    actual = record(archive_path)
    require(actual["bytes"] == archive_record["bytes"] and actual["sha256"] == archive_record["sha256"],
            "Source archive bytes/hash changed")
    expected = {row["path"]: row for row in rows}
    require(len(expected) == len(rows) == count, "Source member count/uniqueness differs")
    for name in expected:
        path_name = PurePosixPath(name)
        require(not path_name.is_absolute() and ".." not in path_name.parts and "\\" not in name,
                "Source path is not a safe relative member")
    raw = {}
    with tarfile.open(archive_path) as stream:
        members = stream.getmembers()
        require(len(members) == count and {m.name for m in members} == set(expected)
                and all(m.isfile() and m.size <= 16*1024*1024 for m in members), "Unexpected source archive members")
        for member in members:
            data = stream.extractfile(member).read()
            row = expected[member.name]
            require(len(data) == row["bytes"] and digest(data) == row["sha256"], "Source member differs:"+member.name)
            raw[member.name] = data
    return {"manifest": record(path), "archive": actual, "verified_regular_members": count}, raw


def control_loop_sha(data):
    loops = [node for node in ast.walk(ast.parse(data.decode())) if isinstance(node, ast.For)
             and isinstance(node.target, ast.Name) and node.target.id == "step"]
    require(len(loops) == 1, "Expected one existing physical control loop")
    return digest(ast.dump(loops[0], include_attributes=False).encode())


def prepare_sources():
    old70, old_raw = verify_archive(STYLE / "navigation_guidance40_source_v1.json", GUIDE40_SOURCE70_SHA, 70)
    new135, new_raw = verify_archive(STYLE / "mixed_curriculum_training_source_v1.json", SOURCE135_SHA, 135)
    entry, _ = verify_archive(ENTRY / "source_freeze_v1.json", ENTRY_FREEZE_SHA, 5, entry=True)
    snapshot_path = STYLE / "ground_support_capture_source_freeze/source_manifest.json"
    require(record(snapshot_path)["sha256"] == "c5f12a1bc9a169b42414ae2c98ec14d1bf53cd5780ff244102ab95043bb4957c",
            "Old exact source snapshot changed")
    snapshot = read(snapshot_path)
    snapshot_records = {}
    for name, row in snapshot["files"].items():
        path = Path(row["snapshot_path"]).resolve()
        require(path.is_relative_to(snapshot_path.parent) and row["source_path"] == str(ROOT/name), "Old snapshot path differs")
        actual = record(path)
        require(actual["sha256"] == row["sha256"] and actual["bytes"] == row["bytes"], "Old selected source bytes changed")
        snapshot_records[name] = actual
    require(len(snapshot_records) == 7, "Old capture snapshot must contain seven exact selected files")
    loops = {"old_negative_selected_snapshot": control_loop_sha(Path(snapshot_records["scripts/train_paper_amp.py"]["path"]).read_bytes()),
             "later_guide40_source70": control_loop_sha(old_raw["scripts/train_paper_amp.py"]),
             "new_source135": control_loop_sha(new_raw["scripts/train_paper_amp.py"])}
    require(set(loops.values()) == {CONTROL_LOOP_SHA}, "Physical control-loop AST changed")
    return {"old_negative_selected_source_snapshot": {"manifest": record(snapshot_path), "files": snapshot_records},
            "later_guide40_archive70": old70, "new_prelaunch_training_archive135": new135,
            "thin_entry_archive": entry, "physical_control_loop_AST": loops,
            "six_deployment_runtime_members_recoverable_from_archives": {
                name: {"later_guide40_70_sha256": digest(old_raw[name]), "new135_sha256": digest(new_raw[name]),
                       "old_negative_selected_snapshot_sha256": snapshot_records.get(name, {}).get("sha256")}
                for name in RUNTIME_SOURCES},
            "archive_python_executed": False,
            "runtime_result_per_file_attestation": False,
            "source_binding_limit": "The70 archive is later guide40 source, not negative20 capture attestation. "
                "Old negative runtime has seven exact selected snapshots, no complete per-file attestation. "
                "Current135 plus thin entry are explicit source bindings; no historical current-tree substitution."}


def prefix_arrays(path, controls, *, plans=False):
    with np.load(path, allow_pickle=False) as archive:
        if plans:
            mask = archive["control_step"] < controls
            return {name: archive[name][mask] for name in archive.files}
        return {name: archive[name] if name in STATIC_GROUND_FIELDS else archive[name][:controls]
                for name in archive.files}


def array_comparison(left, right):
    """Report measured differences; never convert absent or unequal bytes to PASS."""
    result = {"left_shape": list(left.shape), "right_shape": list(right.shape),
              "left_dtype": str(left.dtype), "right_dtype": str(right.dtype),
              "left_raw_SHA256": digest(left.tobytes()), "right_raw_SHA256": digest(right.tobytes())}
    compatible = left.shape == right.shape and left.dtype == right.dtype
    equal = compatible and left.tobytes() == right.tobytes()
    result.update({"same_shape_and_dtype": compatible, "bitwise_equal": equal,
                   "first_differing_record": None, "maximum_absolute_difference": None})
    if compatible and not equal:
        if left.dtype.kind in "biuf":
            differences = left != right
            if differences.any():
                result["first_differing_record"] = int(np.flatnonzero(differences.reshape(len(left), -1).any(-1))[0])
            # Signed zero can differ in bytes without a numerical discrepancy.
            result["maximum_absolute_difference"] = float(np.max(np.abs(left.astype(np.float64)-right.astype(np.float64))))
        else:
            result["first_differing_record"] = int(np.flatnonzero((left != right).reshape(len(left), -1).any(-1))[0])
    return result


def compare_fields(left, right):
    require(set(left) == set(right), "Raw captured field schema differs")
    return {name: array_comparison(left[name], right[name]) for name in sorted(left)}


def assert_argument_contract(baseline, candidate):
    old, new = dict(baseline), dict(candidate)
    require("near_horizon_waypoint_mass" not in old or old["near_horizon_waypoint_mass"] == 0,
            "Historical baseline did not use eta0")
    old.setdefault("near_horizon_waypoint_mass", 0.0)
    require(set(old) == set(new) and len(new) == 57, "Current parsed57 fields differ")
    require(old["evaluate_steps"] == 6000 and old["episode_steps"] == 6001
            and new["evaluate_steps"] == 900 and new["episode_steps"] == 901, "Only declared200s vs30s/cap budgets are admitted")
    for name in ("config", "manifest", "diffusion_checkpoint", "resume", "scene_config"):
        require((ROOT/old[name]).resolve() == (ROOT/new[name]).resolve(), "Actual resource path differs:"+name)
        new[name] = old[name]
    for name in ("output", "evaluate_steps", "episode_steps"):
        new[name] = old[name]
    require(old == new, "Deployment parameters differ beyond paths/output/duration/cap")


def artifact_identity(directory, result):
    records = {name: record(directory/name) for name in ("result.json", "physical_states.npz", "reference_plans.npz", "ground_support.npz")}
    published = result["ground_support_measurement"]["capture"]
    require(published["sha256"] == records["ground_support.npz"]["sha256"]
            and published["bytes"] == records["ground_support.npz"]["bytes"], "Runtime-published ground capture identity differs")
    return records


def resource_identity(result):
    arguments = result["arguments"]
    names = {"controller_checkpoint": "resume", "diffusion_checkpoint": "diffusion_checkpoint",
             "profile": "config", "scene": "scene_config", "manifest": "manifest"}
    resources = {name: record(ROOT/arguments[key]) for name, key in names.items()}
    diffusion = result.get("evaluation_planner_change", {}).get("evaluation_sha256", result["provenance"]["diffusion_sha256"])
    require(resources["diffusion_checkpoint"]["sha256"] == diffusion
            and resources["manifest"]["sha256"] == result["manifest_gate"]["manifest_sha256"], "Actual diffusion/manifest differs from runtime declaration")
    require(resources["scene"]["sha256"] == "7971206da6fd50d9981dafd068b6005f326c6e7b6a8768e5a717aeb517573e9f", "Scene fixture bytes differ")
    return resources


def prefix_statistics(physical, ground, plans):
    feet_load = np.linalg.norm(ground["foot_floor_normal_force_w"].sum(-2), axis=-1)
    contact = feet_load > 1.
    require(np.array_equal(contact, ground["foot_contact_substeps"]), "Actual prefix strict1N contact reconstruction differs")
    side = contact.reshape(900, 4, 4, 2, 2).any(-1)
    return {"controls": 900, "duration_s": 30, "actors": 4, "control_actor_frames": 3600,
            "physics_substeps": 3600, "substep_actor_frames": 14400, "foot_substep_observations": 57600,
            "submitted_physics_step_before_first": int(ground["physics_step_before"][0]),
            "submitted_physics_step_after_last": int(ground["physics_step_end"][-1]),
            "native_engine_time_observed": False, "terminated_actor_frames": int(physical["terminated"].sum()),
            "truncated_actor_frames": int(physical["truncated"].sum()),
            "reset_actor_frames_after_initial": int(physical["reset_mask"][1:].sum()),
            "reached_vertices_per_actor": physical["route_vertex_reached"].sum(0).tolist(),
            "static_contact_actor_frames": int(physical["static_contact"].sum()),
            "peer_contact_unordered_pair_frames": int(np.triu(physical["peer_contact"] | physical["peer_contact"].swapaxes(-1, -2), 1).sum()),
            "double_support_fraction": float(side.all(-1).mean()),
            "no_support_fraction": float((~side.any(-1)).mean()), "floor_threshold_n": 1.,
            "COM_residual_max_m": float(ground["com_composition_max_error_m"].max()),
            "plan_rows_in_prefix": len(plans["control_step"]), "selected_sample_values": np.unique(plans["selected_sample"]).tolist(),
            "pre_step_initial_GT_pose_captured": False,
            "first_post_control_pelvis_xy": physical["root_pos"][0, :, :2].tolist(),
            "first_raw_plan_anchor_xy": plans["raw_xy"][:4, 0].tolist()}


def validate_prefix(physical, ground, plans, helper):
    helper.verify_shapes(helper.Audit(), ground, physical, 900, 4, 1)
    require(np.array_equal(ground["control_step"], np.arange(900))
            and np.array_equal(ground["control_step_after"], np.arange(1, 901)), "Actual900 control clocks differ")
    before = int(ground["physics_step_before"][0])
    require(np.array_equal(ground["physics_step"].reshape(-1), np.arange(before+1, before+3601)), "Actual3600 submitted substep clocks differ")
    require(ground["valid"].all() and (ground["substep_valid_count"] == 4).all()
            and (ground["actor_substep_valid_count"] == 4).all(), "Partial/padded controls are not admitted")
    for name in ("episode_ids", "reset_mask", "terminated", "truncated"):
        require(np.array_equal(ground[name], physical[name]), "Ground/physical same-frame event differs:"+name)
    require(np.array_equal(ground["body_link_pos_w"][:, -1], physical["body_pos"]), "Final-substep body positions differ from physical state")
    require((plans["control_step"] >= 0).all() and (plans["control_step"] < 900).all(), "Plan prefix clock bounds differ")
    require(len(set(zip(plans["control_step"].tolist(), plans["actor_ids"].tolist()))) == len(plans["control_step"]), "Duplicate actualplan actor/tick")
    require(set(plans["actor_ids"].tolist()) == {0, 1, 2, 3} and (plans["selected_sample"] == 0).all(), "Actual S1 joint scene selection differs")
    for array in plans.values():
        require(array.dtype.kind in "biuf" and (array.dtype.kind != "f" or np.isfinite(array).all()), "Nonfinite/invalid plan capture")


def baseline_oracle(sources, helper):
    require(record(BASELINE/"result.json")["sha256"] == BASELINE_RESULT_SHA, "Authoritative negative200 result changed")
    report_path = STYLE / "ground_support_200s_actual_verification.json"
    require(record(report_path)["sha256"] == BASELINE_AUDIT_SHA, "Original native full200 CPU audit changed")
    native = read(report_path)
    require(native["status"] == "passed" and native["actual_native_capture_validated"] is True, "Original actual native capture was not admitted")
    result = read(BASELINE/"result.json")
    identities = artifact_identity(BASELINE, result)
    for old, name in (("result", "result.json"), ("ground_support", "ground_support.npz"), ("physical_states", "physical_states.npz")):
        require(native["artifacts"][old] == identities[name], "Old actual capture differs from previous native audit")
    resources = resource_identity(result)
    require(resources == native["sources"]["resources_full_sha256_recomputed"], "Old exactweights/config/fixture/manifest changed")
    arrays = {"physical": prefix_arrays(BASELINE/"physical_states.npz", 900),
              "ground": prefix_arrays(BASELINE/"ground_support.npz", 900),
              "plans": prefix_arrays(BASELINE/"reference_plans.npz", 900, plans=True)}
    validate_prefix(arrays["physical"], arrays["ground"], arrays["plans"], helper)
    return result, arrays, {"identities": identities, "resources": resources,
        "previous_full200_native_audit": record(report_path), "old_prefix_statistics": prefix_statistics(**arrays),
        "sources": sources, "actual_baseline_only": True, "candidate_comparison_executed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--preparation-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only environment required")
    require(not args.output.exists(), "Evidence output must be a fresh directory")
    require(args.preparation_only or args.candidate_dir is not None, "Actual candidate directory is required for a comparison")
    require(not (args.preparation_only and args.candidate_dir), "Preparation cannot pretend a candidate was compared")
    helper = load_helper()
    sources = prepare_sources()
    baseline, old_arrays, oracle = baseline_oracle(sources, helper)
    output = {"schema": "explicit_pedestrian_entry.actual900_prefix_comparison.v1", "CPU_only": True,
              "GPU_used": False, "SDK_started": False, "new34_outputs_read": False, "reader": record(__file__),
              "baseline_oracle": oracle, "candidate_capture_available": False,
              "candidate_comparison_executed": False, "paired_bitwise_equal": None,
              "status": "prepared_waiting_for_actual30s_capture"}
    if not args.preparation_only:
        candidate = args.candidate_dir.resolve()
        result = read(candidate/"result.json")
        require(result["status"] == "passed" and result["actual_control_steps"] == 900, "New physical run did not complete900 controls")
        assert_argument_contract(baseline["arguments"], result["arguments"])
        entry_report = read(candidate/"deployment_entry.json")
        require(entry_report["parsed_driver_arguments"] == result["arguments"], "Physical result argv differs from actual thin-entry forwarded argv")
        for name in ("runtime_numerics", "physics", "evaluation_dataset", "evaluation_scene", "training_config"):
            left, right = baseline[name], result[name]
            if name == "training_config":
                left = dict(left)
                left.setdefault("expert_sampling", None)
            require(left == right, "Evaluation contract differs:"+name)
        resources = resource_identity(result)
        require(resources == oracle["resources"], "Actualweight/diffusion/profile/scene/manifest identities differ")
        actual = {"physical": prefix_arrays(candidate/"physical_states.npz", 900),
                  "ground": prefix_arrays(candidate/"ground_support.npz", 900),
                  "plans": prefix_arrays(candidate/"reference_plans.npz", 900, plans=True)}
        validate_prefix(actual["physical"], actual["ground"], actual["plans"], helper)
        original_verify_sources = helper.verify_sources
        try:
            # Only source admission differs. The pinned native measurement/force/
            # COM/episode formula auditor is reused unchanged on the actual900 run.
            def candidate_source_admission(audit, measured, **kwargs):
                return {"source_archive_binding": sources, "resources_full_sha256_recomputed": resource_identity(measured),
                        "source_admission_callback_only": True, "actual_runtime_per_file_attestation": False}

            helper.verify_sources = candidate_source_admission
            audit = helper.Audit()
            native = helper.audit_capture(candidate, audit)
            native["status"] = "passed"
            native["checks"] = audit.checks
            native["actual_native_auditor_sha256"] = HELPER_SHA
        finally:
            helper.verify_sources = original_verify_sources
        comparisons = {name: compare_fields(old_arrays[name], actual[name]) for name in actual}
        equal = all(row["bitwise_equal"] for block in comparisons.values() for row in block.values())
        output.update({"candidate_capture_available": True, "candidate_comparison_executed": True,
                       "paired_bitwise_equal": equal, "status": "measured_exact_prefix" if equal else "measured_differences",
                       "candidate_artifacts": artifact_identity(candidate, result), "candidate_entry_report": record(candidate/"deployment_entry.json"),
                       "candidate_prefix_statistics": prefix_statistics(**actual), "all_raw_field_comparisons": comparisons,
                       "candidate_actual_native_measurement_audit": native,
                       "first_post_control_physical_frame": {name: array_comparison(old_arrays["physical"][name][:1], actual["physical"][name][:1])
                            for name in ("root_pos", "body_pos", "root_rot", "dof_pos", "dof_vel")},
                       "pre_step_initial_GT_pose_bitwise_claim": False})
    output["limits"] = ["Old200 fullrun remains authoritative; a30s prefix never replaces its200s quality/stability evidence.",
        "Actual captures contain first post-control state, not a pre-step mocap/GT initial pose or initial history array.",
        "Exact measured dtype/bytes are reported separately from native recording consistency; unequal arrays are retained.",
        "Submitted-step clocks plus declared120Hz dt are observed; no native PhysX engine timestamp or bitwise guarantee is invented.",
        "Feet body-origin tangent velocity remains a proxy; no contact-point slip/friction groundtruth is fabricated.",
        "The later guide40 source70 archive is not substituted for the oldnegative20 complete captured source."]
    args.output.mkdir(parents=True)
    with (args.output/"comparison.json").open("x") as handle:
        json.dump(output, handle, indent=2, allow_nan=False)
        handle.write("\n")
    text = ("# 显式入口30秒 / 旧基线前900步\n\n" + output["status"] + "\n\n"
            "candidate_comparison_executed=" + str(output["candidate_comparison_executed"]) + "; paired_bitwise_equal="
            + str(output["paired_bitwise_equal"]) + "。\n\n" + "\n\n".join(output["limits"]) + "\n")
    with (args.output/"report.md").open("x") as handle:
        handle.write(text)
    print({"status": output["status"], "candidate_comparison_executed": output["candidate_comparison_executed"],
           "paired_bitwise_equal": output["paired_bitwise_equal"]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
