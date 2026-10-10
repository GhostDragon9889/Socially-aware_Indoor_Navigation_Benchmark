"""Retain every termination/reset timestamp and bind matched navigation inputs.

This is a metadata/event sidecar to the frozen native measurement audit. It
does not compute style, stagnation, new contact metrics, or a training gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/style_improvement_20261002"
INPUT = ROOT / "output/diffusion_amp/shared_amass_root_curriculum_ground_support_200s_v1"
AUDIT = DOCS / "amass_root_step1000_navigation_ground_actual_verification_v1.json"
SOURCE = DOCS / "amass_root_step1000_navigation_source_v1.json"


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.output.exists():
        parser.error("Empty CUDA visibility and a fresh output path are required")
    audit = json.loads(AUDIT.read_text())
    source = json.loads(SOURCE.read_text())
    prep_path = ROOT/source["preparation"]["path"]
    prep = json.loads(prep_path.read_text())
    result_path = INPUT/"result.json"
    result = json.loads(result_path.read_text())
    baseline_path = Path(prep["baseline_result"])
    baseline = json.loads(baseline_path.read_text())
    paths = {"native_audit": AUDIT, "source_manifest": SOURCE, "prelaunch_preparation": prep_path,
             "result": result_path, "baseline_result": baseline_path,
             "physical_states": INPUT/"physical_states.npz", "ground_support": INPUT/"ground_support.npz"}
    before = {name: identity(path) for name, path in paths.items()}
    assert audit["status"] == "passed" and audit["actual_native_capture_validated"] is True
    assert all(item["passed"] for item in audit["checks"])
    assert before["source_manifest"]["sha256"] == "c9be09f46086c5909b864806bf9242d9f422a64122f1da88888443b760d2b752"
    assert before["prelaunch_preparation"]["sha256"] == source["preparation"]["sha256"]
    assert before["baseline_result"]["sha256"] == prep["baseline_sha256"]
    for name in ("result", "physical_states", "ground_support"):
        assert before[name] == audit["artifacts"][name]
    expected = dict(baseline["arguments"])
    assert set(prep["changed_arguments"]) == {"config", "output", "resume", "near_horizon_waypoint_mass"}
    for name, item in prep["changed_arguments"].items():
        assert expected.get(name) == item["baseline"]
        expected[name] = item["candidate"]
    assert result["arguments"] == expected
    assert "near_horizon_waypoint_mass" not in baseline["arguments"]
    assert expected["near_horizon_waypoint_mass"] == 0.0
    # This one known older result predates the explicit uniform-sampler field.
    # Preserve that metadata difference; do not call the JSON objects identical.
    baseline_config = baseline["training_config"]
    candidate_config = result["training_config"]
    assert set(candidate_config)-set(baseline_config) == {"expert_sampling"}
    assert not set(baseline_config)-set(candidate_config)
    assert candidate_config["expert_sampling"] is None
    assert all(candidate_config[name] == value for name, value in baseline_config.items())
    for name, key in (("amp", "controller_checkpoint"), ("diffusion", "diffusion_checkpoint"), ("manifest", "manifest")):
        actual = audit["sources"]["resources_full_sha256_recomputed"][key]
        declared = source["resources"][name]
        assert actual["sha256"] == declared["sha256_recomputed"] and actual["bytes"] == declared["bytes"]
        assert Path(actual["path"]) == (ROOT/declared["path"]).resolve()
    with np.load(INPUT/"physical_states.npz", allow_pickle=False) as archive:
        physical = {name: archive[name] for name in ("terminated", "truncated", "reset_mask", "episode_ids",
            "root_pos", "root_rot", "static_contact", "peer_contact", "static_contact_max_force_n", "peer_contact_max_force_n")}
    with np.load(INPUT/"ground_support.npz", allow_pickle=False) as archive:
        ground = {name: archive[name] for name in ("terminated", "truncated", "reset_mask", "episode_ids",
            "control_step", "control_step_after", "physics_step", "physics_step_end",
            "side_contact_substeps", "mass_weighted_world_com")}
    t, n = result["arguments"]["evaluate_steps"], result["arguments"]["num_envs"]
    assert (t, n) == (6000, 4)
    for name in ("terminated", "truncated", "reset_mask", "episode_ids"):
        assert np.array_equal(physical[name], ground[name])
    assert np.array_equal(ground["control_step"], np.arange(t))
    assert np.array_equal(ground["control_step_after"], np.arange(1, t+1))
    assert np.array_equal(ground["physics_step"].reshape(-1), np.arange(1, 4*t+1))
    endings, resets = [], []
    for frame, actor in np.argwhere(physical["terminated"] | physical["truncated"]):
        frame, actor = int(frame), int(actor)
        q = physical["root_rot"][frame, actor]
        assert frame+1 < t and physical["reset_mask"][frame+1, actor]
        assert physical["episode_ids"][frame+1, actor] == physical["episode_ids"][frame, actor]+1
        endings.append({"actor": actor, "capture_frame_index": frame,
            "control_step_before": int(ground["control_step"][frame]),
            "control_step_after": int(ground["control_step_after"][frame]),
            "completed_control_time_s": (frame+1)/30,
            "completed_submitted_physics_step": int(ground["physics_step_end"][frame]),
            "episode_id": int(physical["episode_ids"][frame, actor]),
            "terminated": bool(physical["terminated"][frame, actor]),
            "truncated": bool(physical["truncated"][frame, actor]),
            "root_xyz_m": physical["root_pos"][frame, actor].tolist(),
            "root_up_z_from_XYZW": float(1-2*(q[0]*q[0]+q[1]*q[1])),
            "true_mass_weighted_COM_xyz_m": ground["mass_weighted_world_com"][frame, -1, actor].tolist(),
            "static_contact_same_control": bool(physical["static_contact"][frame, actor]),
            "static_contact_same_control_max_force_n": float(physical["static_contact_max_force_n"][frame, actor]),
            "peer_actors_same_control": np.flatnonzero(physical["peer_contact"][frame, actor]).tolist(),
            "peer_same_control_max_force_n": float(physical["peer_contact_max_force_n"][frame, actor].max()),
            "left_right_support_each_completed_substep": ground["side_contact_substeps"][frame, :, actor].tolist(),
            "next_first_post_reset_capture_frame": frame+1,
            "next_episode_id": int(physical["episode_ids"][frame+1, actor])})
    for frame, actor in np.argwhere(physical["reset_mask"]):
        frame, actor = int(frame), int(actor)
        resets.append({"actor": actor, "first_post_reset_capture_frame_index": frame,
            "initialization": frame == 0, "new_episode_id": int(physical["episode_ids"][frame, actor]),
            "reset_before_current_control_time_s": frame/30,
            "first_completed_control_time_s": (frame+1)/30,
            "first_completed_submitted_physics_step": int(ground["physics_step_end"][frame]),
            "preceding_termination_capture_frame": None if frame == 0 else frame-1})
    assert len(endings) == result["fall_resets"] == 13 and not physical["truncated"].any()
    assert len(resets) == result["resets"]+n == 17
    assert sum(not row["initialization"] for row in resets) == 13
    after = {name: identity(path) for name, path in paths.items()}
    assert before == after
    report = {"schema": "amass_root_navigation_event_and_source_binding.v1", "status": "passed",
        "CPU_only": True, "GPU_used": False, "SDK_started": False, "heldout34_test_outputs_read": False,
        "execution_completed": True, "actual_native_capture_validated": True,
        "closed_loop_quality_accepted": False,
        "quality_boundary": "Complete execution and recording admission do not accept the observed13 falls and5 route vertices as navigation-quality success.",
        "inputs": after, "script": identity(__file__), "time_basis": {
            "control_hz": 30, "physics_hz": 120, "native_engine_time_observed": False,
            "meaning": "Declared control dt and actual consecutive submitted-step counters; termination state captured before reset, following capture is first poststep of new episode."},
        "all_frames_retained": t, "actor_frames": t*n,
        "terminations_all": endings, "resets_all_including_initialization": resets,
        "all_other_deployment_arguments_equal": True,
        "explicit_added_legacy_default": {"near_horizon_waypoint_mass": {"baseline_absent": True, "candidate": 0.0, "meaning": "disabled engineering redistribution; not a changed guidance strength"}},
        "changed_arguments_exact_prelaunch": prep["changed_arguments"], "TrainingConfig_original_fields_exact_equal": True,
        "TrainingConfig_explicit_metadata_addition": {"expert_sampling": {
            "baseline_field_absent": True, "candidate": None,
            "meaning": "known legacy uniform sampler; candidate opt-in expert sampling disabled",
            "JSON_objects_claimed_identical": False}},
        "fixed_diffusion_and_data_match_prelaunch_full_SHA": True,
        "AMP_identity": audit["sources"]["resources_full_sha256_recomputed"]["controller_checkpoint"],
        "historical_native_capture_probe_passed": False, "historical_native_physical_probe_passed": False,
        "limits": ["Same CLI/seed does not make subsequent physical states equal across different controllers.",
                   "The trained controller differs in AMASS-root command/reset/end curriculum; this is not a single-loss causal ablation.",
                   "Same-frame contacts are event context, not demonstrated causes of falls.",
                   "Foot body-origin contact-conditioned speed is a proxy, not true patch slip; reference true COM/contact/friction unavailable.",
                   "No style/MMD, stagnation, or new contact metric is computed here."]}
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "passed", "terminations": len(endings), "resets_including_initialization": len(resets),
                      "closed_loop_quality_accepted": False}))


if __name__ == "__main__":
    main()
