"""Minimal CPU peer: unchanged raw math, actual v2 wiring and entry ABI.

The metadata fixture is synthetic. No future checkpoint/capture is opened.
"""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CORE = ROOT / "docs/style_improvement_20261003/state_prior_pacer_reward_native_reader_preparation_v1"
REV = ROOT / "docs/style_improvement_20261003/state_prior_pacer_reward_native_reader_metadata_revision_v2"
OLD = ROOT / "docs/style_improvement_20261003/state_prior_native_reader_preparation_v1"


def record(path):
    path = Path(path).absolute()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load(path, name):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    for directory, name, sha in ((CORE, "source_freeze_v1.json", "bbc81cc438188ab5b9f23c31a229d51c9cf043300874d304f8cc01c6d186d6f3"),
                                 (REV, "source_freeze_v2.json", "0df9e9d1d0bfed9b7e17c814519c8b9bc443b29b98cfc810fce347f4df1d4de1")):
        assert record(directory / name)["sha256"] == sha
        freeze = json.loads((directory / name).read_text())
        for row in freeze["files"] + [freeze["archive"]]:
            assert record(row["path"]) == row and not Path(row["path"]).stat().st_mode & 0o222
    original = (OLD / "audit_state_prior_native.py").read_text()
    normalized = (CORE / "audit_state_prior_native.py").read_text()
    for before, after in {
        "pacer_reward.native200s_CPU_reader_registration.v1": "public_state_prior.native200s_CPU_reader_registration.v1",
        "pacer_reward.native200s_independent_CPU_audit.v1": "public_state_prior.native200s_independent_CPU_audit.v1",
        "Prepared before actual new4/15 captures; checkpoint SHA and actual terminal/native evidence remain pending. Actual closed owner records are required for execution.": "Reader protocol recorded while native4 may already run; actual preflight and owner prelaunch source snapshots are bound separately. This is not a claim that this CPU reader preceded GPU launch.",
        'proof = {"base_source_archive_index_changes": 1': 'return native, {"base_source_archive_index_changes": 1',
        "    return native, json.loads(json.dumps(proof, allow_nan=False))\n": "",
    }.items():
        normalized = normalized.replace(before, after)
    assert normalized == original and ast.dump(ast.parse(normalized)) == ast.dump(ast.parse(original))
    for name in ("frozen_v1.py", "frozen_v2.py", "frozen_routes.py"):
        assert (CORE / name).read_bytes() == (OLD / name).read_bytes()
    revision = load(REV / "audit_native_checkpoint_metadata_v2.py", "independent_native_metadata_revision")
    core = revision.load_core()
    helper = sys.modules["native_evidence"]
    assert Path(helper.__file__).resolve() == REV / "native_evidence_checkpoint_v2.py"
    assert core.admit_closed_native is helper.admit_closed_native
    assert Path(core.__file__).resolve() == CORE / "audit_state_prior_native.py"
    for actors in (4, 15):
        registered = json.loads((CORE / f"native{actors}_registration.json").read_text())
        _, proof = core.readers(actors)
        assert proof == registered["adapter_proof"]
        assert registered["native_execution_audited"] is False
        assert registered["evidence"]["future_checkpoint"]["sha256"] is None
        revised = json.loads((REV / f"native{actors}_revision_registration.json").read_text())
        assert revised["original_math_executor"] == record(CORE / "audit_state_prior_native.py")
        assert revised["metadata_helper"] == record(REV / "native_evidence_checkpoint_v2.py")
        assert revised["future_checkpoint_sha256"] is None and revised["native_execution_audited"] is False
    checkpoint = {"path": "/synthetic/selected_step_2000.pt", "bytes": 123, "sha256": "a" * 64}
    args = {"evaluate_steps": 6000, "num_envs": 4, "output": str(helper.run_path(4)),
            "resume": checkpoint["path"], "config": "/synthetic/profile.yaml", "diffusion_samples": 1,
            "seed": 7, "contact_threshold_n": 1, "ground_support_threshold_n": 1,
            "power_source": "projected_joint_effort_substep_mean", "near_horizon_waypoint_mass": 0,
            "guidance_weight": 20, "scene_collision_radius": .8}
    args.update({name: True for name in ("physics_only", "deterministic_evaluation", "pairwise_contacts",
                                       "measure_ground_support", "synchronized_replanning")})
    resource = {"path": "/synthetic/resource.json", "bytes": 5, "sha256": "b" * 64}
    entry = {"status": "SDK_free_preflight_passed", "SDK_started": False, "Torch_imported_by_entry": False,
             "physical_execution_verified": False, "parsed_driver_arguments": args,
             "control_steps": 6000, "physical_transitions_planned": 24000, "actual_duration_seconds": 200.0,
             "evaluation_actor_request": {"base_checkpoint": {key: checkpoint[key] for key in ("path", "sha256")},
                                          "alpha": 0, "style_checkpoint": None, "experimental_arm_style": False},
             "native_scene_request": {name: resource for name in ("configuration_file", "preflight", "native_map", "probe_script_snapshot")}}
    training = {"provenance": {"synthetic": True}, "training_config": {"warm_start_action_discriminator_from_state_prior": True}}
    actor = {"facade": "EvaluationActor", "checkpoint": checkpoint["path"], "checkpoint_sha256": checkpoint["sha256"],
             "update_count": 2000, "physical_transition_count": 16384000, "full_training_provenance": training["provenance"],
             "config": training["training_config"], "AMP_reward_available": False,
             "critic_value_available": False, "saved_training_rng_restored": False}
    result = {"status": "passed", "actual_control_steps": 6000, "arguments": args, "evaluation_actor": actor}
    old_helper = load(CORE / "native_evidence.py", "independent_old_native_metadata")
    # Only synthetic resource identity calls are replaced. No future file is read.
    def fixture_record(path):
        return resource if str(path) == resource["path"] else {"sha256": "6d76dbfad6bc3827904af7efaf2fea7849ba5f5ff441971745ea17501b7f1bae"}
    old_helper.record = helper.record = fixture_record
    rejected_old = False
    try:
        old_helper.validate_result_metadata(result, entry, training, checkpoint, 4)
    except ValueError as error:
        rejected_old = "Exact trained checkpoint" in str(error)
    assert rejected_old
    helper.validate_result_metadata(result, entry, training, checkpoint, 4)
    bad = deepcopy(entry)
    bad["evaluation_actor_request"]["base_checkpoint"]["sha256"] = "c" * 64
    rejected_sha = False
    try:
        helper.validate_result_metadata(result, bad, training, checkpoint, 4)
    except ValueError as error:
        rejected_sha = "Exact trained checkpoint" in str(error)
    assert rejected_sha
    assert not any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules)
    report = {"schema": "pacer_reward.native_reader_independent_CPU_peer.v1", "status": "passed", "required_findings": [],
              "reader": record(__file__), "core_source_freeze": record(CORE / "source_freeze_v1.json"),
              "revision_source_freeze": record(REV / "source_freeze_v2.json"),
              "actual_entry": record(REV / "audit_native_checkpoint_metadata_v2.py"),
              "actual_metadata_helper": record(REV / "native_evidence_checkpoint_v2.py"),
              "raw_math_reverse_metadata_full_bytes_AST_exact": True,
              "frozen_v1_v2_routes_entire_bytes_exact": True, "actual_v2_helper_import_no_fallback": True,
              "actual_original_main_admission_function_is_v2_helper": True,
              "both_actor_adapter_proofs_JSON_equal_frozen_registrations": True,
              "producer_shaped_synthetic_ABI": {"old_extra_bytes_requirement_rejected": rejected_old,
                                                 "new_path_SHA_without_bytes_accepted": True,
                                                 "wrong_SHA_still_rejected": rejected_sha,
                                                 "synthetic_fixture_is_actual_run": False},
              "sealed_checkpoint_full_record_readonly_nlink_terminal_checks_unchanged": True,
              "metadata_actual_entry_and_helper_reported_separately_from_math_executor": True,
              "future_checkpoint_or_capture_read": False, "sealed34_outputs_read": False,
              "old_suite_rerun": False, "GPU_used": False, "SDK_started": False,
              "actual_native_physical_quality_claimed": False}
    with (HERE / "final_reader_peer.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({"status": "passed", "required_findings": [], "report": record(HERE / "final_reader_peer.json")}))


if __name__ == "__main__":
    main()
