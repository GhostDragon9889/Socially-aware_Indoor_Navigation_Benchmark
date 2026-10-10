"""Cold admission of the completed S3-R optimizer recovery for evaluation.

This family has a real OS-owner completion and a recovery-specific CPU audit.
It does not claim the interrupted original S3 process finished or that its
physical state was restored. Checkpoint contents remain checked by the existing
strict frozen reader before inference; no Torch or simulator is imported here.
"""
from __future__ import annotations

from protomotions.repository import repository_root

import json
from pathlib import Path

from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record
from CrowdSim.humanoid.runtime import social_popart_training

ROOT = repository_root(__file__)
TRAINING = ROOT / "docs/style_improvement_20261007/shared_social_S3_optimizer_recovery_v2"
OWNER = TRAINING / "owner_capacity_v3"
RUN = ROOT / "output/diffusion_amp/shared_social_budget12500_recovery_training_v2"
CHECKPOINT_SHA = "97ba4c6475817e1bbde4aa56377a45191058a0725cad48fcbf3a9b769accb675"
SCHEMA = "shared_social.actual_optimizer_recovery_training_execution.v1"
PINS = {
    "completion": (OWNER / "completion.json", "d2495ceb1d2fe8223a39eb41b006bfd9fab350a31fbb64f39f65f433963e0a87"),
    "result": (RUN / "result.json", "87bf93cbe6c67a2d9e852cd7b8d9b095de525fe97019e6747fab4ec1427467bb"),
    "audit": (OWNER / "training_CPU_audit.json", "3d81788c369e4edc16ca9ae6f25b932ea6dbd73d53a5ab233e5e6e03be5e3d57"),
    "external_verification": (OWNER / "external_verification.json", "959a6eb89acace413184a2db264f188ad7bdd569f6f5cde6615f4019c1396b99"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_integer(value, expected):
    return type(value) is int and value == expected


def readonly_identity(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and not path.stat().st_mode & 0o222,
            "Require an independent readonly recovery artifact: " + str(path))
    return record(path)


def validate_documents(completion, result, audit, external, identities, numeric):
    """Validate the mutually bound actual completion documents, without IO."""
    require(completion.get("schema") == "shared_social.durable_recovery_owner_state.v1"
            and completion.get("status") == "completed_training_and_CPU_audit",
            "Require the completed durable recovery owner")
    steps = completion.get("steps", [])
    require([row.get("name") for row in steps] == ["training", "terminal_CPU_audit"]
            and all(exact_integer(row.get("actual_exit_code"), 0)
                    and row.get("exit_observation") == "subprocess.Popen.wait"
                    and row.get("status") == "passed" for row in steps),
            "Require both actual recovery child exits observed by OS wait")
    checkpoint = completion.get("checkpoint")
    require(isinstance(checkpoint, dict)
            and checkpoint.get("path") == str(RUN / "step_012500.pt")
            and checkpoint.get("sha256") == CHECKPOINT_SHA
            and exact_integer(checkpoint.get("bytes"), 651040366),
            "Select only the preselected recovered terminal checkpoint")
    require(completion.get("training_result") == identities["result"]
            and completion.get("audit") == identities["audit"],
            "OS completion and terminal identities disagree")
    for key, count in (("retained_updates", 12000), ("actual_segment_updates", 500),
                       ("effective_total_updates", 12500), ("actual_segment_transitions", 4096000),
                       ("effective_training_transitions", 102400000)):
        require(exact_integer(completion.get(key), count), "Incomplete recovery exposure: " + key)
    require(completion.get("physical_world_restarted") is True
            and completion.get("actual_child_exit_codes_observed") is True
            and completion.get("original_continuous_S3_run_completed") is False,
            "Keep the optimizer-recovery versus continuous-physics distinction")
    runtime = result.get("profile", {}).get("runtime", {})
    require(result.get("status") == "passed_local_training"
            and result.get("training_loop_completed") is True
            and result.get("cleanup_succeeded") is True and result.get("cleanup_errors") == []
            and result.get("source_profile_before_after_equal") is True
            and exact_integer(result.get("completed_updates"), 12500)
            and exact_integer(result.get("training_update_returned_count"), 500)
            and exact_integer(result.get("actual_training_transitions"), 102400000)
            and all(exact_integer(runtime.get(key), value) for key, value in
                    (("actors", 256), ("rollout_controls", 32), ("updates", 12500), ("seed", 7))),
            "Recovery terminal result, full budget, profile or cleanup disagree")
    recovery = result.get("recovery", {})
    require(recovery.get("schema") == "shared_social.optimizer_recovery_segment.v1"
            and exact_integer(recovery.get("initial_update_count"), 12000)
            and exact_integer(recovery.get("initial_physical_transition_count"), 98304000)
            and recovery.get("physical_world_restarted") is True
            and recovery.get("physical_world_curriculum_history_restored") is False
            and recovery.get("continuous_original_S3_execution_claimed") is False
            and recovery.get("model_both_Adam_four_RMS_replay_restored") is True
            and recovery.get("global_training_rng_restored_after_setup") is True,
            "Require the disclosed optimizer recovery and restarted physical state")
    canonical = result.get("canonical_entry", {})
    require(canonical.get("strategy") == "popart" and canonical.get("migration") == numeric
            and recovery.get("binding", {}).get("canonical_binding") == canonical,
            "Recovered numeric family differs from the strict Pop-Art reader")
    require(audit.get("schema") == "shared_social.completed_optimizer_recovery_CPU_audit.v1"
            and audit.get("status") == "passed" and audit.get("CUDA_initialized") is False
            and audit.get("checkpoint") == checkpoint and audit.get("recovery") == recovery
            and audit.get("explicit_value_normalization") == numeric
            and exact_integer(audit.get("updates"), 12500)
            and exact_integer(audit.get("physical_transitions"), 102400000)
            and exact_integer(audit.get("actual_recovery_segment_updates"), 500)
            and exact_integer(audit.get("actual_recovery_segment_transitions"), 4096000)
            and exact_integer(audit.get("retained_updates"), 12000)
            and exact_integer(audit.get("finite_tensors_checked"), 202)
            and audit.get("checkpoint_physical_resume_supported") is False,
            "Require the complete recovery-specific CPU audit")
    require(external.get("schema") == "shared_social.completed_optimizer_recovery_external_verification.v1"
            and external.get("status") == "passed"
            and external.get("completion") == identities["completion"]
            and external.get("result") == identities["result"]
            and external.get("audit") == identities["audit"]
            and external.get("checkpoint") == checkpoint
            and exact_integer(external.get("actual_training_child_exit_code"), 0)
            and exact_integer(external.get("actual_CPU_audit_child_exit_code"), 0)
            and external.get("training_exit_inferred_from_metrics_or_PID") is False,
            "Independent cold verification does not bind this actual completed recovery")


def completed_documents():
    identities, documents = {}, {}
    for role, (path, sha) in PINS.items():
        identities[role] = readonly_identity(path)
        require(identities[role]["sha256"] == sha, "Pinned recovery terminal artifact changed: " + role)
        documents[role] = json.loads(path.read_text())
    numeric = social_popart_training.describe()
    validate_documents(documents["completion"], documents["result"], documents["audit"],
                       documents["external_verification"], identities, numeric)
    recovery_binding = documents["result"]["recovery"]["binding"]
    for role in ("registration", "entry", "helper", "configuration"):
        expected = recovery_binding[role]
        require(record(expected["path"]) == expected, "Actual recovery source/config changed: " + role)
    for role in ("entry", "parent"):
        expected = documents["result"]["canonical_entry"][role]
        require(record(expected["path"]) == expected, "Canonical recovery ancestor changed: " + role)
    profile = json.loads(Path(recovery_binding["configuration"]["path"]).read_text())
    require(profile == documents["result"]["profile"], "Saved recovery profile differs from registration")
    checkpoint = documents["completion"]["checkpoint"]
    path = Path(checkpoint["path"])
    require(path.is_file() and not path.is_symlink() and path.stat().st_mode & 0o222 == 0
            and path.stat().st_size == checkpoint["bytes"], "Recovered terminal checkpoint storage changed")
    return documents, identities, numeric


def receipt_document():
    """Construct a new receipt from actual recovery evidence, never fresh-run fields."""
    documents, identities, _ = completed_documents()
    return {"schema": SCHEMA, "run_directory": str(RUN), "terminal_exit_code": 0,
            "completion_identity": identities["completion"], "result_identity": identities["result"],
            "audit_identity": identities["audit"],
            "checkpoint_identity": documents["completion"]["checkpoint"],
            "external_verification_identity": identities["external_verification"],
            "physical_world_restarted": True, "original_continuous_S3_run_completed": False}


def admit_training(receipt_path):
    path = Path(receipt_path)
    before = readonly_identity(path)
    receipt = json.loads(path.read_text())
    require(receipt.get("schema") == SCHEMA and exact_integer(receipt.get("terminal_exit_code"), 0),
            "Require an explicit recovery execution receipt; fresh-run receipts are separate")
    documents, identities, numeric = completed_documents()
    expected = {"schema": SCHEMA, "run_directory": str(RUN), "terminal_exit_code": 0,
                "completion_identity": identities["completion"], "result_identity": identities["result"],
                "audit_identity": identities["audit"], "checkpoint_identity": documents["completion"]["checkpoint"],
                "external_verification_identity": identities["external_verification"],
                "physical_world_restarted": True, "original_continuous_S3_run_completed": False}
    require(receipt == expected and record(path) == before, "Recovery receipt does not bind the actual completion")
    checkpoint = documents["completion"]["checkpoint"]
    return Path(checkpoint["path"]), checkpoint["sha256"], {
        "receipt": before, "result": identities["result"], "audit": identities["audit"],
        "checkpoint": checkpoint, "terminal_exit_code": 0,
        "value_normalization": "popart", "numeric_migration": numeric,
        "optimizer_recovery": {"completion": identities["completion"],
            "external_verification": identities["external_verification"], "admission_source": record(__file__),
            "retained_updates": 12000, "segment_updates": 500, "effective_updates": 12500,
            "physical_world_restarted": True, "original_continuous_S3_run_completed": False,
            "checkpoint_hash_required_by_strict_reader_before_inference": True,
            "quality_evaluated_by_metadata_admission": False}}
