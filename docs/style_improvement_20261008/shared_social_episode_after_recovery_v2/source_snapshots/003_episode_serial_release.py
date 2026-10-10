"""Explicitly release the episode candidate after completed S3-R evaluation.

The original S3 continuous-run certificate is unavailable. This separate family
requires the actual optimizer recovery and combined evaluation completions;
it never creates, substitutes or changes that missing original certificate.
Import and admission do not load Torch or an SDK.
"""
from __future__ import annotations

import json
from pathlib import Path

from CrowdSim.humanoid.application import recovery_evaluation_admission as recovery
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record
from scripts import resume_shared_social_recovery_evaluation as resumed
from scripts import train_shared_social_episode_reference as candidate

ROOT = candidate.ROOT
SCHEMA = "shared_social.episode_candidate_after_optimizer_recovery_release.v1"
EVALUATION = ROOT / "docs/style_improvement_20261008/shared_social_S3R_navigation_retry_v1/completion.json"
COMPARISON = ROOT / "docs/style_improvement_20261008/shared_social_S3R_comparison_v1/result.json"
OBSERVATION = COMPARISON.parent / "actual_execution.json"
PINS = {
    EVALUATION: "7de1439ae51b94a9e7957c1d5239988fd201881a527c3788887e1ebd9aca5094",
    COMPARISON: "4aa60dcf484f4f3d0e139b093dd101f1736d039258722189f0aa1c6bbdd1da31",
    OBSERVATION: "7a704a5a378b3fe06ad947a932cb4ae190ee1cb0f7611be5cc3a9de77ec4fc33",
}
PLAN = candidate.HERE / "execution_plan.json"
PLAN_SHA = "811f993aab84661437025681a1e43f896add05e0651ef71bf697c63d2cb60926"


def stages():
    candidate.pinned(PLAN, PLAN_SHA, readonly=True)
    plan = json.loads(PLAN.read_text())
    recovery.require([row["stage"] for row in plan["stages"]] == ["smoke8", "smoke256", "pilot2000"],
                     "Keep the registered complete episode candidate stage order")
    return plan["stages"]


def release_document():
    documents, identities, _ = recovery.completed_documents()
    evidence = {}
    for path, sha in PINS.items():
        item = recovery.readonly_identity(path)
        recovery.require(item["sha256"] == sha, "Completed S3-R evidence changed: " + str(path))
        evidence[path] = item
    resumed.admit_completion(EVALUATION)
    comparison = json.loads(COMPARISON.read_text())
    observed = json.loads(OBSERVATION.read_text())
    recovery.require(comparison.get("status") == "passed_comparison"
                     and comparison.get("recovery_evaluation_completion") == evidence[EVALUATION]
                     and observed.get("comparison") == evidence[COMPARISON]
                     and observed.get("resumed_evaluation_completion") == evidence[EVALUATION]
                     and all(recovery.exact_integer(observed.get(key), 0) for key in (
                         "actual_navigation_child_exit_code", "actual_original_CPU_audit_child_exit_code",
                         "actual_CPU_comparison_exit_code"))
                     and observed.get("original_failed_execution_overwritten") is False,
                     "Require actually completed, separately retained S3-R measurements and comparison")
    module, binding = candidate.prepare_entry()
    module._sources()
    for row in stages():
        _, profile = module._profile(Path(row["profile"]["path"]))
        recovery.require(profile == row["profile"], "Candidate registered profile changed")
    return {"schema": SCHEMA, "recovery_training_completion": identities["completion"],
            "recovered_checkpoint": documents["completion"]["checkpoint"],
            "completed_evaluation": evidence[EVALUATION], "completed_comparison": evidence[COMPARISON],
            "actual_comparison_observation": evidence[OBSERVATION],
            "candidate_execution_plan": record(PLAN), "candidate_training_binding": binding,
            "release_basis": "completed_optimizer_recovery_and_resumed_independent_evaluation",
            "original_S3_continuous_completion_claimed": False, "original_certificate_created": False,
            "candidate_fresh_initialization": True, "weights_or_moments_transferred": False,
            "managed_GPU_jobs_concurrent_max": 1, "external_GPU_processes_preserved": True,
            "candidate_quality_improvement_proven": False, "default_model_promoted": False}


def admit_release(path):
    path = Path(path).resolve()
    identity = recovery.readonly_identity(path)
    document = json.loads(path.read_text())
    recovery.require(document == release_document() and record(path) == identity,
                     "Use the explicit completed-S3-R serial release with unchanged candidate sources")
    return document, identity
