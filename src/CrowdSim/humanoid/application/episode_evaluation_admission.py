"""Terminal admission for the registered episode-reference AMP experiment.

Metadata and source checks only. Numeric checkpoint and all physical/quality
measurements remain owned by the existing independent Pop-Art pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record
from scripts import train_shared_social_episode_reference as training

ROOT = training.ROOT
PREPARATION = ROOT / "docs/style_improvement_20261006/shared_social_episode_evaluation_preparation_v1"
REGISTRATION = PREPARATION / "registration.json"
ENTRY = ROOT / "scripts/run_shared_social_episode_evaluation.py"
PARENT = ROOT / "scripts/run_shared_social_post_training.py"
PARENT_SHA = "2890d628361fbe1c548d822a82bbcbedde215f6395db44d7327a1903a7199cf8"
PLAN = training.HERE / "execution_plan.json"
PLAN_SHA = "811f993aab84661437025681a1e43f896add05e0651ef71bf697c63d2cb60926"
PROBE = training.HERE / "actual_entry_probe.json"
PROBE_SHA = "c52bf0140784d54507eb080490ae56e02db1b45ff42fce5015df19992662b03a"
OUTPUT = ROOT / "output/evaluation/shared_social_episode_reference_pilot2000_v1"
CURRICULUM_SOURCE = "engineering_fixed_actor_amass_episode_procedural_mixture"


def _fixed(path, sha):
    item = record(path)
    if item["sha256"] != sha or Path(path).stat().st_mode & 0o222:
        raise ValueError("Sealed episode experiment input changed: " + str(path))
    return item


def registration_document():
    """Build the complete expected registration from pinned, cold sources."""
    plan_identity, probe_identity = _fixed(PLAN, PLAN_SHA), _fixed(PROBE, PROBE_SHA)
    parent = record(PARENT)
    if parent["sha256"] != PARENT_SHA:
        raise ValueError("Independent post-training pipeline changed")
    plan = json.loads(PLAN.read_text())
    stages = [stage for stage in plan["stages"] if stage["stage"] == "pilot2000"]
    if len(stages) != 1:
        raise ValueError("Require the single registered episode pilot2000 stage")
    stage = stages[0]
    module, binding = training.prepare_entry()
    entry_sources = module._sources()
    # The original loader refuses any previously imported private module. This
    # helper is needed only for its cold source inventory; release our own alias
    # so a second admission in this same process can perform the full check.
    name = "_episode_evaluation_setup_sources"
    probe = module._load(module.PINS["probe_setup"][0], name, entry_sources["probe_setup"])
    try:
        setup_sources = probe._sources()
    finally:
        if sys.modules.get(name) is probe:
            del sys.modules[name]
    profile_path = Path(stage["profile"]["path"])
    profile, profile_identity = module._profile(profile_path)
    if (profile_identity != stage["profile"] or profile["runtime"] != stage["runtime"]
            or plan["entry"] != binding["entry"] or plan["World"] != binding["World"]
            or plan["curriculum"] != binding["curriculum"] or plan["reference"] != binding["reference_primitive"]
            or profile["runtime"]["actors"] != 256 or profile["runtime"]["updates"] != 2000
            or profile["runtime"]["rollout_controls"] != 32):
        raise ValueError("Episode candidate plan and actual entry/profile disagree")
    expected_reference = json.loads(PROBE.read_text())["cohorts"]["training.json"]["actual_entry_report"][
        "episode_reference_experiment"]["actual_reference_configuration"]
    return {"schema": "shared_social.popart_paired_experiment_registration.v1",
        "configuration": profile_identity, "run_directory": stage["output_directory"],
        "preselected_checkpoint": Path(plan["selected_terminal_checkpoint"]).name,
        "episode_reference_experiment": {"schema": "shared_social.episode_terminal_evaluation_registration.v1",
            "entry": record(ENTRY), "admission_module": record(__file__), "parent": parent,
            "execution_plan": plan_identity, "real_corpus_CPU_probe": probe_identity,
            "training_binding": binding, "entry_sources": entry_sources, "setup_sources": setup_sources,
            "expected_reference_configuration": expected_reference,
            "evaluation_output": str(OUTPUT), "future81_combined": False,
            "reference_change_scope": plan["reference_change_scope"],
            "evaluation_numerics_changed": False, "author_protocol_equivalence_claimed": False}}


def admit_registration(path=REGISTRATION):
    path = Path(path).resolve()
    if path != REGISTRATION or path.is_symlink() or path.stat().st_mode & 0o222:
        raise ValueError("Use the exact readonly episode evaluation registration")
    identity = record(path)
    registration = json.loads(path.read_text())
    if registration != registration_document() or record(path) != identity:
        raise ValueError("Episode evaluation registration or registered sources changed")
    return registration, identity


def validate_episode_result(result, registration):
    """Match actual terminal metadata to the fixed candidate, before inference."""
    expected = registration["episode_reference_experiment"]
    experiment = result.get("episode_reference_experiment")
    if not isinstance(experiment, dict) or experiment.get("binding") != expected["training_binding"]:
        raise ValueError("Require the actual registered episode-reference training binding")
    reference = expected["expected_reference_configuration"]
    if (experiment.get("actual_reference_configuration") != reference
            or experiment.get("caller_probe_namespace_World_selection_changed") is not True
            or experiment.get("original_source_files_modified") is not False
            or experiment.get("author_recipe_equivalence_claimed") is not False):
        raise ValueError("Actual episode reference configuration/ownership differs")
    profile = json.loads(Path(registration["configuration"]["path"]).read_text())
    if (result.get("profile") != profile or type(result.get("completed_updates")) is not int
            or result["completed_updates"] != 2000
            or type(result.get("actual_training_transitions")) is not int
            or result["actual_training_transitions"] != 16384000
            or result.get("arguments", {}).get("output") != registration["run_directory"]
            or result.get("arguments", {}).get("config") != registration["configuration"]["path"]):
        raise ValueError("Require the full fresh registered episode pilot2000 budget/output/profile")
    for name in ("entry_sources", "setup_sources"):
        if result.get(name) != expected[name] or result.get(name + "_after") != expected[name]:
            raise ValueError("Actual episode training source inventory differs: " + name)
    origin = result.get("actual_import_origins", {}).get("episode_reference_curriculum", {})
    if origin != {"module": "_shared_social_original_training.episode_reference_curriculum",
                  **expected["training_binding"]["curriculum"]}:
        raise ValueError("Actual curriculum import is not the registered episode source")
    world = result.get("final_world_description", {})
    curriculum = world.get("curriculum", {})
    statistics = curriculum.get("statistics", {})
    if (world.get("num_actors") != 256 or world.get("control_step") != 64000
            or curriculum.get("source") != CURRICULUM_SOURCE or curriculum.get("config") != reference
            or curriculum.get("author_recipe_equivalence_claimed") is not False
            or statistics.get("control_steps") != 64000
            or statistics.get("amass_actor_transitions") != 8192000
            or statistics.get("procedural_actor_transitions") != 8192000
            or type(statistics.get("procedural_plans")) is not int
            or type(statistics.get("procedural_plan_views")) is not int
            or not 0 < statistics["procedural_plans"] <= statistics["procedural_plan_views"]):
        raise ValueError("Actual terminal curriculum is not the complete episode-reference run")


def admit_episode_training(receipt_path, registration, base_admit):
    """Keep the original actual-exit/audit gate, then admit the candidate family."""
    checkpoint, sha, admission = base_admit(Path(receipt_path))
    if (str(checkpoint.parent) != registration["run_directory"]
            or checkpoint.name != registration["preselected_checkpoint"]
            or admission["value_normalization"] != "popart"):
        raise ValueError("Require the preregistered episode terminal checkpoint")
    result_path = Path(admission["result"]["path"])
    if record(result_path) != admission["result"]:
        raise ValueError("Actual episode training result changed")
    result = json.loads(result_path.read_text())
    validate_episode_result(result, registration)
    resolved = checkpoint.parent / "resolved_configuration.json"
    resolved_identity = record(resolved)
    configuration = json.loads(resolved.read_text())
    provenance = configuration["provenance"]
    if (configuration["entry_profile"] != result["profile"]
            or configuration["training_configuration"] != result["actual_configuration"]
            or provenance["entry_sources"] != result["entry_sources"]
            or provenance["setup_sources"] != result["setup_sources"]
            or provenance["profile"] != registration["configuration"]
            or provenance["initialization"]["weights_or_moments_transferred"] is not False
            or record(resolved) != resolved_identity):
        raise ValueError("Actual saved checkpoint provenance differs from the candidate")
    return checkpoint, sha, {**admission, "episode_reference_experiment": {
        "registration": record(REGISTRATION), "registered_sources": registration["episode_reference_experiment"],
        "resolved_configuration": resolved_identity, "actual_training_sources_matched": True,
        "quality_improvement_proven": False}}
