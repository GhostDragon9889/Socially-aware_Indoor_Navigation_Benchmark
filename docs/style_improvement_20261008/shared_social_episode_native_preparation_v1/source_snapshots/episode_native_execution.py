"""Release registered native DP/SFM tasks for the actual episode2000 candidate.

The frozen seeded SDK, initial capture, raw arrays and style kernels stay intact.
Only the preselected training receipt, output namespace and owner admission are
bound to the separately completed episode run; no old S3 certificate is made.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import ModuleType

from . import native_execution_bridge as original
from scripts import compare_shared_social_episode_evidence as candidate

ROOT, record, require = original.ROOT, original.record, original.require
PARENT_SHA = "a9eccb65600be8b2f3f057ae8b13df297c36da405922398c84f55886e4bcd2a9"
HERE = ROOT / "docs/style_improvement_20261008/shared_social_episode_native_preparation_v1"
REGISTRATION = HERE / "registration.json"
OUTPUT_ROOT = ROOT / "output/diffusion_amp/native_episode2000_seeded_registered_v1"
OLD_OUTPUT_ROOT = ROOT / "output/diffusion_amp/native_S3_seeded_registered_v1"
SCHEMA = "shared_social.local_native_episode2000_multiseed_registration.v1"


def registration_document():
    require(record(original.__file__)["sha256"] == PARENT_SHA, "Frozen original native exit bridge changed")
    registered, candidate_identity = candidate.admit_registration(candidate.REGISTRATION)
    matrix, identity = original.readonly_json(original.MATRIX)
    require(identity["sha256"] == original.MATRIX_SHA
            and matrix["protocol_source"] == record(original.protocol.__file__)
            and matrix["sources"] == original.raw.entry.source_bindings(), "Original native registration/source changed")
    for item in matrix["inputs"]:
        require(record(item["path"]) == item, "Original native resource registration input changed")
    require(len(matrix["cases"]) == 156 and matrix["resource_admitted_cohorts"] == 144
            and matrix["resource_rejected_cohorts"] == 12 and matrix["scenes"] == list(original.protocol.SCENES)
            and matrix["seeds"] == list(original.protocol.SEEDS), "Preserve all native cohorts and resource refusals")
    checkpoint = registered["preselected_candidate_checkpoint"]
    result = deepcopy(matrix)
    result["schema"] = SCHEMA
    result.pop("preselected_S3_terminal_checkpoint")
    result["preselected_episode_terminal_checkpoint"] = checkpoint
    for case in result["cases"]:
        require(case["same_AMP_checkpoint_for_both_arms"] == str(original.CHECKPOINT),
                "Original cohort selects another AMP checkpoint")
        case["same_AMP_checkpoint_for_both_arms"] = checkpoint
        for arm in case["arms"]:
            output = Path(arm["output"])
            require(output.is_relative_to(OLD_OUTPUT_ROOT), "Original native output is outside its registered namespace")
            arm["output"] = str(OUTPUT_ROOT / output.relative_to(OLD_OUTPUT_ROOT))
    # Demonstrate an exact inverse of every inherited field. Added bindings are
    # separate; no resource, order, seed, rejection, spacing or count is edited.
    inverse = deepcopy(result)
    inverse["schema"] = matrix["schema"]
    inverse["preselected_S3_terminal_checkpoint"] = inverse.pop("preselected_episode_terminal_checkpoint")
    inverse["preselected_S3_terminal_checkpoint"] = str(original.CHECKPOINT)
    for case in inverse["cases"]:
        case["same_AMP_checkpoint_for_both_arms"] = str(original.CHECKPOINT)
        for arm in case["arms"]:
            arm["output"] = str(OLD_OUTPUT_ROOT / Path(arm["output"]).relative_to(OUTPUT_ROOT))
    require(inverse == matrix, "Episode native registration changed another inherited field")
    result.update(
        original_native_registration=identity, native_execution_adapter=record(__file__),
        episode_native_registration_entry=record(ROOT / "scripts/prepare_shared_social_episode_native.py"),
        original_exit_bridge=record(original.__file__), episode_comparison_registration=candidate_identity,
        episode_training_owner_plan=registered["candidate_owner_plan"],
        episode_completion_path=registered["candidate_completion_path"],
        episode_training_receipt_path=registered["candidate_training_receipt_path"],
        registration_derivation={"complete_inherited_inverse": True,
            "changed_inherited_fields": ["schema", "terminal_checkpoint_selector", "cohort_AMP_checkpoint", "arm_output_namespace"],
            "resource_cases_seeds_rejections_or_metric_math_changed": False},
    )
    return result


def admit_registration(path=REGISTRATION):
    value, identity = original.readonly_json(path)
    require(value == registration_document(), "Episode native registration or declared sources changed")
    return value, identity


def admit_episode_owner_completion(path, *, registration_path=REGISTRATION):
    registered, identity = admit_registration(registration_path)
    require(Path(path).resolve() == Path(registered["episode_completion_path"]),
            "Require this episode candidate's actual completed owner, never the old S3 certificate")
    comparison_registration, _ = candidate.admit_registration(candidate.REGISTRATION)
    bundle, _, completion = candidate.admit_candidate(comparison_registration)
    checkpoint, sha, base = candidate.child_admission.admit_training(candidate.RECEIPT)
    require(str(checkpoint) == registered["preselected_episode_terminal_checkpoint"]
            and bundle["tracking"]["training_execution_admission"] == base
            and record(original.DIFFUSION)["sha256"] == original.DIFFUSION_SHA,
            "Native actor/diffusion selection differs from the fully completed episode candidate")
    return {"episode_owner_completion": completion, "episode_native_registration": identity,
            "training": base, "checkpoint_sha256": sha,
            "actual_training_and_independent_evaluation_admitted": True,
            "original_S3_certificate_created": False}


def _namespace(registration_path=REGISTRATION):
    registered, identity = admit_registration(registration_path)
    source = Path(original.__file__)
    require(record(source)["sha256"] == PARENT_SHA, "Frozen native exit bridge source changed")
    namespace = ModuleType("CrowdSim.humanoid.application._owned_episode_native_exit_bridge")
    # https://docs.python.org/3.12/library/types.html#types.ModuleType
    namespace.__package__ = "CrowdSim.humanoid.application"
    namespace.__file__ = str(Path(__file__).resolve())
    # Exact original bytes; the module globals below are private to this call.
    exec(compile(source.read_bytes(), str(source) + ":episode_owner_admission", "exec"), namespace.__dict__)
    namespace.MATRIX = Path(registration_path).resolve()
    namespace.MATRIX_SHA = identity["sha256"]
    namespace.CHECKPOINT = Path(registered["preselected_episode_terminal_checkpoint"])
    namespace.TRAINING_RECEIPT = Path(registered["episode_training_receipt_path"])
    namespace.admit_S3_owner_completion = lambda path: admit_episode_owner_completion(
        path, registration_path=registration_path)
    return namespace


def prepare_case(case_id, *, registration_path=REGISTRATION):
    plan = _namespace(registration_path).prepare_case(case_id)
    plan["schema"] = "shared_social.registered_episode_native_case_tool_plan.v1"
    plan["episode_native_binding"] = {
        "registration": record(registration_path), "adapter": record(__file__),
        "original_exit_bridge": record(original.__file__), "original_bridge_source_executed_unchanged": True,
        "training_release_requires_actual_episode_owner_completion": True,
        "SDK_entry_raw_capture_style_and_metric_math_changed": False,
        "original_S3_certificate_created": False,
    }
    return plan


def publish_sdk_exit(case_id, *, planner, controls, observation_path, owner_completion,
                     registration_path=REGISTRATION):
    return _namespace(registration_path).publish_sdk_exit(case_id, planner=planner, controls=controls,
        observation_path=observation_path, owner_completion=owner_completion)


def publish_raw_CPU_exit(case_id, *, planner, controls, observation_path, owner_completion,
                         registration_path=REGISTRATION):
    return _namespace(registration_path).publish_raw_CPU_exit(case_id, planner=planner, controls=controls,
        observation_path=observation_path, owner_completion=owner_completion)


def write_registration(path=REGISTRATION):
    path = Path(path)
    require(not path.exists(), "Preserve existing native registrations")
    document = registration_document()
    original._write_new(path, document)
    return admit_registration(path)[1]
