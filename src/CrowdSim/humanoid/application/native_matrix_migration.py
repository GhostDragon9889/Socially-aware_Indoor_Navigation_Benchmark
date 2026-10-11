"""Current native156 registration, preserving every case and refusal."""
from copy import deepcopy

from CrowdSim.humanoid.analysis import native_seed_protocol as protocol
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record
from social_navigation.artifact_identity import artifact_path, same_artifact

ROOT = protocol.ROOT
MATRIX = ROOT / "configs/compatibility/native_seeded_sim61/registered_S3_native_matrix.json"
OUTPUT = ROOT / "output/diffusion_amp/native_S3_seeded_registered_sim61_v2"


def build(historical, identity):
    current = protocol.build_registration(OUTPUT)
    if len(historical["cases"]) != len(current["cases"]):
        raise ValueError("Native migration changed the registered case count")
    for old, new in zip(historical["cases"], current["cases"], strict=True):
        restored = deepcopy(new)
        restored["same_AMP_checkpoint_for_both_arms"] = old["same_AMP_checkpoint_for_both_arms"]
        if artifact_path(new["same_AMP_checkpoint_for_both_arms"]) != artifact_path(old["same_AMP_checkpoint_for_both_arms"]):
            raise ValueError("Native migration changed the preselected checkpoint")
        if old["arms"]:
            migration = restored.pop("scene_configuration_migration")
            if not same_artifact(migration["historical"], old["scene_configuration"]):
                raise ValueError("Native migration replaced an original scene")
            restored["scene_configuration"] = old["scene_configuration"]
            for old_arm, new_arm in zip(old["arms"], restored["arms"], strict=True):
                new_arm["output"] = old_arm["output"]
        if restored != old:
            raise ValueError("Native migration changed registered cases, seeds or refusals")
    invariant_keys = ("seeds", "scenes", "case_order", "registered_cohorts", "resource_admitted_cohorts",
        "resource_rejected_cohorts", "planned_full200s_arm_runs_after_all_gates", "resource_rejected_arm_slots",
        "planned_paired_actor_controls", "actual_training_exit_admitted", "actual_checkpoint_identity",
        "same_initial_body_map_RNG_required", "native_seeded_raw_and_style_admission_still_required",
        "no_best_or_intermediate_checkpoint_selection", "GPU_or_SDK_started", "paper_effect_achieved",
        "default_model_promoted")
    if any(current[key] != historical[key] for key in invariant_keys):
        raise ValueError("Native migration changed the original evaluation contract")
    current["registration_migration"] = {
        "historical_matrix": identity, "adapter": record(__file__),
        "all_cases_seeds_refusals_checkpoint_preserved": True,
        "new_output_root": str(OUTPUT), "SDK_execution_verified": False}
    return current
