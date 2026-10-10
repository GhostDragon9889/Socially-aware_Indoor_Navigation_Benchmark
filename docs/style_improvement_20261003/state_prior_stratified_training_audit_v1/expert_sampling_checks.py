"""One saved-expert boundary; restored draws stay in the original auditor."""
from __future__ import annotations

import hashlib


def validate_saved_sampling_contract(state, expected_config, identity, check, prefix=""):
    """Pure metadata check, using identities rebuilt from the actual bank."""
    saved = state["runner_state"].get("expert_sampling")
    update = state["update_count"]
    check(prefix + "explicit_stratified_config_and_complete_saved_outer_schema",
          expected_config is not None and state["config"]["expert_sampling"] == expected_config
          and isinstance(saved, dict) and set(saved) == {"schema", "configuration", "sampler"}
          and saved["schema"] == "navisaaclab.expert_reference_sampler.v1"
          and saved["configuration"] == expected_config)
    sampler = saved["sampler"]
    check(prefix + "saved_sampler_exact_inner_schema_and_actual_bank_identities",
          isinstance(sampler, dict) and set(sampler) == set(identity)
          and all(type(sampler[key]) is type(identity[key]) and sampler[key] == identity[key]
                  for key in ("schema", "config", "config_sha256", "registry_sha256",
                              "manifest_sha256", "probabilities_sha256")))
    check(prefix + "actual_expert_counters_one4096_batch_per_update",
          type(update) is int and update > 0
          and type(sampler["sampled_windows"]) is int and sampler["sampled_windows"] == update * 4096
          and type(sampler["sampled_batches"]) is int and sampler["sampled_batches"] == update)
    return saved


def validate_expert_sampling(state, corpus, expected_config, check, same, prefix, result):
    """Rebuild q on the real train bank and load the real saved private RNG."""
    import torch
    from CrowdSim.humanoid.training.expert_sampling import ExpertReferenceSampler

    global_before = torch.get_rng_state().clone()
    shared_before = corpus.generator.get_state().clone()
    sampler = ExpertReferenceSampler(corpus, expected_config)
    identity = sampler.state_dict()["sampler"]
    initial_description = sampler.describe()
    saved = validate_saved_sampling_contract(state, sampler.config, identity, check, prefix)
    check(prefix + "initial_saved_expert_provenance_actual_registry_and_zero_counters",
          same(state["provenance"]["expert_sampling"], initial_description))
    sampler.load_state_dict(saved)
    check(prefix + "actual_sampler_loader_saved_config_hashes_private_RNG_counters_exact",
          same(sampler.state_dict(), saved)
          and torch.equal(sampler.generator.get_state(), saved["sampler"]["generator_state"]))
    check(prefix + "sampler_creation_and_restore_leave_global_and_shared_CPU_RNG_unchanged",
          torch.equal(global_before, torch.get_rng_state())
          and torch.equal(shared_before, corpus.generator.get_state()))
    if result is not None and result["update_count"] == state["update_count"]:
        check(prefix + "closed_result_actual_final_expert_description_exact_saved_state",
              same(result["expert_sampling"], sampler.describe())
              and result["training_config"]["expert_sampling"] == sampler.config)
    proof = {"saved_schema": saved["schema"], "configuration": sampler.config,
             "registry_sha256": identity["registry_sha256"], "probabilities_sha256": identity["probabilities_sha256"],
             "manifest_sha256": identity["manifest_sha256"], "config_sha256": identity["config_sha256"],
             "private_CPU_RNG_saved_and_actual_loader_exact": True,
             "private_CPU_RNG_sha256": hashlib.sha256(saved["sampler"]["generator_state"].numpy().tobytes()).hexdigest(),
             "sampled_windows": saved["sampler"]["sampled_windows"], "sampled_batches": saved["sampler"]["sampled_batches"],
             "actual_bank_registry_rebuilt": True, "importance_correction_applied": False,
             "kinematic_proxy_labels_are_author_labels": False,
             "historical_draws_reexecuted_to_infer_private_RNG": False,
             "limit": "Checks saved private RNG exact load and next logical resume repeat; does not replay all historical draws or physics."}
    return sampler, proof
