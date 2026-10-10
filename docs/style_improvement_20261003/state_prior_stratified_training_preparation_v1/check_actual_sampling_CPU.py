"""One real train-bank mapping/RNG/resume check; no NN, Kit or GPU."""
from collections import Counter
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BUNDLE = ROOT / "output/isolated_training/mixed_state_action_terrain_state_prior_stratified_v1"
PROFILE = BUNDLE / "code/configs/diffusion_amp/isolated_terrain_state_prior_stratified_absolute.yaml"


def same_reference(left, right, torch):
    for a, b in zip(left[:2], right[:2], strict=True):
        assert vars(a).keys() == vars(b).keys()
        for field in vars(a):
            assert torch.equal(getattr(a, field), getattr(b, field)), field
    for a, b in zip(left[2:], right[2:], strict=True):
        assert torch.equal(a, b)


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    started = time.monotonic()
    specification = importlib.util.spec_from_file_location("actual_stratified_bundle_launcher", BUNDLE / "launch_isolated.py")
    launcher = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(launcher)
    manifest = launcher.verify()
    assert len(manifest["files"]) == 935
    ledger = launcher.ImportLedger(HERE / "actual_CPU_imports.json")
    launcher.activate(ledger)
    ledger.phase = "running"
    ledger.write("guard_installed")
    import torch
    from CrowdSim.humanoid.amass_corpus import AMASSCorpus
    from CrowdSim.humanoid.application.configuration import load_amp_profile, parse_amp_args
    from CrowdSim.humanoid.application.settings import TrainingConfig, normalize_expert_sampling_config
    from CrowdSim.humanoid.training.expert_sampling import ExpertReferenceSampler, STRATA
    torch.set_num_threads(2)
    profile = load_amp_profile(PROFILE)
    config = TrainingConfig(**profile["training"])
    config.validate()
    plan = json.loads((HERE / "root_GPU_launch_commands.json").read_text())
    command = plan["commands"]["fullfresh2000"]
    args = parse_amp_args(command[command.index("--") + 1:])
    assert args.resume is None and args.updates == 2000 and args.num_envs == 256 and args.rollout_steps == 32
    assert args.trajectory_source == "mixed_amass_procedural" and args.seed == 7
    assert config.amp_reward_weight == config.amp_reward_scale == 2 and config.task_reward_weight == .5
    assert config.warm_start_action_discriminator_from_state_prior is True
    bank = Path(profile["data_manifest"])

    def corpus(seed=7):
        return AMASSCorpus(bank, "train", device="cpu", seed=seed, dof_velocity_frame="child_joint")

    legacy, replacement = corpus(), corpus()
    global_rng = torch.get_rng_state().clone()
    shared_rng = replacement.generator.get_state().clone()
    bridge = ExpertReferenceSampler(replacement, config.expert_sampling)
    assert torch.equal(shared_rng, replacement.generator.get_state())
    assert torch.equal(global_rng, torch.get_rng_state())
    registry = bridge.sampler
    counts = replacement.data["motion_num_frames"][registry.motion_ids] - 10
    assert len(replacement.entry["clips"]) == len(torch.unique(registry.motion_ids))
    legacy_probability = 1. / (len(replacement.entry["clips"]) * counts.double())
    assert torch.isclose(legacy_probability.sum(), torch.tensor(1., dtype=torch.float64), atol=1e-12, rtol=0)
    legacy_ess = float(1 / legacy_probability.square().sum())
    nominal_old_mass = {name: float(legacy_probability[registry.stratum_ids == index].sum()) for index, name in enumerate(STRATA)}
    exact_new_mass = {name: float(registry.expected_probabilities[registry.stratum_ids == index].sum()) for index, name in enumerate(STRATA)}
    assert all(abs(exact_new_mass[name] - config.expert_sampling["target_weights"][name]) < 1e-12 for name in STRATA)
    reset_left, reset_right = legacy.sample_reference(32), replacement.sample_reference(32)
    same_reference(reset_left, reset_right, torch)
    legacy.sample_reference(4096)
    actual = bridge.sample_reference(4096)
    assert torch.equal(legacy.generator.get_state(), replacement.generator.get_state())
    starts = replacement.data["length_starts"][actual[3]]
    frame_counts = replacement.data["motion_num_frames"][actual[3]]
    assert torch.all(actual[4] >= 9) and torch.all(actual[4] + 1 < frame_counts)
    assert torch.equal(actual[2], replacement.data["dps"][starts + actual[4] + 1])
    state, shared_state = bridge.state_dict(), replacement.generator.get_state().clone()
    expected = bridge.sample_reference(128)
    expected_reset = replacement.sample_reference(32)
    recovered_corpus = corpus(99)
    recovered = ExpertReferenceSampler(recovered_corpus, config.expert_sampling)
    recovered_corpus.generator.set_state(shared_state)
    recovered.load_state_dict(state)
    same_reference(expected, recovered.sample_reference(128), torch)
    same_reference(expected_reset, recovered_corpus.sample_reference(32), torch)
    assert torch.equal(bridge.generator.get_state(), recovered.generator.get_state())
    assert torch.equal(replacement.generator.get_state(), recovered_corpus.generator.get_state())
    bad = deepcopy(state)
    bad["sampler"]["probabilities_sha256"] = "0" * 64
    rejected_changed_registry = False
    try:
        recovered.load_state_dict(bad)
    except ValueError:
        rejected_changed_registry = True
    assert rejected_changed_registry
    assert normalize_expert_sampling_config(None) != normalize_expert_sampling_config(config.expert_sampling)
    assert torch.equal(global_rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
    assert not any(name.split(".")[0] in {"isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules)
    described = bridge.describe()
    original = json.loads((ROOT / "docs/style_improvement_20261002/expert_sampling_integration_v7_verification.json").read_text())
    assert described["sampler"]["registry_sha256"] == original["description"]["sampler"]["registry_sha256"]
    assert described["sampler"]["probabilities_sha256"] == original["description"]["sampler"]["probabilities_sha256"]
    bank_counts = {split: len(entry["clips"]) for split, entry in replacement.manifest["splits"].items()}
    assert bank_counts == {"train": 2848, "validation": 22, "test": 34}
    description = described["sampler"]
    report = {"schema": "stratified_prior.actual_train_bank_CPU_readiness.v1", "status": "passed",
              "CPU_only": True, "GPU_used": False, "SDK_started": False, "sealed34_outputs_read": False,
              "source_manifest_sha256": launcher.digest(BUNDLE / "source_manifest.json"),
              "actual_loaded_profile": str(PROFILE), "profile_sha256": launcher.digest(PROFILE),
              "actual_fresh2000_parser_passed": True, "full_resource_hashes_verified_by_unchanged_launcher": True,
              "bank_clip_counts": bank_counts, "sampler": described,
              "nominal_legacy_uniform_clip_uniform_center_prior": {"window_ESS": legacy_ess, "stratum_mass": nominal_old_mass},
              "new_exact_stratum_mass": exact_new_mass,
              "new_to_legacy_nominal_window_ESS_ratio": description["center_distribution_ess"] / legacy_ess,
              "minimum_positive_center_probability": float(registry.expected_probabilities.min()),
              "actual4096_expert_rows": {"all_history_and_next_bounds_valid": True,
                                         "next_DOF_PD_proxy_matches_actual_bank": True,
                                         "motion_id_counts": dict(Counter(actual[3].tolist()))},
              "paired_reset_payload_and_following_shared_RNG_equal": True,
              "restored_private_sampler_and_shared_corpus_next_expert_then_reset_exact": True,
              "changed_registry_resume_rejected": rejected_changed_registry,
              "original_runner_None_to_enabled_resume_configuration_guard_rejects": True,
              "global_Torch_CPU_rng_unchanged": True,
              "original_v7_mapping_probability_hashes_reproduced": True,
              "elapsed_seconds": time.monotonic() - started,
              "limitations": ["ESS is the exact discrete center probability ESS, not a realized training effective batch size.",
                              "Targets and kinematic root-history labels are local proxies; no author style IDs or hyperparameters.",
                              "No importance correction: this deliberately changes the expert prior.",
                              "Shared RNG compensation is exact only for matched call schedules; changed falls/reset times can diverge.",
                              "No NN/public weights deserialization or physical training executed; pilot/full payload gate remains future."]}
    with (HERE / "actual_CPU_readiness.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    ledger.finish("passed")
    print(json.dumps({"status": "passed", "clips": bank_counts, "legal_centers": description["legal_centers"],
                      "registry_sha256": description["registry_sha256"], "probabilities_sha256": description["probabilities_sha256"],
                      "legacy_nominal_ESS": legacy_ess, "stratified_ESS": description["center_distribution_ess"],
                      "CPU_seconds": report["elapsed_seconds"], "mapping_RNG_resume": "passed"}))


if __name__ == "__main__":
    main()
