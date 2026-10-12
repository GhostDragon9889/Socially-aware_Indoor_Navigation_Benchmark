#!/usr/bin/env python3
"""Strict CPU terminal-D rescoring and input sensitivity, never a training ablation.

Scores the complete saved terminal policy and expert populations with the final
checkpoint D and final AMP RMS. These are not the rewards at collection time,
held-out discrimination accuracy, locomotion quality, or causal evidence.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCOPE = ("Terminal D offline rescoring/input sensitivity only; not original-time reward, "
         "not a training ablation, not held-out accuracy or causal motion-quality evidence.")
SUPPORT = {
    "scripts/evaluate_shared_social_tracking.py": "ff866433ce91feb2a006a137ed437ae7bab92b8b6b0c9daaf09f244521b3767c",
    "scripts/diagnose_shared_social_training.py": "dc85d0628b55fa53b3385c122c0bef237aed546eb7a4c5a3a256b7baf65581f4",
    "CrowdSim/humanoid/runtime/social_popart_inference.py": "39a5c09c39902c276f1af4a03d2f3e1de62aa17d0b823e5eb73dd1462db07385",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare_sources():
    # Hash before importing definitions. These imports are stdlib-only.
    for relative, expected in SUPPORT.items():
        source = (ROOT / relative).read_text()
        digest = hashlib.sha256(source.encode()).hexdigest()
        if digest != expected and relative == "scripts/evaluate_shared_social_tracking.py":
            from CrowdSim.humanoid.runtime.evaluation_source_migration import verify_entry
            verify_entry(ROOT / relative, expected)
            digest = expected
        if digest != expected and relative == "CrowdSim/humanoid/runtime/social_popart_inference.py":
            require(digest == "1b40cfe449b1adda7fc78769e445845747ecdbaede81b8ee09feb07adae7b250", "Registered strict Pop-Art reader changed")
            digest = expected
        require(digest == expected, "Reviewed diagnostic support changed: " + relative)
    from scripts.diagnose_shared_social_training import identity
    from scripts.evaluate_shared_social_tracking import admit_training
    from CrowdSim.humanoid.runtime.social_popart_inference import prepare_reader

    reader, binding = prepare_reader()
    return identity, admit_training, reader, binding


def tensor_digest(value):
    """CPU tensor content hash; never edits its storage or changes caller RNG."""
    require(value.device.type == "cpu", "Tensor digest requires CPU storage")
    array = value.detach().contiguous().numpy()
    digest = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def statistics(value):
    import torch
    value = value.detach().flatten().double()
    require(len(value) > 0 and value.device.type == "cpu" and bool(torch.isfinite(value).all()),
            "Statistics require nonempty finite CPU inputs")
    return {"count": len(value), "mean": float(value.mean()), "std": float(value.std(unbiased=False)),
            "min": float(value.min()), "max": float(value.max()),
            "q01_q50_q99": torch.quantile(value, value.new_tensor([.01, .5, .99])).tolist()}


def separation(policy, expert):
    """Balanced BCE/accuracy and tie-correct rank AUC over the complete populations."""
    import torch
    from torch.nn import functional as F
    combined = torch.cat((policy, expert)).double()
    values, order = combined.sort()
    _, inverse, counts = torch.unique_consecutive(values, return_inverse=True, return_counts=True)
    ranks = torch.arange(1, len(values) + 1, dtype=torch.float64)
    totals = torch.zeros(len(counts), dtype=torch.float64).scatter_add_(0, inverse, ranks)
    sorted_ranks = (totals / counts)[inverse]
    restored = torch.empty_like(sorted_ranks)
    restored[order] = sorted_ranks
    npolicy, nexpert = len(policy), len(expert)
    auc = (restored[npolicy:].sum() - nexpert * (nexpert + 1) / 2) / (npolicy * nexpert)
    return {"expert_minus_policy_mean_logit": float(expert.mean() - policy.mean()),
            "expert_minus_policy_mean_probability": float(expert.sigmoid().mean() - policy.sigmoid().mean()),
            "balanced_BCE": float(.5 * (F.softplus(policy).mean() + F.softplus(-expert).mean())),
            "balanced_accuracy_at_logit0": float(.5 * ((policy < 0).double().mean() + (expert >= 0).double().mean())),
            "expert_positive_fraction": float((expert >= 0).double().mean()),
            "policy_negative_fraction": float((policy < 0).double().mean()), "rank_AUC": float(auc)}


def score(discriminator, features, batch_size):
    import torch
    require(type(batch_size) is int and batch_size > 0, "Positive score batch size required")
    with torch.no_grad():
        return torch.cat([discriminator(batch).flatten() for batch in features.split(batch_size)])


def gradient_groups(discriminator, features, batch_size):
    """d(single-row logit)/d(normalized/clipped D input); no batch-mean scaling."""
    import torch
    parts = {name: [] for name in ("history_L2", "history_RMS", "action_L2", "action_RMS")}
    for batch in features.split(batch_size):
        owned = batch.detach().clone().requires_grad_(True)
        derivative = torch.autograd.grad(discriminator(owned).sum(), owned)[0].detach()
        for name, columns in (("history", derivative[:, :1950]), ("action", derivative[:, 1950:])):
            parts[name + "_L2"].append(columns.square().sum(-1).sqrt())
            parts[name + "_RMS"].append(columns.square().mean(-1).sqrt())
    return {name: torch.cat(rows) for name, rows in parts.items()}


def analyze(discriminator, policy, expert, *, seed, batch_size):
    """Matched-history action interventions; private RNG and owned feature copies."""
    import torch
    require(policy.ndim == expert.ndim == 2 and policy.shape[1] == expert.shape[1] == 2019,
            "Sensitivity requires full 1950 history plus 69 action inputs")
    require(min(len(policy), len(expert)) > 1, "At least two rows per population required")
    before = {name: tensor_digest(values) for name, values in (("policy", policy), ("expert", expert))}
    rng = torch.get_rng_state().clone()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutations = {name: torch.randperm(len(values), generator=generator)
                    for name, values in (("policy", policy), ("expert", expert))}
    cross = {"policy": torch.randint(len(expert), (len(policy),), generator=generator),
             "expert": torch.randint(len(policy), (len(expert),), generator=generator)}
    raw = {"index_policy_permutation": permutations["policy"], "index_expert_permutation": permutations["expert"],
           "index_policy_expert_action_donor": cross["policy"], "index_expert_policy_action_donor": cross["expert"]}
    summaries = {}
    for condition in ("original", "actions_zero", "actions_within_population_shuffled", "actions_other_population"):
        logits = {}
        for name, values, other in (("policy", policy, expert), ("expert", expert, policy)):
            changed = values.clone()
            if condition == "actions_zero":
                changed[:, 1950:] = 0
            elif condition == "actions_within_population_shuffled":
                changed[:, 1950:] = values[permutations[name], 1950:]
            elif condition == "actions_other_population":
                changed[:, 1950:] = other[cross[name], 1950:]
            logits[name] = score(discriminator, changed, batch_size)
            raw[f"{condition}_{name}_logit"] = logits[name]
        row = {"separation": separation(logits["policy"], logits["expert"])}
        for name, values in logits.items():
            row[name] = {"logit": statistics(values), "probability": statistics(values.sigmoid())}
            if condition != "original":
                original = raw[f"original_{name}_logit"]
                row[name].update(logit_delta=statistics(values - original),
                                 probability_delta=statistics(values.sigmoid() - original.sigmoid()),
                                 absolute_logit_delta=statistics((values - original).abs()))
        summaries[condition] = row
    gradients = {}
    for name, values in (("policy", policy), ("expert", expert)):
        arrays = gradient_groups(discriminator, values, batch_size)
        gradients[name] = {label: statistics(array) for label, array in arrays.items()}
        raw.update({f"{name}_gradient_{label}": array for label, array in arrays.items()})
    require(torch.equal(rng, torch.get_rng_state()), "Sensitivity changed caller CPU RNG")
    require(before == {name: tensor_digest(values) for name, values in (("policy", policy), ("expert", expert))},
            "Sensitivity mutated source feature inputs")
    return {"conditions": summaries, "input_gradients": gradients, "input_content_SHA256": before,
            "caller_CPU_RNG_preserved": True, "input_storage_unchanged": True,
            "gradient_units": "logit per normalized/clipped input coordinate; history1950/action69 L2 and per-coordinate RMS"}, raw


def build_discriminator(module, reader, state, config):
    """Use the exact D builder/RMS definitions, never the full-network factory."""
    torch = module.torch
    rng = torch.get_rng_state().clone()
    dimension = 195 if config.amp_normalization == "per_frame" else 1950
    with torch.device("cpu"), torch.random.fork_rng(devices=[]):
        discriminator = module._base._mlp(2019, config.discriminator_layers, 1)
        normalizer = module._base.RunningNormalizer(dimension, config.normalization_clip)
    saved = state["model"]
    selected = {name.removeprefix("discriminator."): value for name, value in saved.items()
                if name.startswith("discriminator.")}
    expected = discriminator.state_dict()
    require(set(selected) == set(expected), "Complete original D keys required; no partial conversion")
    for name, template in expected.items():
        reader._cpu_tensor(module, selected[name], template, "discriminator." + name)
    discriminator.load_state_dict(selected, strict=True, assign=False)
    rms = state["normalizers"]["amp_normalizer"]
    require(set(rms) == set(normalizer.state_dict()), "Complete AMP RMS state required")
    for name, template in normalizer.state_dict().items():
        reader._cpu_tensor(module, rms[name], template, "amp_normalizer." + name)
    require(bool((rms["variance"] >= 0).all()) and float(rms["count"]) > 0, "Invalid AMP RMS moments")
    normalizer.load_state_dict(rms, strict=True, assign=False)
    discriminator.eval().requires_grad_(False)
    normalizer.eval().requires_grad_(False)
    require(torch.equal(rng, torch.get_rng_state()), "D/RMS construction changed caller CPU RNG")
    return discriminator, normalizer


def normalize_features(history, actions, normalizer, config):
    import torch
    if config.amp_normalization == "per_frame":
        normalized = normalizer.normalize(history)
    else:
        normalized = normalizer.normalize(history.flatten(-2)).reshape(history.shape)
    return torch.cat((normalized.flatten(-2), actions), -1)


def admit_inputs(receipt_path, stack):
    from social_navigation.artifact_identity import artifact_path, same_artifact
    receipt_path = artifact_path(receipt_path)
    identity, admit_training, reader, binding = prepare_sources()
    receipt = json.loads(receipt_path.read_text())
    require(type(receipt.get("CPU_audit_exit_code")) is int and receipt["CPU_audit_exit_code"] == 0, "Actual CPU audit exit0 receipt required")
    external = receipt["external_exit_metadata"]
    require(same_artifact(identity(artifact_path(external["path"])), external), "Observed exit evidence changed")
    observed = json.loads(artifact_path(external["path"]).read_text())
    for name, chunk in (("training", receipt["terminal_tool_chunk"]), ("CPU_audit", receipt["CPU_audit_terminal_tool_chunk"])):
        require(type(observed[name]["exit_code"]) is int and observed[name]["exit_code"] == 0 and observed[name]["chunk_id"] == chunk,
                "Actual exit evidence disagrees with sealed receipt")
    checkpoint, sha, admission = admit_training(receipt_path)
    require(admission["value_normalization"] == "popart", "Explicit original Pop-Art format required")
    result = json.loads(artifact_path(admission["result"]["path"]).read_text())
    audit = json.loads(artifact_path(admission["audit"]["path"]).read_text())
    def source_identity(path):
        return identity(artifact_path(path))
    require(same_artifact(audit["reader"], source_identity(ROOT / "scripts/audit_shared_social_popart.py")),
            "Original full CPU audit source changed")
    source_rows = reader._sources()
    for row in result["expert_contract"]["sources"].values():
        require(same_artifact(source_identity(row["path"]), row), "Expert source contract changed")
    manifest = result["expert_contract"]["manifest"]
    require(same_artifact(identity(artifact_path(manifest["path"])), manifest) and same_artifact(manifest, result["manifest"]), "Expert manifest changed")
    module = reader._owned_module("_D_diagnostic_popart_trainer", "trainer")
    torch = module.torch
    require(not torch.cuda.is_initialized() and torch.get_default_dtype() == torch.float32
            and not torch.is_inference_mode_enabled(), "Ordinary CPU float32 mode required")
    artifacts = {}
    for name, selected in (("checkpoint", audit["checkpoint"]), ("rollout", audit["rollout"])):
        path = artifact_path(selected["path"])
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        stack.callback(os.close, descriptor)
        bound = reader._checkpoint_record(path, selected["sha256"], descriptor)
        require(same_artifact({key: bound[key] for key in selected}, selected), "Selected artifact identity differs")
        artifacts[name] = (path, descriptor, bound)
    state = torch.load(f"/proc/self/fd/{artifacts['checkpoint'][1]}", weights_only=True, map_location="cpu", mmap=True)
    from CrowdSim.humanoid.runtime.popart_checkpoint_family import select_inference_definitions
    module = select_inference_definitions(module, state)
    config = module.SocialAMPTrainer._admit_metadata(state)
    require(module._equal_json(config.to_document(), result["actual_configuration"]), "Checkpoint/result complete config mismatch")
    require(state["update_count"] == result["completed_updates"] == receipt["completed_updates"]
            and state["physical_transition_count"] == audit["physical_transitions"] == receipt["actual_training_transitions"],
            "Completed physical counters mismatch")
    initialization = {"strategy": "fresh_local_social_CNN_and_state_action_D", "seed": config.seed,
                      "weights_or_moments_transferred": False, "author_navisaaclab_weights": False}
    require(module._equal_json(state["provenance"].get("initialization"), initialization),
            "Fresh initialization provenance mismatch")
    require(state["runner_state"] is None and state["physical_runner_state_included"] is False, "Unexpected physical runner state")
    saved = torch.load(f"/proc/self/fd/{artifacts['rollout'][1]}", weights_only=True, map_location="cpu", mmap=True)
    return identity, reader, module, binding, source_rows, admission, result, artifacts, state, config, saved


def validate_populations(module, saved, state, config, result):
    torch = module.torch
    rollout = saved["collection"]["rollout"]
    metadata = saved["collection"]["metadata"]
    expert = saved["expert"]
    mask = torch.tensor(config.action_mask, dtype=torch.float32)
    steps, actors = module._base.validate_rollout(rollout, self_dim=config.self_dim, task_dim=config.task_dim,
        history_frames=config.history_frames, amp_frame_dim=config.amp_frame_dim, action_mask=mask)
    require(steps == result["profile"]["runtime"]["rollout_controls"] and actors == config.actor_count,
            "Terminal rollout full population shape mismatch")
    require(module._equal_json(metadata["configuration"], config.to_document())
            and metadata["training_update_count"] + 1 == state["update_count"]
            and metadata["training_physical_transition_count"] + steps * actors == state["physical_transition_count"],
            "Rollout/checkpoint timing or configuration mismatch")
    count = min(steps * actors, 4096)
    for name, shape, dtype in (("history", (count, 10, 195), torch.float32), ("actions", (count, 69), torch.float32),
                                ("motion_ids", (count,), torch.int64), ("frame_indices", (count,), torch.int64)):
        value = expert[name]
        require(isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.dtype == dtype
                and tuple(value.shape) == shape and bool(torch.isfinite(value).all()), "Invalid actual expert " + name)
    require(int(mask.sum()) == 57 and bool((expert["actions"].abs() <= 1).all())
            and not bool(expert["actions"][:, mask == 0].any()), "Expert bounds/frozen57 contract failed")
    manifest = json.loads(Path(result["manifest"]["path"]).read_text())
    clips = {row["motion_id"]: row["frames"] for row in manifest["splits"]["train"]["clips"]}
    require(len(clips) == 2848 and all(int(motion) in clips and 9 <= int(frame) < clips[int(motion)] - 1
        for motion, frame in zip(expert["motion_ids"], expert["frame_indices"], strict=True)), "Expert IDs/frames violate fixed train2848 windows")
    applied, raw = rollout["applied_actions"].flatten(0, 1), rollout["actions"].flatten(0, 1)
    require(torch.equal(applied, raw.clamp(-1, 1) * mask), "Actual raw/applied clamp+mask equality failed")
    amp_count = state["update_count"] * (steps * actors + count) * (10 if config.amp_normalization == "per_frame" else 1)
    require(float(state["normalizers"]["amp_normalizer"]["count"]) == amp_count, "Final AMP RMS physical/expert count mismatch")
    contract = {"policy_rows": steps * actors, "expert_rows": count, "effective_action_dimensions": 57,
        "raw_vs_applied_exact_clamp_mask": True, "expert_bounds_and_frozen_columns_verified": True,
        "expert_clip_boundary_coordinates": int((expert["actions"].abs() == 1).sum()),
        "policy_raw_outside_bounds_coordinates": int((raw.abs() > 1).sum()),
        "policy_raw_frozen_nonzero_coordinates": int((raw[:, mask == 0] != 0).sum()),
        "raw_minus_applied_abs": statistics((raw - applied).abs()),
        "expert_action_source": result["expert_contract"]["expert_action"],
        "next_mocap_position_in_saved_packet": False, "inverse_PD_action_recomputed_from_bank": False,
        "mask_clip_observation_does_not_prove_expert_controls_reproduce_motion": True,
        "train_manifest": result["manifest"], "bank_opened_or_resampled": False}
    return rollout, expert, saved["collection"]["diagnostics"], contract


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-receipt", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Set CUDA_VISIBLE_DEVICES='' for CPU-only diagnostics")
    require(args.batch_size > 0 and args.seed >= 0, "Positive batch size and nonnegative diagnostic seed required")
    require(not args.output_directory.exists(), "Use a new output directory; existing evidence cannot be replaced")
    with ExitStack() as stack:
        admitted = admit_inputs(args.execution_receipt.resolve(), stack)
        identity, reader, module, binding, source_rows, admission, result, artifacts, state, config, saved = admitted
        torch = module.torch
        torch.set_num_threads(4)
        rollout, expert, diagnostics, contract = validate_populations(module, saved, state, config, result)
        discriminator, normalizer = build_discriminator(module, reader, state, config)
        originals = {"policy_history": rollout["amp_history"], "policy_actions": rollout["actions"],
                     "policy_applied": rollout["applied_actions"], "expert_history": expert["history"], "expert_actions": expert["actions"]}
        original_hashes = {name: tensor_digest(value) for name, value in originals.items()}
        policy_features = normalize_features(rollout["amp_history"].flatten(0, 1), rollout["applied_actions"].flatten(0, 1), normalizer, config)
        expert_features = normalize_features(expert["history"], expert["actions"], normalizer, config)
        sensitivity, arrays = analyze(discriminator, policy_features, expert_features, seed=args.seed, batch_size=args.batch_size)
        arrays.update(policy_flat_index=torch.arange(len(policy_features)),
                      policy_control_index=torch.arange(len(policy_features)) // config.actor_count,
                      policy_actor_index=torch.arange(len(policy_features)) % config.actor_count,
                      policy_episode_id=rollout["episode_ids"].flatten(), expert_motion_id=expert["motion_ids"].clone(),
                      expert_frame_index=expert["frame_indices"].clone())
        for name in ("terminated", "truncated"):
            arrays["policy_" + name] = rollout[name].flatten().clone()
        for name in ("physical_fall", "nonfoot_normal_contact", "collateral_truncated", "source_end_truncated", "cap_truncated"):
            arrays["policy_" + name] = diagnostics[name].flatten().clone()
        # Within physical shared groups, exchange each actor's action with the
        # next actor at the SAME control step. Every history/terminal row stays.
        actors = config.actor_count
        group = torch.arange(actors // 4)
        rows = torch.stack((2 * group, 2 * group + 1, actors // 2 + 2 * group, actors // 2 + 2 * group + 1), -1)
        donor = torch.empty(actors, dtype=torch.int64)
        donor[rows] = rows.roll(-1, dims=1)
        donor = (arrays["policy_control_index"] * actors + donor[arrays["policy_actor_index"]])
        changed = policy_features.clone()
        changed[:, 1950:] = policy_features[donor, 1950:]
        logits = score(discriminator, changed, args.batch_size)
        arrays["index_policy_same_control_group_action_donor"] = donor
        arrays["actions_physical_group_exchange_policy_logit"] = logits
        sensitivity["physical_group_action_exchange"] = {
            "policy_logit": statistics(logits),
            "policy_probability": statistics(logits.sigmoid()),
            "policy_logit_delta": statistics(logits - arrays["original_policy_logit"]),
            "policy_probability_delta": statistics(logits.sigmoid() - arrays["original_policy_logit"].sigmoid()),
            "policy_absolute_logit_delta": statistics((logits - arrays["original_policy_logit"]).abs()),
            "separation_against_unchanged_expert": separation(logits, arrays["original_expert_logit"]),
            "groups": rows.tolist(), "all_original_histories_and_failure_rows_retained": True}
        require(original_hashes == {name: tensor_digest(value) for name, value in originals.items()}, "Saved populations mutated")
        for path, descriptor, before in artifacts.values():
            require(reader._checkpoint_record(path, before["sha256"], descriptor) == before, "Input changed during diagnostic")
        require(reader._sources() == source_rows and not torch.cuda.is_initialized(), "Source or CPU-only admission changed")
        require(not any(name.startswith(("isaacsim", "isaaclab", "omni.", "carb", "pxr")) for name in sys.modules), "SDK was imported")
        report = {"schema": "shared_social.terminal_D_input_sensitivity.v1", "status": "passed", "scope": SCOPE,
            "reader": identity(Path(__file__)), "training_admission": admission, "loader_binding": binding,
            "numeric_sources": source_rows, "artifacts": {name: row[2] for name, row in artifacts.items()},
            "configuration": config.to_document(), "seed": args.seed, "batch_size": args.batch_size,
            "population_contract": contract, "sensitivity": sensitivity, "original_tensor_SHA256": original_hashes,
            "normalization": {"selected": "terminal checkpoint AMP RMS, not frozen collection RMS",
                "mode": config.amp_normalization, "clip": 5., "epsilon": 1e-5,
                "terminal_count": float(normalizer.count), "collection_count": float(saved["collection"]["metadata"]["frozen_rms"]["amp_normalizer"]["count"])},
            "construct_or_restore": {"constructed": ["original_D2019", "original_AMP_RMS"],
                "full_network_constructed": False, "actor_critic_encoder_optimizer_replay_restored": False,
                "saved_RNG_restored": False, "complete_other_model_tensors_bound_to_prior_actual0_CPU_audit": True},
            "CUDA_initialized": False, "SDK_imported": False, "training_ablation_performed": False,
            "original_time_reward_reproduced": False, "causal_quality_evidence": False}
        args.output_directory.mkdir(parents=True)
        array_path = args.output_directory / "per_sample.pt"
        with array_path.open("xb") as stream:
            torch.save({"schema": "shared_social.terminal_D_per_sample.v1", "scope": SCOPE, "arrays": arrays}, stream)
        report["per_sample"] = identity(array_path)
        with (args.output_directory / "diagnostic.json").open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": "passed", "scope": SCOPE, "output_directory": str(args.output_directory.resolve()),
            "original_separation": sensitivity["conditions"]["original"]["separation"], "CUDA_initialized": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
