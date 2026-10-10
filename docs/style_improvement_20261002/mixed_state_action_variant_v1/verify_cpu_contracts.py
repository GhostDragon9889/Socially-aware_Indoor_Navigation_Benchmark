"""Actual public warm-start/v7 input verification, never a physical rollout."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from CrowdSim.humanoid.amp.controller import PacerAMPController, action_to_pd
from CrowdSim.humanoid.amp.observations import PacerSMPLMapping, build_amp_observation
from CrowdSim.humanoid.amass_corpus import AMASSCorpus
from CrowdSim.humanoid.application.configuration import load_amp_profile
from CrowdSim.humanoid.application.settings import TrainingConfig
from CrowdSim.humanoid.training.trainer import PaperAMPTrainer
from protomotions.robot_configs.factory import robot_config


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def record(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def tensor_record(value):
    array = value.detach().cpu().contiguous().numpy()
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(), "finite": bool(torch.isfinite(value).all())}


def main():
    profile_path = ROOT / "configs/diffusion_amp/amp_style_mixed_state_action.yaml"
    profile = load_amp_profile(profile_path)
    checkpoint = ROOT / profile["initialization"]["checkpoint"]
    rig = robot_config("smpl")
    lower, upper = rig.kinematic_info.dof_limits_lower, rig.kinematic_info.dof_limits_upper
    mapping = PacerSMPLMapping(body_names=tuple(rig.kinematic_info.body_names), dof_names=tuple(rig.kinematic_info.dof_names))
    trainer = PaperAMPTrainer(TrainingConfig(**profile["training"]), device="cpu")
    before = {key: value.clone() for key, value in trainer.model.discriminator.state_dict().items()}
    initialization = trainer.warm_start_pacer(checkpoint, lower, upper,
        body_names=mapping.body_names, dof_names=mapping.dof_names)
    reference = PacerAMPController.from_checkpoint(checkpoint, lower, upper,
        body_names=mapping.body_names, dof_names=mapping.dof_names)
    discriminator_unchanged = all(torch.equal(value, before[key]) for key, value in trainer.model.discriminator.state_dict().items())
    assert discriminator_unchanged and trainer.model.discriminator[0].weight.shape == (1024, 2019)
    assert trainer.amp_normalizer.mean.shape == (1950,) and trainer.amp_normalizer.count == 0
    own_generator = torch.Generator().manual_seed(20261002)
    own, task = torch.randn(6, 357, generator=own_generator), torch.randn(6, 1044, generator=own_generator)
    with torch.no_grad():
        public_mean, public_sigma = reference.network.eval_actor(reference.observation_normalizer(torch.cat((own, task), -1)))
        distribution, _ = trainer.model.policy(*trainer._observations(own, task))
    deterministic = trainer.act(own, task, deterministic=True)
    stochastic = trainer.act(own, task, deterministic=False)
    assert torch.equal(deterministic["actions"], public_mean)
    assert torch.equal(distribution.stddev, public_sigma.exp())
    assert torch.equal(stochastic["applied_actions"], stochastic["actions"].clamp(-1, 1) * trainer.action_mask)
    assert torch.equal(stochastic["log_prob"], (distribution.log_prob(stochastic["actions"]) * trainer.action_mask).sum(-1))
    corpus = AMASSCorpus(ROOT / profile["data_manifest"], device="cpu", seed=7, dof_velocity_frame="child_joint")
    corpus_rng_before = corpus.generator.get_state().clone()
    tree = ast.parse((ROOT / "CrowdSim/humanoid/amp_world.py").read_text())
    world_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AMPWorld")
    method = deepcopy(next(node for node in world_class.body if isinstance(node, ast.FunctionDef) and node.name == "expert_batch"))
    namespace = {"torch": torch, "build_amp_observation": build_amp_observation, "action_to_pd": action_to_pd}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "actual_expert_batch_CPU", "exec"), namespace)
    view = SimpleNamespace(corpus=corpus, history_frames=10, device=torch.device("cpu"), mapping=mapping, lower=lower, upper=upper)
    history, expert_actions = namespace["expert_batch"](view, 32)
    corpus_rng_after = corpus.generator.get_state().clone()
    corpus.generator.set_state(corpus_rng_before)
    absolute, ids, frames = corpus._indices(32, past=9, future=1)
    assert torch.equal(corpus.generator.get_state(), corpus_rng_after)
    expected = build_amp_observation(corpus.state_at((absolute[:, None] - torch.arange(10)).flatten()), mapping=mapping).reshape(32, 10, 195)
    assert torch.equal(history, expected)
    target = corpus.data["dps"][absolute + 1].clone()
    limits = action_to_pd(torch.ones(32, 69), lower, upper, dof_names=mapping.dof_names).abs()
    target = target.clamp(-limits, limits)
    frozen_physical = [i for i, name in enumerate(mapping.dof_names) if name.rsplit("_", 1)[0] in {"L_Hand", "R_Hand", "L_Toe", "R_Toe"}]
    target[:, frozen_physical] = 0
    decoded = action_to_pd(expert_actions, lower, upper, dof_names=mapping.dof_names)
    pd_error = float((decoded-target).abs().max())
    assert pd_error <= 3e-7 and torch.count_nonzero(expert_actions[:, trainer.action_mask == 0]) == 0
    _, normalized_actions = trainer._amp_inputs(history, expert_actions)
    inputs = normalized_actions.detach().clone().requires_grad_(True)
    normalized_history, _ = trainer._amp_inputs(history, inputs)
    gradient = torch.autograd.grad(trainer.model.disc_logits(normalized_history, inputs).sum(), inputs)[0]
    assert torch.count_nonzero(gradient[:, trainer.action_mask == 1]) > 0
    rewards = trainer.amp_reward(history, expert_actions)
    assert rewards.shape == (32,) and torch.isfinite(rewards).all()
    assert not torch.cuda.is_initialized()
    output = {"schema": "mixed_state_action_actual_cpu_contract.v1", "status": "passed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "profile": record(profile_path),
        "public_initialization_checkpoint": record(checkpoint), "initialization": initialization,
        "discriminator_before_after_warm_start_exact": discriminator_unchanged,
        "discriminator_input_dimensions": 2019, "AMP_normalizer_dimensions": 1950,
        "AMP_normalizer_initial": {"mean": 0., "variance": 1., "count": float(trainer.amp_normalizer.count)},
        "public_actor_transfer_max_abs_error": float((deterministic["actions"]-public_mean).abs().max()),
        "public_actor_transfer_tensor_equal": True, "Gaussian_std_transfer_tensor_equal": True,
        "Gaussian_quantities": {name: tensor_record(value) for name, value in (
            ("mean", distribution.mean), ("std", distribution.stddev), ("raw_action", stochastic["actions"]),
            ("applied_action", stochastic["applied_actions"]), ("log_probability", stochastic["log_prob"]), ("value", stochastic["values"]))},
        "effective_action_coordinates": int(trainer.action_mask.sum()), "frozen_coordinates": int((trainer.action_mask==0).sum()),
        "configuration": asdict(trainer.config), "v7_manifest": record(corpus.manifest_path), "corpus_provenance": corpus.provenance(),
        "bank_full_sha_recomputed_in_this_probe": False, "actual_bank_bytes": corpus.entry["bytes"],
        "expert_history": tensor_record(history), "expert_action": tensor_record(expert_actions),
        "history_order": "current,current-1,...,current-9", "history_source_exact": True,
        "expert_source_ids": ids.tolist(), "expert_control_frames": frames.tolist(),
        "expert_PD_decode_vs_clipped_next_pose_max_error_rad": pd_error,
        "expert_action_semantics": "normalized next-frame PD pose proxy; not measured expert command/torque or inverse dynamics",
        "new_discriminator_effective_action_gradient_nonzero": True, "AMP_reward": tensor_record(rewards),
        "training_started": False, "physical_rollouts_collected": False, "Kit_started": False, "CUDA_initialized": False,
        "sealed34_test_outputs_read": False,
        "sources": [record(ROOT/path) for path in ("CrowdSim/humanoid/training/trainer.py",
            "CrowdSim/humanoid/application/settings.py", "CrowdSim/humanoid/application/configuration.py",
            "CrowdSim/humanoid/amp_world.py", "CrowdSim/humanoid/amass_corpus.py",
            "CrowdSim/humanoid/amp/controller.py", "CrowdSim/humanoid/amp/observations.py")]}
    with (OUT / "actual_cpu_contract.json").open("x") as stream:
        stream.write(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "passed", "D_input": 2019, "actor_transfer_max_error": 0.,
                      "expert_PD_error_rad": pd_error, "training_started": False}))


if __name__ == "__main__":
    main()
