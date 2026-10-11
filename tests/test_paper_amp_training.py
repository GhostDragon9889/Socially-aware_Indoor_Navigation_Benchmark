"""Meaningful optimizer, termination, normalization and full resume checks."""

from pathlib import Path
import random

import numpy as np
import pytest
import torch

from CrowdSim.humanoid.training.contracts import generalized_advantages
from CrowdSim.humanoid.training.trainer import PaperAMPTrainer, RunningNormalizer, TrainingConfig


def tiny_config(**overrides):
    defaults = dict(self_dim=4, task_dim=7, amp_frame_dim=3, history_frames=10,
        task_layers=(12, 6), policy_layers=(16, 8), discriminator_layers=(16, 8),
        update_epochs=2, minibatch_size=8, discriminator_batch_size=8,
        learning_rate=.002, discriminator_learning_rate=.003, normalize_observations=False)
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def rollout_for(trainer, steps=4, envs=4):
    self_obs = torch.randn(steps, envs, trainer.config.self_dim)
    task_obs = torch.randn(steps, envs, trainer.config.task_dim)
    flat = trainer.act(self_obs.flatten(0, 1), task_obs.flatten(0, 1))
    data = {key: flat[key].reshape(steps, envs, -1).squeeze(-1)
            for key in ("actions", "applied_actions", "log_prob", "values")}
    data.update(self_obs=self_obs, task_obs=task_obs,
        amp_history=torch.randn(steps, envs, trainer.config.history_frames, trainer.config.amp_frame_dim),
        next_values=torch.zeros(steps, envs), rewards=torch.arange(steps * envs).reshape(steps, envs).float() / 10,
        terminated=torch.zeros(steps, envs, dtype=torch.bool), truncated=torch.zeros(steps, envs, dtype=torch.bool))
    data["terminated"][-1, 0] = True
    if envs > 1:
        data["truncated"][-1, 1] = True
    return data


def expert_for(trainer, count=20):
    return (torch.randn(count, trainer.config.history_frames, trainer.config.amp_frame_dim) + 2,
            torch.zeros(count, 69))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a real CUDA checkpoint destination")
def test_gpu_resume_preserves_cpu_runner_rng_and_moves_optimizer_state(tmp_path):
    trainer = PaperAMPTrainer(tiny_config())
    trainer.update(rollout_for(trainer), *expert_for(trainer))
    generator = torch.Generator(device="cpu").manual_seed(42)
    runner = {"corpus_rng": generator.get_state(), "physical": {"root_pos": torch.zeros(4, 3)}}
    path = tmp_path / "cpu_runner.pt"
    trainer.save(path, runner_state=runner)
    resumed, restored_runner = PaperAMPTrainer.load(path, device="cuda:0", restore_rng=False)
    # CPU sampling generators cannot consume RNG state relocated to the GPU.
    generator.set_state(restored_runner["corpus_rng"])
    assert next(resumed.model.parameters()).device.type == "cuda"
    assert resumed.optimizer.state
    for state in resumed.optimizer.state.values():
        assert state["exp_avg"].device.type == "cuda"
        assert state["exp_avg_sq"].device.type == "cuda"
    resumed.restore_rng()


def test_real_optimizer_update_changes_policy_critic_discriminator_and_reports_finite_losses():
    trainer = PaperAMPTrainer(tiny_config())
    rollout = rollout_for(trainer)
    before = {name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters()}
    metrics = trainer.update(rollout, *expert_for(trainer))
    for prefix in ("actor.", "critic.", "task_encoder.", "discriminator."):
        assert any(not torch.equal(parameter, before[name]) for name, parameter in trainer.model.named_parameters() if name.startswith(prefix))
    assert trainer.optimizer.state and trainer.discriminator_optimizer.state
    assert metrics["physical_transitions"] == 16 and metrics["optimizer_minibatches"] == 4
    assert all(np.isfinite(value) for value in metrics.values())


def test_discriminator_learns_expert_vs_policy_separation_in_actual_gradient_updates():
    trainer = PaperAMPTrainer(tiny_config(discriminator_gradient_penalty=0., discriminator_logit_regularization=0., discriminator_weight_decay=0.))
    agent = torch.full((32, 10, 3), -2.)
    expert = torch.full((32, 10, 3), 2.)
    actions = torch.zeros(32, 69)
    with torch.no_grad():
        initial = torch.sigmoid(trainer.model.disc_logits(expert, actions)).mean() - torch.sigmoid(trainer.model.disc_logits(agent, actions)).mean()
    for _ in range(40):
        trainer._discriminator_update(agent, actions, expert, actions)
    with torch.no_grad():
        expert_probability = trainer.model.disc_logits(expert, actions).sigmoid().mean()
        agent_probability = trainer.model.disc_logits(agent, actions).sigmoid().mean()
    assert expert_probability - agent_probability > initial + .4
    assert expert_probability > .75 and agent_probability < .25


def test_terminal_and_time_limit_bootstrap_are_distinct_and_do_not_cross_reset():
    rewards = torch.tensor([[1., 1.], [100., 100.]])
    values = torch.zeros(2, 2)
    next_values = torch.tensor([[10., 10.], [0., 0.]])
    terminated = torch.tensor([[True, False], [False, False]])
    truncated = torch.tensor([[False, True], [False, False]])
    advantages, _ = generalized_advantages(rewards, values, next_values, terminated, truncated, gamma=.9, gae_lambda=.95)
    torch.testing.assert_close(advantages[0], torch.tensor([1., 10.]))


def test_singleton_update_and_population_normalizer_are_finite():
    normalizer = RunningNormalizer(2)
    normalizer.update(torch.tensor([[3., 4.]]))
    torch.testing.assert_close(normalizer.variance, torch.zeros(2, dtype=torch.float64))
    trainer = PaperAMPTrainer(tiny_config(normalize_observations=True))
    metrics = trainer.update(rollout_for(trainer, 1, 1), *expert_for(trainer, 1))
    assert all(np.isfinite(value) for value in metrics.values())


def test_complete_checkpoint_restores_optimizers_moments_rng_and_continuation(tmp_path):
    trainer = PaperAMPTrainer(tiny_config(normalize_observations=True))
    trainer.update(rollout_for(trainer), *expert_for(trainer))
    path = tmp_path / "actual_training_state.pt"
    runner = {"step": 16, "physical_state": torch.randn(4, 7), "history": torch.randn(4, 10, 3)}
    trainer.save(path, runner_state=runner)
    expected_rng = (random.random(), np.random.random(), torch.randn(3))
    observations = torch.randn(4, 4)
    tasks = torch.randn(4, 7)
    expected_action = trainer.act(observations, tasks, deterministic=True)
    resume, saved_runner = PaperAMPTrainer.load(path)
    assert saved_runner["step"] == 16
    torch.testing.assert_close(saved_runner["physical_state"], runner["physical_state"])
    actual_rng = (random.random(), np.random.random(), torch.randn(3))
    assert actual_rng[:2] == expected_rng[:2]
    torch.testing.assert_close(actual_rng[2], expected_rng[2])
    for key, value in expected_action.items():
        torch.testing.assert_close(resume.act(observations, tasks, deterministic=True)[key], value)
    assert resume.optimizer.state and resume.discriminator_optimizer.state
    # Identical real second update from the same RNG point produces identical
    # weights and metrics, beyond merely comparing state_dict serialization.
    next_rollout = rollout_for(trainer)
    next_expert = expert_for(trainer)
    rng = torch.get_rng_state()
    expected_metrics = trainer.update(next_rollout, *next_expert)
    torch.set_rng_state(rng)
    actual_metrics = resume.update(next_rollout, *next_expert)
    assert actual_metrics == pytest.approx(expected_metrics)
    for name, value in trainer.model.state_dict().items():
        torch.testing.assert_close(resume.model.state_dict()[name], value)


@pytest.mark.parametrize("problem", ["empty", "nonfinite", "clamp", "cross_reset"])
def test_invalid_physical_rollouts_fail_before_optimizer_mutation(problem):
    trainer = PaperAMPTrainer(tiny_config())
    rollout = rollout_for(trainer)
    if problem == "empty":
        rollout["rewards"] = torch.empty(0, 4)
    elif problem == "nonfinite":
        rollout["self_obs"][0, 0, 0] = float("nan")
    elif problem == "clamp":
        rollout["applied_actions"][0, 0, 0] = .9
    else:
        rollout["episode_ids"] = torch.zeros(4, 4, dtype=torch.long)
        rollout["history_episode_ids"] = torch.ones(4, 4, 10, dtype=torch.long)
    before = {name: parameter.detach().clone() for name, parameter in trainer.model.named_parameters()}
    with pytest.raises(ValueError):
        trainer.update(rollout, *expert_for(trainer))
    assert not trainer.optimizer.state and not trainer.discriminator_optimizer.state
    for name, parameter in trainer.model.named_parameters():
        torch.testing.assert_close(parameter, before[name])


@pytest.mark.needs_lfs
def test_real_verified_pacer_warmstart_actor_parity_and_independent_new_discriminator():
    from CrowdSim.humanoid.amp.controller import PacerAMPController
    from protomotions.robot_configs.factory import robot_config
    path = Path(__file__).resolve().parents[1] / "data/pretrained_models/paper_references/pacer/output/release/pacer_no_shape/Humanoid.pth"
    if not path.is_file() or path.stat().st_size < 10000:
        pytest.skip("Verified public PACER no_shape checkpoint is not installed")
    cfg = robot_config("smpl")
    lower = cfg.kinematic_info.dof_limits_lower
    upper = cfg.kinematic_info.dof_limits_upper
    reference = PacerAMPController.from_checkpoint(path, lower, upper)
    trainer = PaperAMPTrainer(TrainingConfig())
    before = {name: value.clone() for name, value in trainer.model.discriminator.state_dict().items()}
    report = trainer.warm_start_pacer(path, lower, upper)
    raw = torch.randn(8, 1401)
    with torch.no_grad():
        mean, _ = reference.network.eval_actor(reference.observation_normalizer(raw))
    torch.testing.assert_close(trainer.act(raw[:, :357], raw[:, 357:], deterministic=True)["actions"], mean, atol=1e-6, rtol=1e-5)
    for name, value in before.items():
        torch.testing.assert_close(trainer.model.discriminator.state_dict()[name], value)
    assert trainer.amp_normalizer.count == 0
    assert report["published_state_only_discriminator_transferred"] is False


def test_strict_mode_requires_verified_manifest_and_does_not_silently_use_reference():
    with pytest.raises(ValueError, match="author manifest"):
        PaperAMPTrainer(tiny_config(mode="strict_author_experiment"))


def test_effective_action_projection_eliminates_dummy_joint_disc_shortcut_without_changing_pd_targets():
    from CrowdSim.humanoid.amp.controller import action_to_pd
    from CrowdSim.humanoid.amp.observations import PACER_DOF_NAMES

    trainer = PaperAMPTrainer(tiny_config())
    own, task = torch.randn(4, 4), torch.randn(4, 7)
    result = trainer.act(own, task)
    frozen = [index for index, name in enumerate(PACER_DOF_NAMES)
              if name.rsplit("_", 1)[0] in {"L_Hand", "R_Hand", "L_Toe", "R_Toe"}]
    assert len(frozen) == 12 and trainer.action_mask.sum() == 57
    assert torch.count_nonzero(result["actions"][:, frozen]) == 48
    assert torch.count_nonzero(result["applied_actions"][:, frozen]) == 0
    lower, upper = torch.full((69,), -1.), torch.full((69,), 1.)
    old_pd = action_to_pd(result["actions"].clamp(-1, 1), lower, upper)
    effective_pd = action_to_pd(result["applied_actions"], lower, upper)
    torch.testing.assert_close(effective_pd, old_pd)
    normalized_self, normalized_task = trainer._observations(own, task)
    distribution, _ = trainer.model.policy(normalized_self, normalized_task)
    torch.testing.assert_close(result["log_prob"], distribution.log_prob(result["actions"]).sum(-1))
    # The discriminator sees the same zero dummy coordinates for reference
    # and real policy, while genuine69D Gaussian likelihood remains intact.
    history, expert_actions = expert_for(trainer, 4)
    trainer.amp_reward(history, result["applied_actions"])
    expert_actions[:, frozen[0]] = .8
    with pytest.raises(ValueError, match="frozen coordinates"):
        trainer.update(rollout_for(trainer), history, expert_actions)
    assert not trainer.optimizer.state


def test_explicit_action_mask_cannot_enable_physically_frozen_joints_and_is_checkpointed(tmp_path):
    with pytest.raises(ValueError, match="frozen by the physical"):
        PaperAMPTrainer(tiny_config(action_mask=(1,) * 69))
    trainer = PaperAMPTrainer(tiny_config())
    path = tmp_path / "effective_mask.pt"
    trainer.save(path)
    loaded, _ = PaperAMPTrainer.load(path)
    assert loaded.config.action_mask == trainer.config.action_mask
    assert loaded.config.freeze_toes is True
    torch.testing.assert_close(loaded.action_mask, trainer.action_mask)


def test_delayed_rng_restore_survives_planner_and_runner_initialization(tmp_path):
    trainer = PaperAMPTrainer(tiny_config())
    path = tmp_path / "pending_rng.pt"
    trainer.save(path)
    expected = (random.random(), np.random.random(), torch.randn(8))
    loaded, _ = PaperAMPTrainer.load(path, restore_rng=False)
    # An actual newly initialized model consumes CPU RNG, as the diffusion
    # planner does before loading its own checkpoint in the physical runner.
    torch.nn.Linear(50, 30)
    random.random()
    np.random.random()
    loaded.restore_rng()
    actual = (random.random(), np.random.random(), torch.randn(8))
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2])
    with pytest.raises(ValueError, match="No pending"):
        loaded.restore_rng()
