"""Explicit state-only adaptation, strict transfer and resume regressions."""
from pathlib import Path
import ast
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from CrowdSim.humanoid.amp.controller import PacerAMPController
from CrowdSim.humanoid.training.trainer import PaperAMPTrainer, TrainingConfig


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_CHECKPOINT = ROOT / "data/pretrained_models/paper_references/pacer/output/release/pacer_no_shape/Humanoid.pth"


def small_config(**options):
    values = dict(self_dim=4, task_dim=7, amp_frame_dim=3,
                  task_layers=(12, 6), policy_layers=(16, 8), discriminator_layers=(16, 8),
                  update_epochs=2, minibatch_size=8, discriminator_batch_size=8,
                  normalize_observations=True)
    values.update(options)
    return TrainingConfig(**values)


def physical_contract_rollout(trainer):
    steps, batch = 4, 2
    own = torch.randn(steps, batch, trainer.config.self_dim)
    task = torch.randn(steps, batch, trainer.config.task_dim)
    actions = trainer.act(own.flatten(0, 1), task.flatten(0, 1))
    result = {key: actions[key].reshape(steps, batch, -1).squeeze(-1)
              for key in ("actions", "applied_actions", "log_prob", "values")}
    result.update(self_obs=own, task_obs=task,
                  amp_history=torch.randn(steps, batch, 10, trainer.config.amp_frame_dim),
                  next_values=torch.zeros(steps, batch), rewards=torch.linspace(0, 1, 8).reshape(steps, batch),
                  terminated=torch.zeros(steps, batch, dtype=torch.bool),
                  truncated=torch.zeros(steps, batch, dtype=torch.bool))
    result["terminated"][-1, 0] = True
    return result


@pytest.fixture(scope="module")
def public_models():
    if not PUBLIC_CHECKPOINT.is_file() or PUBLIC_CHECKPOINT.stat().st_size < 10000:
        pytest.skip("Complete publicly verified PACER checkpoint is not installed")
    lower, upper = -torch.ones(69), torch.ones(69)
    reference = PacerAMPController.from_checkpoint(PUBLIC_CHECKPOINT, lower, upper)
    trainer = PaperAMPTrainer(TrainingConfig(discriminator_input="state",
        amp_normalization="per_history_coordinate", warm_start_discriminator=True))
    trainer.warm_start_pacer(PUBLIC_CHECKPOINT, lower, upper)
    return reference, trainer


@pytest.mark.needs_lfs
def test_real_public_actor_state_discriminator_and_history_statistics_parity(public_models):
    reference, trainer = public_models
    raw = torch.randn(8, 1401)
    history = torch.randn(8, 10, 195)
    actions = torch.zeros(8, 69)
    with torch.no_grad():
        actor, _ = reference.network.eval_actor(reference.observation_normalizer(raw))
        critic = reference.network.eval_critic(reference.observation_normalizer(raw)).squeeze(-1)
        own, task = trainer._observations(raw[:, :357], raw[:, 357:])
        _, transferred_critic = trainer.model.policy(own, task)
        expected_logits = reference.eval_discriminator(history).squeeze(-1)
        normalized, _ = trainer._amp_inputs(history, actions)
        logits = trainer.model.disc_logits(normalized, actions)
    torch.testing.assert_close(trainer.act(raw[:, :357], raw[:, 357:], deterministic=True)["actions"], actor)
    torch.testing.assert_close(transferred_critic, critic)
    torch.testing.assert_close(logits, expected_logits, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(normalized.flatten(1), reference.discriminator_normalizer(history.flatten(1)))
    for actual, expected in ((trainer.amp_normalizer.mean, reference.discriminator_normalizer.running_mean),
                             (trainer.amp_normalizer.variance, reference.discriminator_normalizer.running_var),
                             (trainer.amp_normalizer.count, reference.discriminator_normalizer.count)):
        assert torch.equal(actual, expected)
    assert trainer.model.discriminator[0].in_features == 1950
    initialization = trainer.provenance["initialization"]
    assert initialization["published_state_only_discriminator_transferred"] is True
    assert initialization["published_amp_history_moments_transferred"] is True
    assert initialization["author_navisaaclab_weights"] is False


@pytest.mark.needs_lfs
def test_public_warmstart_inference_leaves_all_moments_frozen(public_models):
    _, trainer = public_models
    normalizers = (trainer.self_normalizer, trainer.task_normalizer,
                   trainer.value_normalizer, trainer.amp_normalizer)
    before = [{key: value.clone() for key, value in module.state_dict().items()} for module in normalizers]
    for _ in range(3):
        trainer.act(torch.randn(4, 357), torch.randn(4, 1044), deterministic=True)
        trainer.value(torch.randn(4, 357), torch.randn(4, 1044))
        trainer.amp_reward(torch.randn(4, 10, 195), torch.zeros(4, 69))
    for snapshot, module in zip(before, normalizers):
        for key, value in module.state_dict().items():
            assert torch.equal(value, snapshot[key])


def test_state_only_reward_ignores_proxy_actions_but_old_default_consumes_them():
    history = torch.randn(5, 10, 3)
    for mode in ("state", "state_action"):
        trainer = PaperAMPTrainer(small_config(discriminator_input=mode))
        zero, changed = torch.zeros(5, 69), .7 * trainer.action_mask.expand(5, -1)
        first, second = trainer.amp_reward(history, zero), trainer.amp_reward(history, changed)
        if mode == "state":
            assert torch.equal(first, second)
            assert trainer.model.discriminator[0].in_features == 30
        else:
            assert not torch.allclose(first, second)
            assert trainer.model.discriminator[0].in_features == 99


def test_history_coordinate_normalization_preserves_time_slot_information():
    trainer = PaperAMPTrainer(small_config(discriminator_input="state", amp_normalization="per_history_coordinate"))
    history = torch.arange(30.).reshape(1, 10, 3)
    trainer.amp_normalizer.mean.copy_(history.flatten().double())
    normalized, _ = trainer._amp_inputs(history, torch.zeros(1, 69))
    assert torch.equal(normalized, torch.zeros_like(history))
    reversed_history, _ = trainer._amp_inputs(history.flip(1), torch.zeros(1, 69))
    assert not torch.equal(reversed_history, normalized)


@pytest.mark.parametrize("normalization", ["per_frame", "per_history_coordinate"])
def test_state_only_actual_gradient_update_and_complete_resume(tmp_path, normalization):
    trainer = PaperAMPTrainer(small_config(discriminator_input="state", amp_normalization=normalization))
    rollout = physical_contract_rollout(trainer)
    expert = torch.randn(16, 10, 3)
    before = {key: value.clone() for key, value in trainer.model.discriminator.state_dict().items()}
    metrics = trainer.update(rollout, expert, torch.zeros(16, 69))
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert any(not torch.equal(value, before[key]) for key, value in trainer.model.discriminator.state_dict().items())
    expected_count = (8+16)*(10 if normalization == "per_frame" else 1)
    assert trainer.amp_normalizer.count.item() == expected_count
    path = tmp_path/"state_only.pt"
    trainer.save(path, runner_state={"actual_physical_step": 8})
    loaded, runner = PaperAMPTrainer.load(path, restore_rng=False)
    assert runner == {"actual_physical_step": 8}
    assert loaded.config.discriminator_input == "state" and loaded.config.amp_normalization == normalization
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, loaded.model.state_dict()[key])
    assert torch.equal(trainer.amp_normalizer.mean, loaded.amp_normalizer.mean)
    assert loaded.optimizer.state and loaded.discriminator_optimizer.state


def test_known_legacy_checkpoint_missing_new_fields_keeps_old_contract(tmp_path):
    trainer = PaperAMPTrainer(small_config())
    path = tmp_path/"legacy.pt"
    trainer.save(path)
    state = torch.load(path, weights_only=True)
    for key in ("discriminator_input", "amp_normalization", "warm_start_discriminator"):
        del state["config"][key]
    torch.save(state, path)
    loaded, _ = PaperAMPTrainer.load(path, restore_rng=False)
    assert loaded.config.discriminator_input == "state_action"
    assert loaded.config.amp_normalization == "per_frame"
    assert loaded.config.warm_start_discriminator is False
    assert loaded.model.discriminator[0].in_features == 99
    assert loaded.amp_normalizer.mean.shape == (3,)
    assert torch.equal(loaded.action_mask, trainer.action_mask)
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(loaded.model.state_dict()[key], value)


@pytest.mark.parametrize("problem", ["missing", "extra", "nan", "negative_variance", "negative_count", "wrong_shape"])
def test_resume_cannot_silently_replace_missing_or_invalid_moments(tmp_path, problem):
    trainer = PaperAMPTrainer(small_config())
    path = tmp_path/"broken.pt"
    trainer.save(path)
    state = torch.load(path, weights_only=True)
    moments = state["normalizers"]["self_normalizer"]
    if problem == "missing":
        del state["normalizers"]["amp_normalizer"]
    elif problem == "extra":
        state["normalizers"]["other"] = moments
    elif problem == "nan":
        moments["mean"][0] = float("nan")
    elif problem == "negative_variance":
        moments["variance"][0] = -1
    elif problem == "negative_count":
        moments["count"] = torch.tensor(-1., dtype=torch.float64)
    else:
        moments["mean"] = moments["mean"][:1]
    torch.save(state, path)
    with pytest.raises(ValueError, match="normalizer"):
        PaperAMPTrainer.load(path, restore_rng=False)


@pytest.mark.parametrize("options", [
    {"normalize_observations": False}, {"normalize_values": False}, {"normalization_clip": 4.},
])
def test_pacer_transfer_rejects_incompatible_observation_or_value_normalization(options):
    values = {"normalize_observations": True}
    values.update(options)
    trainer = PaperAMPTrainer(small_config(**values))
    with pytest.raises(ValueError, match="normalized observations/values"):
        trainer.warm_start_pacer("file_must_not_be_opened.pt", -torch.ones(69), torch.ones(69))


@pytest.mark.parametrize("options", [
    {"discriminator_input": "unknown"}, {"amp_normalization": "unknown"},
    {"warm_start_discriminator": 1}, {"warm_start_discriminator": True},
    {"warm_start_discriminator": True, "discriminator_input": "state"},
    {"warm_start_discriminator": True, "discriminator_input": "state", "amp_normalization": "per_history_coordinate", "history_frames": 11},
    {"normalize_observations": "false"}, {"normalize_values": 1}, {"seed": -1}, {"seed": True}, {"seed": 2**32},
])
def test_invalid_stage_settings_are_rejected_without_torch_dependencies(options):
    with pytest.raises(ValueError):
        TrainingConfig(**options).validate()


def test_settings_import_and_validation_does_not_import_torch_or_kit():
    code = "from CrowdSim.humanoid.application.settings import TrainingConfig; import sys; TrainingConfig().validate(); assert 'torch' not in sys.modules; assert not any(n.startswith(('isaacsim','omni.')) for n in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_adaptation_profile_is_explicit_state_only_non_author_ablation():
    data = yaml.safe_load((ROOT/"configs/diffusion_amp/amp_pacer_adaptation.yaml").read_text())
    config = TrainingConfig(**data["training"])
    config.validate()
    assert config.discriminator_input == "state" and config.warm_start_discriminator is True
    assert config.amp_normalization == "per_history_coordinate"
    assert data["author_experiment_reproduction"] is False
    assert any("differs from NavIsaacLab Eq2" in assumption for assumption in data["assumptions"])


@pytest.mark.parametrize("mode", ["state_action", "state"])
def test_actual_runner_preserves_legacy_pre_action_and_state_only_post_step_history(mode):
    source = (ROOT/"scripts/train_paper_amp.py").read_text()
    tree = ast.parse(source)
    loop = next(node for node in ast.walk(tree) if isinstance(node, ast.For)
                and isinstance(node.target, ast.Name) and node.target.id == "step")
    begin = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.Assign)
                 and ast.unparse(node.value) == "world.state()")
    end = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.AugAssign)
               and isinstance(node.target, ast.Name) and node.target.id == "control_step")
    fragment = ast.Module(body=loop.body[begin:end+1], type_ignores=[])
    events = []

    class ResponseWorld:
        validation_scene = None
        device = "cpu"
        amp_history = torch.ones(2, 10, 3)
        amp_history_episode_ids = torch.full((2, 10), 7, dtype=torch.long)
        episode_ids = torch.full((2,), 7, dtype=torch.long)

        def state(self):
            return "pre_state"

        def observations(self, trajectory, state):
            return torch.zeros(2, 4), torch.zeros(2, 7)

        def step(self, action):
            events.append("physics")
            self.amp_history = torch.full((2, 10, 3), 9.)
            return "terminal_pre_reset_state", torch.tensor([True, False]), torch.zeros(2, dtype=torch.bool)

    class ResponseTrainer:
        config = SimpleNamespace(discriminator_input=mode)

        def act(self, own, task, deterministic=False):
            events.append("actor")
            return {"applied_actions": torch.zeros(2, 69)}

        def amp_reward(self, history, applied):
            events.append(("D", float(history[0, 0, 0])))
            return history.flatten(1).sum(-1)

    namespace = {"world": ResponseWorld(), "trainer": ResponseTrainer(),
                 "task_trajectory": lambda: torch.zeros(2, 10, 2), "args": SimpleNamespace(evaluate_steps=0, num_envs=2),
                 "torch": torch, "route_indices": torch.zeros(2, dtype=torch.long),
                 "time": SimpleNamespace(perf_counter=lambda: 1.), "section_started": 0.,
                 "timing": {"policy_observation_wall_s": 0.}, "control_step": 0}
    exec(compile(ast.fix_missing_locations(fragment), "actual_runner_reward_timing", "exec"), namespace)
    expected = 1. if mode == "state_action" else 9.
    assert namespace["amp_reward"].tolist() == [30*expected, 30*expected]
    assert torch.all(namespace["history"] == expected)
    assert namespace["control_step"] == 1
    assert torch.all(namespace["history_episode"] == namespace["episode"][:, None])
    assert events == (["actor", ("D", 1.), "physics"] if mode == "state_action"
                      else ["actor", "physics", ("D", 9.)])
    transition = next(node.value for node in loop.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "transition" for target in node.targets))
    fields = {key.value: value for key, value in zip(transition.keys, transition.values)}
    assert ast.unparse(fields["amp_history"]) == "history"
    assert ast.unparse(fields["history_episode_ids"]) == "history_episode"
