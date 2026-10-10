"""Behavioral regressions for explicitly selected AMP style improvements."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from CrowdSim.humanoid.application.configuration import parse_amp_args
from CrowdSim.humanoid.application.settings import TrainingConfig
from CrowdSim.humanoid.application.resources import ResourceError, _manifest_resources
from CrowdSim.humanoid.training.replay import AMPPolicyReplay
from CrowdSim.humanoid.training.rewards import paper_total_reward
from CrowdSim.humanoid.training.trainer import PaperAMPTrainer


def config(**options):
    fields = dict(self_dim=4, task_dim=7, amp_frame_dim=3,
        task_layers=(12, 6), policy_layers=(16, 8), discriminator_layers=(16, 8),
        update_epochs=3, minibatch_size=8, discriminator_batch_size=8,
        discriminator_input="state", discriminator_updates_per_update=2,
        discriminator_replay_capacity=8, effective_action_likelihood=True)
    fields.update(options)
    return TrainingConfig(**fields)


def rollout(trainer):
    self_obs, task_obs = torch.randn(8, 4), torch.randn(8, 7)
    actions = trainer.act(self_obs, task_obs)
    values = {key: actions[key].reshape(4, 2, -1).squeeze(-1)
              for key in ("actions", "applied_actions", "log_prob", "values")}
    values.update(self_obs=self_obs.reshape(4, 2, 4), task_obs=task_obs.reshape(4, 2, 7),
        amp_history=torch.randn(4, 2, 10, 3), rewards=torch.arange(8.).reshape(4, 2),
        next_values=torch.zeros(4, 2), terminated=torch.zeros(4, 2, dtype=torch.bool),
        truncated=torch.zeros(4, 2, dtype=torch.bool))
    return values


def test_replay_overflow_and_resume_keep_actual_old_policy_samples():
    replay = AMPPolicyReplay(5, (10, 3))
    history = torch.arange(9.).reshape(9, 1, 1).expand(-1, 10, 3).clone()
    actions = torch.zeros(9, 69)
    replay.add(history[:3], actions[:3])
    replay.add(history[3:7], actions[3:7])
    assert sorted(replay.history[:, 0, 0].tolist()) == [2., 3., 4., 5., 6.]
    replay.add(history[7:], actions[7:], keep_probability=0.)
    assert replay.accepted == 7 and replay.size == 5
    recovered = AMPPolicyReplay(5, (10, 3))
    recovered.load_state_dict(replay.state_dict())
    assert torch.equal(recovered.history, replay.history)
    assert recovered.cursor == replay.cursor
    torch.manual_seed(58)
    expected = replay.sample(32)
    torch.manual_seed(58)
    actual = recovered.sample(32)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.parametrize("corruption", ["nan", "counter", "dtype", "action", "missing"])
def test_replay_resume_rejects_incomplete_or_invalid_data(corruption):
    replay = AMPPolicyReplay(8, (10, 3))
    replay.add(torch.zeros(4, 10, 3), torch.zeros(4, 69))
    state = copy.deepcopy(replay.state_dict())
    if corruption == "nan":
        state["history"][0, 0, 0] = float("nan")
    elif corruption == "counter":
        state["cursor"] = 2
    elif corruption == "dtype":
        state["history"] = state["history"].double()
    elif corruption == "action":
        state["actions"][0, 0] = 2.
    else:
        del state["actions"]
    with pytest.raises(ValueError):
        replay.load_state_dict(state)


def test_frozen_action_noise_does_not_change_effective_behavior_likelihood():
    trainer = PaperAMPTrainer(config())
    own, task = torch.randn(8, 4), torch.randn(8, 7)
    distribution, _ = trainer.model.policy(own, task)
    first = distribution.mean.detach().clone()
    changed = first.clone()
    changed[:, trainer.action_mask == 0] += 3.
    assert torch.equal(trainer._policy_log_prob(distribution, first), trainer._policy_log_prob(distribution, changed))
    changed[:, trainer.action_mask != 0] += .1
    assert not torch.equal(trainer._policy_log_prob(distribution, first), trainer._policy_log_prob(distribution, changed))


def test_independent_discriminator_schedule_and_historical_policy_replay():
    trainer = PaperAMPTrainer(config())
    expert = torch.randn(16, 10, 3)
    metrics = trainer.update(rollout(trainer), expert, torch.zeros(16, 69))
    assert metrics["discriminator_minibatches"] == 2
    assert metrics["optimizer_minibatches"] == 3
    assert metrics["disc_replay_samples"] == 0
    metrics = trainer.update(rollout(trainer), expert, torch.zeros(16, 69))
    assert metrics["disc_replay_samples"] == 4
    assert metrics["disc_live_samples"] == 4 and metrics["disc_expert_samples"] == 8
    assert metrics["replay_size"] == 8


def test_kl_guard_does_not_suppress_independent_discriminator_updates():
    trainer = PaperAMPTrainer(config(target_kl=1e-10, learning_rate=.01))
    metrics = trainer.update(rollout(trainer), torch.randn(16, 10, 3), torch.zeros(16, 69))
    assert metrics["policy_target_kl_stopped"]
    assert metrics["optimizer_minibatches"] == 1
    assert metrics["discriminator_minibatches"] == 2


def test_checkpoint_resumes_full_replay_and_its_torch_rng(tmp_path):
    trainer = PaperAMPTrainer(config())
    trainer.update(rollout(trainer), torch.randn(16, 10, 3), torch.zeros(16, 69))
    path = tmp_path / "full.pt"
    trainer.save(path, runner_state={"physical_control_step": 4})
    expected = trainer.policy_replay.sample(24)
    loaded, runner = PaperAMPTrainer.load(path)
    actual = loaded.policy_replay.sample(24)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
    assert runner["physical_control_step"] == 4
    assert loaded.config.effective_action_likelihood
    broken = torch.load(path, weights_only=True)
    del broken["policy_replay"]
    torch.save(broken, path)
    with pytest.raises(ValueError, match="complete saved replay"):
        PaperAMPTrainer.load(path)


def test_explicit_reward_weighting_and_default_paper_coefficients():
    amp, task = torch.tensor([2.]), {"task": torch.tensor([3.])}
    assert paper_total_reward(amp, task).item() == 3.5
    assert paper_total_reward(amp, task, amp_weight=2., task_weight=.5).item() == 5.5
    with pytest.raises(ValueError):
        paper_total_reward(amp, task, amp_weight=True)


@pytest.mark.parametrize("field,value", [("discriminator_replay_capacity", True),
    ("discriminator_updates_per_update", 0), ("warm_start_normalizer_count_cap", -1),
    ("discriminator_replay_fraction", 1.1), ("amp_reward_weight", float("nan")),
    ("effective_action_likelihood", 1), ("target_kl", True), ("motion_symmetry_coefficient", -1),
    ("expert_dof_velocity_frame", "exp_map_derivative")])
def test_style_settings_fail_before_sdk(field, value):
    with pytest.raises(ValueError):
        config(**{field: value}).validate()


def test_explicit_reference_velocity_frame_is_checkpointed_and_legacy_default_is_retained(tmp_path):
    trainer = PaperAMPTrainer(config(expert_dof_velocity_frame="child_joint"))
    path = tmp_path / "child.pt"
    trainer.save(path)
    loaded, _ = PaperAMPTrainer.load(path, restore_rng=False)
    assert loaded.config.expert_dof_velocity_frame == "child_joint"
    state = torch.load(path, weights_only=True)
    del state["config"]["expert_dof_velocity_frame"]
    torch.save(state, path)
    legacy, _ = PaperAMPTrainer.load(path, restore_rng=False)
    assert legacy.config.expert_dof_velocity_frame == "legacy_stored"


def test_explicit_curriculum_limits_and_invalid_preflight_args(tmp_path):
    args = parse_amp_args(["--trajectory-source", "pacer_curriculum", "--curriculum-speed-max-mps", "2",
                           "--curriculum-accel-max-mps2", "1.5", "--terrain-extent-m", "128", "--spawn-margin-m", "48",
                           "--output", str(tmp_path / "fresh")])
    assert args.curriculum_speed_max_mps == 2. and args.curriculum_accel_max_mps2 == 1.5
    with pytest.raises(SystemExit):
        parse_amp_args(["--curriculum-speed-max-mps", "nan"])


def test_segment_parent_identity_is_safe_across_splits(tmp_path):
    # Deliberately shallow ZIP-signature fixtures: never loaded as model/data.
    splits = {}
    signature = b"PK\x03\x04signature-only-not-motion-data"
    for name in ("train", "validation", "test"):
        bank = tmp_path / f"{name}.pt"
        bank.write_bytes(signature)
        parent = f"{name}/walk.npz"
        clips = [{"motion_id": 0, "source_path": parent + "#frames=0:250", "parent_source_path": parent}]
        if name == "train":
            clips.append({"motion_id": 1, "source_path": parent + "#frames=250:500", "parent_source_path": parent})
        splits[name] = dict(path=bank.name, bytes=len(signature), sha256=hashlib.sha256(signature).hexdigest(),
                            clip_count=len(clips), clips=clips)
    path = tmp_path / "manifest.json"
    def write():
        path.write_text(json.dumps({"schema": "navisaaclab.amass_reconstruction.v1", "splits": splits}))
    write()
    _, counts = _manifest_resources(path)
    assert counts["train"] == 2
    record = splits["test"]["clips"][0]
    record.update(source_path="train/walk.npz#frames=500:750", parent_source_path="train/walk.npz")
    write()
    with pytest.raises(ResourceError, match="identity leakage"):
        _manifest_resources(path)
    record["parent_source_path"] = "test/fake.npz"
    write()
    with pytest.raises(ResourceError, match="parent identity disagrees"):
        _manifest_resources(path)


@pytest.mark.needs_lfs
def test_public_moments_cap_preserves_means_variances_and_declares_changed_counts():
    checkpoint = Path(__file__).resolve().parents[1] / "data/pretrained_models/paper_references/pacer/output/release/pacer_no_shape/Humanoid.pth"
    if not checkpoint.exists():
        pytest.skip("Public PACER resource absent")
    options = dict(discriminator_input="state", amp_normalization="per_history_coordinate", warm_start_discriminator=True)
    original = PaperAMPTrainer(TrainingConfig(**options))
    capped = PaperAMPTrainer(TrainingConfig(**options, warm_start_normalizer_count_cap=200000.))
    for trainer in (original, capped):
        trainer.warm_start_pacer(checkpoint, -torch.ones(69), torch.ones(69))
    for key in ("self_normalizer", "task_normalizer", "amp_normalizer", "value_normalizer"):
        a, b = getattr(original, key), getattr(capped, key)
        assert torch.equal(a.mean, b.mean) and torch.equal(a.variance, b.variance)
        assert b.count == min(a.count, 200000.)
    assert capped.provenance["initialization"]["normalizer_count_cap"] == 200000.
