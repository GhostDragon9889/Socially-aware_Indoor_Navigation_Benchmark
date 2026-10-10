"""CPU integration of opt-in RMS protection; synthetic rollouts are explicit."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from CrowdSim.humanoid.application.resources import ResourceError, preflight_amp_resources
from CrowdSim.humanoid.application.settings import TrainingConfig
from CrowdSim.humanoid.training.terrain_statistics import PUBLIC_PACER_SHA256
from CrowdSim.humanoid.training.trainer import PaperAMPTrainer, RunningNormalizer


ROOT = Path(os.environ.get("RMS_SOURCE_ROOT", Path(__file__).resolve().parents[1]))
PUBLIC = ROOT / "data/pretrained_models/paper_references/pacer/output/release/pacer_no_shape/Humanoid.pth"


def compact_config(**overrides):
    options = dict(self_dim=357, task_dim=1044, amp_frame_dim=195,
                   task_layers=(12, 6), policy_layers=(16, 8), discriminator_layers=(16, 8),
                   update_epochs=1, minibatch_size=4, discriminator_batch_size=4,
                   warm_start_normalizer_count_cap=200000.,
                   task_terrain_statistics_mode="preserve_declared_suffix")
    options.update(overrides)
    return TrainingConfig(**options)


def bind_numerical_fixture(trainer):
    """Not a public model transfer: small network fixture for corruption tests."""
    initialization = dict(mode="transfer_warm_start_public_pacer_actor_critic",
                          path=str(PUBLIC), sha256=PUBLIC_PACER_SHA256,
                          actor_critic_task_transferred=True,
                          observation_value_moments_transferred=True,
                          normalizer_count_cap=200000.,
                          normalizer_counts_before_cap={"task_normalizer": 61931520001.})
    trainer.provenance["initialization"] = initialization
    trainer.task_normalizer.mean.copy_(torch.linspace(-1, 1, 1044, dtype=torch.float64))
    trainer.task_normalizer.variance.copy_(torch.linspace(.1, 3, 1044, dtype=torch.float64))
    trainer.task_normalizer.count.fill_(200000.)
    trainer.terrain_statistics_policy.bind_public_warm_start(trainer.task_normalizer,
        initialization=initialization, reference_count=61931520001., warm_start_task_count=200000.)


def rollout(trainer):
    own = torch.randn(2, 2, 357)
    task = torch.randn(2, 2, 1044)
    task[..., 20:] = 0  # declared flat-terrain synthetic observation
    decision = trainer.act(own.flatten(0, 1), task.flatten(0, 1))
    data = {key: decision[key].reshape(2, 2, -1).squeeze(-1)
            for key in ("actions", "applied_actions", "log_prob", "values")}
    data.update(self_obs=own, task_obs=task, amp_history=torch.randn(2, 2, 10, 195),
                next_values=torch.zeros(2, 2), rewards=torch.tensor([[.1, .2], [.3, .4]]),
                terminated=torch.zeros(2, 2, dtype=torch.bool),
                truncated=torch.zeros(2, 2, dtype=torch.bool))
    return data


def experts():
    return torch.randn(4, 10, 195), torch.zeros(4, 69)


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("value", [True, None, "", "unknown"])
def test_mode_rejects_nonexplicit_values(value):
    with pytest.raises(ValueError, match="task_terrain_statistics_mode"):
        TrainingConfig(task_terrain_statistics_mode=value).validate()


def test_unbound_cannot_act_update_or_publish_checkpoint(tmp_path):
    trainer = PaperAMPTrainer(compact_config())
    for operation in (lambda: trainer.act(torch.zeros(1, 357), torch.zeros(1, 1044)),
                      lambda: trainer.update({}, torch.zeros(1, 10, 195), torch.zeros(1, 69)),
                      lambda: trainer.save(tmp_path / "invalid.pt")):
        with pytest.raises(ValueError, match="unbound"):
            operation()
    assert not (tmp_path / "invalid.pt").exists()
    assert not trainer.optimizer.state and not trainer.discriminator_optimizer.state


def test_cold_frontend_rejects_from_scratch_before_resource_reads():
    args = SimpleNamespace(seed=7, from_scratch=True, trajectory_source="diffusion")
    profile = {"training": {"task_terrain_statistics_mode": "preserve_declared_suffix"}}
    with pytest.raises(ResourceError, match="from-scratch"):
        preflight_amp_resources(args, profile)


def test_real_update_preserves_suffix_and_original_prefix_count_math():
    trainer = PaperAMPTrainer(compact_config())
    bind_numerical_fixture(trainer)
    initial = {key: value.clone() for key, value in trainer.task_normalizer.state_dict().items()}
    data = rollout(trainer)
    ordinary = RunningNormalizer(1044)
    ordinary.load_state_dict(initial)
    ordinary.update(data["task_obs"].flatten(0, 1))
    weights = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    trainer.update(data, *experts())
    assert torch.equal(trainer.task_normalizer.mean[:20], ordinary.mean[:20])
    assert torch.equal(trainer.task_normalizer.variance[:20], ordinary.variance[:20])
    assert torch.equal(trainer.task_normalizer.count, ordinary.count)
    assert torch.equal(trainer.task_normalizer.mean[20:], initial["mean"][20:])
    assert torch.equal(trainer.task_normalizer.variance[20:], initial["variance"][20:])
    assert trainer.task_normalizer.count == 200004
    assert trainer.terrain_statistics_policy.post_update_calls == trainer.update_count == 1
    assert trainer.physical_transition_count == 4
    assert any(not torch.equal(value, weights[key]) for key, value in trainer.model.state_dict().items())


def test_complete_resume_next_optimizer_update_and_rng_are_bitwise(tmp_path):
    trainer = PaperAMPTrainer(compact_config())
    bind_numerical_fixture(trainer)
    trainer.update(rollout(trainer), *experts())
    path = tmp_path / "protected.pt"
    trainer.save(path, runner_state={"physical_counter": 4})
    expected_rng = (random.random(), np.random.random(), torch.rand(3))
    saved = torch.load(path, weights_only=True)
    restored, physical = PaperAMPTrainer.load(path)
    assert physical == {"physical_counter": 4}
    actual_rng = (random.random(), np.random.random(), torch.rand(3))
    assert expected_rng[:2] == actual_rng[:2]
    assert torch.equal(expected_rng[2], actual_rng[2])
    assert_tree_equal(trainer.optimizer.state_dict(), restored.optimizer.state_dict())
    assert_tree_equal(trainer.discriminator_optimizer.state_dict(), restored.discriminator_optimizer.state_dict())
    assert_tree_equal(saved["task_terrain_statistics_policy"], restored.terrain_statistics_policy.state_dict())
    next_rollout, next_expert = rollout(trainer), experts()
    rng = torch.get_rng_state()
    expected_metrics = trainer.update(next_rollout, *next_expert)
    torch.set_rng_state(rng)
    assert restored.update(next_rollout, *next_expert) == expected_metrics
    assert_tree_equal(trainer.model.state_dict(), restored.model.state_dict())
    assert_tree_equal(trainer.optimizer.state_dict(), restored.optimizer.state_dict())
    assert_tree_equal(trainer.task_normalizer.state_dict(), restored.task_normalizer.state_dict())


@pytest.mark.parametrize("corruption", ["missing", "extra", "calls", "source", "count", "moments", "mode"])
def test_resume_rejects_saved_policy_corruption_without_replacing_statistics(tmp_path, corruption):
    trainer = PaperAMPTrainer(compact_config())
    bind_numerical_fixture(trainer)
    path = tmp_path / "corrupt.pt"
    trainer.save(path)
    state = torch.load(path, weights_only=True)
    policy = state["task_terrain_statistics_policy"]
    if corruption == "missing":
        del state["task_terrain_statistics_policy"]
    elif corruption == "extra":
        policy["extra"] = None
    elif corruption == "calls":
        policy["post_update_calls"] = 1
    elif corruption == "source":
        state["provenance"]["initialization"]["sha256"] = "0" * 64
    elif corruption == "count":
        state["normalizers"]["task_normalizer"]["count"] += 1
    elif corruption == "moments":
        state["normalizers"]["task_normalizer"]["mean"][20] += 1
    else:
        state["config"]["task_terrain_statistics_mode"] = "legacy"
    torch.save(state, path)
    with pytest.raises(ValueError):
        PaperAMPTrainer.load(path, restore_rng=False)


def test_old_missing_mode_and_policy_loads_as_legacy_without_new_state(tmp_path):
    trainer = PaperAMPTrainer(compact_config(task_terrain_statistics_mode="legacy"))
    path = tmp_path / "old.pt"
    trainer.save(path)
    state = torch.load(path, weights_only=True)
    assert "task_terrain_statistics_policy" not in state
    del state["config"]["task_terrain_statistics_mode"]
    torch.save(state, path)
    restored, _ = PaperAMPTrainer.load(path, restore_rng=False)
    assert restored.config.task_terrain_statistics_mode == "legacy"
    assert restored.terrain_statistics_policy.state_dict() is None
    assert_tree_equal(trainer.model.state_dict(), restored.model.state_dict())


def test_policy_binding_hooks_validation_do_not_consume_caller_rng():
    trainer = PaperAMPTrainer(compact_config())
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    bind_numerical_fixture(trainer)
    trainer.terrain_statistics_policy.validate_checkpoint(trainer.task_normalizer,
        update_count=0, physical_transition_count=0)
    trainer.terrain_statistics_policy.state_dict()
    assert random.getstate() == before[0]
    current_numpy = np.random.get_state()
    assert current_numpy[0] == before[1][0]
    assert np.array_equal(current_numpy[1], before[1][1])
    assert current_numpy[2:] == before[1][2:]
    assert torch.equal(torch.get_rng_state(), before[2])
    assert not torch.cuda.is_initialized()


def test_wrong_public_source_refuses_before_any_weights_or_moments_copy(monkeypatch):
    from CrowdSim.humanoid.amp.controller import PacerAMPController
    trainer = PaperAMPTrainer(TrainingConfig(task_terrain_statistics_mode="preserve_declared_suffix"))
    weights = deepcopy(trainer.model.state_dict())
    moments = deepcopy(trainer.task_normalizer.state_dict())
    monkeypatch.setattr(PacerAMPController, "from_checkpoint",
        lambda *args, **kwargs: SimpleNamespace(checkpoint_sha256="0" * 64))
    with pytest.raises(ValueError, match="checkpoint SHA"):
        trainer.warm_start_pacer("explicit_synthetic_source.pth", -torch.ones(69), torch.ones(69))
    assert_tree_equal(weights, trainer.model.state_dict())
    assert_tree_equal(moments, trainer.task_normalizer.state_dict())
    assert not trainer.terrain_statistics_policy.bound


@pytest.mark.needs_lfs
def test_actual_public_warm_start_binds_original_count_and_repeat_refuses_before_copy(tmp_path):
    if not PUBLIC.is_file() or PUBLIC.stat().st_size < 10000:
        pytest.skip("Complete verified public PACER reference is not installed")
    config = TrainingConfig(task_terrain_statistics_mode="preserve_declared_suffix",
                            warm_start_normalizer_count_cap=200000.)
    trainer = PaperAMPTrainer(config)
    trainer.warm_start_pacer(PUBLIC, -torch.ones(69), torch.ones(69))
    policy = trainer.terrain_statistics_policy.describe()
    assert policy["reference_count"] == 61931520001.
    assert policy["warm_start_task_count"] == 200000.
    assert policy["reference_initialization"]["sha256"] == PUBLIC_PACER_SHA256
    before = {key: value.clone() for key, value in trainer.model.state_dict().items()}
    moments = deepcopy(trainer.task_normalizer.state_dict())
    # No public file is even opened: the protected phase guard executes first.
    with pytest.raises(ValueError, match="cannot be repeated"):
        trainer.warm_start_pacer(tmp_path / "does_not_exist.pth", -torch.ones(69), torch.ones(69))
    assert_tree_equal(before, trainer.model.state_dict())
    assert_tree_equal(moments, trainer.task_normalizer.state_dict())
    path = tmp_path / "public_protected.pt"
    trainer.save(path)
    from CrowdSim.humanoid.runtime.amp_inference import ReconstructionAMPInference
    rng = torch.get_rng_state()
    inference = ReconstructionAMPInference.from_checkpoint(path)
    assert torch.equal(rng, torch.get_rng_state())
    own, task = torch.randn(2, 357), torch.randn(2, 1044)
    old = trainer.act(own, task, deterministic=True)
    result = inference.act(own, task)
    assert torch.equal(old["actions"], result["actions"])
    assert torch.equal(old["applied_actions"], result["applied_actions"])
    assert set(inference.model._modules) == {"actor", "task_encoder"}
