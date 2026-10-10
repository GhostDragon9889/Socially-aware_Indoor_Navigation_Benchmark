"""Paired shared RNG, actual expert payload and full checkpoint integration."""

import ast
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import yaml

from CrowdSim.humanoid.amass_corpus import AMASSCorpus
from CrowdSim.humanoid.amp import action_to_pd, build_amp_observation
from CrowdSim.humanoid.amp.observations import PacerSMPLMapping
from CrowdSim.humanoid.application.settings import TrainingConfig, normalize_expert_sampling_config
from CrowdSim.humanoid.training.expert_sampling import ExpertReferenceSampler, STRATA
from CrowdSim.humanoid.training.trainer import PaperAMPTrainer


ROOT = Path(__file__).resolve().parents[1]


def sampling_config(**options):
    config = {"mode": "stratified_windows", "seed": 7,
              "target_weights": {name: float(name == "fast") for name in STRATA},
              "minimum_parents_per_bucket": 1, "minimum_windows_per_bucket": 1}
    config.update(options)
    return config


def corpus_file(tmp_path):
    chunks, rows, starts, files = [], [], [], []
    total = 0
    for index, speed in enumerate((0., .8, 1., 1., 2.2, .8, 2.5)):
        frames = 11 if index == 4 else 35+index
        yaw = torch.arange(frames)*(.04 if index == 2 else 0.)
        velocity = torch.zeros(frames, 24, 3)
        velocity[:, 0, 0] = speed*yaw.cos()
        velocity[:, 0, 1] = speed*yaw.sin()
        if index == 3:
            velocity[:, 0, 0], velocity[:, 0, 1] = 0., speed
        position = torch.zeros(frames, 24, 3)
        position[:, :, 2] = .9
        position[:, 0, :2] = velocity[:, 0, :2]*torch.arange(frames)[:, None]/30
        rotation = torch.zeros(frames, 24, 4)
        rotation[..., 3] = 1
        rotation[:, 0, 2], rotation[:, 0, 3] = (yaw/2).sin(), (yaw/2).cos()
        dof = (1000*index+torch.arange(frames)).float()[:, None].expand(-1, 69).clone()/10000
        chunks.append({"gts": position, "grs": rotation, "gvs": velocity,
                       "gavs": torch.zeros_like(velocity), "dps": dof, "dvs": torch.zeros_like(dof)})
        source = f"/processed/smpl/{'A' if index % 2 else 'B'}/subject/motion{index}.motion"
        rows.append({"motion_id": index, "source_path": source, "frames": frames, "dt": 1/30, "diffusion_eligible": False})
        starts.append(total)
        files.append(source)
        total += frames
    bank = {key: torch.cat([chunk[key] for chunk in chunks]) for key in chunks[0]}
    bank.update(length_starts=torch.tensor(starts), motion_num_frames=torch.tensor([row["frames"] for row in rows]),
                motion_dt=torch.full((len(rows),), 1/30), motion_files=files, dof_velocity_frame="child_joint")
    path = tmp_path/"bank.pt"
    torch.save(bank, path)
    manifest = {"schema": "navisaaclab.amass_reconstruction.v1", "splits": {
        "train": {"path": str(path), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "dof_velocity_frame": "child_joint", "clips": rows},
        "validation": {"clips": []}, "test": {"clips": []}}}
    target = tmp_path/"manifest.json"
    target.write_text(json.dumps(manifest))
    return target


def new_corpus(path, seed=29):
    return AMASSCorpus(path, device="cpu", seed=seed, dof_velocity_frame="child_joint")


def assert_reference_equal(left, right):
    for a, b in zip(left[:2], right[:2], strict=True):
        assert vars(a).keys() == vars(b).keys()
        for key in vars(a):
            assert torch.equal(getattr(a, key), getattr(b, key)), key
    for a, b in zip(left[2:], right[2:], strict=True):
        assert torch.equal(a, b)


def world_methods():
    # Execute the owned actual methods, without importing an SDK/creating Kit.
    tree = ast.parse((ROOT/"CrowdSim/humanoid/amp_world.py").read_text())
    world = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AMPWorld")
    names = {"expert_batch", "configure_expert_sampling", "expert_sampling_state_dict", "load_expert_sampling_state_dict"}
    world.body = [node for node in world.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(body=[world], type_ignores=[])
    namespace = {"torch": torch, "build_amp_observation": build_amp_observation, "action_to_pd": action_to_pd}
    exec(compile(module, str(ROOT/"CrowdSim/humanoid/amp_world.py"), "exec"), namespace)
    return namespace["AMPWorld"]


def make_world(corpus):
    cls = world_methods()
    world = cls.__new__(cls)
    world.corpus, world.device, world.history_frames = corpus, torch.device("cpu"), 10
    world.mapping = PacerSMPLMapping()
    world.lower, world.upper = torch.full((69,), -torch.pi), torch.full((69,), torch.pi)
    return world


def test_interleaved_reset_draws_and_global_rng_are_exactly_preserved(tmp_path):
    path = corpus_file(tmp_path)
    legacy, replacement = new_corpus(path), new_corpus(path)
    global_before = torch.get_rng_state().clone()
    bridge = ExpertReferenceSampler(replacement, sampling_config())
    assert torch.equal(legacy.generator.get_state(), replacement.generator.get_state())
    changed = 0
    for reset_batch, expert_batch in ((1, 7), (5, 123), (2, 1), (9, 61), (1, 300)):
        assert_reference_equal(legacy.sample_reference(reset_batch), replacement.sample_reference(reset_batch))
        original = legacy.sample_reference(expert_batch)
        actual = bridge.sample_reference(expert_batch)
        changed += int(not torch.equal(original[3], actual[3]))
        assert set(actual[3].tolist()) <= {4, 6}
        assert torch.equal(legacy.generator.get_state(), replacement.generator.get_state())
    assert changed > 0
    assert torch.equal(global_before, torch.get_rng_state())
    assert_reference_equal(legacy.sample_reference(100), replacement.sample_reference(100))


def test_replacement_payload_matches_actual_sampler_history_and_next_indices(tmp_path):
    corpus = new_corpus(corpus_file(tmp_path))
    bridge = ExpertReferenceSampler(corpus, sampling_config())
    before = bridge.sampler.state_dict()
    actual = bridge.sample_reference(512)
    after = bridge.sampler.state_dict()
    bridge.sampler.load_state_dict(before)
    selected = bridge.sampler.sample(512)
    assert_reference_equal(actual, (corpus.state_at(selected.absolute_indices),
        corpus.state_at(selected.history_indices.reshape(-1)), corpus.data["dps"][selected.next_indices],
        selected.motion_ids, selected.frame_indices))
    assert torch.equal(bridge.sampler.generator.get_state(), after["generator_state"])
    short = selected.motion_ids == 4
    assert short.any() and torch.all(selected.frame_indices[short] == 9)
    assert torch.all(selected.next_indices < corpus.data["length_starts"][selected.motion_ids]+corpus.data["motion_num_frames"][selected.motion_ids])


def test_actual_world_disabled_branch_and_enabled_effective_expert_features(tmp_path):
    path = corpus_file(tmp_path)
    baseline, world = new_corpus(path), make_world(new_corpus(path))
    _, history, _, _, _ = baseline.sample_reference(17)
    actual, _ = world.expert_batch(17)
    assert torch.equal(actual, build_amp_observation(history, mapping=world.mapping).reshape(17, 10, 195))
    assert torch.equal(baseline.generator.get_state(), world.corpus.generator.get_state())
    assert world.expert_sampling_state_dict() is None
    world.load_expert_sampling_state_dict(None)
    with pytest.raises(ValueError, match="disabled"):
        world.load_expert_sampling_state_dict({})
    world.configure_expert_sampling(sampling_config())
    source_before = world.expert_reference_sampler.state_dict()
    features, actions = world.expert_batch(100)
    world.expert_reference_sampler.load_state_dict(source_before)
    _, replacement_history, _, _, _ = world.expert_reference_sampler.sample_reference(100)
    assert torch.equal(features, build_amp_observation(replacement_history, mapping=world.mapping).reshape(100, 10, 195))
    assert actions.shape == (100, 69) and torch.isfinite(actions).all()
    frozen = [i for i, name in enumerate(world.mapping.dof_names) if name.rsplit("_", 1)[0] in {"L_Hand", "R_Hand", "L_Toe", "R_Toe"}]
    assert torch.all(actions[:, frozen] == 0)
    with pytest.raises(ValueError, match="already configured"):
        world.configure_expert_sampling(sampling_config())
    with pytest.raises(ValueError, match="complete saved state"):
        world.load_expert_sampling_state_dict(None)


def test_full_trainer_checkpoint_resumes_experts_resets_and_private_rng(tmp_path):
    path = corpus_file(tmp_path)
    corpus = new_corpus(path)
    world = make_world(corpus)
    config = sampling_config()
    world.configure_expert_sampling(config)
    trainer = PaperAMPTrainer(TrainingConfig(self_dim=4, task_dim=7, amp_frame_dim=195,
        task_layers=(12, 6), policy_layers=(16, 8), discriminator_layers=(16, 8),
        update_epochs=1, minibatch_size=8, discriminator_batch_size=8, discriminator_input="state",
        expert_sampling=config))
    expert_history, expert_actions = world.expert_batch(8)
    own, task = torch.randn(8, 4), torch.randn(8, 7)
    act = trainer.act(own, task)
    rollout = {key: act[key].reshape(4, 2, -1).squeeze(-1) for key in ("actions", "applied_actions", "log_prob", "values")}
    rollout.update(self_obs=own.reshape(4, 2, 4), task_obs=task.reshape(4, 2, 7),
        amp_history=expert_history.reshape(4, 2, 10, 195), rewards=torch.ones(4, 2), next_values=torch.zeros(4, 2),
        terminated=torch.zeros(4, 2, dtype=torch.bool), truncated=torch.zeros(4, 2, dtype=torch.bool))
    trainer.update(rollout, expert_history, expert_actions)
    checkpoint = tmp_path/"checkpoint.pt"
    trainer.save(checkpoint, runner_state={"corpus_rng": corpus.generator.get_state(), "expert_sampling": world.expert_sampling_state_dict()})
    expected_experts = world.expert_batch(111)
    expected_resets = corpus.sample_reference(61)
    expected_global = torch.rand(9)
    loaded, saved = PaperAMPTrainer.load(checkpoint)
    restored = make_world(new_corpus(path, seed=100))
    restored.configure_expert_sampling(loaded.config.expert_sampling)
    restored.corpus.generator.set_state(saved["corpus_rng"])
    restored.load_expert_sampling_state_dict(saved["expert_sampling"])
    actual_experts = restored.expert_batch(111)
    assert all(torch.equal(a, b) for a, b in zip(actual_experts, expected_experts, strict=True))
    assert_reference_equal(expected_resets, restored.corpus.sample_reference(61))
    assert torch.equal(expected_global, torch.rand(9))
    assert loaded.update_count == trainer.update_count == 1
    assert loaded.optimizer.state_dict()["state"]
    assert normalize_expert_sampling_config(loaded.config.expert_sampling) == normalize_expert_sampling_config(config)


@pytest.mark.parametrize("change", ["configuration", "registry", "probabilities", "missing", "counters", "literal_config"])
def test_enabled_resume_rejects_corrupted_or_changed_contract(tmp_path, change):
    bridge = ExpertReferenceSampler(new_corpus(corpus_file(tmp_path)), sampling_config())
    state = copy.deepcopy(bridge.state_dict())
    if change == "configuration":
        state["configuration"]["seed"] += 1
    elif change == "registry":
        state["sampler"]["registry_sha256"] = "0"*64
    elif change == "probabilities":
        state["sampler"]["probabilities_sha256"] = "0"*64
    elif change == "missing":
        del state["sampler"]["config"]
    elif change == "counters":
        state["sampler"].update(sampled_windows=0, sampled_batches=1)
    else:
        state["sampler"]["config"]["target_weights"]["fast"] = True
    with pytest.raises(ValueError):
        bridge.load_state_dict(state)


@pytest.mark.parametrize("options", [{"mode": "unknown"}, {"seed": True}, {"dataset_cap": True},
    {"minimum_windows_per_bucket": True}, {"thresholds": {"net_turn_angle_rad": float("nan")}},
    {"target_weights": {name: True for name in STRATA}}, {"unknown": 1}, {"dataset_weighting": []}])
def test_nested_sampling_invalid_config_fails_without_sdk(options):
    with pytest.raises(ValueError):
        TrainingConfig(expert_sampling=sampling_config(**options)).validate()


def test_settings_remains_torch_and_sdk_free():
    code = f"import sys; sys.path.insert(0, {str(ROOT)!r}); from CrowdSim.humanoid.application.settings import TrainingConfig; TrainingConfig().validate(); assert 'torch' not in sys.modules; assert not any(k.startswith('omni.') for k in sys.modules)"
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_new_profile_changes_only_expert_prior_and_old_checkpoint_default(tmp_path):
    base = yaml.safe_load((ROOT/"configs/diffusion_amp/amp_style_amass_backward_penalty.yaml").read_text())
    opted = yaml.safe_load((ROOT/"configs/diffusion_amp/amp_style_amass_stratified_experts.yaml").read_text())
    difference = {key for key in opted["training"] if base["training"].get(key) != opted["training"][key]}
    assert difference == {"expert_sampling"}
    assert base["initialization"] == opted["initialization"]
    assert base["data_manifest"] == opted["data_manifest"]
    TrainingConfig(**opted["training"]).validate()
    trainer = PaperAMPTrainer(TrainingConfig(self_dim=4, task_dim=7, amp_frame_dim=3,
        task_layers=(12, 6), policy_layers=(16, 8), discriminator_layers=(16, 8)))
    target = tmp_path/"old.pt"
    trainer.save(target, runner_state={"corpus_rng": torch.Generator().get_state()})
    state = torch.load(target, weights_only=True)
    del state["config"]["expert_sampling"]
    torch.save(state, target)
    loaded, saved = PaperAMPTrainer.load(target, restore_rng=False)
    assert loaded.config.expert_sampling is None and "expert_sampling" not in saved
    assert asdict(loaded.config)["expert_sampling"] is None
