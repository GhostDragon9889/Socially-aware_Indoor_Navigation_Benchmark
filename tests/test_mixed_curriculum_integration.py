"""Real reset/runner methods over CPU tensors; these do not emulate PhysX."""

from dataclasses import asdict
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import MethodType, SimpleNamespace

import pytest
import torch

from CrowdSim.humanoid.application.configuration import load_amp_profile, parse_amp_args, validate_training_source
from CrowdSim.humanoid.application.settings import TrainingConfig
from CrowdSim.humanoid.training.trajectories import PacerTrajectoryConfig


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'configs/diffusion_amp/amp_style_mixed_amass_procedural.yaml'


def helper():
    path = ROOT / 'tests/test_amass_root_curriculum.py'
    spec = importlib.util.spec_from_file_location('mixed_existing_contract_helpers', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def argv(tmp_path, *extra):
    checkpoint = tmp_path / 'diffusion.ckpt'
    checkpoint.write_bytes(b'CPU existence contract only')
    return ['--output', str(tmp_path / 'fresh'), '--diffusion-checkpoint', str(checkpoint),
            '--trajectory-source', 'mixed_amass_procedural', '--num-envs', '4',
            '--terrain-extent-m', '192', '--spawn-margin-m', '48', '--episode-steps', '300',
            '--replan-steps', '15', '--synchronized-replanning', '--curriculum-mode', 'smooth',
            '--curriculum-speed-max-mps', '2', '--curriculum-accel-max-mps2', '1.5', *extra]


def test_same_core_configuration_and_explicit_procedural_baseline(tmp_path):
    old = load_amp_profile(ROOT / 'configs/diffusion_amp/amp_style_amass_backward_penalty.yaml')
    mixed = load_amp_profile(PROFILE)
    assert asdict(TrainingConfig(**old['training'])) == asdict(TrainingConfig(**mixed['training']))
    assert mixed['training']['expert_sampling'] is None
    args = parse_amp_args(argv(tmp_path))
    validate_training_source(args, mixed)
    assert args.curriculum_speed_max_mps == 2 and args.curriculum_accel_max_mps2 == 1.5


@pytest.mark.parametrize('extra', [('--num-envs', '3'), ('--num-envs', '1'),
    ('--evaluate-steps', '10', '--resume', 'actor.pt'), ('--compile-diffusion-forward',),
    ('--guidance-weight', '20'), ('--diffusion-guidance-mode', 'trace_clean')])
def test_invalid_mixed_cli_fails_before_sdk(tmp_path, extra):
    with pytest.raises(SystemExit) as error:
        parse_amp_args(argv(tmp_path, *extra))
    assert error.value.code == 2


@pytest.mark.parametrize('extra', [('--curriculum-speed-max-mps', '3'),
    ('--curriculum-accel-max-mps2', '2'), ('--episode-steps', '200'), ('--replan-steps', '10'),
    ('--terrain-extent-m', '200'), ('--spawn-margin-m', '60')])
def test_declared_recipe_refuses_silent_baseline_changes(tmp_path, extra):
    args = parse_amp_args(argv(tmp_path, *extra))
    with pytest.raises(ValueError):
        validate_training_source(args, load_amp_profile(PROFILE))


def test_missing_declaration_and_wrong_fixed_assignment_rejected(tmp_path):
    args = parse_amp_args(argv(tmp_path))
    with pytest.raises(ValueError, match='explicit profile'):
        validate_training_source(args, {})
    profile = load_amp_profile(PROFILE)
    profile['trajectory_training']['assignment'] = 'per_reset_bernoulli'
    with pytest.raises(ValueError, match='fixed half'):
        validate_training_source(args, profile)


def test_profile_preflight_imports_no_torch_or_sdk(tmp_path):
    code = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
from CrowdSim.humanoid.application.configuration import parse_amp_args, load_amp_profile, validate_training_source
validate_training_source(parse_amp_args({argv(tmp_path)!r}),load_amp_profile({str(PROFILE)!r}))
assert not any(n=='torch' or n.startswith(('isaacsim','isaaclab','omni','pxr')) for n in sys.modules)
"""
    subprocess.run([sys.executable, '-I', '-c', code], capture_output=True, text=True, check=True)


def mixed_world(tmp_path):
    from CrowdSim.humanoid.training.mixed_trajectories import MixedTrajectoryCurriculum
    h = helper()
    corpus = h.corpus(tmp_path, counts=(91, 169))
    state = corpus.state_at(torch.tensor([9, 9, 9, 9]))
    state.dof_forces = torch.zeros(4, 69)
    writes, actions = [], []

    def reset(values, env_ids):
        writes.append(env_ids.clone())
        for name in ('root_pos', 'root_rot', 'root_vel', 'root_ang_vel', 'dof_pos', 'dof_vel'):
            getattr(state, name)[env_ids] = getattr(values, name)

    def step(value):
        actions.append(value.clone())

    world = SimpleNamespace(corpus=corpus, device=torch.device('cpu'), num_envs=4, history_frames=10,
        episode_steps=300, origins=torch.tensor([[48., 48.], [48., 52.], [52., 48.], [52., 52.]]),
        lower=torch.zeros(69), upper=torch.ones(69), mapping=SimpleNamespace(dof_names=[]),
        simulator=SimpleNamespace(reset_envs=reset, step=step), age=torch.zeros(4, dtype=torch.long),
        episode_ids=torch.zeros(4, dtype=torch.long), reference_ids=torch.zeros(4, dtype=torch.long),
        amp_history=torch.zeros(4, 10, 195), amp_history_episode_ids=torch.zeros(4, 10, dtype=torch.long),
        planar_history=torch.zeros(4, 91, 5), history_available=torch.zeros(4, 91, dtype=torch.bool),
        ground_height_at=lambda xy: torch.zeros(xy.shape[:-1]), state=lambda: state)
    world.mixed_trajectory_curriculum = MixedTrajectoryCurriculum(corpus, 4, bounds_xy=(0., 0., 192., 192.),
        tail_policy='linear_velocity', procedural_config=PacerTrajectoryConfig(speed_max_mps=2.,
            scalar_accel_max_mps2=1.5, vector_accel_max_mps2=1.5))
    for name, method in h.world_methods(ROOT / 'CrowdSim/humanoid/amp_world.py').items():
        if name in ('reset', 'step', 'planar'):
            setattr(world, name, MethodType(method, world))
    return world, writes, actions


def test_real_world_methods_reset_subset_and_clipend_only_source_group(tmp_path):
    world, writes, actions = mixed_world(tmp_path)
    all_ids = torch.arange(4)
    world.reset(all_ids)
    curriculum = world.mixed_trajectory_curriculum
    for _ in range(310):
        _, fallen, truncated = world.step(torch.zeros(4, 69))
        assert not fallen.any()
        assert not bool((truncated[2:] & (world.age[2:] < 300)).any())
        reset_ids = torch.nonzero(truncated).flatten()
        if len(reset_ids):
            unchanged = torch.ones(4, dtype=torch.bool)
            unchanged[reset_ids] = False
            old_ages = world.age.clone()
            old_episodes = world.episode_ids.clone()
            world.reset(reset_ids)
            assert torch.equal(world.age[unchanged], old_ages[unchanged])
            assert torch.equal(world.episode_ids[unchanged], old_episodes[unchanged])
    stats = curriculum.describe()['statistics']
    assert len(actions) == 310 and all(len(value) == 4 for value in actions)
    assert sum(len(value) for value in writes) > 4
    assert stats['amass_actor_transitions'] == stats['procedural_actor_transitions'] == 620
    assert world.episode_ids[2:].tolist() == [2, 2]
    assert bool((world.episode_ids[:2] > 2).all())


def test_actual_runner_replan_reward_and_future_targets_preserve_branch_phases(tmp_path):
    h = helper()
    world, _, _ = mixed_world(tmp_path)
    world.reset(torch.arange(4))
    curriculum = world.mixed_trajectory_curriculum
    namespace = dict(torch=torch, args=SimpleNamespace(trajectory_source='mixed_amass_procedural'),
        world=world, curriculum=curriculum, plans=torch.zeros(4, 151, 2), goals=torch.zeros(4, 2),
        plan_age=torch.full((4,), 15), plan_frequency_hz=30, replan_count=0, planner_forward_calls=0)
    replan = h.runner_function('replan', namespace)
    replan(torch.tensor([3, 0, 2, 1]))
    assert namespace['plan_age'].tolist() == [0]*4
    assert namespace['replan_count'] == 4 and namespace['planner_forward_calls'] == 0
    interpolate = h.runner_function('interpolate', namespace)
    targets = h.runner_function('task_trajectory', namespace)
    source = curriculum.amass
    for step in range(1, 15):
        world.step(torch.zeros(4, 69))
        namespace['plan_age'] += 1
        mid = source.motion_ids
        expected = world.corpus.data['gts'][world.corpus.data['length_starts'][mid]+source.initial_frames+step, 0, :2]+source.offset_xy
        assert torch.equal(interpolate(torch.full((4, 1), step/30))[:2, 0], expected)
        actual = targets()
        assert actual.shape == (4, 10, 2)
        assert torch.equal(actual[:2], source.future_targets(world.age[:2], episode_ids=world.episode_ids[:2]))
    rngs = (world.corpus.generator.get_state().clone(), source.corpus.generator.get_state().clone(), curriculum.generator.get_state().clone())
    targets()
    assert torch.equal(rngs[0], world.corpus.generator.get_state())
    assert torch.equal(rngs[1], source.corpus.generator.get_state())
    assert torch.equal(rngs[2], curriculum.generator.get_state())


def test_mapped_checkpoint_restores_exact_three_cpu_rng_buffers_without_mutating_payload(tmp_path):
    spec = importlib.util.spec_from_file_location('mixed_real_driver', ROOT / 'scripts/train_paper_amp.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    world, _, _ = mixed_world(tmp_path)
    world.reset(torch.arange(4))
    curriculum = world.mixed_trajectory_curriculum
    state = curriculum.state_dict()
    moved = []

    class MappedGeneratorBuffer:
        """Stand-in for deserializer device transport, never a simulated GPU."""
        def __init__(self, tensor, name):
            self.tensor, self.name = tensor, name

        def cpu(self):
            moved.append(self.name)
            return self.tensor.clone()

    state['generator_state'] = MappedGeneratorBuffer(state['generator_state'], 'top')
    for name in ('amass_state', 'procedural_state'):
        state[name]['generator_state'] = MappedGeneratorBuffer(state[name]['generator_state'], name)
    restored = driver.curriculum_rng_to_cpu(state, mixed=True)
    assert moved == ['top', 'amass_state', 'procedural_state']
    assert isinstance(state['generator_state'], MappedGeneratorBuffer)
    assert all(isinstance(state[name]['generator_state'], MappedGeneratorBuffer) for name in ('amass_state', 'procedural_state'))
    curriculum.load_state_dict(restored)  # Actual strict child contracts accept the result.
    assert restored['config'] is state['config']
    assert restored['procedural_actors'] is state['procedural_actors']
    missing = dict(state)
    del missing['amass_state']
    with pytest.raises(KeyError):
        driver.curriculum_rng_to_cpu(missing, mixed=True)


def test_report_and_resume_group_budget_cannot_disagree_with_committed_trainer_clock(tmp_path):
    spec = importlib.util.spec_from_file_location('mixed_budget_driver', ROOT / 'scripts/train_paper_amp.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    world, _, _ = mixed_world(tmp_path)
    world.reset(torch.arange(4))
    for _ in range(4):
        world.step(torch.zeros(4, 69))
    curriculum = world.mixed_trajectory_curriculum
    driver.validate_mixed_transition_exposure(curriculum, 16, 4, 4)
    with pytest.raises(ValueError, match='committed trainer budget'):
        driver.validate_mixed_transition_exposure(curriculum, 20, 5, 4)
    with pytest.raises(ValueError, match='exact trainer/control clock'):
        driver.validate_mixed_transition_exposure(curriculum, 18, 4, 4)
    curriculum.statistics['procedural_actor_transitions'] += 1
    with pytest.raises(ValueError, match='committed trainer budget'):
        driver.validate_mixed_transition_exposure(curriculum, 16, 4, 4)
