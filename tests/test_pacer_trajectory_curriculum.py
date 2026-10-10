"""CPU command bounds, independent source parity and deterministic continuation."""
import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from CrowdSim.humanoid.training.trajectories import PacerTrajectoryConfig, PacerTrajectoryCurriculum


def state(batch=16, speed=1.5):
    position = torch.zeros(batch, 3)
    position[:, :2] = torch.arange(batch)[:, None].float()
    position[:, 2] = .9
    velocity = torch.zeros(batch, 3)
    velocity[:, 0] = speed
    rotation = torch.zeros(batch, 4)
    rotation[:, 3] = 1
    return SimpleNamespace(root_pos=position, root_vel=velocity, root_rot=rotation)


def test_real_smooth_plan_speed_vector_acceleration_and_current_root_contract():
    physical = state(128)
    original = {key: value.clone() for key, value in vars(physical).items()}
    curriculum = PacerTrajectoryCurriculum(seed=7)
    plan = curriculum.build_plan(physical)
    assert plan.xy.shape == (128, 51, 2)
    assert torch.equal(plan.xy[:, 0], physical.root_pos[:, :2])
    assert torch.equal(plan.times_s, torch.arange(51) * .1)
    assert plan.velocities_xy.norm(dim=-1).max() <= 3 + 1e-6
    assert (plan.velocities_xy[:, 1:] - plan.velocities_xy[:, :-1]).norm(dim=-1).max() / .1 <= 2 + 1e-5
    assert (plan.xy[:, 1:] - plan.xy[:, :-1]).norm(dim=-1).max() / .1 <= 3 + 1e-4
    assert plan.sharp_turn_mask.any()  # actual rare-turn branch is covered
    assert not plan.boundary_reflection_mask.any()
    assert plan.metadata['vector_acceleration_bound_mps2'] == 2
    for key, value in original.items():
        assert torch.equal(getattr(physical, key), value)


def test_public_knot_branch_parity_against_actual_official_reset_ast():
    source = Path(__file__).parents[1] / 'third_party/pacer/pacer/env/util/traj_generator.py'
    if not source.is_file():
        pytest.skip('Optional fixed official reference checkout absent')
    tree = ast.parse(source.read_text())
    official = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'reset')
    compiled = compile(ast.fix_missing_locations(ast.Module(body=[official], type_ignores=[])), str(source), 'exec')
    rng = torch.Generator().manual_seed(42)
    torch_proxy = SimpleNamespace(**{name: getattr(torch, name) for name in ('ones_like', 'zeros_like', 'clip', 'cumsum', 'stack', 'cos', 'sin')})
    torch_proxy.rand = lambda *args, **kwargs: torch.rand(*args, **kwargs, generator=rng)
    torch_proxy.bernoulli = lambda value: torch.bernoulli(value, generator=rng)
    namespace = {'torch': torch_proxy, 'np': np, 'flags': SimpleNamespace(fixed_path=False, slow=False, real_path=False)}
    exec(compiled, namespace)
    config = PacerTrajectoryConfig(mode='pacer_knots', horizon_s=5., knot_dt_s=.1)
    physical = state(4)
    reference = SimpleNamespace(_dt=.1, _dtheta_max=2., _speed_min=0., _speed_max=3., _accel_max=2.,
                                _sharp_turn_prob=.02, _device='cpu', _verts=torch.zeros(4, 51, 3), get_num_verts=lambda: 51)
    namespace['reset'](reference, torch.arange(4), physical.root_pos)
    curriculum = PacerTrajectoryCurriculum(config, seed=42)
    plan = curriculum.build_plan(physical)
    # Source performs cumsum of deltas with the initial position in delta0.
    torch.testing.assert_close(plan.xy, reference._verts[..., :2], rtol=1e-6, atol=2e-6)
    assert plan.metadata['vector_acceleration_bound_mps2'] is None
    assert torch.equal(rng.get_state(), curriculum.generator.get_state())


def test_zero_speed_still_receives_independent_acceleration_commands():
    stationary = state(128, speed=0.)
    plan = PacerTrajectoryCurriculum(seed=7).build_plan(stationary)
    assert plan.speed_goals_mps.mean() > 1.3
    assert (plan.speed_goals_mps > 1.).sum() > 64
    assert plan.velocities_xy[:, 5].norm(dim=-1).mean() > .65
    assert plan.velocities_xy[:, -1].norm(dim=-1).mean() > 1.
    assert plan.velocities_xy[:, 0].norm(dim=-1).max() == 0
    # Desired speed distribution is independent of actual tracking speed.
    moving = PacerTrajectoryCurriculum(seed=7).build_plan(state(128, speed=2.5))
    assert torch.equal(plan.speed_goals_mps, moving.speed_goals_mps)


def test_private_rng_resume_reproduces_full_future_without_global_rng_mutation():
    torch.manual_seed(99)
    global_before = torch.get_rng_state().clone()
    original = PacerTrajectoryCurriculum(seed=7)
    original.build_plan(state(3))
    saved = original.state_dict()
    future = original.build_plan(state(4), actor_ids=[3, 1])
    restored = PacerTrajectoryCurriculum(seed=999)
    restored.load_state_dict(saved)
    repeated = restored.build_plan(state(4), actor_ids=[3, 1])
    for field in ('xy', 'velocities_xy', 'sharp_turn_mask', 'boundary_reflection_mask'):
        assert torch.equal(getattr(future, field), getattr(repeated, field))
    assert torch.equal(torch.get_rng_state(), global_before)
    assert restored.plans_generated == original.plans_generated == 5
    assert torch.equal(future.xy[:, 0], state(4).root_pos[[3, 1], :2])


def test_initial_high_speed_is_explicit_command_clip_not_physical_change():
    physical = state(3, speed=5.)
    plan = PacerTrajectoryCurriculum().build_plan(physical)
    assert plan.initial_velocity_clipped.all()
    assert torch.equal(physical.root_vel[:, 0], torch.full((3,), 5.))
    assert plan.velocities_xy.norm(dim=-1).max() <= 3 + 1e-6
    assert torch.equal(plan.xy[:, 0], physical.root_pos[:, :2])


def test_explicit_reflection_stays_in_bounds_and_declares_loss_of_acceleration_guarantee():
    physical = state(1, speed=3.)
    physical.root_pos[:, 0] = .8
    config = PacerTrajectoryConfig(bounds_xy=(-1., -1., 1., 1.), boundary_mode='reflection')
    plan = PacerTrajectoryCurriculum(config).build_plan(physical)
    assert plan.xy.min() >= -1 and plan.xy.max() <= 1
    assert plan.boundary_reflection_mask.any()
    assert plan.metadata['vector_acceleration_bound_mps2'] is None
    assert torch.equal(plan.xy[:, 0], physical.root_pos[:, :2])
    assert physical.root_pos[0, 0] == .8


@pytest.mark.parametrize('change', [
    {'speed_max_mps': True}, {'horizon_s': 0}, {'knot_dt_s': .3}, {'speed_min_mps': -1},
    {'scalar_accel_max_mps2': 0}, {'sharp_turn_probability': 1.1}, {'mode': 'diffusion'},
    {'boundary_mode': 'reflection'}, {'bounds_xy': (1., 0., 0., 1.)}])
def test_invalid_configs_fail_before_generator_creation(change):
    with pytest.raises(ValueError):
        PacerTrajectoryCurriculum(replace(PacerTrajectoryConfig(), **change))


@pytest.mark.parametrize('case', ['nan', 'quat', 'duplicate_ids', 'float_ids', 'outside', 'empty'])
def test_invalid_physical_inputs_do_not_advance_private_rng(case):
    config = PacerTrajectoryConfig(bounds_xy=(-5., -5., 5., 5.), boundary_mode='reflection')
    curriculum = PacerTrajectoryCurriculum(config)
    physical = state(3)
    ids = None
    if case == 'nan':
        physical.root_vel[0, 0] = float('nan')
    elif case == 'quat':
        physical.root_rot[0] = 0
    elif case == 'duplicate_ids':
        ids = [1, 1]
    elif case == 'float_ids':
        ids = [1.5]
    elif case == 'outside':
        physical.root_pos[0, 0] = 6
    else:
        ids = []
    previous = curriculum.generator.get_state().clone()
    with pytest.raises(ValueError):
        curriculum.build_plan(physical, actor_ids=ids)
    assert torch.equal(previous, curriculum.generator.get_state())


@pytest.mark.parametrize('case', ['extra', 'config', 'count', 'rng'])
def test_malformed_resume_is_rejected_without_mutation(case):
    curriculum = PacerTrajectoryCurriculum()
    saved = curriculum.state_dict()
    if case == 'extra':
        saved['unexpected'] = 1
    elif case == 'config':
        saved['config']['speed_max_mps'] = 4
    elif case == 'count':
        saved['plans_generated'] = True
    else:
        saved['generator_state'] = torch.zeros(3)
    previous = curriculum.generator.get_state().clone()
    with pytest.raises(ValueError):
        curriculum.load_state_dict(saved)
    assert torch.equal(previous, curriculum.generator.get_state())
