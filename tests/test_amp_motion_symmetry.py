"""Geometric and official-source parity for raw PACER mirror regularization."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import torch

from CrowdSim.humanoid.amp.observations import (
    PACER_BODY_NAMES,
    PACER_DOF_NAMES,
    build_self_observation,
    build_task_observation,
)
from CrowdSim.humanoid.training.symmetry import mirror_actions, mirror_policy_observations
from protomotions.utils.rotations import (
    exp_map_to_quat,
    matrix_to_quaternion,
    quaternion_to_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
# This is the original public no-masterfoot24-body swap order, not derived
# from the mirror implementation under test.
OFFICIAL_BODY_SWAP = (0, 5, 6, 7, 8, 1, 2, 3, 4, 9, 10, 11, 12, 13, 19, 20, 21, 22, 23, 14, 15, 16, 17, 18)
OFFICIAL_ACTION_SWAP = (4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10, 11, 12, 18, 19, 20, 21, 22, 13, 14, 15, 16, 17)


def state_fixture():
    generator = torch.Generator().manual_seed(31)
    positions = torch.randn(2, 24, 3, generator=generator, dtype=torch.float64)
    positions += torch.tensor([3., -2., 1.], dtype=torch.float64)
    rotation = exp_map_to_quat(torch.randn(2, 24, 3, generator=generator, dtype=torch.float64) * .4, w_last=True)
    return SimpleNamespace(rigid_body_pos=positions, rigid_body_rot=rotation,
        rigid_body_vel=torch.randn(2, 24, 3, generator=generator, dtype=torch.float64),
        rigid_body_ang_vel=torch.randn(2, 24, 3, generator=generator, dtype=torch.float64),
        dof_pos=torch.zeros(2, 69, dtype=torch.float64), dof_vel=torch.zeros(2, 69, dtype=torch.float64))


def geometric_reflection(state):
    # Independent reflection of SO(3) matrices R -> S R S. Angular velocity
    # is an axial vector det(S)*S*w, unlike position and linear velocity.
    reflection = torch.diag(torch.tensor([1., -1., 1.], dtype=torch.float64))
    swap = list(OFFICIAL_BODY_SWAP)
    matrices = quaternion_to_matrix(state.rigid_body_rot[:, swap], w_last=True)
    return SimpleNamespace(rigid_body_pos=state.rigid_body_pos[:, swap] @ reflection,
        rigid_body_rot=matrix_to_quaternion(reflection @ matrices @ reflection, w_last=True),
        rigid_body_vel=state.rigid_body_vel[:, swap] @ reflection,
        rigid_body_ang_vel=state.rigid_body_ang_vel[:, swap] @ -reflection,
        dof_pos=state.dof_pos, dof_vel=state.dof_vel)


def terrain_height(xy):
    return .03 * xy[..., 1] + .005 * xy[..., 0] * xy[..., 1] + .01 * xy[..., 0].square()


def test_raw_observation_matches_independently_reflected_physical_state_and_terrain():
    state = state_fixture()
    reflected = geometric_reflection(state)
    path = torch.stack((torch.linspace(2., 5., 10), torch.linspace(-1., 3., 10)), -1).double()[None].expand(2, -1, -1)
    self_obs = build_self_observation(state)
    task_obs = build_task_observation(state, path, terrain_height)
    actual_self, actual_task = mirror_policy_observations(self_obs, task_obs)
    expected_self = build_self_observation(reflected)
    reflected_path = path * torch.tensor([1., -1.])

    def reflected_terrain(xy):
        return terrain_height(xy * xy.new_tensor([1., -1.]))

    expected_task = build_task_observation(reflected, reflected_path, reflected_terrain)
    torch.testing.assert_close(actual_self, expected_self, atol=2e-7, rtol=2e-7)
    torch.testing.assert_close(actual_task, expected_task, atol=2e-7, rtol=2e-7)


def test_action_exp_maps_match_reflected_rotation_matrices_and_official_order():
    original = torch.linspace(-.8, .8, 69, dtype=torch.float64)[None]
    result = mirror_actions(original)
    matrices = quaternion_to_matrix(exp_map_to_quat(original.reshape(1, 23, 3), w_last=True), w_last=True)
    reflection = torch.diag(torch.tensor([1., -1., 1.], dtype=torch.float64))
    expected = reflection @ matrices[:, list(OFFICIAL_ACTION_SWAP)] @ reflection
    actual = quaternion_to_matrix(exp_map_to_quat(result.reshape(1, 23, 3), w_last=True), w_last=True)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    # An isolated left knee Y target becomes right knee Y with no sign flip;
    # its X/Z rotation axes flip, and sagittal torso axes follow the same rule.
    action = torch.zeros(1, 69)
    action[0, PACER_DOF_NAMES.index("L_Knee_y")] = .6
    action[0, PACER_DOF_NAMES.index("L_Knee_x")] = .2
    action[0, PACER_DOF_NAMES.index("Torso_z")] = -.3
    reflected = mirror_actions(action)
    assert reflected[0, PACER_DOF_NAMES.index("R_Knee_y")] == .6
    assert reflected[0, PACER_DOF_NAMES.index("R_Knee_x")] == -.2
    assert reflected[0, PACER_DOF_NAMES.index("Torso_z")] == .3


def test_official_task_method_parity_without_importing_isaacgym():
    source = ROOT / "third_party/pacer/pacer/env/tasks/humanoid_pedestrain_terrain.py"
    if not source.is_file():
        pytest.skip("Pinned optional PACER reference checkout unavailable")
    tree = ast.parse(source.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_compute_flip_task_obs")
    namespace = {"torch": torch, "np": np}
    # Execute only this pure method from the actual pinned source. SDK imports
    # and all other third-party functions remain unexecuted.
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    fake = SimpleNamespace(terrain_obs=True, velocity_map=False, num_height_points=1024, _divide_group=False, _group_obs=False)
    task = torch.arange(2 * 1044, dtype=torch.float32).reshape(2, 1044)
    _, result = mirror_policy_observations(torch.zeros(2, 357), task)
    official = namespace["_compute_flip_task_obs"](fake, task, None)
    assert torch.equal(result, official)


def test_official_symmetry_action_loss_and_two_branch_gradients_parity():
    source = ROOT / "third_party/pacer/pacer/learning/amp_continuous.py"
    if not source.is_file():
        pytest.skip("Pinned optional PACER reference checkout unavailable")
    tree = ast.parse(source.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_sym_loss")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    original = torch.linspace(-.8, .8, 138).reshape(2, 69).requires_grad_()
    flipped = torch.linspace(.3, -.5, 138).reshape(2, 69).requires_grad_()
    marker = torch.zeros(2, 1401)

    def eval_actor(observations):
        # Non-leaf copies avoid official method's in-place edits on orig_a.
        return (original * 1 if observations is marker else flipped * 1), None

    fake = SimpleNamespace(vec_env=SimpleNamespace(env=SimpleNamespace(task=SimpleNamespace(left_to_right_index_action=OFFICIAL_ACTION_SWAP))),
                           model=SimpleNamespace(a2c_network=SimpleNamespace(eval_actor=eval_actor)))
    official = namespace["_sym_loss"](fake, torch.ones_like(marker), marker)["sym_loss"]
    expected = (mirror_actions(original)-flipped).square().mean(-1)*50
    torch.testing.assert_close(official, expected)
    left, right = torch.autograd.grad(official.mean(), (original, flipped))
    expected_left, expected_right = torch.autograd.grad(expected.mean(), (original, flipped))
    torch.testing.assert_close(left, expected_left)
    torch.testing.assert_close(right, expected_right)
    assert left.abs().sum() > 0 and right.abs().sum() > 0


@pytest.mark.parametrize("leading", [(2,), (2, 3)])
def test_mirror_involution_preserves_norms_inputs_and_all_batch_dimensions(leading):
    generator = torch.Generator().manual_seed(17)
    state = torch.randn(*leading, 357, generator=generator)
    task = torch.randn(*leading, 1044, generator=generator)
    action = torch.randn(*leading, 69, generator=generator)
    originals = [x.clone() for x in (state, task, action)]
    mirrored = mirror_policy_observations(state, task)
    recovered = mirror_policy_observations(*mirrored)
    assert all(torch.equal(a, b) for a, b in zip(recovered, (state, task)))
    assert torch.equal(mirror_actions(mirror_actions(action)), action)
    assert all(torch.equal(a, b) for a, b in zip(originals, (state, task, action)))
    assert all(torch.allclose(a.norm(dim=-1), b.norm(dim=-1)) for a, b in zip(mirrored, (state, task)))


def test_both_policy_branches_and_mirrored_actions_have_gradients():
    generator = torch.Generator().manual_seed(8)
    state = torch.randn(2, 357, generator=generator, requires_grad=True)
    task = torch.randn(2, 1044, generator=generator, requires_grad=True)
    actor = torch.nn.Linear(1401, 69)
    left = actor(torch.cat((state, task), -1))
    mirrored_state, mirrored_task = mirror_policy_observations(state, task)
    right = actor(torch.cat((mirrored_state, mirrored_task), -1))
    left.retain_grad()
    right.retain_grad()
    (mirror_actions(left)-right).square().mean().backward()
    for value in (left.grad, right.grad, state.grad, task.grad, actor.weight.grad):
        assert value is not None and torch.isfinite(value).all() and torch.count_nonzero(value)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("which,invalid", [("self", torch.zeros(1, 358)), ("task", torch.zeros(1, 1043)),
                                         ("self", torch.zeros(0, 357)), ("self", torch.zeros(357)),
                                         ("task", torch.zeros(1, 1044, dtype=torch.int64)),
                                         ("self", torch.full((1, 357), float("nan"))),
                                         ("task", torch.full((1, 1044), float("inf")))])
def test_observation_shape_dtype_and_finite_gates(which, invalid):
    state, task = torch.zeros(1, 357), torch.zeros(1, 1044)
    with pytest.raises(ValueError):
        mirror_policy_observations(invalid if which == "self" else state, invalid if which == "task" else task)


def test_observation_batch_device_and_name_gates():
    with pytest.raises(ValueError, match="batch"):
        mirror_policy_observations(torch.zeros(1, 357), torch.zeros(2, 1044))
    with pytest.raises(ValueError, match="device"):
        mirror_policy_observations(torch.zeros(1, 357), torch.empty(1, 1044, device="meta"))
    with pytest.raises(ValueError, match="bijection"):
        mirror_policy_observations(torch.zeros(1, 357), torch.zeros(1, 1044), body_names=("unknown",)*24)
    with pytest.raises(ValueError, match="Pelvis first"):
        mirror_policy_observations(torch.zeros(1, 357), torch.zeros(1, 1044), body_names=PACER_BODY_NAMES[1:]+PACER_BODY_NAMES[:1])


@pytest.mark.parametrize("invalid", [torch.zeros(1, 68), torch.zeros(69), torch.zeros(0, 69),
                                   torch.ones(1, 69, dtype=torch.int64), torch.full((1, 69), float("inf"))])
def test_action_gates_and_named_order(invalid):
    with pytest.raises(ValueError):
        mirror_actions(invalid)
    with pytest.raises(ValueError, match="bijection"):
        mirror_actions(torch.zeros(1, 69), dof_names=("unknown",)*69)
