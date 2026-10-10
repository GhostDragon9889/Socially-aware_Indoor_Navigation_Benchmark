"""CPU regressions against isolated official PACER mathematical kernels.

The reference AST loader executes selected local, reviewed numerical functions
only. It never imports Isaac Gym, runs upstream installation, or loads pickle
data. Optional public checkpoint tests use weights_only=True and fixed SHA256.
"""

from __future__ import annotations

from protomotions.repository import repository_root

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from CrowdSim.humanoid.amp import (
    PACER_BODY_NAMES,
    PACER_DOF_NAMES,
    PacerAMPController,
    PacerAMPNetwork,
    PacerSMPLMapping,
    RunningNormalizer,
    action_to_pd,
    build_amp_observation,
    build_self_observation,
    build_task_observation,
)
from protomotions.utils import rotations

ROOT = repository_root(__file__)
REFERENCE = ROOT / "third_party/pacer/pacer"
CHECKPOINT = ROOT / "data/pretrained_models/paper_references/pacer/output/release/pacer_no_shape/Humanoid.pth"


def _functions(path, names, namespace, *, class_name=None):
    tree = ast.parse(path.read_text())
    nodes = tree.body
    if class_name:
        for class_component in class_name.split("."):
            nodes = next(node.body for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_component)
    selected = []
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            if isinstance(node, ast.FunctionDef):
                node.decorator_list = []
            selected.append(node)
    assert {node.name for node in selected} == set(names)
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


@pytest.fixture(scope="module")
def official():
    if not (REFERENCE / "utils/torch_utils.py").is_file():
        pytest.skip("Fixed public PACER source checkout is required for source parity")
    namespace = {
        "torch": torch, "np": np, "nn": nn,
        "quat_mul": lambda q, p: rotations.quat_mul(q, p, w_last=True),
        "quat_conjugate": lambda q: torch.cat((-q[..., :3], q[..., 3:]), -1),
        "quat_from_angle_axis": lambda angle, axis: rotations.quat_from_angle_axis(angle, axis, w_last=True),
        "normalize_angle": rotations.normalize_angle,
        "normalize": lambda q: torch.nn.functional.normalize(q, dim=-1, eps=1e-9),
        "quat_apply": lambda q, v: rotations.quat_rotate(q.reshape(-1, 4), v.reshape(-1, 3), w_last=True).reshape(v.shape),
        "to_torch": lambda values, device=None: torch.tensor(values, dtype=torch.float32, device=device),
        "flags": SimpleNamespace(height_debug=False, divide_group=False),
    }
    _functions(REFERENCE / "utils/torch_utils.py", (
        "my_quat_rotate", "quat_to_tan_norm", "exp_map_to_angle_axis",
        "exp_map_to_quat", "calc_heading", "calc_heading_quat", "calc_heading_quat_inv",
    ), namespace)
    namespace["torch_utils"] = SimpleNamespace(**namespace)
    _functions(REFERENCE / "env/tasks/humanoid.py", (
        "remove_base_rot", "dof_to_obs_smpl", "compute_humanoid_observations_smpl_max",
    ), namespace)
    _functions(REFERENCE / "env/tasks/humanoid.py", ("_build_pd_action_offset_scale",), namespace, class_name="Humanoid")
    _functions(REFERENCE / "env/tasks/humanoid_amp.py", ("build_amp_observations_smpl",), namespace)
    _functions(REFERENCE / "env/tasks/humanoid_pedestrain_terrain.py", (
        "compute_location_observations", "quat_apply_yaw",
    ), namespace)
    methods = ("init_center_height_points", "init_square_height_points", "get_center_heights",
               "get_heights", "get_head_pose", "_compute_task_obs")
    _functions(REFERENCE / "env/tasks/humanoid_pedestrain_terrain.py", methods, namespace, class_name="HumanoidPedestrianTerrain")
    namespace["TerrainReference"] = type("TerrainReference", (), {name: namespace[name] for name in methods})
    _functions(REFERENCE / "utils/running_mean_std.py", ("RunningMeanStd",), namespace)
    _functions(REFERENCE / "learning/amp_network_sept_builder.py", ("eval_task", "eval_actor", "eval_critic"), namespace, class_name="AMPSeptBuilder.Network")
    _functions(REFERENCE / "learning/amp_network_builder.py", ("eval_disc",), namespace, class_name="AMPBuilder.Network")
    return namespace


@pytest.fixture
def state():
    generator = torch.Generator().manual_seed(731)
    shape = (4, 24)
    quaternion = torch.randn(*shape, 4, generator=generator)
    quaternion /= quaternion.norm(dim=-1, keepdim=True)
    return SimpleNamespace(
        rigid_body_pos=torch.randn(*shape, 3, generator=generator),
        rigid_body_rot=quaternion,
        rigid_body_vel=torch.randn(*shape, 3, generator=generator),
        rigid_body_ang_vel=torch.randn(*shape, 3, generator=generator),
        dof_pos=torch.randn(4, 69, generator=generator) * 0.3,
        dof_vel=torch.randn(4, 69, generator=generator),
    )


class SlopedTerrain:
    def height_at(self, xy):
        return (0.13 * xy[..., 0] - 0.21 * xy[..., 1] + 0.07 * xy[..., 0].sin())

    def sample_height_points(self, xyz, **_):
        return self.height_at(xyz[..., :2])


def test_self_observation_matches_official_max_coordinate_kernel(state, official):
    expected = official["compute_humanoid_observations_smpl_max"](
        state.rigid_body_pos, state.rigid_body_rot, state.rigid_body_vel,
        state.rigid_body_ang_vel, torch.zeros(4, 17), torch.zeros(4, 10),
        True, False, True, False, False,
    )
    assert expected.shape == (4, 357)
    torch.testing.assert_close(build_self_observation(state), expected, atol=2e-6, rtol=1e-6)


def test_amp_observation_matches_official_subset_and_keybody_order(state, official):
    subset = torch.tensor([i for i, name in enumerate(PACER_DOF_NAMES)
                           if name.rsplit("_", 1)[0] not in {"L_Hand", "R_Hand", "L_Toe", "R_Toe"}])
    keys = [PACER_BODY_NAMES.index(name) for name in ("R_Ankle", "L_Ankle", "R_Wrist", "L_Wrist")]
    expected = official["build_amp_observations_smpl"](
        state.rigid_body_pos[:, 0], state.rigid_body_rot[:, 0],
        state.rigid_body_vel[:, 0], state.rigid_body_ang_vel[:, 0],
        state.dof_pos, state.dof_vel, state.rigid_body_pos[:, keys],
        torch.zeros(4, 17), torch.zeros(4, 10), subset,
        True, False, True, False, False, True,
    )
    assert expected.shape == (4, 195)
    torch.testing.assert_close(build_amp_observation(state), expected, atol=2e-6, rtol=1e-6)


def test_task_observation_matches_official_head_grid_and_root_center(state, official):
    reference = official["TerrainReference"]()
    reference.num_envs, reference.device = 4, "cpu"
    reference.sensor_extent, reference.sensor_res = 2, 32
    reference.smpl_humanoid, reference._has_upright_start = True, True
    reference._body_names = list(PACER_BODY_NAMES)
    reference._rigid_body_pos, reference._rigid_body_rot = state.rigid_body_pos, state.rigid_body_rot
    reference._humanoid_root_states = torch.cat((state.rigid_body_pos[:, 0], state.rigid_body_rot[:, 0]), -1)
    reference.cfg = {"env": {"terrain": {"terrainType": "trimesh"}, "use_center_height": True}}
    reference.terrain = SlopedTerrain()
    reference.terrain_obs, reference.terrain_obs_root = True, "head"
    reference._divide_group, reference._group_obs, reference.velocity_map = False, False, False
    reference.height_meas_scale = 5
    reference.height_points = reference.init_square_height_points()
    reference.center_height_points = reference.init_center_height_points()
    generator = torch.Generator().manual_seed(87)
    trajectory = torch.randn(4, 10, 3, generator=generator)
    reference._fetch_traj_samples = lambda env_ids: trajectory
    expected = reference._compute_task_obs()
    actual = build_task_observation(state, trajectory, SlopedTerrain())
    assert expected.shape == (4, 1044)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-6)
    local = expected[:, :20].reshape(4, 10, 2)
    torch.testing.assert_close(build_task_observation(state, local, SlopedTerrain(), trajectory_frame="local_heading"), actual)


def test_named_body_and_dof_permutations_preserve_observations_and_targets(state):
    body_permutation = torch.randperm(24, generator=torch.Generator().manual_seed(13))
    dof_permutation = torch.randperm(69, generator=torch.Generator().manual_seed(17))
    mapping = PacerSMPLMapping(tuple(PACER_BODY_NAMES[i] for i in body_permutation),
                               tuple(PACER_DOF_NAMES[i] for i in dof_permutation))
    permuted = SimpleNamespace(**{
        name: getattr(state, name)[:, body_permutation] for name in
        ("rigid_body_pos", "rigid_body_rot", "rigid_body_vel", "rigid_body_ang_vel")
    }, dof_pos=state.dof_pos[:, dof_permutation], dof_vel=state.dof_vel[:, dof_permutation])
    torch.testing.assert_close(build_self_observation(permuted, mapping=mapping), build_self_observation(state))
    torch.testing.assert_close(build_amp_observation(permuted, mapping=mapping), build_amp_observation(state))
    lower, upper = -torch.linspace(0.1, 2.0, 69), torch.linspace(0.3, 2.5, 69)
    action = torch.ones(4, 69) * 0.2
    expected = action_to_pd(action, lower, upper)
    actual = action_to_pd(action, lower[dof_permutation], upper[dof_permutation], dof_names=mapping.dof_names)
    torch.testing.assert_close(actual, expected[:, dof_permutation])


def test_pd_scale_matches_official_false_bias_branch_knees_and_freezes(official):
    lower, upper = -torch.linspace(0.1, 2.0, 69), torch.linspace(0.3, 2.5, 69)
    reference = SimpleNamespace(_dof_offsets=np.arange(0, 70, 3), _bias_offset=False,
                                dof_limits_lower=lower.clone(), dof_limits_upper=upper.clone(),
                                device="cpu", smpl_humanoid=True, _dof_names=list(PACER_BODY_NAMES[1:]),
                                _has_smpl_pd_offset=False)
    official["_build_pd_action_offset_scale"](reference)
    raw = torch.linspace(-0.9, 0.9, 69).repeat(4, 1)
    expected = reference._pd_action_offset + reference._pd_action_scale * raw
    for name in ("L_Hand", "R_Hand", "L_Toe", "R_Toe"):
        start = (PACER_BODY_NAMES.index(name) - 1) * 3
        expected[:, start:start + 3] = 0
    torch.testing.assert_close(action_to_pd(raw, lower, upper), expected)
    assert reference._pd_action_scale[PACER_DOF_NAMES.index("L_Knee_y")] == 5
    assert reference._pd_action_scale[PACER_DOF_NAMES.index("R_Knee_y")] == 5


def test_normalizer_matches_official_inference_and_singleton_update_is_finite(official):
    current, reference = RunningNormalizer(7), official["RunningMeanStd"](7).eval()
    state = {"running_mean": torch.linspace(-2, 2, 7).double(),
             "running_var": torch.linspace(0.01, 3, 7).double(), "count": torch.tensor(523.0).double()}
    current.load_state_dict(state)
    reference.load_state_dict(state)
    values = torch.linspace(-10, 10, 28).reshape(4, 7)
    torch.testing.assert_close(current(values), reference(values))
    torch.testing.assert_close(current(values, unnorm=True), reference(values, unnorm=True))
    current.update(values[:1])
    assert all(torch.isfinite(value).all() for value in current.state_dict().values())


@pytest.mark.parametrize("bad", ["duplicate", "missing", "unknown"])
def test_mapping_rejects_incompatible_skeleton(bad):
    names = list(PACER_BODY_NAMES)
    if bad == "duplicate":
        names[1] = names[0]
    elif bad == "missing":
        names.pop()
    else:
        names[1] = "SOMA_hip"
    with pytest.raises(ValueError, match="bijection"):
        PacerSMPLMapping(body_names=tuple(names))


def test_checkpoint_hash_failure_does_not_deserialize_or_fallback(tmp_path, monkeypatch):
    path = tmp_path / "wrong.ckpt"
    path.write_bytes(b"not a PACER checkpoint")
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("Unverified checkpoint must not deserialize"))
    with pytest.raises(ValueError, match="SHA-256"):
        PacerAMPController.from_checkpoint(path, -torch.ones(69), torch.ones(69))
    controller = PacerAMPController(-torch.ones(69), torch.ones(69))
    with pytest.raises(RuntimeError, match="verified checkpoint"):
        controller.act(None, None, None)


@pytest.fixture(scope="module")
def real_controller():
    if not CHECKPOINT.is_file() or CHECKPOINT.stat().st_size != 119755239:
        pytest.skip("Downloaded complete public PACER no_shape checkpoint is required")
    threads = torch.get_num_threads()
    torch.set_num_threads(4)
    result = PacerAMPController.from_checkpoint(CHECKPOINT, -torch.full((69,), np.pi), torch.full((69,), np.pi))
    yield result
    torch.set_num_threads(threads)


@pytest.mark.needs_lfs
def test_real_actor_critic_discriminator_and_pd_targets_are_finite(real_controller, state):
    own = build_self_observation(state)
    task = build_task_observation(state, torch.zeros(4, 10, 2), SlopedTerrain())
    observation = torch.cat((own, task), -1)
    mu, log_sigma = real_controller.network.eval_actor(real_controller.observation_normalizer(observation))
    assert mu.shape == log_sigma.shape == (4, 69)
    assert real_controller.evaluate_value(observation).shape == (4, 1)
    history = build_amp_observation(state)[:, None].expand(-1, 10, -1).clone()
    assert real_controller.eval_discriminator(history).shape == (4, 1)
    assert torch.isfinite(real_controller.discriminator_reward(history)).all()
    raw = real_controller.act(state, torch.zeros(4, 10, 2), SlopedTerrain(), return_raw=True)
    pd = real_controller.act(state, torch.zeros(4, 10, 2), SlopedTerrain())
    torch.testing.assert_close(pd, real_controller.decode_action(raw))
    assert torch.isfinite(pd).all()
    for name in ("L_Hand", "R_Hand", "L_Toe", "R_Toe"):
        start = (PACER_BODY_NAMES.index(name) - 1) * 3
        assert torch.count_nonzero(pd[:, start:start + 3]) == 0


@pytest.mark.needs_lfs
def test_loaded_network_forward_matches_official_actor_critic_and_disc_methods(real_controller, official):
    network = real_controller.network
    reference = SimpleNamespace(
        self_obs_size=357, task_obs_size=1044, task_obs_size_detail={"traj": 20, "heightmap": 1024},
        _task_mlp=network._task_mlp, actor_cnn=nn.Identity(), actor_mlp=network.actor_mlp,
        critic_mlp=network.critic_mlp, value_act=nn.Identity(), value=network.value,
        mu_act=nn.Identity(), mu=network.mu, sigma_act=nn.Identity(), sigma=network.sigma,
        is_discrete=False, is_multi_discrete=False, is_continuous=True,
        space_config={"fixed_sigma": True}, _disc_mlp=network._disc_mlp,
        _disc_logits=network._disc_logits,
    )
    reference.eval_task = MethodType(official["eval_task"], reference)
    generator = torch.Generator().manual_seed(613)
    normalized = torch.randn(4, 1401, generator=generator).clamp(-5, 5)
    expected_mu, expected_sigma = official["eval_actor"](reference, normalized)
    mu, sigma = network.eval_actor(normalized)
    torch.testing.assert_close(mu, expected_mu, atol=0, rtol=0)
    torch.testing.assert_close(sigma, expected_sigma, atol=0, rtol=0)
    torch.testing.assert_close(network.eval_critic(normalized), official["eval_critic"](reference, normalized), atol=0, rtol=0)
    history = torch.randn(4, 1950, generator=generator).clamp(-5, 5)
    torch.testing.assert_close(network.eval_disc(history), official["eval_disc"](reference, history), atol=0, rtol=0)


def test_network_actor_and_task_encoder_are_trainable_without_reference_checkpoint():
    torch.manual_seed(19)
    network = PacerAMPNetwork(discriminator_obs_size=2214, learn_sigma=True)
    observation = torch.zeros(2, 1401)
    mu, sigma = network.eval_actor(observation)
    loss = mu.square().mean() + sigma.mean() + network.eval_critic(observation).square().mean() + network.eval_disc(torch.zeros(2, 2214)).square().mean()
    loss.backward()
    assert network.mu.weight.grad is not None
    assert network._task_mlp[0].weight.grad is not None
    assert network.sigma.grad is not None
    assert network._disc_logits.weight.grad is not None
