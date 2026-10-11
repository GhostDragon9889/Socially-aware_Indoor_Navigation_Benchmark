"""Narrow CPU regressions for optional vendored source, without IsaacGym import.

Pure modules are loaded normally. Functions coupled to obsolete simulator or
training dependencies are compiled directly from their source definitions and
executed with their external boundaries supplied explicitly. These are component
tests, not evidence of an integrated PACER/FLUX training or simulation run.
"""
import ast
import importlib.util
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from protomotions.repository import repository_root

ROOT = repository_root(__file__)


def _module(relative):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location("vendor_regression_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _definitions(relative, names, namespace):
    """Execute original definitions while avoiding external module side effects."""
    path = ROOT / relative
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = []
    for name in names:
        matches = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name]
        assert len(matches) == 1, (name, len(matches))
        nodes.append(matches[0])
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2, -math.pi / 2, 0.43])
def test_flux_world_local_transform_round_trip(yaw):
    ns = _definitions(
        "third_party/flux/baselines/flux/deployment_server.py",
        ["local_to_world", "world_to_local"], {"np": np, "math": math},
    )
    pose = [1.4, -2.5, yaw]
    goal = [0.7, -0.4, 0.0]
    world = ns["local_to_world"](pose, goal)
    np.testing.assert_allclose(ns["world_to_local"](pose, world)[0], goal, atol=1e-7)


def test_flux_point_goal_prediction_executes_on_configured_device():
    ns = _definitions(
        "third_party/flux/baselines/flux/policy_network.py",
        ["predict_pointgoal_action", "_build_trajectory", "_select_trajectories"], {"torch": torch},
    )
    model = SimpleNamespace(device="cpu", predict_size=4, cfm_num_steps=2)
    model.point_encoder = nn.Linear(3, 4)
    model.rgbd_encoder = lambda images, depths: torch.zeros(images.shape[0], 2, 4)
    model.sample_ode_euler = lambda goal, rgbd, batch, steps: torch.ones(batch, 4, 3)
    model.predict_critic = lambda actions, rgbd: torch.arange(actions.shape[0], dtype=torch.float32)
    model.denormalize_action_tensor = lambda action: action
    model._build_trajectory = lambda *args: ns["_build_trajectory"](model, *args)
    model._select_trajectories = lambda *args: ns["_select_trajectories"](model, *args)
    trajectories, critics, positive, negative = ns["predict_pointgoal_action"](
        model, np.ones((2, 3), dtype=np.float32), torch.zeros(2, 1), torch.zeros(2, 1), sample_num=3,
    )
    assert trajectories.shape == (2, 3, 4, 3)
    np.testing.assert_allclose(trajectories[:, :, -1, :], 1.0)
    np.testing.assert_array_equal(critics, [[0, 1, 2], [3, 4, 5]])
    assert positive.shape == negative.shape == (2, 2, 4, 3)


@pytest.mark.parametrize("vendor", ["flux", "navdp"])
def test_logoplanner_plot_handles_empty_history(vendor, tmp_path, monkeypatch):
    module = _module(f"third_party/{vendor}/baselines/logoplanner/deployment/visualization.py")
    monkeypatch.chdir(tmp_path)
    figure = module.plot_trajectory(np.array([[0., 0.], [1., 1.]]), [], [], traj_name=vendor)
    assert (tmp_path / f"trajectory_{vendor}.png").stat().st_size > 0
    assert len(figure.axes) == 2
    import matplotlib.pyplot as plt
    plt.close(figure)


@pytest.mark.parametrize("relative", ["third_party/pacer/pacer/utils/flags.py", "third_party/pacer/uhc/utils/flags.py"])
def test_pacer_flags_export_only_defined_public_symbols(relative):
    module = _module(relative)
    exported = {name: getattr(module, name) for name in module.__all__}
    assert list(exported) == ["flags"]
    assert exported["flags"].debug is False


def test_pacer_standard_epoch_schedule_uses_aggregated_kl():
    ns = _definitions(
        "third_party/pacer/pacer/learning/common_agent.py", ["train_epoch"],
        {"torch": torch, "time": time, "torch_ext": SimpleNamespace(mean_list=lambda items: torch.stack(items).mean())},
    )
    observed = []
    agent = SimpleNamespace(
        is_rnn=False, play_steps=lambda: {"played_frames": 8}, set_train=lambda: None,
        prepare_dataset=lambda batch: None, algo_observer=SimpleNamespace(after_steps=lambda: None),
        has_central_value=False, mini_epochs_num=2, dataset=[1.0, 3.0],
        train_actor_critic=lambda kl: {"kl": torch.tensor(kl)},
        schedule_type="standard_epoch", multi_gpu=True,
        hvd=SimpleNamespace(average_value=lambda kl, name: kl + 1),
        last_lr=0.01, entropy_coef=0.2, epoch_num=7,
        scheduler=SimpleNamespace(update=lambda lr, entropy, epoch, unused, kl: (observed.append(kl) or lr, entropy)),
        update_lr=lambda lr: None, _record_train_batch_info=lambda batch, info: None,
    )
    result = ns["train_epoch"](agent)
    assert observed == [3.0]
    assert len(result["kl"]) == 4 and agent.curr_frames == 8


def test_pacer_residual_default_runs_and_missing_attention_fails_explicitly():
    ns = _definitions(
        "third_party/pacer/pacer/learning/network_builder.py",
        ["Conv2dAuto", "ConvBlock", "ResidualBlock"], {"torch": torch, "nn": nn},
    )
    block = ns["ResidualBlock"](channels=4)
    value = torch.randn(2, 4, 5, 5)
    torch.testing.assert_close(block(value), value)
    with pytest.raises(NotImplementedError, match="does not vendor"):
        ns["ResidualBlock"](channels=4, use_attention=True)


@pytest.mark.parametrize("class_name", ["SMPL_M_Renderer", "SMPL_M_Viewer"])
def test_smpl_viewer_thread_creation_error_does_not_raise_name_error(class_name, capsys):
    relative = "third_party/pacer/uhc/smpllib/smpl_mujoco.py"
    path = ROOT / relative
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "show_pose_thread")
    def fail_thread(*args, **kwargs):
        raise OSError("thread creation failed")
    namespace = {"threading": SimpleNamespace(Thread=fail_thread)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    viewer = SimpleNamespace(show_pose=lambda: None, show_pose_in_thread=lambda: None)
    namespace["show_pose_thread"](viewer)
    assert "thread creation failed" in capsys.readouterr().out


def test_pacer_de_heading_removes_yaw_and_retains_roll():
    module = _module("third_party/pacer/uhc/utils/torch_utils.py")
    half_yaw, half_roll = 0.3, 0.2
    yaw = torch.tensor([math.cos(half_yaw), 0., 0., math.sin(half_yaw)], dtype=torch.float64)
    roll = torch.tensor([math.cos(half_roll), math.sin(half_roll), 0., 0.], dtype=torch.float64)
    pose = module.quaternion_multiply_batch(yaw[None], roll[None])[0]
    torch.testing.assert_close(module.de_heading(pose), roll, atol=1e-12, rtol=1e-12)
    with pytest.raises(ValueError, match="shape"):
        module.de_heading(pose[None])


def test_fsq_configuration_return_type_is_resolvable():
    import typing
    from protomotions.agents.ppo.config import PPOAgentConfig
    module = _module("examples/experiments/mimic/fsq.py")
    assert typing.get_type_hints(module.agent_config)["return"] is PPOAgentConfig


@pytest.mark.parametrize("relative", [
    "third_party/flux/configs/tasks/usd_utils.py",
    "third_party/navdp/configs/tasks/usd_utils.py",
    "third_party/navdp/baselines/x-navdp/src/environment/tasks/event_utils.py",
])
def test_point_light_name_query_does_not_create_extra_prims(relative):
    existing = {"/World/Lights/point_light", "/World/Lights/point_light_1"}
    queries = []
    created = []
    poses = []
    def get_prim(path):
        queries.append(path)
        assert len(queries) < 10, "name lookup should terminate without creating queried prims"
        return SimpleNamespace(IsValid=lambda: path in existing)
    def create_prim(*args, **kwargs):
        path = kwargs.get("prim_path", args[0] if args else None)
        created.append(path)
        assert len(created) == 1, "only the selected SphereLight may be created"
        return SimpleNamespace(path=path)
    class Transform:
        def __init__(self, path): self.path = path
        def set_world_pose(self, position): poses.append((self.path, position))
    class Light:
        def __init__(self, prim): self.prim = prim
        def CreateIntensityAttr(self, value): self.intensity = value
        def CreateColorAttr(self, value): self.color = value
        def CreateRadiusAttr(self, value): self.radius = value
    ns = _definitions(relative, ["add_point_light"], {
        "torch": torch, "get_prim_at_path": get_prim, "create_prim": create_prim,
        "UsdLux": SimpleNamespace(SphereLight=Light), "Gf": SimpleNamespace(Vec3f=lambda *v: tuple(v)),
        "XFormPrim": Transform, "SingleXFormPrim": Transform,
    })
    light = ns["add_point_light"](torch.tensor([1.0, 2.0, 3.0]))
    assert queries == ["/World/Lights/point_light", "/World/Lights/point_light_1", "/World/Lights/point_light_2"]
    assert created == ["/World/Lights/point_light_2"]
    assert light.prim.path == created[0]
    np.testing.assert_allclose(poses[0][1], [1, 2, 3])


def test_navdp_preflight_accepts_real_target_metadata_without_loading_kit():
    import importlib.metadata
    try:
        sim = importlib.metadata.version("isaacsim")
        lab = importlib.metadata.version("isaaclab")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("Target Isaac Sim/Lab metadata is not installed in this CPU test environment")
    if not sim.startswith("6.1.") or not lab.startswith("17."):
        pytest.skip("Requires the target Sim6.1 / Lab3EA distribution metadata")
    module = _module("third_party/navdp/runtime_compat.py")
    result = module.require_supported_runtime()
    assert result["isaacsim"].startswith("6.1.")
    assert result["isaaclab_distribution"].startswith("17.")


@pytest.mark.parametrize("python,sim,lab,expected", [
    ((3, 10, 0), "6.1.0.0", "17.0.2", "Python 3.10"),
    ((3, 12, 13), "4.5.0.0", "17.0.2", "Isaac Sim 4.5"),
    ((3, 12, 13), "6.0.0.1", "17.0.2", "Isaac Sim 6.0"),
    ((3, 12, 13), "6.1.0.0", "2.0.2", "distribution 2.0.2"),
])
def test_navdp_preflight_rejects_legacy_environments_before_kit(python, sim, lab, expected):
    module = _module("third_party/navdp/runtime_compat.py")
    with pytest.raises(RuntimeError, match=expected):
        module.require_supported_runtime(python_version=python, version_lookup=lambda name: {"isaacsim": sim, "isaaclab": lab}[name])


def test_navdp_preflight_missing_dependency_is_actionable():
    import importlib.metadata
    module = _module("third_party/navdp/runtime_compat.py")
    def missing(name): raise importlib.metadata.PackageNotFoundError(name)
    with pytest.raises(RuntimeError, match="Missing installed package: isaacsim"):
        module.require_supported_runtime(python_version=(3, 12, 13), version_lookup=missing)


def test_navdp_all_legacy_imports_migrated_and_preflight_precedes_app_import():
    root = ROOT / "third_party/navdp"
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names] if isinstance(node, ast.Import) else []
            assert not any(name and name.startswith("omni.isaac") for name in names), str(path)
    entries = list(root.glob("*_wheeled.py"))
    assert len(entries) == 7
    for path in entries:
        tree = ast.parse(path.read_text())
        app = next(node.lineno for node in tree.body if isinstance(node, ast.ImportFrom) and node.module == "isaaclab.app")
        preflight = next(node.lineno for node in tree.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and getattr(node.value.func, "id", None) == "require_supported_runtime")
        assert preflight < app


def test_pacer_trajectory_constructor_and_cpu_reset_avoid_removed_numpy_int():
    ns = _definitions("third_party/pacer/pacer/env/util/traj_generator.py", ["TrajGenerator"], {
        "np": np, "torch": torch, "flags": SimpleNamespace(fixed_path=False, slow=False, real_path=False),
    })
    generator = ns["TrajGenerator"](2, 2.0, 9, "cpu", 0.3, 0.5, 1.5, 0.5, 0.0)
    starts = torch.tensor([[0., 0., 0.], [1., 1., 0.]])
    generator.reset(torch.tensor([0, 1]), starts)
    assert generator._verts.shape == (2, 9, 3)
    assert torch.isfinite(generator._verts).all()
    torch.testing.assert_close(generator._verts[:, 0, :2], starts[:, :2])
    speeds = torch.linalg.vector_norm(torch.diff(generator._verts[:, :, :2], dim=1), dim=-1) / generator._dt
    assert speeds.min() >= 0.5 - 1e-6 and speeds.max() <= 1.5 + 1e-6


@pytest.mark.parametrize("vendor", ["flux", "navdp"])
def test_differential_controller_batch_preserves_scalar_speed_limits(vendor):
    from abc import ABC, abstractmethod
    ns = {"np": np, "torch": torch, "ABC": ABC, "abstractmethod": abstractmethod, "ArticulationAction": SimpleNamespace}
    _definitions(f"third_party/{vendor}/wheeled_robots/controllers/base_controller.py", ["BaseController"], ns)
    _definitions(f"third_party/{vendor}/wheeled_robots/controllers/differential_controller.py", ["DifferentialController"], ns)
    controller = ns["DifferentialController"]("cpu_test", 0.0591, 0.22616, max_linear_speed=0.8, max_angular_speed=0.5, max_wheel_speed=12.0)
    commands = np.array([[2., 2.], [-2., -2.], [0.2, -0.1], [0., 0.]])
    expected = np.stack([controller.forward(c).joint_velocities for c in commands])
    np.testing.assert_allclose(controller.forward_batch(commands).numpy(), expected, atol=1e-6)
    assert np.abs(expected).max() <= 12.0
    with pytest.raises(ValueError, match="shape"):
        controller.forward_batch(np.zeros((2, 3)))


def test_navdp_optional_dependencies_fail_before_gpu_startup():
    module = _module("third_party/navdp/runtime_compat.py")
    with pytest.raises(RuntimeError, match="casadi, pathfinding"):
        module.require_baseline_dependencies(module_lookup=lambda name: None if name in {"casadi", "pathfinding"} else object())
    with pytest.raises(RuntimeError, match="pynput"):
        module.require_baseline_dependencies(teleop=True, module_lookup=lambda name: None if name == "pynput" else object())
    assert "pynput" not in module.require_baseline_dependencies(module_lookup=lambda _: object())


@pytest.mark.parametrize("reset_name", ["exploration_reset", "pointnav_reset", "imagenav_reset", "pixelnav_reset"])
def test_navdp_reset_lab_xyzw_preserves_core_wxyz_goal_boundary(reset_name, tmp_path):
    from scipy.spatial.transform import Rotation
    # The deprecated IsaacSim core rotation utility returns WXYZ. Lab 3 writes XYZW.
    # Use real numeric rotations; inject only scene/asset boundaries, without Kit.
    yaw = np.array([0.0, math.pi / 2])
    points = np.column_stack(([1., 2.], [3., 4.], [5., 6.], [7., 8.], yaw))
    path = tmp_path / "episodes.npy"
    np.save(path, points)
    writes, goals = [], []
    class Proxy:
        def __init__(self, value): self.torch = value
        def __getattr__(self, name): raise AssertionError("implicit ProxyArray access: " + name)
    asset = SimpleNamespace(data=SimpleNamespace(root_pos_w=Proxy(torch.tensor([[0., 0., 0.7], [0., 0., 0.8]]))),
        write_root_pose_to_sim=lambda pose, env_ids: writes.append((pose, env_ids)))
    class Scene(dict):
        env_origins = torch.zeros(2, 3)
    env = SimpleNamespace(scene=Scene(robot=asset))
    class PrimView:
        def __init__(self, prim_paths_expr, name): self.path = prim_paths_expr
        def set_world_poses(self, position, orientation):
            assert position.shape == (1, 3) and orientation.shape == (1, 4)
            goals.append((self.path, position.clone(), orientation.clone()))
    ns = _definitions("third_party/navdp/configs/tasks/wheeled_task.py", [reset_name], {
        "np": np, "torch": torch, "reset_counter": 0,
        "ManagerBasedEnv": object, "RigidObject": object, "Articulation": object,
        "SceneEntityCfg": lambda name: SimpleNamespace(name=name), "XFormPrimView": PrimView,
        "rot_utils": SimpleNamespace(euler_angles_to_quats=lambda value: Rotation.from_euler("xyz", value).as_quat(scalar_first=True)),
    })
    ids = torch.tensor([1, 0])
    kwargs = dict(init_point_path=str(path), height_offset=0.2, robot_visible=True, light_enabled=False)
    if reset_name == "imagenav_reset": kwargs["camera_offset"] = 0.3
    ns[reset_name](env, ids, **kwargs)
    pose = writes[0][0]
    expected_xyzw = Rotation.from_euler("z", yaw[:, None]).as_quat()
    np.testing.assert_allclose(pose[:, 3:].numpy(), expected_xyzw, atol=1e-6)
    np.testing.assert_allclose(Rotation.from_quat(pose[:, 3:].numpy()).apply(np.array([[1., 0., 0.], [1., 0., 0.]])), [[1., 0., 0.], [0., 1., 0.]], atol=1e-6)
    expected_wxyz = expected_xyzw[:, [3, 0, 1, 2]]
    regular_goals = [item for item in goals if item[0].endswith("/Goal")]
    for i, (_, _, quaternion) in enumerate(regular_goals):
        np.testing.assert_allclose(quaternion.numpy()[0], expected_wxyz[i], atol=1e-6)
    if reset_name == "imagenav_reset":
        image_goals = [item for item in goals if item[0].endswith("/goal_cam")]
        np.testing.assert_allclose([item[1][0, 2].item() for item in image_goals], [1.1, 1.0], atol=1e-6)


def test_navdp_oracle_observation_unwraps_lab3_proxy_for_math():
    from scipy.spatial.transform import Rotation
    class Proxy:
        def __init__(self, value): self.torch = value
        def __getattr__(self, name): raise AssertionError("implicit ProxyArray access: " + name)
    root = torch.tensor([[1., 2., 0.], [3., 4., 0.]])
    quat = torch.tensor(Rotation.from_euler("z", np.array([[0.], [math.pi/2]])).as_quat(), dtype=torch.float32)
    goals = torch.tensor([[2., 2., 0.], [3., 5., 0.]])
    def matrix(quaternion):
        assert isinstance(quaternion, torch.Tensor), "scripted Lab math requires a Tensor"
        return torch.tensor(Rotation.from_quat(quaternion.numpy()).as_matrix(), dtype=torch.float32)
    asset = SimpleNamespace(data=SimpleNamespace(root_pos_w=Proxy(root), root_quat_w=Proxy(quat)))
    ns = _definitions("third_party/navdp/configs/tasks/wheeled_task.py", ["oracle_imu_pose_data"], {
        "torch": torch, "ManagerBasedEnv": object, "SceneEntityCfg": lambda name: SimpleNamespace(name=name),
        "math_utils": SimpleNamespace(matrix_from_quat=matrix),
        "XFormPrimView": lambda **kwargs: SimpleNamespace(get_world_poses=lambda: (goals, None)),
    })
    result = ns["oracle_imu_pose_data"](SimpleNamespace(scene={"robot": asset}))
    torch.testing.assert_close(result, torch.tensor([[1., 0., 0.], [1., 0., 0.]]), atol=1e-6, rtol=1e-6)


def test_xnavdp_guidance_receives_lab3_xyzw_without_double_conversion():
    quat = torch.tensor([[0., 0., math.sin(0.25), math.cos(0.25)]])
    pose = torch.tensor([[1., 2., 3.]])
    robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=pose), root_quat_w=SimpleNamespace(torch=quat)))
    env = SimpleNamespace(unwrapped=SimpleNamespace(scene={"robot": robot}))
    ns = _definitions("third_party/navdp/baselines/x-navdp/eval/scripts/evaluate_pointgoal.py", ["add_robot_state"], {})
    result = ns["add_robot_state"]({"rgb": 1}, env)
    assert result["robot_rot"] is quat and result["robot_pose"] is pose



def test_navdp_stuck_termination_is_per_environment_and_snapshots_shared_memory():
    from collections import deque
    positions = torch.zeros(2, 3)
    asset = SimpleNamespace(data=SimpleNamespace(root_pos_w=SimpleNamespace(torch=positions)))
    env = SimpleNamespace(scene={"robot": asset})
    ns = _definitions("third_party/navdp/configs/tasks/wheeled_task.py", ["stuck_terminal_check"], {
        "torch": torch, "deque": deque, "ManagerBasedEnv": object,
        "SceneEntityCfg": lambda name: SimpleNamespace(name=name),
    })
    check = ns["stuck_terminal_check"]
    assert check(env, window_size=3).tolist() == [False, False]
    positions[1, 0] = 1.0
    assert check(env, window_size=3).tolist() == [False, False]
    positions[1, 0] = 2.0
    assert check(env, window_size=3).tolist() == [True, False]
    with pytest.raises(ValueError, match="window_size"):
        check(env, window_size=1)
