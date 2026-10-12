"""CPU contracts for SDK boundaries; these tests do not replace physics tests."""
from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from protomotions.simulator.isaaclab.utils.runtime_compat import (
    as_torch,
    supports_native_usd_only_replication,
    write_root_link_state,
    write_root_link_velocity,
)

ROOT = Path(__file__).resolve().parents[1]
SIMULATOR = ROOT / "src/protomotions/simulator/isaaclab/simulator.py"


def load_simulator_method(name):
    """Exercise a real adapter method without importing Kit at collection time."""
    module = ast.parse(SIMULATOR.read_text())
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "IsaacLabSimulator")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    isolated = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
    from protomotions.simulator.base_simulator.simulator_state import ObjectState, StateConversion
    from protomotions.simulator.isaaclab.utils.materials import set_material_friction
    namespace = {"torch": torch, "as_torch": as_torch, "write_root_link_state": write_root_link_state,
                 "ObjectState": ObjectState, "StateConversion": StateConversion,
                 "wp": wp, "set_material_friction": set_material_friction,
                 "write_root_link_velocity": write_root_link_velocity}
    exec(compile(ast.fix_missing_locations(isolated), str(SIMULATOR), "exec"), namespace)
    return namespace[name]


@pytest.fixture(scope="module", autouse=True)
def initialize_cpu_warp():
    wp.init()


def test_warp_torch_zero_copy_and_torch_identity():
    source = torch.tensor([1.0, 2.0, 3.0])
    assert as_torch(source) is source
    array = wp.from_torch(source)
    view = as_torch(array)
    view[1] = 4.0
    assert source[1].item() == 4.0


@pytest.mark.parametrize("sdk_root", [
    Path(os.environ.get("ISAACLAB_ROOT", ROOT.parent.parent / "IsaacLab")),
    Path(os.environ.get("SOCIAL_NAVIGATION_REFERENCE_LAB_ROOT", ROOT.parent / "IsaacLab-Protomotions")),
])
def test_actual_sdk_proxy_array_explicit_view(sdk_root):
    path = Path(sdk_root) / "source/isaaclab/isaaclab/utils/warp/proxy_array.py"
    if not path.is_file():
        pytest.skip("Local SDK source is unavailable")
    spec = importlib.util.spec_from_file_location("contract_sdk_proxy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    proxy = module.ProxyArray(wp.from_torch(source))
    assert as_torch(proxy).data_ptr() == source.data_ptr()
    as_torch(proxy)[0, 1] = 9.0
    assert source[0, 1].item() == 9.0


def test_replication_protocol_capability_without_version_guess():
    def old(plan, *, stage):
        pass
    def new(plan, *, stage, replicate_physics=True):
        pass
    assert not supports_native_usd_only_replication(old)
    assert supports_native_usd_only_replication(new)


def test_root_link_state_preserves_xyzw_and_link_velocities():
    calls = []
    asset = SimpleNamespace(
        data=SimpleNamespace(
            body_com_pos_b=torch.zeros(3, 1, 3),
            root_link_quat_w=torch.tensor([[0., 0., 0., 1.]]).repeat(3, 1),
        ),
        write_root_link_pose_to_sim_index=lambda **kw: calls.append(("pose", kw)),
        write_root_com_velocity_to_sim_index=lambda **kw: calls.append(("velocity", kw)),
    )
    state = torch.tensor([[1., 2., 3., 0., 0., 0., 1., 4., 5., 6., 7., 8., 9.]])
    ids = torch.tensor([2])
    write_root_link_state(asset, state, env_ids=ids)
    assert [name for name, _ in calls] == ["pose", "velocity"]
    torch.testing.assert_close(calls[0][1]["root_pose"], state[:, :7])
    torch.testing.assert_close(calls[1][1]["root_velocity"], state[:, 7:])
    assert calls[0][1]["env_ids"] is ids
    assert calls[1][1]["env_ids"] is ids


@pytest.mark.parametrize("ids", [None, torch.tensor([3, 1]), torch.tensor([], dtype=torch.long)])
def test_object_contacts_select_environments_after_object_concat(ids):
    first = torch.arange(4 * 1 * 2 * 3, dtype=torch.float32).reshape(4, 1, 2, 3)
    second = first + 100
    method = load_simulator_method("_get_simulator_object_contact_buf")
    host = SimpleNamespace(
        scene_lib=SimpleNamespace(num_scenes=lambda: 1, num_objects_per_scene=2),
        _object_contact_sensor=[
            SimpleNamespace(data=SimpleNamespace(force_matrix_w=first)),
            SimpleNamespace(data=SimpleNamespace(force_matrix_w=second)),
        ],
        num_envs=4, device="cpu",
    )
    actual = method(host, ids)
    expected = torch.cat([first, second], dim=1)
    if ids is not None:
        expected = expected[ids]
    torch.testing.assert_close(actual, expected)


def test_object_contact_no_scene_subset():
    method = load_simulator_method("_get_simulator_object_contact_buf")
    host = SimpleNamespace(scene_lib=SimpleNamespace(num_scenes=lambda: 0), num_envs=4, device="cpu")
    actual = method(host, torch.tensor([3, 1]))
    assert actual.shape == (2, 1, 1, 3)
    assert not actual.any()


def test_projectile_write_does_not_mutate_caller_positions():
    method = load_simulator_method("_set_projectile_root_states")
    writes = []
    asset = SimpleNamespace(
        data=SimpleNamespace(
            body_com_pos_b=torch.zeros(4, 1, 3),
            root_link_quat_w=torch.tensor([[0., 0., 0., 1.]]).repeat(4, 1),
        ),
        write_root_link_pose_to_sim_index=lambda **kw: writes.append(kw["root_pose"].clone()),
        write_root_com_velocity_to_sim_index=lambda **kw: None,
    )
    host = SimpleNamespace(_proj_config=SimpleNamespace(hide_z=-50), _projectile_objects=[asset])
    position = torch.tensor([[0., 0., -60.], [4., 5., 1.]])
    previous = position.clone()
    method(host, torch.tensor([0, 0]), position, torch.tensor([[0., 0., 0., 1.]]).repeat(2, 1),
           torch.zeros(2, 3), torch.zeros(2, 3), torch.tensor([3, 2]))
    torch.testing.assert_close(position, previous)
    torch.testing.assert_close(writes[0][0, :3], torch.tensor([3., 3., -60.]))
    torch.testing.assert_close(writes[0][1, :3], previous[1])


def test_mjcf_cache_changes_for_sdk_upgrade(tmp_path, monkeypatch):
    # Load real cache functions without pulling in the optional MJCF parser or Kit.
    import hashlib
    path = ROOT / "src/protomotions/simulator/isaaclab/utils/mjcf_to_usd.py"
    tree = ast.parse(path.read_text())
    names = {"default_usd_cache_dir", "conversion_cache_key", "_absolute_path", "_normalized_options"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"Path": Path, "hashlib": hashlib, "_MJCF_CONVERTER_CACHE_VERSION": "contract", "_mjcf_fingerprint": lambda _: "unchanged-input"}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[
        ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes
    ], type_ignores=[])), str(path), "exec"), namespace)
    source = str(tmp_path / "humanoid.xml")
    namespace["_sdk_converter_versions"] = lambda: (("isaacsim", "6.0.0.1"),)
    old_dir = namespace["default_usd_cache_dir"](source, {})
    cfg = {"asset_path": source, "usd_dir": str(tmp_path)}
    old_key = namespace["conversion_cache_key"](cfg)
    namespace["_sdk_converter_versions"] = lambda: (("isaacsim", "6.1.0.0"),)
    assert namespace["default_usd_cache_dir"](source, {}) != old_dir
    assert namespace["conversion_cache_key"](cfg) != old_key


def test_simulation_cfg_explicitly_preserves_trained_dynamics_defaults():
    tree = ast.parse(SIMULATOR.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    sim_call = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "SimulationCfg")
    physics_call = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "PhysxCfg")
    assert next(kw.value.value for kw in sim_call.keywords if kw.arg == "use_newton_actuators") is False
    assert next(kw.value.value for kw in physics_call.keywords if kw.arg == "enable_external_forces_every_iteration") is False


@pytest.mark.parametrize("ids", [None, torch.tensor([3, 1]), torch.tensor([], dtype=torch.long)])
def test_empty_scene_has_zero_object_root_state(ids):
    method = load_simulator_method("_get_simulator_object_root_state")
    host = SimpleNamespace(_object=[], num_envs=4, device="cpu")
    state = method(host, ids)
    n = 4 if ids is None else len(ids)
    assert state.root_pos.shape == (n, 0, 3)
    assert state.root_rot.shape == (n, 0, 4)
    assert state.root_vel.shape == (n, 0, 3)
    assert state.root_ang_vel.shape == (n, 0, 3)


@pytest.mark.parametrize("action", ["owner", "actions", "register", "unregister_all"])
def test_key_scope_rejects_unreachable_action_names(action):
    from protomotions.simulator.base_simulator.user_interface import UserInterface
    ui = UserInterface()
    scope = ui.scope("test")
    with pytest.raises(ValueError, match="shadows"):
        scope.register("K", action, "Reserved action")
    assert not ui.registered_keys


def test_mass_randomization_scales_inertia_from_pre_write_mass():
    # Lab 3 setters update the live buffers in place; deprecated default_mass
    # captures its snapshot only on first access, including after a mass write.
    class AssetData:
        def __init__(self):
            self.body_mass = torch.tensor([[2., 3., 4.], [5., 6., 8.]])
            self.body_inertia = torch.arange(1., 55.).reshape(2, 3, 9)
            self._default_mass = None

        @property
        def default_mass(self):
            if self._default_mass is None:
                self._default_mass = self.body_mass.clone()
            return SimpleNamespace(torch=self._default_mass)

        @property
        def default_inertia(self):
            return SimpleNamespace(torch=self.body_inertia.clone())

    data = AssetData()
    original_mass = data.body_mass.clone()
    original_inertia = data.body_inertia.clone()
    body_ids = torch.tensor([2, 0])
    target_mass = torch.tensor([[16., 10.], [16., 10.]])

    def set_masses(*, masses, body_ids, env_ids):
        data.body_mass[env_ids[:, None], body_ids] = masses

    def set_inertias(*, inertias, body_ids, env_ids):
        data.body_inertia[env_ids[:, None], body_ids] = inertias

    robot = SimpleNamespace(
        data=data,
        find_bodies=lambda names, preserve_order: ([2, 0], names),
        set_masses_index=set_masses,
        set_inertias_index=set_inertias,
        root_view=SimpleNamespace(
            get_material_properties=lambda: torch.zeros(2, 3, 3),
            set_material_properties=lambda *args: None,
        ),
    )
    host = SimpleNamespace(
        config=SimpleNamespace(num_envs=2, default_robot_friction=1.),
        num_envs=2, device="cpu", _robot=robot,
        robot_config=SimpleNamespace(kinematic_info=SimpleNamespace(body_names=["root", "left", "right"])),
        _domain_randomization={"body_mass": {"body_indices": [0, 2], "mass": target_mass[:1]}},
        _apply_scene_object_properties_after_spawn=lambda ids: None,
    )
    load_simulator_method("_apply_domain_randomization_if_needed")(host)
    expected_inertia = original_inertia.clone()
    expected_inertia[:, body_ids] *= (target_mass / original_mass[:, body_ids])[..., None]
    torch.testing.assert_close(data.body_mass[:, body_ids], target_mass)
    torch.testing.assert_close(data.body_inertia, expected_inertia)
    torch.testing.assert_close(data.body_mass[:, 1], original_mass[:, 1])


@pytest.mark.parametrize("world_translation", [0., 1_000_000.])
@pytest.mark.parametrize("num_bodies", [1, 2])
def test_root_link_state_round_trip_with_offset_com_and_reordered_subset(
    world_translation, num_bodies
):
    # The inspected Lab 3 PhysX articulation link setter passes link velocities
    # to a backend setter that expects COM velocities. Model that exact boundary.
    poses = torch.zeros(3, 7)
    poses[:, 6] = 1.
    velocities = torch.zeros(3, 6)
    offsets = torch.tensor([[.2, -.3, .4], [.1, .2, .3], [-.1, .25, -.2]])
    local_com = torch.full((3, num_bodies, 3), 9.)
    local_com[:, 0] = torch.tensor([[.2, -.3, .4], [.1, .2, .3], [.25, .1, -.2]])

    class Data:
        @property
        def root_pos_w(self):
            return poses[:, :3]

        @property
        def root_link_quat_w(self):
            return poses[:, 3:7]

        @property
        def body_com_pos_b(self):
            return local_com

        @property
        def root_com_pos_w(self):
            return poses[:, :3] + offsets

        @property
        def root_link_vel_w(self):
            link_velocity = velocities.clone()
            link_velocity[:, :3] -= torch.cross(velocities[:, 3:], offsets, dim=-1)
            return link_velocity

    def write_pose(*, root_pose, env_ids):
        poses[env_ids] = root_pose

    def write_backend_velocity(*, root_velocity, env_ids):
        velocities[env_ids] = root_velocity

    asset = SimpleNamespace(
        data=Data(),
        write_root_link_pose_to_sim_index=write_pose,
        write_root_link_velocity_to_sim_index=write_backend_velocity,
        write_root_com_velocity_to_sim_index=write_backend_velocity,
    )
    ids = torch.tensor([2, 0])
    state = torch.tensor([
        [3., 4., 5., 0., 0., .70710678, .70710678, .1, .2, .3, 1., 2., 3.],
        [-2., 1., 6., 0., 0., 0., 1., .4, .5, .6, -3., 2., 1.],
    ])
    state[:, :3] += world_translation
    write_root_link_state(asset, state, env_ids=ids)
    actual_link_velocity = velocities[ids, :3] - torch.cross(
        velocities[ids, 3:], offsets[ids], dim=-1
    )
    torch.testing.assert_close(actual_link_velocity, state[:, 7:10])
    torch.testing.assert_close(velocities[ids, 3:], state[:, 10:13])
    assert not velocities[1].any()
    linear_impulse = torch.tensor([[.1, -.2, .3], [-.4, .5, -.6]])
    angular_impulse = torch.tensor([[.4, -.5, .6], [-.1, .2, -.3]])
    load_simulator_method("_apply_root_velocity_impulse")(
        SimpleNamespace(_robot=asset), linear_impulse, angular_impulse, ids
    )
    torch.testing.assert_close(
        asset.data.root_link_vel_w[ids],
        state[:, 7:] + torch.cat([linear_impulse, angular_impulse], dim=-1),
    )
    assert not velocities[1].any()
