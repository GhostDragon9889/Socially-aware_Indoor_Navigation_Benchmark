"""CPU contracts and USD checks, not a live PhysX ground-contact probe."""
from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest
import torch

from CrowdSim.humanoid.foot_support import (
    FOOT_NAMES,
    FootSupportMonitor,
    body_link_world_com,
    body_name_permutation,
    body_origin_tangential_speed,
    mass_weighted_world_com,
    resolve_floor_collision_paths,
)


def paths(actors=1):
    return [{name: f"/World/envs/env_{i}/Robot/nested/{name}" for name in FOOT_NAMES}
            for i in range(actors)]


FLOORS = ("/World/ground/terrain/mesh", "/World/ground/second/mesh")


class ReusedBufferView:
    """Explicit mock with native metadata reordering and overwritten buffers."""

    def __init__(self, actor_paths, frames, *, permutation=None, rows=None):
        sensors = [actor[name] for actor in actor_paths for name in FOOT_NAMES]
        permutation = list(range(len(sensors))) if permutation is None else permutation
        self.sensor_paths = [sensors[i] for i in permutation]
        self.sensor_count, self.filter_count = len(sensors), len(FLOORS)
        self.filter_paths = [list(FLOORS) for _ in sensors] if rows is None else rows
        self.frames = list(frames)
        self.buffer = torch.empty_like(self.frames[0])
        self.calls = []

    def get_contact_force_matrix(self, *, dt):
        self.buffer.copy_(self.frames[len(self.calls)])
        self.calls.append(dt)
        return self.buffer


def monitor(frames=None, actor_paths=None, **kwargs):
    actor_paths = paths() if actor_paths is None else actor_paths
    frames = [torch.zeros(4 * len(actor_paths), 2, 3)] * 4 if frames is None else frames
    view = ReusedBufferView(actor_paths, frames)
    result = FootSupportMonitor(view, actor_paths, FLOORS, contact_reporting_enabled=True, **kwargs)
    result.begin_control_step(episode_ids=[0] * len(actor_paths), control_step=0, physics_step_before=10)
    return result, view


def sample_all(sensor, *, velocities=None, positions=None, episodes=None, start=11):
    episodes = [0] * sensor.num_actors if episodes is None else episodes
    for i in range(sensor.substeps):
        sensor.sample_completed_substep(1 / 120, episode_ids=episodes, physics_step=start + i,
                                        foot_body_origin_vel_w=velocities, foot_body_origin_pos_w=positions)


def test_known_transient_support_mean_peak_and_airborne_intervals_survive_buffer_reuse():
    frames = torch.zeros(4, 4, 2, 3)
    frames[0, 0, 0, 2] = 80.  # left ankle only
    frames[1, 1, 0, 2] = 40.  # left toe
    frames[1, 2, 1, 2] = 60.  # right ankle, simultaneous double support
    frames[2, 3, 0, 2] = 100.  # right toe only
    sensor, view = monitor(list(frames))
    sample_all(sensor)
    observed = sensor.measurements()
    assert view.calls == [1 / 120] * 4
    assert observed["side_normal_load_mean_n"].tolist() == [[30., 40.]]
    assert observed["side_normal_load_peak_n"].tolist() == [[80., 100.]]
    assert observed["side_contact_substeps"].tolist() == [[[True, False]], [[True, True]], [[False, True]], [[False, False]]]
    assert observed["double_support_fraction"].item() == .25
    assert observed["no_support_fraction"].item() == .25
    assert observed["actor_substep_valid_count"].tolist() == [4]
    assert observed["foot_floor_normal_force_w"][0, 0, 0, 0, 2] == 80
    # Mutating returned measurements and the reused native buffer cannot corrupt stored snapshots.
    observed["foot_floor_normal_force_w"].fill_(999)
    view.buffer.fill_(444)
    assert sensor.measurements()["foot_floor_normal_force_w"][0, 0, 0, 0, 2] == 80


def test_metadata_reorders_two_actors_and_per_sensor_ground_columns():
    actors = paths(2)
    permutation = [7, 0, 4, 2, 5, 1, 6, 3]
    rows = [list(reversed(FLOORS)) if i % 2 else list(FLOORS) for i in range(8)]
    native = torch.zeros(8, 2, 3)
    for row, canonical in enumerate(permutation):
        for col, floor in enumerate(rows[row]):
            native[row, col, 2] = 10 * canonical + FLOORS.index(floor) + 1
    view = ReusedBufferView(actors, [native] * 4, permutation=permutation, rows=rows)
    sensor = FootSupportMonitor(view, actors, FLOORS, contact_reporting_enabled=True)
    sensor.begin_control_step(episode_ids=[0, 4], control_step=12, physics_step_before=100)
    sample_all(sensor, episodes=[0, 4], start=101)
    force = sensor.measurements()["foot_floor_normal_force_w"][0].reshape(8, 2, 3)
    assert force[:, :, 2].tolist() == [[10 * i + 1, 10 * i + 2] for i in range(8)]


@pytest.mark.parametrize("layout", ["shared", "flat_rows"])
def test_explicit_native_filter_layouts_are_resolved_without_guessing_order(layout):
    sensor, view = monitor()
    view.filter_paths = list(FLOORS) if layout == "shared" else list(FLOORS) * 4
    resolved = FootSupportMonitor(view, paths(), FLOORS, contact_reporting_enabled=True)
    assert resolved._filter_indices == ((0, 1),) * 4


def test_factory_receives_only_exact_foot_and_floor_paths_and_no_net_force_api():
    calls = []

    def factory(sources, **kwargs):
        calls.append((sources, kwargs))
        return ReusedBufferView(paths(2), [torch.zeros(8, 2, 3)] * 4)

    sensor = FootSupportMonitor.from_factory(factory, paths(2), FLOORS, contact_reporting_enabled=True)
    assert calls[0][0] == [actor[name] for actor in paths(2) for name in FOOT_NAMES]
    assert calls[0][1]["filter_patterns"] == [list(FLOORS)] * 8
    assert sensor.manifest["native_physical_probe_passed"] is False
    assert sensor.manifest["author_metric_equivalence"] is False


@pytest.mark.parametrize("mutation", ["sensor_missing", "sensor_duplicate", "filter_wall", "filter_missing", "count"])
def test_changed_or_non_floor_native_partners_fail_closed(mutation):
    view = ReusedBufferView(paths(), [torch.zeros(4, 2, 3)] * 4)
    if mutation == "sensor_missing":
        view.sensor_paths[0] = "/World/other/L_Ankle"
    elif mutation == "sensor_duplicate":
        view.sensor_paths[0] = view.sensor_paths[1]
    elif mutation == "filter_wall":
        view.filter_paths[1][0] = "/World/wall"
    elif mutation == "filter_missing":
        view.filter_paths[0] = [FLOORS[0]]
    else:
        view.filter_count = 1
    with pytest.raises(RuntimeError):
        FootSupportMonitor(view, paths(), FLOORS, contact_reporting_enabled=True)


def test_semantic_feet_paths_reporting_activation_and_duplicates_must_be_explicit():
    view = ReusedBufferView(paths(), [torch.zeros(4, 2, 3)] * 4)
    with pytest.raises(ValueError, match="enabled"):
        FootSupportMonitor(view, paths(), FLOORS, contact_reporting_enabled=False)
    incorrect = paths()
    incorrect[0]["L_Ankle"] = "/World/envs/env_0/Robot/R_Toe"
    with pytest.raises(ValueError, match="semantics"):
        FootSupportMonitor(view, incorrect, FLOORS, contact_reporting_enabled=True)
    with pytest.raises(ValueError, match="duplicate"):
        FootSupportMonitor(view, paths(), [FLOORS[0], FLOORS[0]], contact_reporting_enabled=True)
    wildcard = paths()
    wildcard[0]["L_Toe"] = "/World/envs/env_*/Robot/L_Toe"
    with pytest.raises(ValueError, match="exact"):
        FootSupportMonitor(view, wildcard, FLOORS, contact_reporting_enabled=True)


@pytest.mark.parametrize("fault", ["control_dt", "repeated_step", "skipped_step", "reset", "shape", "nan"])
def test_bad_substep_poisoning_prevents_partial_or_stale_measurements(fault):
    sensor, view = monitor()
    if fault == "shape":
        view.frames[0] = torch.zeros(3, 2, 3)
        view.buffer = torch.empty(3, 2, 3)
    elif fault == "nan":
        view.frames[0] = torch.full((4, 2, 3), float("nan"))
    dt = 1 / 30 if fault == "control_dt" else 1 / 120
    step = 10 if fault == "repeated_step" else 12 if fault == "skipped_step" else 11
    episodes = [1] if fault == "reset" else [0]
    with pytest.raises((ValueError, RuntimeError, FloatingPointError)):
        sensor.sample_completed_substep(dt, episode_ids=episodes, physics_step=step)
    with pytest.raises(RuntimeError, match="partial"):
        sensor.measurements()
    with pytest.raises(RuntimeError, match="reset"):
        sensor.begin_control_step(episode_ids=[1], control_step=10, physics_step_before=20)
    sensor.reset()
    assert sensor.reset_count == 1
    with pytest.raises(RuntimeError):
        sensor.measurements()


def test_partial_extra_substeps_and_control_gaps_are_rejected():
    sensor, _ = monitor()
    sensor.sample_completed_substep(1 / 120, episode_ids=[0], physics_step=11)
    with pytest.raises(RuntimeError, match="partial"):
        sensor.measurements()
    with pytest.raises(RuntimeError, match="reset"):
        sensor.begin_control_step(episode_ids=[0], control_step=1, physics_step_before=11)
    for i in range(12, 15):
        sensor.sample_completed_substep(1 / 120, episode_ids=[0], physics_step=i)
    with pytest.raises(RuntimeError, match="incomplete"):
        sensor.sample_completed_substep(1 / 120, episode_ids=[0], physics_step=15)
    with pytest.raises(RuntimeError, match="consecutive"):
        sensor.begin_control_step(episode_ids=[0], control_step=2, physics_step_before=14)
    with pytest.raises(RuntimeError, match="Reset"):
        sensor.begin_control_step(episode_ids=[1], control_step=1, physics_step_before=14)
    sensor.begin_control_step(episode_ids=[0], control_step=1, physics_step_before=14)
    with pytest.raises(RuntimeError, match="partial"):
        sensor.measurements()


def test_load_uses_vector_magnitude_and_contact_is_strict_per_body():
    frame = torch.zeros(4, 2, 3)
    frame[0, 0] = torch.tensor([.6, .8, 0.])  # exactly threshold, not contact
    frame[1, 0] = torch.tensor([0., 0., .75])
    frame[2, 1] = torch.tensor([.8, .8, 0.])  # magnitude > threshold
    sensor, _ = monitor([frame] * 4)
    sample_all(sensor)
    observed = sensor.measurements()
    assert observed["side_normal_load_mean_n"][0, 0] == pytest.approx(1.75)
    assert observed["side_contact_any"].tolist() == [[False, True]]


def test_body_origin_slip_proxy_excludes_vertical_speed_and_has_explicit_no_contact_validity():
    frame = torch.zeros(4, 2, 3)
    frame[0, 0, 2] = 20
    velocity = torch.tensor([[[3., 4., 100.], [7., 0., 0.], [0., 0., 9.], [1., 0., 0.]]])
    position = torch.ones(1, 4, 3)
    sensor, _ = monitor([frame] * 4)
    sample_all(sensor, velocities=velocity, positions=position)
    velocity.fill_(999)
    position.fill_(777)
    with pytest.raises(ValueError, match="plane normal"):
        sensor.measurements()
    observed = sensor.measurements(ground_normal_w=torch.tensor([0., 0., 1.]))
    assert observed["body_origin_tangential_speed_proxy_mps"][0, 0].tolist() == [5., 7., 0., 1.]
    assert observed["supported_speed_proxy_valid"].tolist() == [[True, False, False, False]]
    assert observed["supported_body_origin_tangential_speed_proxy_mean_mps"][0, 0] == 5
    assert observed["foot_body_origin_pos_w"][0, 0, 0, 0] == 1


def test_optional_velocity_channel_cannot_disappear_mid_control():
    sensor, _ = monitor()
    sensor.sample_completed_substep(1 / 120, episode_ids=[0], physics_step=11,
                                    foot_body_origin_vel_w=torch.zeros(1, 4, 3))
    with pytest.raises(ValueError, match="availability"):
        sensor.sample_completed_substep(1 / 120, episode_ids=[0], physics_step=12)


def test_proxy_array_shape_does_not_override_its_explicit_torch_vector_view():
    class Proxy:
        shape = (4, 2)  # Warp vector array omits vector components.

        def __init__(self):
            self.torch = torch.zeros(4, 2, 3)
            self.torch[0, 0, 2] = 30

    proxy = Proxy()
    sensor, view = monitor()
    view.get_contact_force_matrix = lambda **kwargs: proxy
    sample_all(sensor)
    assert sensor.measurements()["side_normal_load_mean_n"].tolist() == [[30., 0.]]


@pytest.mark.skipif(importlib.util.find_spec("warp") is None, reason="CPU Warp conversion unavailable")
def test_actual_cpu_warp_vec3_array_is_converted_before_shape_validation():
    import warp as wp

    wp.init()
    backing = torch.zeros(4, 2, 3)
    backing[2, 1, 2] = 55
    warp_array = wp.from_torch(backing, dtype=wp.vec3f)
    assert tuple(warp_array.shape) == (4, 2)
    sensor, view = monitor()
    view.get_contact_force_matrix = lambda **kwargs: warp_array
    sample_all(sensor)
    assert sensor.measurements()["side_normal_load_mean_n"].tolist() == [[0., 55.]]
    # This exercises a real CPU frontend, not a PhysX contact backend.


def test_finite_inputs_that_overflow_reductions_are_not_published_as_valid_support():
    sensor, _ = monitor([torch.full((4, 2, 3), 1e38)] * 4)
    sample_all(sensor)
    with pytest.raises(FloatingPointError, match="loads"):
        sensor.measurements()


def test_local_com_composition_rotates_offset_and_mass_weighting_is_not_body_origin_mean():
    position = torch.tensor([[[10., 0., 0.], [0., 0., 0.]]], dtype=torch.float64)
    angle = torch.tensor(torch.pi / 4, dtype=torch.float64)
    quat = torch.tensor([[[0., 0., angle.sin(), angle.cos()], [0., 0., 0., 1.]]], dtype=torch.float64)
    offset = torch.tensor([[[2., 0., 0.], [0., 0., 3.]]], dtype=torch.float64)
    com = body_link_world_com(position, quat, offset)
    torch.testing.assert_close(com, torch.tensor([[[10., 2., 0.], [0., 0., 3.]]], dtype=torch.float64))
    weighted = mass_weighted_world_com(com, torch.tensor([[1., 3.]], dtype=torch.float64))
    torch.testing.assert_close(weighted, torch.tensor([[2.5, .5, 2.25]], dtype=torch.float64))
    assert not torch.allclose(weighted, position.mean(1))
    assert body_name_permutation(["R_Toe", "L_Toe"], ["L_Toe", "R_Toe"]) == (1, 0)


def test_com_names_quaternions_masses_and_tangent_contracts_fail_closed():
    with pytest.raises(ValueError, match="bijection"):
        body_name_permutation(["L_Toe", "L_Toe"], ["L_Toe", "R_Toe"])
    position = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match="unit"):
        body_link_world_com(position, torch.zeros(1, 2, 4), position)
    with pytest.raises(ValueError, match="positive"):
        mass_weighted_world_com(position, torch.tensor([[1., 0.]]))
    with pytest.raises(FloatingPointError):
        mass_weighted_world_com(position, torch.tensor([[1., float("nan")]]))
    with pytest.raises(ValueError, match="unit"):
        body_origin_tangential_speed(position, torch.tensor([0., 0., 2.]))


@pytest.mark.skipif(importlib.util.find_spec("pxr") is None, reason="Standalone USD bindings unavailable")
def test_real_cpu_usd_floor_resolution_excludes_disabled_collider_and_xform_root():
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/ground")
    floor = UsdGeom.Mesh.Define(stage, "/World/ground/terrain/mesh").GetPrim()
    UsdPhysics.CollisionAPI.Apply(floor)
    disabled = UsdGeom.Cube.Define(stage, "/World/ground/disabled").GetPrim()
    UsdPhysics.CollisionAPI.Apply(disabled).CreateCollisionEnabledAttr(False)
    UsdGeom.Xform.Define(stage, "/World/empty")
    assert resolve_floor_collision_paths(stage, ["/World/ground"]) == (FLOORS[0],)
    with pytest.raises(ValueError, match="enabled"):
        resolve_floor_collision_paths(stage, ["/World/empty"])
    with pytest.raises(ValueError, match="does not exist"):
        resolve_floor_collision_paths(stage, ["/World/missing"])


def test_import_does_not_bootstrap_sdk_or_warp():
    code = "import sys; import CrowdSim.humanoid.foot_support; assert not any(k.startswith(('omni.', 'isaaclab', 'isaacsim', 'warp')) for k in sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
