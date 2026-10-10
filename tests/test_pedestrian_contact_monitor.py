"""CPU aggregation contracts; these do not claim a live PhysX contact probe."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import warp as wp

from CrowdSim.humanoid.contact_monitor import (
    PedestrianContactMonitor,
    contact_filter_layout,
    resolve_static_collision_paths,
)


@pytest.fixture(scope="module", autouse=True)
def initialize_cpu_warp():
    wp.init()


class BufferView:
    def __init__(self, frames):
        self.frames = frames
        self.calls = []

    def get_contact_force_matrix(self, *, dt):
        tensor = self.frames[min(len(self.calls), len(self.frames) - 1)]
        self.calls.append(dt)
        return wp.from_torch(tensor, dtype=wp.vec3f)


def monitor(views, actors=3):
    result = PedestrianContactMonitor.__new__(PedestrianContactMonitor)
    result.num_envs = actors
    result.device = torch.device("cpu")
    result.threshold_n = 1.
    result.views = [tuple(view) + (1,) if len(view) == 6 else tuple(view) for view in views]
    result.static_max_force_n = torch.zeros(actors)
    result.peer_max_force_n = torch.zeros(actors, actors)
    result.substep_count = 0
    return result


def test_exact_partner_layout_excludes_floor_and_self_and_preserves_actor_columns():
    bodies = [[f"/World/envs/env_{i}/Robot/{name}" for name in ("Pelvis", "nested/L_Ankle")]
              for i in range(3)]
    obstacles = ["/World/PaperValidation/Obstacle_0/geometry/mesh"]
    layouts = contact_filter_layout(bodies, obstacles)
    middle = layouts[1]
    assert middle["sources"] == tuple(bodies[1])
    assert middle["filters"] == tuple(obstacles + bodies[0] + bodies[2])
    assert middle["partner_ids"] == (-1, 0, 0, 2, 2)
    assert set(middle["sources"]).isdisjoint(middle["filters"])
    assert all("ground" not in path for path in middle["filters"])


@pytest.mark.parametrize("bodies,static", [
    ([["/body"], ["/body"]], []),
    ([["/body"]], ["/body"]),
    ([["/body"], ["/other", "/other2"]], []),
    ([["/body"]], ["/World/ground/terrain/mesh"]),
    ([["/World/envs/env_*/Robot/Pelvis"]], []),
    ([["/body"]], ["/static", "/static"]),
])
def test_invalid_partner_or_floor_paths_do_not_silently_report_no_collision(bodies, static):
    with pytest.raises(ValueError):
        contact_filter_layout(bodies, static)


def test_transient_contact_and_multiple_source_bodies_survive_four_substeps():
    zero = torch.zeros(1, 5, 3)
    first = zero.clone()
    first[0, 0] = torch.tensor([0., 0., 2.])
    first[0, 2] = torch.tensor([3., 4., 0.])
    second_body = zero.clone()
    second_body[0, 3] = torch.tensor([0., 0., 7.])
    view1 = BufferView([first, zero, zero, zero])
    view2 = BufferView([zero, second_body, zero, zero])
    sensor = monitor([(0, view1, 1, [1, 2], 2, [-1, 1, 1, 2, 2]),
                      (0, view2, 1, [1, 2], 2, [-1, 1, 1, 2, 2])])
    sensor.begin_control_step()
    for _ in range(4):
        sensor.sample(1 / 120)
    observed = sensor.measurements()
    assert sensor.substep_count == 4
    assert view1.calls == [1 / 120] * 4
    assert observed["static_contact"].tolist() == [True, False, False]
    assert observed["peer_contact"][0].tolist() == [False, True, True]
    assert observed["static_contact_max_force_n"].tolist() == [2., 0., 0.]
    assert observed["peer_contact_max_force_n"][0].tolist() == [0., 5., 7.]
    assert not observed["peer_contact"].diagonal().any()
    sensor.begin_control_step()
    with pytest.raises(RuntimeError, match="completed physics substep"):
        sensor.measurements()
    sensor.sample(1 / 120)
    assert not sensor.measurements()["peer_contact"].any()
    # Returned force data owns its memory across accumulator resets.
    assert observed["peer_contact_max_force_n"][0, 2] == 7


def test_force_threshold_is_strict_and_normal_vector_magnitude_is_used():
    force = torch.tensor([[[1., 0., 0.], [0., .999, 0.], [.8, .8, 0.]]])
    sensor = monitor([(0, BufferView([force]), 1, [1], 2, [-1, 1, 1])], actors=2)
    sensor.sample(1 / 120)
    values = sensor.measurements()
    assert not values["static_contact"][0]
    assert values["peer_contact"][0, 1]
    assert values["peer_contact_max_force_n"][0, 1] == pytest.approx(2 ** .5 * .8)


def test_actor_body_batch_reduction_matches_independent_views_across_substeps():
    frames = torch.zeros(4, 24, 25, 3)
    frames[0, 13, 0, 2] = 3
    frames[1, 7, 11, :2] = torch.tensor([6., 8.])
    frames[2, 23, 24, 1] = 11
    frames[3, 0, 0, 2] = 2
    independent = monitor([(0, BufferView(list(frames[:, body:body + 1])), 1, [1], 24,
                            [-1] + [1] * 24) for body in range(24)], actors=2)
    batched = monitor([(0, BufferView(list(frames)), 1, [1], 24,
                        [-1] + [1] * 24, 24)], actors=2)
    for _ in range(4):
        independent.sample(1 / 120)
        batched.sample(1 / 120)
    for name, value in independent.measurements().items():
        torch.testing.assert_close(batched.measurements()[name], value, rtol=0, atol=0)
    assert batched.measurements()["static_contact_max_force_n"].tolist() == [3., 0.]
    assert batched.measurements()["peer_contact_max_force_n"][0, 1] == 11


def test_actor_body_batch_requires_all_declared_sensors():
    sensor = monitor([(0, BufferView([torch.zeros(23, 25, 3)]), 1, [1], 24,
                       [-1] + [1] * 24, 24)], actors=2)
    with pytest.raises(RuntimeError, match="expected.*24"):
        sensor.sample(1 / 120)


@pytest.mark.parametrize("dt", [0., -1., float("nan"), float("inf"), True])
def test_invalid_physics_duration_fails_before_querying_view(dt):
    view = BufferView([torch.zeros(1, 1, 3)])
    sensor = monitor([(0, view, 1, [], 2, [-1])])
    with pytest.raises(ValueError, match="duration"):
        sensor.sample(dt)
    assert not view.calls


@pytest.mark.parametrize("force,expected", [
    (torch.zeros(2, 5, 3), RuntimeError),
    (torch.zeros(1, 4, 3), RuntimeError),
    (torch.full((1, 5, 3), float("nan")), FloatingPointError),
])
def test_changed_force_shape_or_nonfinite_measurement_cannot_be_false_zero(force, expected):
    sensor = monitor([(0, BufferView([force]), 1, [1, 2], 2, [-1, 1, 1, 2, 2])])
    with pytest.raises(expected):
        sensor.sample(1 / 120)
    assert sensor.substep_count == 0


def test_actual_usd_collision_descendants_replace_the_visual_xform_filter():
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    root = "/World/PaperValidation/Obstacle_0"
    UsdGeom.Xform.Define(stage, root)
    collider = UsdGeom.Cube.Define(stage, root + "/geometry/mesh")
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    disabled = UsdGeom.Cube.Define(stage, root + "/disabled")
    UsdPhysics.CollisionAPI.Apply(disabled.GetPrim()).CreateCollisionEnabledAttr(False)
    assert resolve_static_collision_paths(stage, [root]) == [root + "/geometry/mesh"]
    assert not stage.GetPrimAtPath(root).HasAPI(UsdPhysics.CollisionAPI)
    with pytest.raises(ValueError, match="no enabled CollisionAPI"):
        resolve_static_collision_paths(stage, [root + "/disabled"])
    with pytest.raises(ValueError, match="does not exist"):
        resolve_static_collision_paths(stage, ["/missing"])


def test_invalid_constructor_values_fail_without_importing_sdk():
    with pytest.raises(ValueError, match="threshold"):
        PedestrianContactMonitor(SimpleNamespace(), threshold_n=float("nan"))
    with pytest.raises(ValueError, match="capacity"):
        PedestrianContactMonitor(SimpleNamespace(), max_contacts_per_view=0)
