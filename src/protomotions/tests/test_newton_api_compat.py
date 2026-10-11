# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU regressions for live Newton APIs and control/state semantics."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

wp = pytest.importorskip("warp")
newton = pytest.importorskip("newton")

from newton.selection import ArticulationView
from newton.sensors import SensorContact

from protomotions.robot_configs.base import ControlType
from protomotions.simulator.newton.api_compat import (
    ModelChangeFlags,
    allocate_contacts,
    contact_total_forces,
    count_contact_sensing_objects,
    create_body_contact_sensor,
    mujoco_solver_options,
    position_target_attribute,
    validate_scene_support,
)
from protomotions.simulator.newton.config import NewtonSimParams
from protomotions.simulator.newton.simulator import (
    NewtonSimulator,
    compute_pd_torques_kernel,
)


@pytest.fixture(autouse=True)
def _cpu_device():
    with wp.ScopedDevice("cpu"):
        yield


def _model(worlds=2, floating=True):
    robot = newton.ModelBuilder(up_axis=newton.Axis.Z)
    root = robot.add_link(mass=1.0, label="root")
    tip = robot.add_link(mass=1.0, label="tip")
    joints = []
    if floating:
        joints.append(robot.add_joint_free(root, label="root_free"))
    else:
        joints.append(robot.add_joint_fixed(-1, root, label="root_fixed"))
    joints.append(robot.add_joint_revolute(root, tip, axis=newton.Axis.Y, label="knee"))
    robot.add_articulation(joints, label="robot")
    robot.add_shape_sphere(root, radius=0.1)
    robot.add_shape_sphere(tip, radius=0.1)
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.replicate(robot, worlds)
    builder.add_ground_plane()
    return builder.finalize(device="cpu")


@pytest.mark.parametrize("coord_layout", [False, True])
def test_position_targets_preserve_free_root_and_select_live_joint_layout(monkeypatch, coord_layout):
    if hasattr(newton, "use_coord_layout_targets"):
        monkeypatch.setattr(newton, "use_coord_layout_targets", coord_layout)
    elif coord_layout:
        pytest.skip("This Newton version only has the legacy target layout")
    model = _model()
    view = ArticulationView(model, pattern="robot", include_joints=["knee"])
    control = model.control()
    attribute = position_target_attribute(control)
    initial = wp.to_torch(getattr(control, attribute)).clone()
    targets = torch.tensor([[[0.35]], [[-0.65]]])
    NewtonSimulator._apply_simulator_pd_targets(
        SimpleNamespace(robot_view=view, control=control), targets.squeeze(1)
    )
    np.testing.assert_allclose(view.get_attribute(attribute, control).numpy(), targets.numpy())
    root_width = 7 if coord_layout and hasattr(newton, "use_coord_layout_targets") else 6
    actual = wp.to_torch(getattr(control, attribute)).reshape(2, root_width + 1)
    assert torch.equal(actual[:, :root_width], initial.reshape(2, root_width + 1)[:, :root_width])
    if coord_layout:
        assert torch.equal(actual[:, 6], torch.ones(2))


def test_solver_options_follow_constructor_and_cpu_backend():
    solver_type = newton.solvers.SolverMuJoCo
    options = mujoco_solver_options(NewtonSimParams(), "cpu", solver_type)
    assert options["use_mujoco_cpu"] is True
    # Actual construction verifies options against the installed library.
    solver = solver_type(_model(worlds=1, floating=False), **options)
    assert solver.use_mujoco_cpu is True
    assert ModelChangeFlags.SHAPE_PROPERTIES == 16


def test_legacy_constructor_keeps_parallel_setting():
    class LegacySolver:
        def __init__(self, model, *, ls_parallel=True, use_mujoco_cpu=False):
            pass

    options = mujoco_solver_options(NewtonSimParams(ls_parallel=False), "cuda:0", LegacySolver)
    assert options["ls_parallel"] is False
    assert "use_mujoco_cpu" not in options
    assert position_target_attribute(SimpleNamespace(joint_target_pos=None)) == "joint_target_pos"


def test_removed_parallel_option_is_explicit_when_serial_was_requested():
    class ModernSolver:
        def __init__(self, model, *, use_mujoco_cpu=False):
            pass

    with pytest.warns(RuntimeWarning, match="cannot be enforced"):
        options = mujoco_solver_options(NewtonSimParams(ls_parallel=False), "cpu", ModernSolver)
    assert "ls_parallel" not in options


def test_sensor_kernel_and_environment_mapping_use_total_force():
    model = _model()
    # Reverse selection order to catch a naive reshape that swaps environments.
    tips = [i for i, name in enumerate(model.body_label) if name.split("/")[-1] == "tip"]
    sensor = create_body_contact_sensor(SensorContact, model, list(reversed(tips)))
    assert count_contact_sensing_objects(sensor) == 2
    pipeline, contacts = allocate_contacts(model)
    assert pipeline is not None
    assert contacts.force is not None
    shape_bodies = model.shape_body.numpy()
    tip_shapes = [int(np.flatnonzero(shape_bodies == body)[0]) for body in tips]
    ground = int(np.flatnonzero(shape_bodies == -1)[0])
    force = np.zeros((contacts.force.shape[0], 6), dtype=np.float32)
    force[0, :3] = [2.0, 3.0, 40.0]
    force[1, :3] = [-5.0, 4.0, 80.0]
    contacts.force.assign(force)
    shapes0 = np.full(contacts.rigid_contact_shape0.shape[0], -1, dtype=np.int32)
    shapes1 = shapes0.copy()
    shapes0[:2] = tip_shapes
    shapes1[:2] = ground
    contacts.rigid_contact_shape0.assign(shapes0)
    contacts.rigid_contact_shape1.assign(shapes1)
    contacts.rigid_contact_count.assign(np.array([2], dtype=np.int32))
    normals = np.zeros((contacts.rigid_contact_normal.shape[0], 3), dtype=np.float32)
    normals[:2, 2] = 1.0
    contacts.rigid_contact_normal.assign(normals)
    sensor.update(model.state(), contacts)
    result = contact_total_forces(sensor, model, 2)
    np.testing.assert_allclose(result.numpy(), force[:2, :3], atol=1e-6)


def test_sensor_rejects_a_missing_environment_instead_of_returning_zeros():
    model = _model()
    tip = next(i for i, name in enumerate(model.body_label) if name.split("/")[-1] == "tip")
    sensor = create_body_contact_sensor(SensorContact, model, [tip])
    with pytest.raises(ValueError, match="exactly one body"):
        contact_total_forces(sensor, model, 2)


def test_legacy_sensor_readings_are_summed():
    readings = np.array([[[1, 2, 3], [4, 5, 6]], [[-1, -2, -3], [2, 3, 4]]], dtype=np.float32)
    sensor = SimpleNamespace(net_force=wp.array(readings, dtype=wp.vec3), sensing_objs=[[1], [2]])
    assert count_contact_sensing_objects(sensor) == 2
    result = contact_total_forces(sensor, None, 2)
    np.testing.assert_array_equal(result.numpy(), readings.sum(axis=1))


def test_explicit_pd_kernel_respects_root_offsets_and_torque_limits():
    q = np.zeros(18, dtype=np.float32)
    qd = np.zeros(16, dtype=np.float32)
    q[[7, 8, 16, 17]] = [0.5, -0.5, 0.0, 0.25]
    qd[[6, 7, 14, 15]] = [0.1, -0.2, 1.0, 0.0]
    forces = wp.zeros(16, dtype=wp.float32, device="cpu")
    wp.launch(
        compute_pd_torques_kernel, dim=4,
        inputs=[
            wp.array(q), wp.array(qd), forces,
            wp.array(np.array([1.0, -1.0, 0.0, 0.0], dtype=np.float32)),
            wp.array(np.array([10.0, 20.0], dtype=np.float32)),
            wp.array(np.array([2.0, 3.0], dtype=np.float32)),
            wp.array(np.array([2.0, 4.0], dtype=np.float32)),
            9, 8, 7, 6, 2,
        ], device="cpu",
    )
    expected = np.zeros(16, dtype=np.float32)
    expected[[6, 7, 14, 15]] = [2.0, -4.0, -2.0, -4.0]
    np.testing.assert_array_equal(forces.numpy(), expected)


@pytest.mark.parametrize("decimation", [1, 2, 3, 4])
def test_simulate_keeps_latest_substep_for_each_decimation(decimation):
    class State:
        def __init__(self):
            self.value = 0

        def clear_forces(self):
            pass

        def assign(self, other):
            self.value = other.value

    class Solver:
        def step(self, source, target, control, contacts, dt):
            target.value = source.value + 1

    sim = SimpleNamespace(
        decimation=decimation, state_0=State(), state_1=State(),
        _max_action_latency_ms=0, control_type=ControlType.BUILT_IN_PD,
        viewer=None, solver=Solver(), control=None, contacts=None, sim_dt=0.1,
    )
    for frame in range(1, 4):
        NewtonSimulator._simulate(sim)
        assert sim.state_0.value == frame * decimation


def test_scene_validation_rejects_unspawned_objects_and_unrepresented_heights():
    empty = SimpleNamespace(num_objects_per_scene=0)
    flat = SimpleNamespace(height_field_raw=np.zeros((2, 2), dtype=np.int16))
    validate_scene_support(empty, None)
    validate_scene_support(empty, flat)
    with pytest.raises(NotImplementedError, match="does not spawn SceneLib"):
        validate_scene_support(SimpleNamespace(num_objects_per_scene=1), flat)
    with pytest.raises(NotImplementedError, match="non-flat terrain"):
        validate_scene_support(empty, SimpleNamespace(height_field_raw=np.array([[0, 1]])))


def test_close_releases_viewer_and_graph_and_is_idempotent():
    class Viewer:
        closed = 0

        def close(self):
            self.closed += 1

    viewer = Viewer()
    sim = SimpleNamespace(viewer=viewer, graph=object(), use_cuda_graph=True, _simulation_running=True)
    NewtonSimulator.close(sim)
    NewtonSimulator.close(sim)
    assert viewer.closed == 1
    assert sim.viewer is None and sim.graph is None
    assert sim.use_cuda_graph is False and sim._simulation_running is False


def test_close_cleans_state_even_when_viewer_close_raises():
    class Viewer:
        def close(self):
            raise RuntimeError("viewer close failed")

    sim = SimpleNamespace(viewer=Viewer(), graph=object(), use_cuda_graph=True)
    with pytest.raises(RuntimeError, match="viewer close failed"):
        NewtonSimulator.close(sim)
    assert sim.viewer is None and sim.graph is None and sim.use_cuda_graph is False
