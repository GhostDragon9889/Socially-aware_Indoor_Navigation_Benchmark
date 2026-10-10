"""CPU contract tests; fake tensors do not certify native forces or locomotion."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from CrowdSim.humanoid.foot_capture import FootCaptureAdapter, native_tensor
from CrowdSim.humanoid.foot_support import FOOT_NAMES, FootSupportMonitor
from CrowdSim.humanoid.motion_tracking import SMPL_BODY_NAMES


class Proxy:
    def __init__(self, tensor):
        self.torch = tensor


class NativeFloor:
    def __init__(self, actors, clock):
        self.clock = clock
        self.sensor_paths = [actor[name] for actor in actors for name in FOOT_NAMES][::-1]
        self.sensor_count, self.filter_count = len(self.sensor_paths), 2
        self.filter_paths = [["/World/floor/B", "/World/floor/A"] for _ in self.sensor_paths]
        self.force = torch.zeros(self.sensor_count, 2, 3)
        self.calls = []

    def get_contact_force_matrix(self, *, dt):
        self.calls.append((self.clock.value, dt))
        self.force.zero_()
        for row, path in enumerate(self.sensor_paths):
            # Only left ankle/toe support, vector magnitude varies each substep.
            if path.endswith(("/L_Ankle", "/L_Toe")):
                self.force[row, 1, 2] = float(self.clock.value)
        return Proxy(self.force)


class Peer:
    def __init__(self, clock, actors):
        self.clock = clock
        self.actors = actors
        self.events = []
        self.manifest = {"threshold_n": 1., "source": "fake peer test", "excluded": ["floor"]}
        self.mutate_on_sample = False

    def begin_control_step(self):
        self.events.append(("begin", self.clock.value))

    def sample(self, dt):
        self.events.append(("sample", self.clock.value, dt))
        if self.mutate_on_sample:
            self.clock.value += 1

    def measurements(self):
        return {"static_contact": torch.zeros(self.actors, dtype=torch.bool),
                "peer_contact": torch.zeros(self.actors, self.actors, dtype=torch.bool)}


def fixture(actors=4):
    clock = SimpleNamespace(value=40)
    names = tuple(reversed(SMPL_BODY_NAMES))
    pos = torch.zeros(actors, 24, 3)
    pos[..., 0] = torch.arange(24)
    quaternion = torch.zeros(actors, 24, 4)
    quaternion[..., 2:] = math.sqrt(.5)  # +90deg Z; local +X offset becomes world +Y.
    velocity = torch.zeros(actors, 24, 3)
    velocity[..., 0] = 3.
    velocity[..., 2] = 4.  # Tangential speed on declared Z plane must remain3.
    local_com = torch.zeros(actors, 24, 7)
    local_com[..., 0], local_com[..., 6] = .25, 1.
    live_com = pos.clone()
    live_com[..., 1] = .25
    masses = torch.arange(1, 25).float().expand(actors, -1).clone()
    robot = SimpleNamespace(body_names=names,
        data=SimpleNamespace(body_link_pos_w=Proxy(pos), body_link_quat_w=Proxy(quaternion),
            body_link_lin_vel_w=Proxy(velocity), body_link_ang_vel_w=Proxy(torch.zeros_like(velocity)),
            body_com_pos_w=Proxy(live_com)),
        root_view=SimpleNamespace(shared_metatype=SimpleNamespace(link_names=names),
            get_coms=lambda: Proxy(local_com), get_masses=lambda: Proxy(masses)))
    sim = SimpleNamespace(get_physics_step_count=lambda: clock.value)
    world = SimpleNamespace(num_envs=actors, episode_ids=torch.arange(actors, dtype=torch.int64),
        mapping=SimpleNamespace(body_names=SMPL_BODY_NAMES),
        simulator=SimpleNamespace(_robot=robot, _sim=sim),
        robot_config=SimpleNamespace(contact_bodies=list(FOOT_NAMES)))
    paths = [{name: f"/World/envs/env_{actor}/Robot/{name}" for name in FOOT_NAMES} for actor in range(actors)]
    native = NativeFloor(paths, clock)
    floor = FootSupportMonitor(native, paths, ["/World/floor/A", "/World/floor/B"], contact_reporting_enabled=True)
    peer = Peer(clock, actors)
    adapter = FootCaptureAdapter(world, floor, peer_monitor=peer)
    return world, clock, native, peer, adapter


def control(clock, adapter, step=700):
    adapter.prepare_control_step(step)
    adapter.begin_control_step()
    for _ in range(4):
        clock.value += 1  # The caller alone simulates a completed physics step.
        adapter.sample(1 / 120)
    return adapter.measurements()


def test_four_actor_four_substep_native_order_com_clock_and_peer_single_sampling():
    world, clock, native, peer, adapter = fixture()
    result = control(clock, adapter)
    assert result["foot_floor_normal_force_w"].shape == (4, 4, 4, 2, 3)
    assert result["physics_step"] == (41, 42, 43, 44)
    assert result["physics_step_before"] == 40 and result["control_step"] == 700
    assert result["episode_ids"] == (0, 1, 2, 3)
    assert native.calls == [(value, 1 / 120) for value in (41, 42, 43, 44)]
    assert peer.events == [("begin", 40)] + [("sample", value, 1 / 120) for value in (41, 42, 43, 44)]
    assert torch.equal(result["side_contact_any"], torch.tensor([[True, False]] * 4))
    assert torch.equal(result["foot_floor_normal_force_w"][:, 0, 0, 0, 2], torch.arange(41, 45).float())
    assert not result["foot_floor_normal_force_w"][..., 1, :].any()
    expected_native_pelvis = tuple(reversed(SMPL_BODY_NAMES)).index("Pelvis")
    assert torch.equal(result["body_link_pos_w"][:, :, 0, 0], torch.full((4, 4), float(expected_native_pelvis)))
    assert torch.allclose(result["body_com_pos_w"][..., 1], torch.full((4, 4, 24), .25))
    assert torch.allclose(result["com_composition_max_error_m"], torch.zeros(4), atol=1e-6)
    # Independent weighted sum in known native order, with a nonzero rotated COM offset.
    expected_x = float((torch.arange(24).float() * torch.arange(1, 25).float()).sum() / torch.arange(1, 25).float().sum())
    assert torch.allclose(result["mass_weighted_world_com"], torch.tensor([expected_x, .25, 0.]).expand(4, 4, 3))
    assert torch.equal(result["body_origin_tangential_speed_proxy_mps"], torch.full((4, 4, 4), 3.))
    assert result["reference_true_com"] is None
    assert result["captured_true_com"] is True and result["extra_physics_steps"] == 0
    assert adapter.manifest["source"] == peer.manifest["source"]
    assert adapter.manifest["foot_support_capture"]["native_capture_probe_passed"] is False
    assert clock.value == 44 and torch.equal(world.episode_ids, torch.arange(4))
    # SDK buffers change later; published data remains the sampled state.
    world.simulator._robot.data.body_link_pos_w.torch.fill_(999.)
    native.force.fill_(999.)
    assert float(result["body_link_pos_w"].max()) == 23.
    assert float(result["foot_floor_normal_force_w"].max()) == 44.


def test_successive_controls_and_explicit_episode_boundary_after_publication():
    world, clock, _, _, adapter = fixture(2)
    first = control(clock, adapter, 100)
    second = control(clock, adapter, 101)
    assert first["physics_step"] == (41, 42, 43, 44)
    assert second["physics_step"] == (45, 46, 47, 48)
    world.episode_ids += 1
    adapter.reset_episode_boundary()
    third = control(clock, adapter, 202)
    assert third["episode_ids"] == (1, 2)
    assert third["physics_step"] == (49, 50, 51, 52)


def test_valid_static_native_channel_transport_does_not_change_common_com_geometry():
    world, clock, _, _, adapter = fixture()
    masses = world.simulator._robot.root_view.get_masses().torch.double()
    coms = world.simulator._robot.root_view.get_coms().torch.double()
    world.simulator._robot.root_view.get_masses = lambda: Proxy(masses)
    world.simulator._robot.root_view.get_coms = lambda: Proxy(coms)
    result = control(clock, adapter)
    assert result["body_masses_kg"].dtype == torch.float32
    assert result["local_com_pose_b"].dtype == torch.float32
    assert torch.allclose(result["body_com_pos_w"][..., 1], torch.full((4, 4, 24), .25))
    assert torch.allclose(result["com_composition_max_error_m"], torch.zeros(4), atol=1e-6)


def test_missing_explicit_control_counter_and_unpublished_samples_fail():
    _, clock, _, _, adapter = fixture()
    with pytest.raises(RuntimeError, match="prepare"):
        adapter.begin_control_step()
    adapter.prepare_control_step(1)
    adapter.begin_control_step()
    clock.value += 1
    adapter.sample(1 / 120)
    for function in (adapter.measurements, adapter.reset_episode_boundary, lambda: adapter.prepare_control_step(2)):
        with pytest.raises(RuntimeError):
            function()
    partial = adapter.discard_interrupted_control(reason="explicit CPU test abort")
    assert partial["valid"] is False and len(partial["completed_state_samples"]) == 1
    assert len(partial["floor_force_samples"]) == 1
    assert not any(isinstance(value, torch.Tensor) and value.shape[0] == 4 for value in partial.values())
    assert clock.value == 41
    assert control(clock, adapter, 2)["physics_step"] == (42, 43, 44, 45)


@pytest.mark.parametrize("operation", ["skip", "repeat", "reset", "peer_step", "nonfinite", "wrong_com", "wrong_dtype", "bad_mass"])
def test_misaligned_or_corrupt_native_samples_are_rejected_not_zero_filled(operation):
    world, clock, _, peer, adapter = fixture()
    adapter.prepare_control_step(1)
    adapter.begin_control_step()
    clock.value += 1
    if operation == "skip":
        clock.value += 1
    elif operation == "repeat":
        clock.value -= 1
    elif operation == "reset":
        world.episode_ids[1] += 1
    elif operation == "peer_step":
        peer.mutate_on_sample = True
    elif operation == "nonfinite":
        world.simulator._robot.data.body_link_lin_vel_w.torch[0, 0, 0] = float("nan")
    elif operation == "wrong_com":
        world.simulator._robot.data.body_com_pos_w.torch[0, 0, 0] += .1
    elif operation == "wrong_dtype":
        world.simulator._robot.data.body_link_ang_vel_w.torch = world.simulator._robot.data.body_link_ang_vel_w.torch.double()
    elif operation == "bad_mass":
        world.simulator._robot.root_view.get_masses().torch[0, 0] = 0.
    with pytest.raises((ValueError, RuntimeError)):
        adapter.sample(1 / 120)
    with pytest.raises(RuntimeError):
        adapter.measurements()
    with pytest.raises(RuntimeError):
        adapter.reset_episode_boundary()


@pytest.mark.parametrize("bad_dt", [1 / 30, True, 0., float("nan")])
def test_only_true_physics_dt_accepted(bad_dt):
    _, clock, _, _, adapter = fixture()
    adapter.prepare_control_step(1)
    adapter.begin_control_step()
    clock.value += 1
    with pytest.raises(ValueError):
        adapter.sample(bad_dt)


@pytest.mark.parametrize("change", ["counter_missing", "counter_bool", "native_order", "body_count", "monitor_actors", "three_substeps", "zero_normal"])
def test_constructor_or_actual_counter_contract_rejects_unknown_layout(change):
    world, _, _, peer, adapter = fixture()
    floor = adapter.floor
    if change == "counter_missing":
        world.simulator._sim.get_physics_step_count = None
    elif change == "counter_bool":
        world.simulator._sim.get_physics_step_count = lambda: True
    elif change == "native_order":
        world.simulator._robot.body_names = SMPL_BODY_NAMES
    elif change == "body_count":
        world.simulator._robot.body_names = world.simulator._robot.root_view.shared_metatype.link_names = ("Pelvis",)
    elif change == "monitor_actors":
        world.num_envs += 1
    elif change == "three_substeps":
        floor.substeps = 3
    with pytest.raises((ValueError, RuntimeError)):
        new = FootCaptureAdapter(world, floor, peer_monitor=peer, ground_normal_w=(0, 0, 0) if change == "zero_normal" else (0, 0, 1))
        new.counter()


@pytest.mark.parametrize("counter", [True, -1, .5])
def test_control_counter_is_explicit_integer(counter):
    _, _, _, _, adapter = fixture()
    with pytest.raises(ValueError):
        adapter.prepare_control_step(counter)


def test_extra_step_or_reset_after_sampling_before_publish_rejected():
    for mutate in ("step", "reset"):
        world, clock, _, _, adapter = fixture()
        adapter.prepare_control_step(1)
        adapter.begin_control_step()
        for _ in range(4):
            clock.value += 1
            adapter.sample(1 / 120)
        if mutate == "step":
            clock.value += 1
        else:
            world.episode_ids += 1
        with pytest.raises(RuntimeError, match="before"):
            adapter.measurements()


def test_contact_reporting_false_rejected_before_factory_or_sdk_lookup():
    world, _, _, _, _ = fixture()
    with pytest.raises(ValueError, match="before physics"):
        FootCaptureAdapter.from_world(world, actor_foot_paths=[], stage=None, floor_roots=[], contact_reporting_enabled=False)


def test_proxy_tensor_resolved_and_invalid_proxy_rejected_without_warp():
    value = torch.tensor([1.])
    assert native_tensor(value) is value
    assert native_tensor(Proxy(value)) is value
    with pytest.raises(TypeError):
        native_tensor(Proxy(None))
