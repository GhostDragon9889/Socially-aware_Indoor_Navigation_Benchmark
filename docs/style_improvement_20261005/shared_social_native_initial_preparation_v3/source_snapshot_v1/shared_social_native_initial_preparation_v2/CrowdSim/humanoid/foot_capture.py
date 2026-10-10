"""Optional read-only four-substep foot support and live COM capture.

The caller initializes physics/contact reporting and explicitly installs this
adapter on the existing PaperPowerSimulator contact hook. Importing the module
does not import Kit or start a simulation. CPU tests prove contracts, not gait.
"""

from __future__ import annotations

from copy import deepcopy
import math

import torch

from CrowdSim.humanoid.foot_support import (
    FOOT_NAMES,
    FootSupportMonitor,
    body_link_world_com,
    body_name_permutation,
    mass_weighted_world_com,
    resolve_floor_collision_paths,
)
from CrowdSim.humanoid.physics_views import resolve_native_contact_factory


def native_tensor(value):
    """Resolve pinned Lab ProxyArray/PhysX data without starting an SDK."""
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "torch"):
        result = value.torch
        if not isinstance(result, torch.Tensor):
            raise TypeError("Native .torch channel is not a Torch tensor")
        return result
    import warp as wp

    return wp.to_torch(value)


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


class FootCaptureAdapter:
    """Combine one floor monitor and an optional existing peer/static monitor.

    Call prepare_control_step(actual_runner_counter), then exactly one existing
    world.step(), then measurements() before any reset. The existing simulator
    invokes begin_control_step()/sample(dt); this adapter never invokes step(),
    update(), set_pose(), or any physics/clock mutation.
    """

    @classmethod
    def from_world(cls, world, *, actor_foot_paths, stage, floor_roots,
                   contact_reporting_enabled, peer_monitor=None, threshold_n=1.,
                   max_contact_data_count=512, com_tolerance_m=1e-4,
                   ground_normal_w=(0., 0., 1.)):
        """Resolve enabled floor CollisionAPI shapes in the initialized stage.

        actor_foot_paths contains exact rigid-body paths, resolved from the
        actual spawned asset by the caller. Contact reporting must have been
        enabled before that articulation initialized; it is never enabled here.
        """
        if contact_reporting_enabled is not True:
            raise ValueError("Contact reporting must be explicitly enabled before physics initialization")
        if not set(FOOT_NAMES).issubset(set(world.robot_config.contact_bodies)):
            raise ValueError("World contact-reporting configuration does not include every semantic foot")
        if len(actor_foot_paths) != world.num_envs:
            raise ValueError("One exact foot-path mapping is required for every actor")
        floors = resolve_floor_collision_paths(stage, floor_roots)
        factory, ownership = resolve_native_contact_factory(world.simulator._sim)
        physics_dt = float(world.simulator._sim.get_physics_dt())
        if world.simulator.decimation != 4 or not math.isclose(physics_dt, 1 / 120, abs_tol=1e-12, rel_tol=0):
            raise ValueError("This capture contract requires four120Hz physics substeps per30Hz control")
        floor = FootSupportMonitor.from_factory(factory, actor_foot_paths, floors,
            contact_reporting_enabled=True, physics_dt_s=physics_dt, substeps_per_control=4,
            threshold_n=threshold_n, max_contact_data_count=max_contact_data_count)
        adapter = cls(world, floor, peer_monitor=peer_monitor, com_tolerance_m=com_tolerance_m,
                      ground_normal_w=ground_normal_w)
        adapter.manifest["foot_support_capture"]["physics_view_ownership"] = ownership
        return adapter

    def __init__(self, world, floor_monitor, *, peer_monitor=None,
                 com_tolerance_m=1e-4, ground_normal_w=(0., 0., 1.)):
        self.world, self.floor, self.peer = world, floor_monitor, peer_monitor
        self.robot = world.simulator._robot
        self.com_tolerance_m = _positive(com_tolerance_m, "COM tolerance")
        if self.floor.num_actors != world.num_envs or self.floor.substeps != 4:
            raise ValueError("Capture requires exactly four substeps and matching actor counts")
        if not math.isclose(self.floor.physics_dt_s, 1 / 120, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Capture requires120Hz physical samples")
        self.counter_getter = getattr(world.simulator._sim, "get_physics_step_count", None)
        if not callable(self.counter_getter):
            raise RuntimeError("Public Lab physics-step counter unavailable; no synthetic fallback")
        self.native_names = tuple(self.robot.root_view.shared_metatype.link_names)
        self.common_names = tuple(world.mapping.body_names)
        if tuple(self.robot.body_names) != self.native_names or len(self.native_names) != 24:
            raise ValueError("Actual robot data/root-view body order requires a verified24-body mapping")
        self.common_indices = body_name_permutation(self.native_names, self.common_names)
        if not set(FOOT_NAMES).issubset(set(self.common_names)):
            raise ValueError("Common body mapping lacks semantic feet")
        self.foot_indices = tuple(self.common_names.index(name) for name in FOOT_NAMES)
        normal = torch.as_tensor(ground_normal_w, dtype=torch.float64)
        if normal.shape != (3,) or not torch.isfinite(normal).all() or not torch.allclose(normal.norm(), normal.new_tensor(1.), atol=1e-6, rtol=0):
            raise ValueError("Ground normal requires an explicit finite unit world vector")
        self.ground_normal = normal
        if self.peer is not None:
            if self.peer is self.floor or any(not callable(getattr(self.peer, name, None)) for name in
                                            ("begin_control_step", "sample", "measurements")):
                raise ValueError("Optional peer monitor must implement the existing contact hook independently")
        capture_manifest = {
            "schema": "navisaaclab.live_foot_support_capture.v1",
            "floor_measurement": deepcopy(self.floor.manifest),
            "counter_source": "Lab SimulationContext.get_physics_step_count()",
            "counter_semantics": "monotonic submitted-step counter; not PhysX native simulation time",
            "control_counter_source": "explicit prepare_control_step(actual_runner_control_step)",
            "episode_source": "world.episode_ids before control and every completed substep",
            "native_body_names": self.native_names, "common_body_names": self.common_names,
            "foot_names": FOOT_NAMES, "quaternion_order": "xyzw",
            "sampling_hook": "existing PaperPowerSimulator.pedestrian_contacts after physics+scene.update",
            "world_com_source": "robot.data.body_com_pos_w",
            "local_com_source": "root_view.get_coms() local position and principal-axis pose",
            "masses_source": "root_view.get_masses() in verified native order",
            "static_channel_transport": "validated get_coms/get_masses values copied to live body-state dtype/device; no geometric recomputation",
            "link_velocity_source": "robot.data.body_link_lin_vel_w; body-origin, not COM alias",
            "com_composition_tolerance_m": self.com_tolerance_m,
            "ground_normal_w": normal.tolist(),
            "extra_physics_steps": 0, "pose_writes": 0, "native_engine_time_observed": False,
            "friction_force_observed": False, "contact_patch_slip_observed": False,
            "reference_true_com_available": False, "reference_true_com": None,
            "author_metric_equivalence": False, "native_capture_probe_passed": False,
        }
        # Existing static/peer measurement metadata keeps its original meaning.
        self.manifest = deepcopy(getattr(self.peer, "manifest", {}))
        if "foot_support_capture" in self.manifest:
            raise ValueError("Existing contact monitor is already composed with a foot capture")
        self.manifest["foot_support_capture"] = capture_manifest
        self._prepared = None
        self._last_control = None
        self._samples = []
        self._active = False
        self._published = True
        self._poisoned = False
        self.reset_count = 0

    def counter(self):
        value = self.counter_getter()
        if type(value) is not int or value < 0:
            raise RuntimeError("Actual Lab counter must be a nonnegative Python integer")
        return value

    def episodes(self):
        values = self.world.episode_ids
        if not isinstance(values, torch.Tensor) or values.shape != (self.world.num_envs,) or values.dtype not in (torch.int32, torch.int64):
            raise ValueError("World episode IDs require one actual integer Tensor value per actor")
        if (values < 0).any():
            raise ValueError("World episode IDs must be nonnegative")
        return tuple(values.detach().cpu().tolist())

    def prepare_control_step(self, control_step):
        """Declare the true runner counter; never infer it from an actor's age."""
        if type(control_step) is not int or control_step < 0:
            raise ValueError("Control step must be a nonnegative Python integer")
        if self._prepared is not None or not self._published or self._poisoned:
            raise RuntimeError("Previous control requires publication or explicit interrupted-data disposal")
        if self._last_control is not None and control_step != self._last_control + 1:
            raise RuntimeError("Runner control counters require consecutive values or explicit episode-boundary reset")
        self._prepared = control_step

    def begin_control_step(self):
        if self._prepared is None or not self._published or self._poisoned:
            raise RuntimeError("Explicit prepare_control_step is required before the existing physics hook")
        self._control = self._prepared
        self._prepared = None
        self._before = self.counter()
        self._episodes = self.episodes()
        self._samples = []
        self._active, self._published = True, False
        try:
            self.floor.begin_control_step(episode_ids=self._episodes, control_step=self._control, physics_step_before=self._before)
            if self.peer is not None:
                self.peer.begin_control_step()
            if self.counter() != self._before or self.episodes() != self._episodes:
                raise RuntimeError("Contact begin mutated physics counter or actor episodes")
            self._last_control = self._control
        except Exception:
            self._poisoned = True
            raise

    def _channel(self, value, name, shape):
        channel = native_tensor(value)
        if tuple(channel.shape) != shape or not channel.is_floating_point() or not torch.isfinite(channel).all():
            raise ValueError(f"Actual native {name} requires finite floating{shape}")
        return channel.detach().clone()

    def _state_snapshot(self):
        count, bodies = self.world.num_envs, len(self.common_names)
        rows = list(self.common_indices)
        result = {}
        channels = (("body_link_pos_w", 3), ("body_link_quat_w", 4), ("body_link_lin_vel_w", 3),
                    ("body_link_ang_vel_w", 3), ("body_com_pos_w", 3))
        for name, width in channels:
            result[name] = self._channel(getattr(self.robot.data, name), name, (count, bodies, width))[:, rows]
        baseline = result["body_link_pos_w"]
        result["local_com_pose_b"] = self._channel(self.robot.root_view.get_coms(), "local COM pose", (count, bodies, 7))[:, rows].to(baseline)
        result["body_masses_kg"] = self._channel(self.robot.root_view.get_masses(), "body masses", (count, bodies))[:, rows].to(baseline)
        if any(value.dtype != baseline.dtype or value.device != baseline.device for value in result.values()):
            raise ValueError("All live body/COM/mass channels must share dtype and device")
        composed = body_link_world_com(baseline, result["body_link_quat_w"], result["local_com_pose_b"][..., :3])
        result["com_composition_max_error_m"] = (composed - result["body_com_pos_w"]).abs().amax()
        if float(result["com_composition_max_error_m"]) > self.com_tolerance_m:
            raise RuntimeError("Live body COM differs from same-instant link/local COM composition")
        result["mass_weighted_world_com"] = mass_weighted_world_com(result["body_com_pos_w"], result["body_masses_kg"])
        return result

    def sample(self, dt):
        """Called once after each existing physics step and scene.update."""
        if not self._active or self._published or self._poisoned or len(self._samples) >= self.floor.substeps:
            raise RuntimeError("No incomplete prepared control for foot capture")
        try:
            duration = _positive(dt, "physics dt")
            if not math.isclose(duration, self.floor.physics_dt_s, rel_tol=0, abs_tol=1e-12):
                raise ValueError("Force capture requires the true120Hz dt, not control duration")
            before_read = self.counter()
            if before_read != self._before + len(self._samples) + 1:
                raise RuntimeError("Missing/repeated physical substep; counter has no synthetic fallback")
            if self.episodes() != self._episodes:
                raise RuntimeError("Actor reset inside sampled control")
            state = self._state_snapshot()
            feet = list(self.foot_indices)
            self.floor.sample_completed_substep(duration, episode_ids=self._episodes, physics_step=before_read,
                foot_body_origin_pos_w=state["body_link_pos_w"][:, feet],
                foot_body_origin_vel_w=state["body_link_lin_vel_w"][:, feet])
            if self.peer is not None:
                self.peer.sample(duration)
            if self.counter() != before_read or self.episodes() != self._episodes:
                raise RuntimeError("Read-only capture advanced physics or changed episode IDs")
            state["physics_step"] = before_read
            self._samples.append(state)
        except Exception:
            self._poisoned = True
            raise

    def measurements(self):
        """Publish completed samples before reset; no partial/zero fallback."""
        if self._poisoned or not self._active or len(self._samples) != self.floor.substeps:
            raise RuntimeError("Foot capture requires every completed substep; partial data cannot be published")
        if self.counter() != self._before + self.floor.substeps or self.episodes() != self._episodes:
            raise RuntimeError("Reset or extra physics step before foot capture publication")
        normal = self.ground_normal.to(self._samples[0]["body_link_pos_w"])
        result = self.floor.measurements(ground_normal_w=normal)
        for key in self._samples[0]:
            values = [row[key] for row in self._samples]
            result[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else tuple(values)
        result.update(physics_step_before=self._before, control_step=self._control, episode_ids=self._episodes,
                      common_body_names=self.common_names, foot_names=FOOT_NAMES,
                      reference_true_com=None, captured_true_com=True, extra_physics_steps=0)
        if self.peer is not None:
            peer = self.peer.measurements()
            if not isinstance(peer, dict) or set(peer) & set(result):
                raise ValueError("Peer capture must be a disjoint measurement dictionary")
            result.update(peer)
        if self.counter() != self._before + self.floor.substeps or self.episodes() != self._episodes:
            raise RuntimeError("Publication unexpectedly mutated physics counter or episodes")
        self._published = True
        return result

    def reset_episode_boundary(self):
        """Caller explicitly starts a new episode boundary after publication."""
        if not self._published or self._prepared is not None or self._poisoned:
            raise RuntimeError("Publish or explicitly archive/dispose interrupted data before episode reset")
        self.floor.reset()
        self._last_control = None
        self._samples = []
        self._active = False
        self.reset_count += 1

    def discard_interrupted_control(self, *, reason):
        """Return failed partial evidence before an explicit caller disposal.

        This is not a valid capture or zero-filled frame. The caller must save
        the returned error record if restarting after a failed/aborted control.
        """
        if not isinstance(reason, str) or not reason.strip() or self._published:
            raise ValueError("Explicit nonempty disposal reason and interrupted control are required")
        evidence = {"valid": False, "reason": reason, "control_step": self._control,
                    "episode_ids": self._episodes, "physics_step_before": self._before,
                    "physics_step_at_disposal": self.counter(), "completed_state_samples": deepcopy(self._samples),
                    "floor_force_samples": tuple(value.detach().clone() for value in self.floor._forces)}
        self.floor.reset()
        self._samples = []
        self._prepared, self._last_control = None, None
        self._active, self._poisoned, self._published = False, False, True
        self.reset_count += 1
        return evidence
