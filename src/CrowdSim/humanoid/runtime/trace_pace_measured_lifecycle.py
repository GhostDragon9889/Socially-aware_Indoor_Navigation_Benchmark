"""Terminal ownership for original N4 AMPWorld with peer and floor measurements.

Source preparation only until the new entry's actual smoke exits successfully.
The simulator/control/contact calculations are not altered by this release.
"""
from __future__ import annotations

from pathlib import Path
from social_navigation.sdk_paths import relocate_sdk_path
import sys
import weakref
import gc

from .social_evaluation_lifecycle import IsolatedTrackingLifecycle, ROOT, record

PINS = {
    "CrowdSim.humanoid.contact_monitor": (ROOT / "CrowdSim/humanoid/contact_monitor.py", "a821bc58e03d4f38612b823cfaa198ed831ea3aadf8f87b8b9bf13e04ad1a7fd"),
    "CrowdSim.humanoid.foot_support": (ROOT / "CrowdSim/humanoid/foot_support.py", "8c9cb88ddcbe55599ebe79abf80ea7a77a85eedfaa17324b937a862b284267e8"),
    "CrowdSim.humanoid.foot_capture": (ROOT / "CrowdSim/humanoid/foot_capture.py", "89c48fde37c05eaa57f4eacf2fb3eb072b087c0d9a46acff3f7d23729f28d39b"),
    "isaaclab_physx.sensors.contact_sensor.contact_sensor": (
        Path(relocate_sdk_path("lab:source/isaaclab_physx/isaaclab_physx/sensors/contact_sensor/contact_sensor.py")),
        "d8a7bb388fd1f6d2aae80a81c357f0c9ebaf6124f1cbd9f9564188860c3c3b37"),
}


class TracePaceMeasuredLifecycle(IsolatedTrackingLifecycle):
    """One vectorized articulation, 24 SDK sensors, four peer views, one floor view."""

    def __init__(self, report):
        super().__init__(report)
        if self._sdk61 is not None:
            return
        self.contact_refs = self.borrowed_refs = self.sensor_refs = ()
        report["evaluation_lifecycle"].update(navigation_implementation=record(Path(__file__)),
            contact_enabled_navigation_supported=True, navigation_actor_count=None,
            expected_owned_contact_views=None, expected_SDK_sensor_objects=24)
        for path, sha in PINS.values():
            if record(path)["sha256"] != sha:
                raise ValueError("Navigation lifecycle source changed: " + str(path))

    @staticmethod
    def _classes():
        classes = {}
        names = {"CrowdSim.humanoid.contact_monitor": "PedestrianContactMonitor",
                 "CrowdSim.humanoid.foot_support": "FootSupportMonitor",
                 "CrowdSim.humanoid.foot_capture": "FootCaptureAdapter",
                 "isaaclab_physx.sensors.contact_sensor.contact_sensor": "ContactSensor"}
        for name, (path, sha) in PINS.items():
            module = sys.modules.get(name)
            if module is None or Path(module.__file__).resolve() != path.resolve() or record(path)["sha256"] != sha:
                raise RuntimeError("Actual native ownership source differs: " + name)
            cls = getattr(module, names[name])
            if cls.__module__ != name or cls.__qualname__ != names[name]:
                raise RuntimeError("Native ownership class origin differs: " + name)
            classes[names[name]] = cls
        return classes

    def _detach_world(self, world, producer):
        simulator = scene = asset = adapter = floor = peer = sensor = backend_view = None
        sensors = rows = classes = None
        try:
            if (world is None or world.num_envs != 4 or world.shared_world is not True
                    or world.scene_provider is not None or world.validation_scene is None):
                raise RuntimeError("Require the complete original N4 cuboid evaluation World")
            simulator = world.simulator
            scene, asset = simulator._scene, simulator._robot
            if (set(scene.articulations) != {"robot"} or scene.articulations["robot"] is not asset
                    or scene.rigid_objects or scene.rigid_object_collections or scene.deformable_objects
                    or simulator._object or simulator._projectile_objects):
                raise RuntimeError("Additional unreviewed physical owners in navigation World")
            classes = self._classes()
            adapter = simulator.pedestrian_contacts
            if type(adapter) is not classes["FootCaptureAdapter"] or adapter.world is not world or adapter.robot is not asset:
                raise RuntimeError("Require this World's exact floor/peer capture owner")
            floor, peer = adapter.floor, adapter.peer
            if (type(floor) is not classes["FootSupportMonitor"] or type(peer) is not classes["PedestrianContactMonitor"]
                    or floor.num_actors != 4 or peer.num_envs != 4):
                raise RuntimeError("Require the exact one-floor/four-peer native measurement owners")
            if (floor.view is not None or peer.views or len(self.contact_refs) != 5
                    or any(ref() is not None for ref in self.contact_refs)):
                raise RuntimeError("Owned contact views were not released before timeline stop")
            sensors = list(scene.sensors.values())
            if (len(sensors) != 24 or len({id(value) for value in sensors}) != 24
                    or len(simulator._contact_sensor_map) != 24
                    or {id(value) for value in simulator._contact_sensor_map.values()} != {id(value) for value in sensors}
                    or any(type(value) is not classes["ContactSensor"] for value in sensors)):
                raise RuntimeError("Require all 24 exact caller-owned SDK contact sensors")
            backend_view = sys.modules["isaaclab_physx.physics.physx_manager"].PhysxManager.get_scene_data_backend().simulation_view
            for sensor in sensors:
                if (sensor.__dict__.get("_physics_sim_view") is not backend_view
                        or sensor.__dict__.get("_body_physx_view") is not None
                        or sensor.__dict__.get("_contact_view") is not None):
                    raise RuntimeError("SDK stop callback has not invalidated the sensor's native subviews")
            self.sensor_refs = tuple(weakref.ref(value) for value in sensors)
            self.borrowed_refs = (weakref.ref(backend_view),)
            # The exact pinned one-asset helper validates context, SDK methods,
            # stopped state and asset/data aliases before writing either field.
            self.release_asset(simulator._sim, (asset,), self.report)
            for sensor in sensors:
                sensor._physics_sim_view = None  # Terminal-only borrowed instance alias; no SDK source/method patch.
            self.report["navigation_SDK_sensor_aliases_released"] = 24
            floor.view = None
            peer.views.clear()
            adapter.world = adapter.robot = adapter.counter_getter = adapter.floor = adapter.peer = None
            simulator.pedestrian_contacts = None
            simulator._contact_sensor_map.clear()
            scene.sensors.clear()
            sensors.clear()
            if producer is not None:
                if producer.world is not world or producer.sim is not simulator._sim:
                    raise RuntimeError("Navigation producer does not belong to this World")
                producer.world = producer.sim = producer.sampler = None
            scene.articulations.clear()
            simulator._robot = simulator._scene = simulator._sim = simulator._simulation_app = None
            world.simulator = None
            self.report["evaluation_world_detached"] = True
        finally:
            if isinstance(rows, list):
                rows.clear()
            if isinstance(sensors, list):
                sensors.clear()
            if isinstance(classes, dict):
                classes.clear()
            simulator = scene = asset = adapter = floor = peer = sensor = backend_view = None
            sensors = rows = classes = world = producer = None

    def _verify_contacts(self):
        report = self.report
        for field, refs, expected in (("navigation_owned_contact_views", self.contact_refs, self.actor_count+1),
                                     ("navigation_SDK_sensor_objects", self.sensor_refs, 24),
                                     ("navigation_borrowed_backend_views", self.borrowed_refs, 1)):
            if len(refs) != expected or any(type(ref) is not weakref.ReferenceType for ref in refs):
                raise RuntimeError("Incomplete native ownership receipt: " + field)
            live = sum(ref() is not None for ref in refs)
            report[field] = {"observed_count": len(refs), "live_before_app_shutdown": live}
            if live:
                raise RuntimeError("Native owners survive before App shutdown: " + field)

    def _release_contacts_before_stop(self, world):
        """Release owned raw contact views while the native timeline is live.

        SDK sensor views are owned by their stop callback. Custom floor/peer
        views have their own lifetime and are detached before that callback.
        This is a terminal ownership adaptation, not a confirmed crash cause.
        """
        import torch
        classes = self._classes()
        self.actor_count=world.num_envs
        if self.actor_count not in (1,4):raise ValueError("Explicit N1/N4 contact ownership required")
        self.report["evaluation_lifecycle"].update(navigation_actor_count=self.actor_count,expected_owned_contact_views=self.actor_count+1)
        adapter = world.simulator.pedestrian_contacts
        if (type(adapter) is not classes["FootCaptureAdapter"] or adapter.world is not world
                or type(adapter.floor) is not classes["FootSupportMonitor"]
                or type(adapter.peer) is not classes["PedestrianContactMonitor"]):
            raise RuntimeError("Complete exact contact owners required before stop")
        floor, peer = adapter.floor, adapter.peer
        tensor_api = sys.modules["omni.physics.tensors.api"]
        rows = list(peer.views)
        if (type(floor.view) is not tensor_api.RigidContactView or len(rows) != world.num_envs
                or any(type(row) is not tuple or len(row) != 7
                    or type(row[1]) is not tensor_api.RigidContactView for row in rows)):
            raise RuntimeError("Exact one-floor/four-peer native layout required")
        self.contact_refs = (weakref.ref(floor.view),) + tuple(weakref.ref(row[1]) for row in rows)
        if len({id(ref()) for ref in self.contact_refs}) != world.num_envs+1:
            raise RuntimeError("Five distinct owned contact views required")
        torch.cuda.synchronize(world.device)
        floor.view = None
        peer.views.clear()
        rows.clear()
        gc.collect()
        torch.cuda.synchronize(world.device)
        if any(ref() is not None for ref in self.contact_refs):
            raise RuntimeError("Owned contact views survived pre-stop release")
        self.report["contacts_released_before_timeline_stop"] = True
        self.report["contact_release_cuda_barriers"] = 2

    def close(self, app, world, producer=None):
        if self._sdk61 is not None:
            return self._sdk61.close(app, world, producer)
        try:
            self._release_contacts_before_stop(world)
        except BaseException as error:
            self.report["cleanup_errors"].append("pre-stop contact release: " + str(error))
        super().close(app, world, producer, native_guard=self._verify_contacts)
