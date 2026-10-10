"""Explicit N4 source-owned Warehouse lifetime for TRACE and measured PACER."""
import sys
import weakref
from .trace_pace_measured_lifecycle import TracePaceMeasuredLifecycle
from .dynbench_scene import DynBenchSceneProvider
from .native_grid_scene import NativeGridScene

class TracePaceNativeLifecycle(TracePaceMeasuredLifecycle):
    @staticmethod
    def _guard_native(world):
        provider, scene = world.scene_provider, world.validation_scene
        if (type(provider) is not DynBenchSceneProvider or type(scene) is not NativeGridScene
                or scene.provider is not provider or provider._query is None
                or getattr(provider._counter, '__self__', None) is not world.simulator._sim):
            raise RuntimeError('Native provider/scene/query must belong to this exact World')

    def _release_contacts_before_stop(self, world):
        # Immediate CPU report subscription owns no physical views. Release it
        # while the actual native scene is still attached, before full cleanup.
        provider=world.scene_provider
        if hasattr(provider,"_contact_report_sub"):
            provider._contact_report_sub=None
            self.report["native_contact_report_subscription_released_before_stop"]=True
        super()._release_contacts_before_stop(world)

    def _detach_world(self, world, producer):
            simulator = scene = asset = adapter = floor = peer = sensor = backend_view = None
            sensors = rows = classes = None
            try:
                if (world is None or world.num_envs not in (1,4) or world.shared_world is not True
                        or world.scene_provider is None or world.validation_scene is None):
                    raise RuntimeError("Require the complete original N1/N4 native evaluation World")
                self._guard_native(world)
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
                        or floor.num_actors != world.num_envs or peer.num_envs != world.num_envs):
                    raise RuntimeError("Require the exact one-floor/exact-actor native measurement owners")
                if (floor.view is not None or peer.views or len(self.contact_refs) != world.num_envs+1
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
                world.scene_provider._query = world.scene_provider._counter = None
                world.scene_provider = world.validation_scene = None
                self.report['native_provider_query_counter_released'] = True
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
