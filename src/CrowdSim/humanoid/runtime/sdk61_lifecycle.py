"""Owned Isaac Lab 3 EA / Sim 6.1 cleanup through the SDK lifecycle APIs.

The Sim 6.0 binary overlay and sealed teardown implementation are never loaded
on this runtime. Physics failure and cleanup failure remain separate evidence.
"""
from __future__ import annotations

import gc
import hashlib
from importlib import metadata
from pathlib import Path
import sys
import weakref


# The terminal workaround below is specific to this inspected Sim 6.1 binding.
# It must not silently mutate an unreviewed Carbonite module on another build.
_EVENTDISPATCHER_BINDING_SHA256 = "316ed2d58583951dac196de9feb525f2e63715b2341e99c54105d18f16d3f616"


def uses_sim61() -> bool:
    """Identify the supported new runtime without importing native modules."""
    try:
        return metadata.version("isaacsim") == "6.1.0.0"
    except metadata.PackageNotFoundError:
        return False


class SDK61Lifecycle:
    """Release the caller's views, then the context, then the complete app."""

    def __init__(self, report):
        self.report = report
        self._pending_world = None
        self._navigation_sensor_refs = self._navigation_backend_refs = ()
        self._eventdispatcher_cleanup_receipt = None
        versions = {name: metadata.version(name) for name in ("isaaclab", "isaacsim", "warp-lang")}
        expected = {"isaaclab": "17.0.2", "isaacsim": "6.1.0.0", "warp-lang": "1.16.0"}
        if versions != expected:
            raise RuntimeError(f"Unsupported SDK61 runtime: {versions}; expected {expected}")
        report["evaluation_lifecycle"] = {
            "implementation": str(Path(__file__).resolve()), "runtime_versions": versions,
            "protocol": "owned views -> SimulationContext.stop/clear_instance -> full App.close",
            "sim60_overlay_loaded": False, "SDK_source_modified": False,
            "contact_enabled_navigation_supported": True,
        }
        report.update(cleanup_succeeded=False, cleanup_errors=[], simulation_stop_returned=False,
                      simulation_clear_instance_returned=False, app_close_returned=False)

    @property
    def kit_args(self):
        return "--/app/renderer/enabled=false --/renderer/enabled=false"

    def verify_loaded(self):
        from isaaclab.sim import SimulationContext
        from isaaclab_physx.physics import PhysxManager
        sim = SimulationContext.instance()
        if sim is not None and sim.physics_manager.get_scene_data_backend() is not PhysxManager.get_scene_data_backend():
            raise RuntimeError("Expected the live owned Kit PhysX simulation context")
        sources = {}
        for name in ("isaaclab.sim.simulation_context", "isaaclab_physx.physics.physx_manager"):
            path = Path(sys.modules[name].__file__).resolve()
            sources[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.report["evaluation_lifecycle"]["loaded_SDK_sources"] = sources
        self.report["evaluation_lifecycle"]["live_context_verified"] = sim is not None

    def _release_contacts(self, world):
        simulator = getattr(world, "simulator", None)
        if simulator is None:
            return
        adapter = getattr(simulator, "pedestrian_contacts", None)
        provider = getattr(world, "scene_provider", None)
        if provider is not None and hasattr(provider, "_contact_report_sub"):
            provider._contact_report_sub = None
            self.report["native_contact_report_subscription_released_before_stop"] = True
        if adapter is None:
            return
        if adapter.world is not world or adapter.robot is not simulator._robot:
            raise RuntimeError("Contact adapter belongs to another world")
        if hasattr(adapter, "floor"):
            floor, peer = adapter.floor, adapter.peer
        else:
            from CrowdSim.humanoid.contact_monitor import PedestrianContactMonitor
            if type(adapter) is not PedestrianContactMonitor:
                raise RuntimeError("Unreviewed contact owner type")
            floor, peer = None, adapter
        views = ([floor.view] if floor is not None and floor.view is not None else [])
        if peer is not None:
            views.extend(row[1] for row in peer.views)
        refs = tuple(weakref.ref(view) for view in views)
        import torch
        if torch.cuda.is_initialized():
            torch.cuda.synchronize(world.device)
        if floor is not None:
            floor.view = None
        if peer is not None:
            peer.views.clear()
            # A combined capture also owns its peer monitor's world aliases.
            if hasattr(peer, "world"):
                peer.world = peer.robot = None
        views.clear()
        gc.collect()
        if any(ref() is not None for ref in refs):
            raise RuntimeError("Caller-owned raw contact views remain live before stop")
        self.report["navigation_owned_contact_views"] = {
            "observed_count": len(refs), "live_before_app_shutdown": 0}
        self.report["contacts_released_before_timeline_stop"] = True

    def _detach_world(self, world, producer):
        if world is None:
            return
        simulator = getattr(world, "simulator", None)
        if simulator is None:
            return
        scene = getattr(simulator, "_scene", None)
        if scene is not None:
            from isaaclab_physx.physics import PhysxManager
            backend_view = PhysxManager.get_scene_data_backend().simulation_view
            articulations = getattr(scene, "articulations", {})
            asset = getattr(simulator, "_robot", None)
            if asset is None and world is self._pending_world:
                asset = articulations.get("robot")
            empty_partial = asset is None and not articulations and world is self._pending_world
            if not empty_partial and (set(articulations) != {"robot"} or articulations["robot"] is not asset):
                raise RuntimeError("Expected this AMP World's single vectorized articulation")
            if asset is not None and (getattr(asset, "_is_initialized", False) or asset.__dict__.get("_root_view") is not None):
                raise RuntimeError("SDK stop callback has not invalidated the owned articulation")
            data = None if asset is None else asset.__dict__.get("_data")
            owners = [] if asset is None else [("articulation", asset)]
            if data is not None:
                owners.append(("articulation_data", data))
            for name, sensor in getattr(scene, "sensors", {}).items():
                if (getattr(sensor, "_is_initialized", False)
                        or sensor.__dict__.get("_body_physx_view") is not None
                        or sensor.__dict__.get("_contact_view") is not None):
                    raise RuntimeError("SDK stop callback has not invalidated owned sensor " + name)
                owners.append(("sensor:" + name, sensor))
            if "navigation_actor_count" in self.report["evaluation_lifecycle"] and world is not self._pending_world:
                sensors = tuple(scene.sensors.values())
                mapped = tuple(getattr(simulator, "_contact_sensor_map", {}).values())
                if (len(sensors) != 24 or len({id(sensor) for sensor in sensors}) != 24
                        or len(mapped) != 24 or {id(sensor) for sensor in mapped} != {id(sensor) for sensor in sensors}
                        or backend_view is None):
                    raise RuntimeError("Require all 24 owned navigation sensors and one backend")
                self._navigation_sensor_refs = tuple(weakref.ref(sensor) for sensor in sensors)
                self._navigation_backend_refs = (weakref.ref(backend_view),)
                sensors = mapped = None
            # GDB located the late native SimulationView destructor in the
            # caller's Articulation and ArticulationData dictionaries. These
            # borrowed aliases survive the SDK's stop callback. Validate all
            # identities before clearing terminal-only instance fields.
            for name, owner in owners:
                value = owner.__dict__.get("_physics_sim_view")
                if value is not None and value is not backend_view:
                    raise RuntimeError("Owned native view belongs to another backend: " + name)
            released = []
            for name, owner in owners:
                if "_physics_sim_view" in owner.__dict__:
                    owner._physics_sim_view = None
                    released.append(name)
            self.report["SDK_owned_borrowed_simulation_view_aliases_released"] = released
            self.report["SDK_owned_borrowed_simulation_view_alias_identity_verified"] = True
            owner = value = backend_view = data = asset = owners = None
        adapter = getattr(simulator, "pedestrian_contacts", None)
        if adapter is not None:
            adapter.world = adapter.robot = adapter.counter_getter = adapter.floor = adapter.peer = None
            simulator.pedestrian_contacts = None
        if producer is not None:
            self._release_producer(world, producer)
        if scene is not None:
            for name in ("sensors", "articulations", "rigid_objects", "rigid_object_collections", "deformable_objects"):
                getattr(scene, name, {}).clear()
        getattr(simulator, "_contact_sensor_map", {}).clear()
        simulator._robot = simulator._scene = simulator._sim = simulator._simulation_app = None
        provider = getattr(world, "scene_provider", None)
        if provider is not None:
            provider._query = provider._counter = None
        world.scene_provider = world.validation_scene = world.simulator = None
        self.report["evaluation_world_detached"] = True

    def _release_producer(self, world, producer):
        """Break both plain and native facade/core ownership cycles at shutdown."""
        if producer.world is not world:
            raise RuntimeError("Observation producer belongs to another world")
        core = getattr(producer, "_core", None)
        if core is None:
            if producer.sim is not world.simulator._sim:
                raise RuntimeError("Observation producer belongs to another context")
            producer.world = producer.sim = producer.sampler = None
            return
        provider, scene = world.scene_provider, world.validation_scene
        if (core.world is not world or core.sim is not world.simulator._sim
                or producer._provider is not provider or producer._scene is not scene
                or getattr(producer._base_describe, "__self__", None) is not core
                or getattr(core.__dict__.get("describe"), "__self__", None) is not producer):
            raise RuntimeError("Native observation facade/core belongs to another owner")
        core.__dict__.pop("describe")
        core.world = core.sim = core.sampler = None
        producer.world = producer._core = producer._base_describe = producer._provider = producer._scene = None
        self.report["native_observation_facade_core_cycle_released"] = True

    def _release_eventdispatcher_cleanup(self):
        """Run the pinned binding's terminal capsule while its plugins are live.

        Cold MJCF conversion leaves native callback data in this binding. GDB
        located the exit crash in its cleanup capsule at Python finalization.
        CPython retains the capsule in both the public module dictionary and
        the native extension's initialization snapshot; deleting only the
        public attribute does not run its destructor. Release precisely those
        two aliases after physics stops and before complete App.close.
        """
        name = "carb.eventdispatcher._eventdispatcher"
        # A reused report is evidence, not authorization for this instance to
        # accept a missing public alias while the native snapshot may survive.
        self.report.pop("SDK_eventdispatcher_terminal_cleanup", None)
        module = sys.modules.get(name)
        if module is None:
            self.report["SDK_eventdispatcher_terminal_cleanup"] = {"capsule_present": False}
            return
        path = Path(module.__file__).resolve()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != _EVENTDISPATCHER_BINDING_SHA256:
            raise RuntimeError("Unreviewed eventdispatcher binding: " + str(path))
        if "_cleanup" not in vars(module):
            receipt = self._eventdispatcher_cleanup_receipt
            if (receipt is None or receipt[0]() is not module
                    or receipt[1:3] != (str(path), digest)):
                raise RuntimeError("eventdispatcher cleanup capsule missing without a verified release")
            self.report["SDK_eventdispatcher_terminal_cleanup"] = dict(receipt[3])
            return
        capsule = module._cleanup
        if (type(capsule).__module__, type(capsule).__name__) != ("builtins", "PyCapsule"):
            raise RuntimeError("Expected the reviewed eventdispatcher cleanup capsule")
        owners = [owner for owner in gc.get_referrers(capsule)
                  if isinstance(owner, dict) and owner.get("_cleanup") is capsule]
        if (len(owners) != 2 or not any(owner is vars(module) for owner in owners)
                or any(owner.get("__name__") != name
                       or owner.get("__file__") != module.__file__
                       or owner.get("IEventDispatcher") is not module.IEventDispatcher
                       for owner in owners)
                or sys.getrefcount(capsule) != len(owners) + 2):
            raise RuntimeError("Unexpected eventdispatcher cleanup capsule owners")
        for owner in owners:
            del owner["_cleanup"]
        # Do not retain the capsule in this frame or in the JSON report.
        capsule = None
        evidence = {
            "capsule_present": True, "module": name, "path": str(path), "sha256": digest,
            "released_dictionary_aliases": len(owners), "released_before_App_close": True,
        }
        self._eventdispatcher_cleanup_receipt = (weakref.ref(module), str(path), digest, dict(evidence))
        self.report["SDK_eventdispatcher_terminal_cleanup"] = evidence

    def _owned_context(self, world):
        return getattr(getattr(world, "simulator", None), "_sim", None)

    def construct_world(self, factory, *args, **kwargs):
        """Retain a failed world's owned scene until terminal cleanup completes."""
        from isaaclab.sim import SimulationContext
        if SimulationContext.instance() is not None or self._pending_world is not None:
            raise RuntimeError("World construction requires an unowned context")
        if not isinstance(factory, type) or factory.__new__ is not object.__new__:
            raise TypeError("World constructor must use ordinary instance allocation")
        world = factory.__new__(factory)
        self._pending_world = world
        factory.__init__(world, *args, **kwargs)
        self._pending_world = None
        return world

    def close(self, app, world, producer=None, **unused):
        """Attempt all cleanup phases and preserve errors, including constructor failures."""
        from isaaclab.sim import SimulationContext
        report, errors = self.report, self.report["cleanup_errors"]
        failed_constructor = self._pending_world is not None
        if failed_constructor:
            if world is not None and world is not self._pending_world:
                raise RuntimeError("Failed constructor belongs to another world")
            world = self._pending_world
            report["partial_construction_cleanup"] = True
        sim = SimulationContext.instance()
        owned = self._owned_context(world)
        if owned is not None and sim is not owned:
            raise RuntimeError("Refuse to clear another caller's SimulationContext")
        def attempt(label, operation):
            try:
                operation()
            except Exception as error:
                errors.append(f"{label}: {type(error).__name__}: {error}")
        if world is not None:
            attempt("caller contact release", lambda: self._release_contacts(world))
        if sim is not None:
            manager = sim.physics_manager
            def stop():
                sim.stop()
                report["simulation_stop_returned"] = True
            attempt("simulation stop", stop)
            if failed_constructor and report["simulation_stop_returned"]:
                def partial_stop():
                    # A failed PHYSICS_READY callback can leave the timeline
                    # already stopped, so stop() emits no STOP event. Notify
                    # the owned SDK listeners before close deletes callbacks.
                    from isaaclab.physics import PhysicsEvent
                    manager.dispatch_event(PhysicsEvent.STOP)
                    manager.raise_callback_exception_if_any()
                    report["partial_constructor_STOP_callbacks_dispatched"] = True
                attempt("partial constructor STOP", partial_stop)
            def clear():
                SimulationContext.clear_instance()
                if SimulationContext.instance() is not None:
                    raise RuntimeError("SimulationContext singleton survived clear_instance")
                report["simulation_clear_instance_returned"] = True
            attempt("simulation clear", clear)
            # Borrowed asset/data/sensor fields must be verified against the
            # backend before its final owning property is cleared.
            attempt("caller world release", lambda: self._detach_world(world, producer))
            def backend_release():
                backend = manager.get_scene_data_backend()
                view = None if backend is None else backend.simulation_view
                reference = None if view is None else weakref.ref(view)
                if backend is not None:
                    backend.simulation_view = None
                view = None
                gc.collect()
                if reference is not None and reference() is not None:
                    raise RuntimeError("Borrowed SimulationView survives before App.close")
                report["public_backend_view_detached"] = True
                report["public_backend_view_live_before_App_close"] = False
            attempt("backend view release", backend_release)
            manager = None
        else:
            report["no_simulation_context_to_clear"] = True
            attempt("caller world release", lambda: self._detach_world(world, producer))
        if world is None:
            report["partial_construction_cleanup"] = True
        if "navigation_actor_count" in report["evaluation_lifecycle"] and not failed_constructor and world is not None:
            def verify_navigation_release():
                gc.collect()
                for field, refs, expected in (
                    ("navigation_SDK_sensor_objects", self._navigation_sensor_refs, 24),
                    ("navigation_borrowed_backend_views", self._navigation_backend_refs, 1),
                ):
                    if len(refs) != expected:
                        raise RuntimeError("Incomplete native ownership receipt: " + field)
                    live = sum(ref() is not None for ref in refs)
                    report[field] = {"observed_count": len(refs), "live_before_app_shutdown": live}
                    if live:
                        raise RuntimeError("Native owners survive before App shutdown: " + field)
            attempt("navigation owner release verification", verify_navigation_release)
        # Isaac Lab registers a native IHookHolder when initializing PhysX.
        # Its module-global alias survives PhysicsManager.close in this SDK;
        # release the subscription while Kit's extension plugin is still alive.
        def enable_hook_release():
            module = sys.modules.get("isaaclab_physx")
            if module is not None and hasattr(module, "_SIMULATION_MANAGER_ENABLE_HOOK"):
                had_hook = module._SIMULATION_MANAGER_ENABLE_HOOK is not None
                module._SIMULATION_MANAGER_ENABLE_HOOK = None
                report["SDK_extension_enable_hook_released"] = had_hook
        attempt("SDK extension hook release", enable_hook_release)
        def timeline_release():
            if SimulationContext.instance() is not None:
                raise RuntimeError("Cannot release Timeline before the simulation context is closed")
            from isaaclab_physx.physics import PhysxManager
            module = sys.modules.get("omni.timeline")
            getter = None if module is None else getattr(module, "get_timeline_interface", None)
            value = getattr(PhysxManager, "_timeline", None)
            if getter is not None and hasattr(getter, "timeline"):
                # SDK 6.1 caches ITimeline separately from the manager's
                # concrete Timeline. They are distinct Python wrappers.
                report["SDK_terminal_timeline_wrappers"] = {
                    "manager_type": type(value).__module__ + "." + type(value).__name__,
                    "cache_type": type(getter.timeline).__module__ + "." + type(getter.timeline).__name__,
                    "same_python_wrapper": getter.timeline is value,
                }
                del getter.timeline
            PhysxManager._timeline = None
            report["SDK_terminal_timeline_aliases_released_before_App_close"] = True
        attempt("SDK terminal Timeline release", timeline_release)
        if app is not None:
            attempt("SDK terminal eventdispatcher cleanup", self._release_eventdispatcher_cleanup)
        self._pending_world = None
        sim = owned = world = producer = None
        gc.collect()
        if app is not None:
            from protomotions.simulator.isaaclab.utils.sdk61_owned_close import _release_physx_replicator_cache
            def release_cache():
                if SimulationContext.instance() is not None:
                    raise RuntimeError("Cannot release terminal PhysX cache before the context is closed")
                _release_physx_replicator_cache(report)
            attempt("SDK terminal PhysX replicator cache", release_cache)
            def close_app():
                app.close(wait_for_replicator=False, skip_cleanup=False,
                          exit_code=1 if errors or str(report.get("status", "")).startswith("failed") else 0)
                report["app_close_returned"] = True
            attempt("App close", close_app)
        report["cleanup_succeeded"] = not errors and report["app_close_returned"] is True and (
            report.get("no_simulation_context_to_clear") is True or (
                report["simulation_stop_returned"] and report["simulation_clear_instance_returned"]
                and report.get("public_backend_view_detached") is True))
