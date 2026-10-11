"""Terminal cleanup for the simulator's owned Sim 6.1 scene.

Imports of Kit, CrowdSim and the SDK are deferred until a matching runtime is
closed. The fixed single-articulation lifecycle remains the low-level owner;
this adapter adds the explicitly supported second CrowdSim articulation and
ordinary rigid objects, and drains rendering before native plugins shut down.
"""

import gc
import hashlib
from importlib import metadata
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace


# This Sim 6.1 Linux/CPython 3.12 binding retains a py::list in a function-static
# replicator callback cache. Its __cxa_atexit destructor decrefs that list after
# Py_Finalize, crashing in list_dealloc/_PyObject_Free. The exact binary and data
# offset were verified against a late-attached native SIGSEGV stack and the
# initializer/destructor disassembly. There is no public cache reset API.
_PHYSX_BINDING_SHA256 = "8feacb575951aa814dbda988b4e83bf8f3230167b0b4856c1e1e44e3f6a28354"
_PHYSX_REPLICATOR_CACHE_OFFSET = 0x193640


def _physx_cache_slot(binding):
    """Locate the reviewed loaded ELF's cache; never assume an ASLR address."""
    import ctypes

    class DlInfo(ctypes.Structure):
        _fields_ = [("fname", ctypes.c_char_p), ("base", ctypes.c_void_p),
                    ("sname", ctypes.c_char_p), ("saddr", ctypes.c_void_p)]

    path = Path(binding.__file__).resolve()
    library = ctypes.CDLL(str(path))
    dladdr = ctypes.CDLL(None).dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    dladdr.restype = ctypes.c_int
    info = DlInfo()
    if (not dladdr(ctypes.cast(library.PyInit__physx, ctypes.c_void_p), ctypes.byref(info))
            or not info.base or not info.fname or Path(info.fname.decode()).resolve() != path):
        raise RuntimeError("Cannot verify loaded PhysX binding address")
    return ctypes.c_void_p.from_address(info.base + _PHYSX_REPLICATOR_CACHE_OFFSET)


def _release_physx_replicator_cache(report):
    """Release one reviewed native static reference while Python is alive.

    Terminal use only: the context and borrowed views have already been cleared,
    and the application is about to close. The static initializer guard remains
    set, so this process must not start another PhysX application after closing.
    The destructor's null check then has no Python object to touch at C++ exit.
    """
    import ctypes

    binding = sys.modules.get("omni.physx.bindings._physx")
    if binding is None:
        report["physx_replicator_terminal_cache"] = {"binding_loaded": False}
        return
    path = Path(binding.__file__).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != _PHYSX_BINDING_SHA256:
        raise RuntimeError("Unreviewed PhysX binding: " + str(path))
    slot = _physx_cache_slot(binding)
    receipt = {"binding_loaded": True, "binding_sha256": digest,
               "global_slot_offset": hex(_PHYSX_REPLICATOR_CACHE_OFFSET)}
    if not slot.value:
        receipt["cached_list_present"] = False
    else:
        cached = ctypes.cast(slot.value, ctypes.py_object).value
        # These are the exclusions returned by this SDK's owned clone callback.
        # Refuse unknown state before modifying any native memory.
        if type(cached) is not list or cached != ["/World/template", "/World/envs"]:
            raise RuntimeError("Unexpected PhysX replicator cache; refusing terminal mutation")
        receipt.update(cached_list_present=True, paths=list(cached))
        pointer = slot.value
        slot.value = None
        # pythonapi preserves the GIL. Remove precisely the native owner's
        # reference; the local alias keeps the object alive until this completes.
        decref = ctypes.pythonapi.Py_DecRef
        decref.argtypes = [ctypes.c_void_p]
        decref.restype = None
        decref(pointer)
        cached = None
        if slot.value is not None:
            raise RuntimeError("PhysX replicator terminal cache survived release")
    receipt["released_before_App_close"] = True
    report["physx_replicator_terminal_cache"] = receipt


def uses_sim61():
    try:
        return metadata.version("isaacsim").startswith("6.1.")
    except metadata.PackageNotFoundError:
        return False


def owned_keyboard_class(base):
    """Give this project's Sim 6.1 keyboard an explicit, idempotent release."""
    if not uses_sim61():
        return base

    class OwnedKeyboard(base):
        def close(self):
            interface = getattr(self, "_input", None)
            subscription = getattr(self, "_keyboard_sub", None)
            try:
                if interface is not None and subscription is not None:
                    interface.unsubscribe_to_keyboard_events(self._keyboard, subscription)
            finally:
                # Preserve an unsubscribe exception, but never retry a native
                # interface from __del__ after its plugin has shut down.
                self._keyboard_sub = None
                getattr(self, "_additional_callbacks", {}).clear()
                self._input = self._keyboard = self._appwindow = None

        def __del__(self):
            self.close()

    return OwnedKeyboard


def _release_exception_tracebacks(error):
    pending, visited = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        pending.extend(item for item in (current.__cause__, current.__context__) if item is not None)
        # A logger may retain the traceback object independently of the error.
        # Clear completed frames before unlinking it; executing frames are
        # preserved by traceback.clear_frames. Diagnostics are formatted first.
        traceback.clear_frames(current.__traceback__)
        current.__traceback__ = None


def _drain_rendering(app, context):
    import asyncio
    import omni.replicator.core as rep

    rep.orchestrator.set_capture_on_play(False)
    context.stop()  # Camera STOP detaches annotators and destroys render products.
    rep.orchestrator.stop()
    rep.orchestrator.wait_until_complete()
    loop = asyncio.get_event_loop()
    targets = {"_Orchestrator._restore_settings.<locals>._delayed_restore_async_rendering",
               "_Orchestrator.stop.<locals>.on_stop"}
    for updates in range(1, 33):
        app.update()
        pending = [task.get_coro().__qualname__ for task in asyncio.all_tasks(loop)
                   if not task.done() and task.get_coro().__qualname__ in targets]
        if updates >= 6 and not pending and rep.orchestrator.get_status().name == "STOPPED":
            return {"public_orchestrator_stop_and_wait": True, "app_updates": updates,
                    "remaining_owned_stop_tasks": [], "orchestrator_status": "STOPPED"}
    raise RuntimeError(f"Replicator stop did not drain after {updates} updates: {pending}")


def _make_lifecycle(report):
    from CrowdSim.humanoid.runtime.sdk61_lifecycle import SDK61Lifecycle

    class OwnedSceneLifecycle(SDK61Lifecycle):
        def _detach_world(self, world, producer):
            simulator = world.simulator
            scene = getattr(simulator, "_scene", None)
            if scene is None:
                return super()._detach_world(world, producer)
            from isaaclab_physx.physics import PhysxManager
            backend_view = PhysxManager.get_scene_data_backend().simulation_view
            articulations = scene.articulations
            names = set(articulations)
            if names not in ({"robot"}, {"robot", "crowdsim_robot"}):
                raise RuntimeError(f"Unsupported owned articulations: {sorted(names)}")
            robot = getattr(simulator, "_robot", None)
            if robot is not None and articulations["robot"] is not robot:
                raise RuntimeError("Simulator robot is not the owned scene articulation")
            for kind in ("rigid_object_collections", "deformable_objects", "cable_objects", "surface_grippers"):
                if getattr(scene, kind, {}):
                    raise RuntimeError(f"Unsupported native owner collection: {kind}")
            assets = list(articulations.items()) + list(getattr(scene, "rigid_objects", {}).items())
            owners = []
            for name, asset in assets:
                if getattr(asset, "_is_initialized", False) or vars(asset).get("_root_view") is not None:
                    raise RuntimeError(f"SDK STOP did not invalidate owned asset {name}")
                owners.append((name, asset))
                data = vars(asset).get("_data")
                if data is not None:
                    owners.append((name + ".data", data))
            for name, sensor in scene.sensors.items():
                if getattr(sensor, "_is_initialized", False) or any(vars(sensor).get(field) is not None
                    for field in ("_body_physx_view", "_physx_body_view", "_contact_view")):
                    raise RuntimeError(f"SDK STOP did not invalidate owned sensor {name}")
                if any(field in vars(sensor) for field in ("_renderer", "_render_data")) and any(
                    vars(sensor).get(field) is not None for field in ("_renderer", "_render_data", "_view")
                ):
                    raise RuntimeError(f"SDK STOP did not release owned camera {name}")
                owners.append((name, sensor))
            # Check every owner before mutating any borrowed alias. External env,
            # navigation and callback references may keep these wrappers alive;
            # their native handles must nevertheless be gone before App.close.
            for name, owner in owners:
                value = vars(owner).get("_physics_sim_view")
                if value is not None and value is not backend_view:
                    raise RuntimeError(f"Owned native view belongs to another backend: {name}")
            released = []
            for name, owner in owners:
                if "_physics_sim_view" in vars(owner):
                    owner._physics_sim_view = None
                    released.append(name)
            report["owned_scene_native_aliases_released"] = released
            report["owned_scene_articulation_names"] = sorted(names)
            report["owned_scene_rigid_object_names"] = sorted(getattr(scene, "rigid_objects", {}))
            report["owned_scene_sensor_names"] = sorted(scene.sensors)
            if robot is None:
                # Scene construction can succeed before _create_simulation
                # assigns this alias. It is the same validated owned asset.
                simulator._robot = articulations["robot"]
                report["partial_initialization_robot_owner_recovered"] = True
            robot = None
            owner = value = backend_view = data = asset = sensor = owners = assets = None
            # The parent keeps its original single-robot guard. Only remove the
            # separately validated and released second articulation at terminal.
            articulations.pop("crowdsim_robot", None)
            for field in ("_object", "_object_contact_sensor", "_projectile_objects"):
                value = getattr(simulator, field, None)
                if isinstance(value, list):
                    value.clear()
            value = None
            super()._detach_world(world, producer)

    return OwnedSceneLifecycle(report)


def close_owned_simulator(simulator):
    """Return False for other SDKs; completely close this Sim 6.1 owner once."""
    if not uses_sim61():
        return False
    previous = getattr(simulator, "_sdk61_close_report", None)
    if previous is not None:
        if previous["cleanup_succeeded"]:
            return True
        raise RuntimeError("Previous Sim 6.1 close failed: " + "; ".join(previous["cleanup_errors"]))

    from isaaclab.sim import SimulationContext
    context = getattr(simulator, "_sim", None)
    if context is None or SimulationContext.instance() is not context:
        raise RuntimeError("Refuse to close a simulator that does not own the active SimulationContext")
    app = simulator._simulation_app
    requested_exit_code = int(getattr(simulator, "_shutdown_exit_code", 0))
    active_error = sys.exception()
    report = {"status": "failed" if requested_exit_code or active_error is not None else "passed",
              "requested_exit_code": requested_exit_code}
    lifecycle = _make_lifecycle(report)
    simulator._sdk61_close_report = report
    lifecycle.verify_loaded()
    if active_error is not None:
        report["active_exception_traceback"] = "".join(traceback.format_exception(active_error))
        print(report["active_exception_traceback"], file=sys.stderr, flush=True)
        _release_exception_tracebacks(active_error)
    active_error = None

    def attempt(label, operation):
        try:
            operation()
        except Exception as error:
            report["cleanup_errors"].append(f"{label}: {type(error).__name__}: {error}")
            traceback.print_exc()
            _release_exception_tracebacks(error)

    def release_ui():
        try:
            keyboard = getattr(simulator, "keyboard_interface", None)
            if keyboard is not None:
                close = getattr(keyboard, "close", None)
                if not callable(close):
                    raise RuntimeError("Owned keyboard lacks an explicit native release")
                try:
                    close()
                finally:
                    simulator.keyboard_interface = None
        finally:
            viewer = getattr(simulator, "_perspective_view", None)
            if viewer is not None:
                viewer.viewport_api = None
                simulator._perspective_view = None
        report["owned_ui_released"] = True

    attempt("owned UI release", release_ui)
    def stop_owned_scene():
        context.stop()
        scene = getattr(simulator, "_scene", None)
        owners = [] if scene is None else [
            owner for kind in ("articulations", "rigid_objects", "sensors")
            for owner in getattr(scene, kind, {}).values()
        ]
        if any(getattr(owner, "_is_initialized", False) or any(
            vars(owner).get(field) is not None for field in (
                "_root_view", "_body_physx_view", "_physx_body_view", "_contact_view",
                "_renderer", "_render_data", "_view")
        ) for owner in owners):
            # A failed PHYSICS_READY callback can leave the timeline already
            # stopped. Notify this owned context before callback removal.
            from isaaclab.physics import PhysicsEvent
            context.physics_manager.dispatch_event(PhysicsEvent.STOP)
            context.physics_manager.raise_callback_exception_if_any()
            report["residual_initialized_owner_STOP_dispatched"] = True
    attempt("owned scene STOP", stop_owned_scene)
    if "omni.replicator.core" in sys.modules:
        attempt("RTX drain", lambda: report.update(rendering_shutdown=_drain_rendering(app, context)))
    # Disable the SDK's process-shortcut configuration for this owned full close.
    # AppLauncher callers may have left its default fast_shutdown=True enabled.
    def require_full_shutdown():
        if getattr(app, "config", {}).get("fast_shutdown", False):
            import carb
            app.config["fast_shutdown"] = False
            carb.settings.get_settings().set_bool("/app/fastShutdown", False)
        report["full_app_shutdown_requested"] = True
    attempt("full shutdown configuration", require_full_shutdown)

    class AppExitAdapter:
        def close(self, **kwargs):
            if requested_exit_code:
                kwargs["exit_code"] = requested_exit_code
            report["app_exit_code"] = kwargs["exit_code"]
            app.close(**kwargs)

    context = None
    gc.collect()
    lifecycle.close(AppExitAdapter(), SimpleNamespace(simulator=simulator))
    if not report["cleanup_succeeded"]:
        raise RuntimeError("Sim 6.1 cleanup failed: " + "; ".join(report["cleanup_errors"]))
    return True
