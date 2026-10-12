"""Production simulator cleanup contracts with CPU native-owner fixtures."""
import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from test_sdk61_lifecycle import make_world, runtime  # noqa: F401
from protomotions.simulator.isaaclab.utils import sdk61_owned_close as close_module


@pytest.fixture
def owned(runtime, monkeypatch):  # noqa: F811 - imported pytest fixture
    from CrowdSim.humanoid.runtime import sdk61_lifecycle
    monkeypatch.setattr(sdk61_lifecycle, "SDK61Lifecycle", runtime.module.SDK61Lifecycle)
    monkeypatch.delitem(sys.modules, "omni.replicator.core", raising=False)
    runtime.Context.current = runtime.Sim()
    world, robot, sensor, robot_data = make_world(runtime)
    simulator = world.simulator
    simulator._simulation_app = runtime.App()
    def asset():
        return SimpleNamespace(_physics_sim_view=runtime.backend.simulation_view, _root_view=None,
            _is_initialized=False, _data=SimpleNamespace(_physics_sim_view=runtime.backend.simulation_view))
    car, box = asset(), asset()
    camera = SimpleNamespace(_is_initialized=False, _renderer=None, _render_data=None, _view=None)
    simulator._scene.articulations["crowdsim_robot"] = car
    simulator._scene.rigid_objects = {"object_0": box}
    simulator._scene.sensors["crowdsim_robot_camera"] = camera
    simulator._object, simulator._projectile_objects = [box], [box]
    # These callers deliberately remain live across close, like env/nav/keyboard
    # callbacks in the real application. Native aliases must still be released.
    clients = SimpleNamespace(env=SimpleNamespace(crowdsim_robot=car, crowdsim_robot_camera=camera),
                              nav=SimpleNamespace(robot=car, drive=SimpleNamespace(robot=car)))
    return SimpleNamespace(simulator=simulator, car=car, box=box, robot=robot, camera=camera,
                           clients=clients, runtime=runtime)


@pytest.mark.parametrize("exit_code", [0, 1, 7])
def test_owned_multi_asset_close_preserves_exit_code_and_is_idempotent(owned, exit_code):
    simulator = owned.simulator
    simulator._shutdown_exit_code = exit_code
    assert close_module.close_owned_simulator(simulator)
    assert close_module.close_owned_simulator(simulator)
    assert simulator._sdk61_close_report["cleanup_succeeded"]
    for asset in (owned.robot, owned.car, owned.box):
        assert asset._physics_sim_view is asset._data._physics_sim_view is None
    assert owned.clients.nav.drive.robot is owned.car
    assert owned.clients.env.crowdsim_robot_camera is owned.camera
    assert simulator._object == simulator._projectile_objects == []
    closes = [event for event in owned.runtime.events if isinstance(event, tuple)]
    assert len(closes) == 1 and closes[0][1]["exit_code"] == exit_code


@pytest.mark.parametrize("missing_context", [False, True])
def test_foreign_context_is_rejected_before_any_cleanup(owned, missing_context):
    owned.simulator._sim = None if missing_context else owned.runtime.Sim()
    with pytest.raises(RuntimeError, match="does not own"):
        close_module.close_owned_simulator(owned.simulator)
    assert not owned.runtime.events
    assert not hasattr(owned.simulator, "_sdk61_close_report")


def test_close_handles_scene_constructed_before_robot_initialization(owned):
    del owned.simulator._robot
    close_module.close_owned_simulator(owned.simulator)
    assert owned.simulator._sdk61_close_report["partial_initialization_robot_owner_recovered"]
    assert owned.robot._physics_sim_view is None


@pytest.mark.parametrize("initialized", [False, True])
def test_stopped_partial_initialization_dispatches_owned_stop(owned, initialized):
    owned.robot._is_initialized = initialized
    owned.robot._root_view = object()
    def invalidate():
        owned.robot._is_initialized = False
        owned.robot._root_view = None
    owned.runtime.Manager.callbacks.append(invalidate)
    with pytest.raises(RuntimeError, match="PHYSICS_READY callback failed"):
        try:
            raise RuntimeError("PHYSICS_READY callback failed")
        finally:
            close_module.close_owned_simulator(owned.simulator)
    assert owned.simulator._sdk61_close_report["cleanup_succeeded"]
    assert owned.simulator._sdk61_close_report["residual_initialized_owner_STOP_dispatched"]
    assert owned.runtime.events.index("explicit_STOP") < owned.runtime.events.index("clear")


@pytest.mark.parametrize("mode,expected_updates", [("drained", 6), ("late_stop", 9), ("pending", 32), ("running", 32)])
def test_rendering_drain_observes_tasks_status_and_update_bound(monkeypatch, mode, expected_updates):
    import asyncio
    events, count = [], [0]
    rep = ModuleType("omni.replicator.core")
    rep.orchestrator = SimpleNamespace(
        set_capture_on_play=lambda value: events.append(("capture", value)),
        stop=lambda: events.append("rep_stop"), wait_until_complete=lambda: events.append("wait"),
        get_status=lambda: SimpleNamespace(name="STOPPED" if mode not in {"late_stop", "running"}
            or mode == "late_stop" and count[0] >= 9 else "STARTED"))
    omni, replicator = ModuleType("omni"), ModuleType("omni.replicator")
    omni.replicator, replicator.core = replicator, rep
    for name, module in (("omni", omni), ("omni.replicator", replicator), ("omni.replicator.core", rep)):
        monkeypatch.setitem(sys.modules, name, module)
    loop = object()
    monkeypatch.setattr(asyncio, "get_event_loop", lambda: loop)
    task = SimpleNamespace(done=lambda: False, get_coro=lambda: SimpleNamespace(
        __qualname__="_Orchestrator._restore_settings.<locals>._delayed_restore_async_rendering"))
    # An unrelated pending task must not be cancelled or awaited by this owner.
    unrelated = SimpleNamespace(done=lambda: False, get_coro=lambda: SimpleNamespace(__qualname__="other_owner"))
    monkeypatch.setattr(asyncio, "all_tasks", lambda active_loop: [unrelated] + (
        [task] if mode == "pending" or count[0] < 5 else []))
    app = SimpleNamespace(update=lambda: count.__setitem__(0, count[0] + 1))
    context = SimpleNamespace(stop=lambda: events.append("context_stop"))
    if mode in {"pending", "running"}:
        with pytest.raises(RuntimeError, match="after 32 updates"):
            close_module._drain_rendering(app, context)
    else:
        assert close_module._drain_rendering(app, context)["app_updates"] == expected_updates
    assert count[0] == expected_updates
    assert events == [("capture", False), "context_stop", "rep_stop", "wait"]


@pytest.mark.parametrize("defect", ["extra_alias", "rigid_data_alias", "camera", "raycaster", "collection"])
def test_unsupported_or_live_native_owners_fail_closed_without_partial_alias_release(owned, defect):
    if defect == "extra_alias":
        owned.car._physics_sim_view = object()
    elif defect == "rigid_data_alias":
        owned.box._data._physics_sim_view = object()
    elif defect == "camera":
        owned.camera._renderer = object()
    elif defect == "raycaster":
        owned.simulator._scene.sensors["lidar"] = SimpleNamespace(_is_initialized=False, _physx_body_view=object())
    else:
        owned.simulator._scene.rigid_object_collections = {"unsupported": object()}
    with pytest.raises(RuntimeError, match="cleanup failed"):
        close_module.close_owned_simulator(owned.simulator)
    assert owned.robot._physics_sim_view is not None
    report = owned.simulator._sdk61_close_report
    assert report["cleanup_succeeded"] is False and report["app_exit_code"] == 1
    previous_count = len(owned.runtime.events)
    with pytest.raises(RuntimeError, match="Previous Sim 6.1 close failed"):
        close_module.close_owned_simulator(owned.simulator)
    assert len(owned.runtime.events) == previous_count


@pytest.mark.parametrize("retain_traceback", [False, True])
def test_active_exception_keeps_diagnostics_without_retaining_native_sdk_frames(owned, capsys, retain_traceback):
    saved_tracebacks = []
    def bad_step():
        native_alias = owned.runtime.backend.simulation_view
        assert native_alias is not None
        try:
            raise ValueError("original backend cause")
        except ValueError as error:
            if retain_traceback:
                saved_tracebacks.append(error.__traceback__)
            raise RuntimeError("native step failed") from error
    with pytest.raises(RuntimeError, match="native step failed"):
        try:
            bad_step()
        finally:
            assert sys.exception() is not None
            close_module.close_owned_simulator(owned.simulator)
    report = owned.simulator._sdk61_close_report
    assert report["cleanup_succeeded"] and report["app_exit_code"] == 1
    assert "bad_step" in report["active_exception_traceback"]
    assert "original backend cause" in capsys.readouterr().err
    if retain_traceback:
        # An external logger can retain the traceback itself after the error's
        # own traceback is unlinked. Its completed frame must not own views.
        assert saved_tracebacks[0].tb_frame.f_locals == {}


def test_owned_keyboard_releases_once_even_with_callback_and_external_owner(owned):
    events = []
    class Keyboard:
        def __init__(self):
            self._keyboard_sub = 123
            self._keyboard = object()
            self._appwindow = object()
            self._input = SimpleNamespace(unsubscribe_to_keyboard_events=lambda key, token: events.append(token))
            self._additional_callbacks = {"C": lambda: owned.clients.env.crowdsim_robot_camera}
    keyboard = close_module.owned_keyboard_class(Keyboard)()
    owned.simulator.keyboard_interface = keyboard
    owned.simulator._perspective_view = SimpleNamespace(viewport_api=object())
    close_module.close_owned_simulator(owned.simulator)
    keyboard.close()
    keyboard.__del__()
    assert events == [123]
    assert keyboard._additional_callbacks == {}
    assert keyboard._input is keyboard._keyboard is keyboard._appwindow is None
    assert owned.simulator.keyboard_interface is owned.simulator._perspective_view is None


def test_fast_shutdown_default_is_disabled_for_complete_close(owned, monkeypatch):
    settings = []
    fake = ModuleType("carb")
    fake.settings = SimpleNamespace(get_settings=lambda: SimpleNamespace(set_bool=lambda *args: settings.append(args)))
    monkeypatch.setitem(sys.modules, "carb", fake)
    owned.simulator._simulation_app.config = {"fast_shutdown": True}
    app = owned.simulator._simulation_app
    close_module.close_owned_simulator(owned.simulator)
    assert app.config["fast_shutdown"] is False
    assert settings == [("/app/fastShutdown", False)]


def test_keyboard_release_failure_still_attempts_complete_app_close(owned, capsys):
    attempts = []
    def bad_unsubscribe(*args):
        attempts.append(args)
        raise RuntimeError("keyboard unsubscribe failure")
    class Keyboard:
        def __init__(self):
            self._keyboard_sub, self._keyboard, self._appwindow = 123, object(), object()
            self._input = SimpleNamespace(unsubscribe_to_keyboard_events=bad_unsubscribe)
            self._additional_callbacks = {"C": lambda: owned.camera}
    keyboard = close_module.owned_keyboard_class(Keyboard)()
    owned.simulator.keyboard_interface = keyboard
    viewer = SimpleNamespace(viewport_api=object())
    owned.simulator._perspective_view = viewer
    with pytest.raises(RuntimeError, match="keyboard unsubscribe failure"):
        close_module.close_owned_simulator(owned.simulator)
    assert "keyboard unsubscribe failure" in capsys.readouterr().err
    report = owned.simulator._sdk61_close_report
    assert report["app_close_returned"] and report["app_exit_code"] == 1
    assert not report["cleanup_succeeded"]
    keyboard.close()
    keyboard.__del__()
    assert len(attempts) == 1
    assert keyboard._input is keyboard._keyboard is keyboard._appwindow is None
    assert keyboard._additional_callbacks == {}
    assert viewer.viewport_api is None
    assert owned.simulator._perspective_view is owned.simulator.keyboard_interface is None


def test_other_sdk_has_no_simulator_or_kit_access(monkeypatch):
    monkeypatch.setattr(close_module, "uses_sim61", lambda: False)
    assert close_module.close_owned_simulator(object()) is False
    assert close_module.owned_keyboard_class(object) is object


def test_production_close_dispatches_and_legacy_exit_code_is_unchanged(monkeypatch):
    """Execute the real close method without importing the native SDK module."""
    source = Path(__file__).resolve().parents[1] / "src/protomotions/simulator/isaaclab/simulator.py"
    tree = ast.parse(source.read_text())
    original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "IsaacLabSimulator")
    method = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == "close")
    events = []
    class Base:
        def close(self):
            events.append("base")
    node = ast.ClassDef(name="IsaacLabSimulator", bases=[ast.Name(id="Base", ctx=ast.Load())],
        keywords=[], body=[method], decorator_list=[])
    namespace = {"Base": Base, "__package__": "protomotions.simulator.isaaclab"}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(source), "exec"), namespace)
    simulator = namespace["IsaacLabSimulator"]()
    simulator._shutdown_exit_code = 7
    simulator._simulation_app = SimpleNamespace(close=lambda **kwargs: events.append(kwargs))
    monkeypatch.setattr(close_module, "close_owned_simulator", lambda obj: events.append("owned") or True)
    simulator.close()
    assert events == ["base", "owned"]
    events.clear()
    monkeypatch.setattr(close_module, "close_owned_simulator", lambda obj: False)
    simulator.close()
    assert events == ["base", {"exit_code": 7}]
