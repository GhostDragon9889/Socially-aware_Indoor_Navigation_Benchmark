"""Terminal ordering and failure propagation; no native physics is mocked as proof."""
from importlib import util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(os.environ.get('SOCIAL_NAV_TEST_REPO', Path(__file__).resolve().parents[1]))


def load(name, relative):
    path = Path(os.environ.get('SOCIAL_NAV_TEST_LIFECYCLE_FILE', ROOT / relative)) if name == 'test_owned_sdk61' else ROOT / relative
    spec = util.spec_from_file_location(name, path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    # Other test modules can load the real binding during collection. Keep it
    # alive in MonkeyPatch's undo stack and restore it unchanged after this
    # synthetic lifecycle; capsule-specific cases install their own module.
    monkeypatch.delitem(sys.modules, 'carb.eventdispatcher._eventdispatcher', raising=False)
    module = load('test_owned_sdk61', 'src/CrowdSim/humanoid/runtime/sdk61_lifecycle.py')
    versions = {'isaaclab': '17.0.2', 'isaacsim': '6.1.0.0', 'warp-lang': '1.16.0'}
    actual_version = module.metadata.version
    monkeypatch.setattr(module.metadata, 'version',
                        lambda name: versions[name] if name in versions else actual_version(name))
    events = []
    class View:
        pass
    backend = SimpleNamespace(simulation_view=View())
    class Manager:
        _timeline = object()
        callbacks = []
        @classmethod
        def dispatch_event(cls, event):
            events.append('explicit_STOP')
            for callback in cls.callbacks:
                callback()
        @staticmethod
        def raise_callback_exception_if_any():
            pass
        @staticmethod
        def get_scene_data_backend():
            return backend
    class Context:
        current = None
        @classmethod
        def instance(cls):
            return cls.current
        @classmethod
        def clear_instance(cls):
            events.append('clear')
            cls.current = None
    class Sim:
        physics_manager = Manager
        def stop(self):
            events.append('stop')
    def getter():
        return getter.timeline
    getter.timeline = object()  # A distinct native wrapper is valid on this SDK.
    for name in ('isaaclab', 'isaaclab.sim', 'isaaclab.sim.simulation_context', 'isaaclab.physics',
                 'isaaclab_physx', 'isaaclab_physx.physics',
                 'isaaclab_physx.physics.physx_manager', 'omni.timeline'):
        fake = ModuleType(name)
        fake.__file__ = str(tmp_path / (name + '.py'))
        Path(fake.__file__).write_text('# exact source fixture\n')
        monkeypatch.setitem(sys.modules, name, fake)
    sys.modules['isaaclab.sim'].SimulationContext = Context
    sys.modules['isaaclab.physics'].PhysicsEvent = SimpleNamespace(STOP='stop')
    sys.modules['isaaclab_physx.physics'].PhysxManager = Manager
    sys.modules['isaaclab_physx']._SIMULATION_MANAGER_ENABLE_HOOK = object()
    sys.modules['omni.timeline'].get_timeline_interface = getter
    report = {'status': 'passed'}
    lifecycle = module.SDK61Lifecycle(report)
    class App:
        def close(self, **kwargs):
            assert Context.instance() is None
            assert backend.simulation_view is None
            assert sys.modules['isaaclab_physx']._SIMULATION_MANAGER_ENABLE_HOOK is None
            assert Manager._timeline is None
            assert not hasattr(getter, 'timeline')
            events.append(('app', kwargs))
    return SimpleNamespace(module=module, versions=versions, events=events, backend=backend,
                           Manager=Manager, Context=Context, Sim=Sim, App=App,
                           report=report, lifecycle=lifecycle, getter=getter)


def test_preconstruction_source_identity_then_live_context(runtime):
    runtime.lifecycle.verify_loaded()
    assert runtime.report['evaluation_lifecycle']['live_context_verified'] is False
    runtime.Context.current = runtime.Sim()
    runtime.lifecycle.verify_loaded()
    assert runtime.report['evaluation_lifecycle']['live_context_verified'] is True
    assert len(runtime.report['evaluation_lifecycle']['loaded_SDK_sources']) == 2


def test_navigation_report_rebinding_reaches_the_sdk_delegate(runtime):
    from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import IsolatedTrackingLifecycle
    wrapper = IsolatedTrackingLifecycle.__new__(IsolatedTrackingLifecycle)
    wrapper._sdk61 = runtime.lifecycle
    wrapper.report = dict(runtime.report, cleanup_errors=[])
    wrapper.verify_loaded()
    assert runtime.lifecycle.report is wrapper.report
    # Navigation finish() replaces the outer report after source preparation.
    wrapper.report = dict(wrapper.report, cleanup_errors=[])
    runtime.Context.current = runtime.Sim()
    wrapper.close(runtime.App(), None)
    assert runtime.lifecycle.report is wrapper.report
    assert wrapper.report['cleanup_succeeded'] is True
    assert runtime.report['cleanup_succeeded'] is False


def test_constructor_rejects_different_sdk(runtime):
    runtime.versions['warp-lang'] = '1.15.0'
    with pytest.raises(RuntimeError, match='Unsupported SDK61'):
        runtime.module.SDK61Lifecycle({})


def test_context_owner_mismatch_is_rejected_before_stop(runtime):
    runtime.Context.current = runtime.Sim()
    foreign_world = SimpleNamespace(simulator=SimpleNamespace(_sim=runtime.Sim()))
    with pytest.raises(RuntimeError, match="another caller"):
        runtime.lifecycle.close(runtime.App(), foreign_world)
    assert runtime.events == []


def test_partial_constructor_context_cleanup_order(runtime):
    runtime.Context.current = runtime.Sim()
    runtime.lifecycle.close(runtime.App(), None)
    assert runtime.events[:2] == ['stop', 'clear']
    assert runtime.events[2][0] == 'app'
    assert runtime.events[2][1] == {'wait_for_replicator': False, 'skip_cleanup': False, 'exit_code': 0}
    assert runtime.report['cleanup_succeeded'] is True
    assert runtime.report['SDK_terminal_timeline_wrappers']['same_python_wrapper'] is False


def make_world(runtime):
    view = runtime.backend.simulation_view
    data = SimpleNamespace(_physics_sim_view=view)
    asset = SimpleNamespace(_physics_sim_view=view, _data=data, _root_view=None, _is_initialized=False)
    sensor = SimpleNamespace(_physics_sim_view=view, _body_physx_view=None,
                             _contact_view=None, _is_initialized=False)
    scene = SimpleNamespace(articulations={'robot': asset}, sensors={'foot': sensor})
    simulator = SimpleNamespace(_sim=runtime.Context.current, _robot=asset, _scene=scene,
                                _contact_sensor_map={'foot': sensor})
    return SimpleNamespace(simulator=simulator, scene_provider=None, validation_scene=None), asset, sensor, data


def test_asset_data_and_sensor_borrowed_aliases_are_released_before_app(runtime):
    runtime.Context.current = runtime.Sim()
    world, asset, sensor, data = make_world(runtime)
    runtime.lifecycle.close(runtime.App(), world)
    assert asset._physics_sim_view is data._physics_sim_view is sensor._physics_sim_view is None
    assert runtime.report['cleanup_succeeded'] is True
    assert runtime.report['public_backend_view_live_before_App_close'] is False
    assert runtime.report['SDK_owned_borrowed_simulation_view_aliases_released'] == [
        'articulation', 'articulation_data', 'sensor:foot']


@pytest.mark.parametrize('defect', [None, 'retained', 'missing', 'different_map'])
def test_navigation_sensor_and_backend_liveness_is_actually_measured(runtime, defect):
    runtime.Context.current = runtime.Sim()
    world = make_world(runtime)[0]
    class Sensor:
        _is_initialized = False
        _body_physx_view = _contact_view = None
        def __init__(self):
            self._physics_sim_view = runtime.backend.simulation_view
    scene = world.simulator._scene
    scene.sensors = {str(i): Sensor() for i in range(24)}
    world.simulator._contact_sensor_map = dict(scene.sensors)
    runtime.report['evaluation_lifecycle']['navigation_actor_count'] = 3
    held = scene.sensors['0'] if defect == 'retained' else None
    if defect == 'missing':
        scene.sensors.pop('0')
    elif defect == 'different_map':
        world.simulator._contact_sensor_map['0'] = Sensor()
    runtime.lifecycle.close(runtime.App(), world)
    assert runtime.report['cleanup_succeeded'] is (defect is None)
    if defect in (None, 'retained'):
        assert runtime.report['navigation_SDK_sensor_objects'] == {
            'observed_count': 24, 'live_before_app_shutdown': int(held is not None)}
    if defect is None:
        assert runtime.report['navigation_borrowed_backend_views'] == {
            'observed_count': 1, 'live_before_app_shutdown': 0}


def test_failed_world_constructor_retains_assets_until_full_app_close(runtime):
    owners = []

    class BrokenWorld:
        def __init__(self):
            runtime.Context.current = runtime.Sim()
            world, asset, sensor, data = make_world(runtime)
            self.__dict__.update(vars(world))
            owners.extend((asset, sensor, data))
            asset._is_initialized, asset._root_view = True, object()
            sensor._body_physx_view = object()
            def invalidate():
                asset._is_initialized, asset._root_view = False, None
                sensor._body_physx_view = None
            runtime.Manager.callbacks.append(invalidate)
            raise RuntimeError('sensor initialization failed')

    with pytest.raises(RuntimeError, match='sensor initialization failed'):
        runtime.lifecycle.construct_world(BrokenWorld)
    runtime.report['status'] = 'failed'
    runtime.lifecycle.close(runtime.App(), None)
    assert runtime.report['partial_construction_cleanup'] is True
    assert runtime.report['partial_constructor_STOP_callbacks_dispatched'] is True
    assert runtime.report['cleanup_succeeded'] is True
    assert runtime.events[-1][1]['exit_code'] == 1
    assert all(owner._physics_sim_view is None for owner in owners)
    assert runtime.lifecycle._pending_world is None


def test_constructor_cannot_replace_a_pending_world(runtime):
    class BrokenWorld:
        def __init__(self):
            raise RuntimeError('failed before context')
    with pytest.raises(RuntimeError, match='failed before context'):
        runtime.lifecycle.construct_world(BrokenWorld)
    with pytest.raises(RuntimeError, match='unowned context'):
        runtime.lifecycle.construct_world(BrokenWorld)
    runtime.lifecycle.close(runtime.App(), None)
    assert runtime.lifecycle._pending_world is None


def test_failure_before_any_scene_asset_is_registered_can_close(runtime):
    class BrokenWorld:
        def __init__(self):
            runtime.Context.current = runtime.Sim()
            self.simulator = SimpleNamespace(_sim=runtime.Context.current,
                                             _scene=SimpleNamespace(articulations={}, sensors={}))
            raise RuntimeError('asset spawn failed')
    with pytest.raises(RuntimeError, match='asset spawn failed'):
        runtime.lifecycle.construct_world(BrokenWorld)
    runtime.report['status'] = 'failed'
    runtime.lifecycle.close(runtime.App(), None)
    assert runtime.report['cleanup_succeeded'] is True
    assert runtime.report['SDK_owned_borrowed_simulation_view_aliases_released'] == []


def test_partial_stop_error_is_retained_even_if_all_views_were_released(runtime):
    class BrokenWorld:
        def __init__(self):
            runtime.Context.current = runtime.Sim()
            world, *_ = make_world(runtime)
            self.__dict__.update(vars(world))
            raise RuntimeError('construction failed')
    with pytest.raises(RuntimeError, match='construction failed'):
        runtime.lifecycle.construct_world(BrokenWorld)
    def fail():
        raise RuntimeError('STOP observer failed')
    runtime.Manager.callbacks.append(fail)
    runtime.lifecycle.close(runtime.App(), None)
    assert runtime.report['cleanup_succeeded'] is False
    assert any('STOP observer failed' in e for e in runtime.report['cleanup_errors'])
    assert runtime.report['public_backend_view_live_before_App_close'] is False
    assert runtime.events[-1][1]['exit_code'] == 1


def test_foreign_borrowed_alias_fails_before_mutating_owned_fields(runtime):
    runtime.Context.current = runtime.Sim()
    world, asset, sensor, data = make_world(runtime)
    foreign = object()
    data._physics_sim_view = foreign
    runtime.lifecycle.close(runtime.App(), world)
    assert data._physics_sim_view is foreign
    assert asset._physics_sim_view is not None
    assert runtime.report['cleanup_succeeded'] is False
    assert any('another backend' in item for item in runtime.report['cleanup_errors'])


def native_producer_fixture(runtime):
    class Core:
        def describe(self):
            return {}

    class Facade:
        def describe(self):
            return self._base_describe()

    runtime.Context.current = runtime.Sim()
    world, _, _, _ = make_world(runtime)
    world.scene_provider = SimpleNamespace(_query=object(), _counter=object())
    world.validation_scene = object()
    core, facade = Core(), Facade()
    core.world, core.sim, core.sampler = world, world.simulator._sim, object()
    facade.world, facade._core = world, core
    facade._provider, facade._scene = world.scene_provider, world.validation_scene
    facade._base_describe = core.describe
    core.describe = facade.describe
    return world, core, facade


def test_native_facade_core_and_borrowed_provider_are_released_before_app(runtime):
    import gc
    import weakref
    world, core, facade = native_producer_fixture(runtime)
    provider = world.scene_provider
    refs = weakref.ref(core), weakref.ref(facade)
    runtime.lifecycle.close(runtime.App(), world, facade)
    assert runtime.report['cleanup_succeeded'] is True
    assert runtime.report['native_observation_facade_core_cycle_released'] is True
    assert core.world is core.sim is core.sampler is None
    assert 'describe' not in core.__dict__
    assert facade.world is facade._core is facade._base_describe is facade._provider is facade._scene is None
    assert provider._query is provider._counter is None
    del core, facade
    gc.collect()
    assert all(ref() is None for ref in refs)


@pytest.mark.parametrize('defect', ['world', 'context', 'provider', 'scene', 'base_describe', 'describe'])
def test_native_facade_rejects_foreign_owner_without_releasing_any_alias(runtime, defect):
    world, core, facade = native_producer_fixture(runtime)
    if defect == 'world':
        core.world = object()
    elif defect == 'context':
        core.sim = object()
    elif defect == 'provider':
        facade._provider = object()
    elif defect == 'scene':
        facade._scene = object()
    elif defect == 'base_describe':
        facade._base_describe = lambda: {}
    else:
        core.describe = lambda: {}
    before = dict(core.__dict__), dict(facade.__dict__)
    with pytest.raises(RuntimeError, match='another owner'):
        runtime.lifecycle._release_producer(world, facade)
    assert (core.__dict__, facade.__dict__) == before


def test_plain_producer_rejects_foreign_context_before_release(runtime):
    runtime.Context.current = runtime.Sim()
    world, _, _, _ = make_world(runtime)
    producer = SimpleNamespace(world=world, sim=object(), sampler=object())
    before = dict(vars(producer))
    with pytest.raises(RuntimeError, match='another context'):
        runtime.lifecycle._release_producer(world, producer)
    assert vars(producer) == before


@pytest.mark.parametrize('mode,expected_views', [('floor', 1), ('peer', 4), ('combined', 5)])
def test_all_supported_contact_owner_layouts_release_before_stop(runtime, mode, expected_views):
    from CrowdSim.humanoid.contact_monitor import PedestrianContactMonitor
    class View:
        pass
    runtime.Context.current = runtime.Sim()
    world, asset, _, _ = make_world(runtime)
    world.num_envs, world.device = 4, 'cpu'
    peer = PedestrianContactMonitor.__new__(PedestrianContactMonitor)
    peer.world, peer.robot = world, asset
    peer.views = [(i, View()) for i in range(4)] if mode != 'floor' else []
    adapter = peer if mode == 'peer' else SimpleNamespace(
        world=world, robot=asset, floor=SimpleNamespace(view=View()), peer=peer if mode == 'combined' else None)
    world.simulator.pedestrian_contacts = adapter
    runtime.lifecycle.close(runtime.App(), world)
    assert runtime.report['cleanup_succeeded'] is True
    assert runtime.report['navigation_owned_contact_views'] == {
        'observed_count': expected_views, 'live_before_app_shutdown': 0}
    assert runtime.report['contacts_released_before_timeline_stop'] is True
    if mode != 'floor':
        assert not peer.views and peer.world is peer.robot is None
    assert adapter.world is adapter.robot is None


def test_live_articulation_is_not_terminally_detached(runtime):
    runtime.Context.current = runtime.Sim()
    world, asset, sensor, data = make_world(runtime)
    asset._is_initialized = True
    runtime.lifecycle.close(runtime.App(), world)
    assert asset._physics_sim_view is not None
    assert runtime.report['cleanup_succeeded'] is False
    assert any('stop callback' in item for item in runtime.report['cleanup_errors'])


def test_stop_error_still_clears_and_full_closes_with_failure(runtime):
    class FailingSim(runtime.Sim):
        def stop(self):
            raise RuntimeError('stop failed')
    runtime.Context.current = FailingSim()
    runtime.lifecycle.close(runtime.App(), None)
    assert runtime.events[0] == 'clear'
    assert runtime.events[-1][1]['exit_code'] == 1
    assert runtime.report['cleanup_succeeded'] is False
    assert any('stop failed' in item for item in runtime.report['cleanup_errors'])


def test_physics_error_preserves_failure_exit_code(runtime):
    runtime.Context.current = runtime.Sim()
    runtime.report['status'] = 'failed_physics'
    runtime.lifecycle.close(runtime.App(), None)
    assert runtime.events[-1][1]['exit_code'] == 1
    assert runtime.report['status'] == 'failed_physics'


def test_full_app_close_failure_is_not_accepted(runtime):
    runtime.Context.current = runtime.Sim()
    class FailingApp:
        def close(self, **kwargs):
            raise RuntimeError('app shutdown failed')
    runtime.lifecycle.close(FailingApp(), None)
    assert runtime.report['app_close_returned'] is False
    assert runtime.report['cleanup_succeeded'] is False
    assert any('app shutdown failed' in item for item in runtime.report['cleanup_errors'])


def test_no_app_cannot_pass_cleanup(runtime):
    runtime.lifecycle.close(None, None)
    assert runtime.report['no_simulation_context_to_clear'] is True
    assert runtime.report['cleanup_succeeded'] is False


def test_runtime_fixture_preserves_preloaded_dispatcher_binding(monkeypatch, tmp_path):
    name = 'carb.eventdispatcher._eventdispatcher'
    preloaded = ModuleType(name)
    preloaded._cleanup = capsule = object()
    monkeypatch.setitem(sys.modules, name, preloaded)

    with pytest.MonkeyPatch.context() as isolated:
        owned = runtime.__wrapped__(isolated, tmp_path)
        assert name not in sys.modules
        owned.Context.current = owned.Sim()
        owned.lifecycle.close(owned.App(), None)
        assert owned.report['cleanup_succeeded'] is True
        assert owned.report['SDK_eventdispatcher_terminal_cleanup'] == {'capsule_present': False}
        assert preloaded._cleanup is capsule

    assert sys.modules[name] is preloaded
    assert preloaded._cleanup is capsule


def test_sdk61_framework_teardown_requires_complete_app_close(runtime):
    module = load('test_sdk61_framework', 'src/CrowdSim/humanoid/runtime/native_framework_shutdown.py')
    with pytest.raises(RuntimeError, match='Full App.close'):
        module.unload_native_plugins({}, full_app_close_returned=False)
    report = {}
    module.unload_native_plugins(report, full_app_close_returned=True)
    assert report['native_framework_shutdown']['manual_plugin_unload'] is False
    assert report['native_framework_shutdown']['actual_process_exit_requires_parent_wait'] is True
    assert report['native_framework_shutdown']['process_exit_masked'] is False


def install_dispatcher_capsule(runtime, monkeypatch, tmp_path):
    """Model the two Python owners; actual native exit is tested separately."""
    name = 'carb.eventdispatcher._eventdispatcher'
    module = ModuleType(name)
    path = tmp_path / '_eventdispatcher.so'
    path.write_bytes(b'reviewed binding fixture')
    module.__file__ = str(path)
    module.IEventDispatcher = type('IEventDispatcher', (), {})
    capsule_type = type('PyCapsule', (), {
        '__module__': 'builtins',
        '__del__': lambda self: runtime.events.append('dispatcher destructor'),
    })
    module._cleanup = capsule_type()
    snapshot = vars(module).copy()
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(runtime.module, '_EVENTDISPATCHER_BINDING_SHA256',
                        runtime.module.hashlib.sha256(path.read_bytes()).hexdigest())
    return module, snapshot


def test_dispatcher_both_capsule_aliases_destroy_before_full_app_close(runtime, monkeypatch, tmp_path):
    module, snapshot = install_dispatcher_capsule(runtime, monkeypatch, tmp_path)
    runtime.Context.current = runtime.Sim()
    runtime.lifecycle.close(runtime.App(), None)
    assert runtime.events[:3] == ['stop', 'clear', 'dispatcher destructor']
    assert runtime.events[3][0] == 'app'
    assert '_cleanup' not in vars(module) and '_cleanup' not in snapshot
    assert module.IEventDispatcher is snapshot['IEventDispatcher']
    assert runtime.report['cleanup_succeeded'] is True
    evidence = runtime.report['SDK_eventdispatcher_terminal_cleanup']
    assert evidence['released_dictionary_aliases'] == 2
    assert evidence['released_before_App_close'] is True


@pytest.mark.parametrize('problem', ['binary', 'foreign_dictionary', 'extra_reference', 'wrong_type'])
def test_dispatcher_unreviewed_owner_fails_without_mutating_aliases(
        runtime, monkeypatch, tmp_path, problem):
    module, snapshot = install_dispatcher_capsule(runtime, monkeypatch, tmp_path)
    runtime.Context.current = runtime.Sim()
    extra_reference = None
    if problem == 'binary':
        monkeypatch.setattr(runtime.module, '_EVENTDISPATCHER_BINDING_SHA256', 'not-reviewed')
    elif problem == 'foreign_dictionary':
        snapshot['__name__'] = 'another.module'
    elif problem == 'extra_reference':
        extra_reference = [module._cleanup]
    elif problem == 'wrong_type':
        module._cleanup = snapshot['_cleanup'] = object()
    runtime.lifecycle.close(runtime.App(), None)
    assert module._cleanup is snapshot['_cleanup']
    if extra_reference is not None:
        assert extra_reference[0] is module._cleanup
    assert runtime.report['cleanup_succeeded'] is False
    assert any('eventdispatcher' in error for error in runtime.report['cleanup_errors'])
    assert runtime.events[-1][1]['exit_code'] == 1


def test_dispatcher_not_terminally_released_without_app(runtime, monkeypatch, tmp_path):
    module, snapshot = install_dispatcher_capsule(runtime, monkeypatch, tmp_path)
    runtime.lifecycle.close(None, None)
    assert module._cleanup is snapshot['_cleanup']
    assert 'dispatcher destructor' not in runtime.events
    assert runtime.report['cleanup_succeeded'] is False


def test_dispatcher_missing_public_alias_does_not_hide_live_snapshot(runtime, monkeypatch, tmp_path):
    module, snapshot = install_dispatcher_capsule(runtime, monkeypatch, tmp_path)
    del module._cleanup
    runtime.Context.current = runtime.Sim()

    runtime.lifecycle.close(runtime.App(), None)

    assert '_cleanup' in snapshot
    assert 'dispatcher destructor' not in runtime.events
    assert runtime.report['cleanup_succeeded'] is False
    assert any('missing without a verified release' in error for error in runtime.report['cleanup_errors'])
    assert runtime.events[-1][1]['exit_code'] == 1


def test_dispatcher_verified_release_is_idempotent_within_same_lifecycle(runtime, monkeypatch, tmp_path):
    module, snapshot = install_dispatcher_capsule(runtime, monkeypatch, tmp_path)
    runtime.lifecycle._release_eventdispatcher_cleanup()
    evidence = dict(runtime.report['SDK_eventdispatcher_terminal_cleanup'])

    runtime.lifecycle._release_eventdispatcher_cleanup()

    assert '_cleanup' not in vars(module) and '_cleanup' not in snapshot
    assert runtime.events.count('dispatcher destructor') == 1
    assert runtime.report['SDK_eventdispatcher_terminal_cleanup'] == evidence


@pytest.mark.parametrize('change', ['new_lifecycle', 'module_identity', 'path', 'binary'])
def test_dispatcher_stale_release_evidence_cannot_authorize_missing_capsule(
        runtime, monkeypatch, tmp_path, change):
    module, snapshot = install_dispatcher_capsule(runtime, monkeypatch, tmp_path)
    runtime.lifecycle._release_eventdispatcher_cleanup()
    lifecycle = runtime.lifecycle
    if change == 'new_lifecycle':
        lifecycle = runtime.module.SDK61Lifecycle(runtime.report)
    elif change == 'module_identity':
        replacement = ModuleType(module.__name__)
        replacement.__dict__.update(vars(module))
        monkeypatch.setitem(sys.modules, module.__name__, replacement)
    elif change == 'path':
        copied_path = tmp_path / 'other_eventdispatcher.so'
        copied_path.write_bytes(Path(module.__file__).read_bytes())
        module.__file__ = str(copied_path)
    else:
        Path(module.__file__).write_bytes(b'changed binding after first release')

    with pytest.raises(RuntimeError, match='eventdispatcher'):
        lifecycle._release_eventdispatcher_cleanup()

    assert runtime.events.count('dispatcher destructor') == 1
    assert not runtime.report.get('SDK_eventdispatcher_terminal_cleanup', {}).get('released_before_App_close', False)
