"""Terminal PhysX cache ownership, without loading Kit or its native binding."""
import ctypes
import hashlib
import sys
from types import SimpleNamespace

import pytest

from protomotions.simulator.isaaclab.utils import sdk61_owned_close as terminal
from test_sdk61_lifecycle import runtime  # noqa: F401
from test_sdk61_owned_close import owned  # noqa: F401


@pytest.fixture
def cache(monkeypatch, tmp_path):
    path = tmp_path / '_physx.so'
    path.write_bytes(b'reviewed native fixture')
    monkeypatch.setattr(terminal, '_PHYSX_BINDING_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())
    monkeypatch.setitem(sys.modules, 'omni.physx.bindings._physx', SimpleNamespace(__file__=str(path)))
    slot = ctypes.c_void_p()
    monkeypatch.setattr(terminal, '_physx_cache_slot', lambda binding: slot)
    return path, slot


def store(slot, value):
    ctypes.pythonapi.Py_IncRef(ctypes.py_object(value))
    slot.value = id(value)


def discard(slot):
    if slot.value:
        value = slot.value
        slot.value = None
        ctypes.pythonapi.Py_DecRef(ctypes.c_void_p(value))


def test_releases_exactly_the_static_reference_and_can_repeat(cache):
    _, slot = cache
    paths = ['/World/template', '/World/envs']
    store(slot, paths)
    before = sys.getrefcount(paths)
    report = {}
    try:
        terminal._release_physx_replicator_cache(report)
        assert slot.value is None
        assert sys.getrefcount(paths) == before - 1
        assert paths == ['/World/template', '/World/envs']
        assert report['physx_replicator_terminal_cache']['released_before_App_close'] is True
        terminal._release_physx_replicator_cache({})
    finally:
        discard(slot)


@pytest.mark.parametrize('value', [('/World/template', '/World/envs'), [], ['/foreign'], [True, '/World/envs']])
def test_foreign_cache_shape_or_paths_are_rejected_before_mutation(cache, value):
    _, slot = cache
    store(slot, value)
    pointer = slot.value
    try:
        with pytest.raises(RuntimeError, match='Unexpected PhysX replicator cache'):
            terminal._release_physx_replicator_cache({})
        assert slot.value == pointer
    finally:
        discard(slot)


def test_unreviewed_binary_is_rejected_before_resolving_memory(cache, monkeypatch):
    path, _ = cache
    path.write_bytes(b'changed native fixture')
    monkeypatch.setattr(terminal, '_physx_cache_slot', lambda binding: pytest.fail('memory access before identity check'))
    with pytest.raises(RuntimeError, match='Unreviewed PhysX binding'):
        terminal._release_physx_replicator_cache({})


def test_unloaded_binding_requires_no_native_import(monkeypatch):
    monkeypatch.delitem(sys.modules, 'omni.physx.bindings._physx', raising=False)
    report = {}
    terminal._release_physx_replicator_cache(report)
    assert report['physx_replicator_terminal_cache']['binding_loaded'] is False


def test_cache_release_occurs_after_native_views_and_before_app_close(owned, monkeypatch):
    calls = []

    def release(report):
        assert owned.runtime.Context.instance() is None
        assert owned.runtime.backend.simulation_view is None
        assert not any(isinstance(event, tuple) for event in owned.runtime.events)
        calls.append(report)

    monkeypatch.setattr(terminal, '_release_physx_replicator_cache', release)
    terminal.close_owned_simulator(owned.simulator)
    terminal.close_owned_simulator(owned.simulator)
    assert calls == [owned.simulator._sdk61_close_report]
    assert owned.simulator._sdk61_close_report['cleanup_succeeded']


def test_common_lifecycle_releases_cache_even_without_production_adapter(runtime, monkeypatch):
    from test_sdk61_lifecycle import make_world
    runtime.Context.current = runtime.Sim()
    world = make_world(runtime)[0]
    calls = []

    def release(report):
        assert runtime.Context.instance() is None
        assert runtime.backend.simulation_view is None
        assert not any(isinstance(event, tuple) for event in runtime.events)
        calls.append(report)

    monkeypatch.setattr(terminal, '_release_physx_replicator_cache', release)
    runtime.lifecycle.close(runtime.App(), world)
    assert calls == [runtime.report]
    assert runtime.report['cleanup_succeeded']


def test_cache_rejection_preserves_failure_and_still_attempts_full_app_close(runtime, monkeypatch):
    from test_sdk61_lifecycle import make_world

    def release(report):
        raise RuntimeError('unreviewed fixture binding')

    monkeypatch.setattr(terminal, '_release_physx_replicator_cache', release)
    runtime.Context.current = runtime.Sim()
    runtime.lifecycle.close(runtime.App(), make_world(runtime)[0])
    assert runtime.report['app_close_returned']
    assert not runtime.report['cleanup_succeeded']
    assert 'unreviewed fixture binding' in runtime.report['cleanup_errors'][0]
    assert runtime.events[-1][1]['exit_code'] == 1


def test_failed_context_clear_never_releases_terminal_cache(runtime, monkeypatch):
    from test_sdk61_lifecycle import make_world
    runtime.Context.current = runtime.Sim()
    monkeypatch.setattr(runtime.Context, 'clear_instance', lambda: None)
    calls, closes = [], []
    monkeypatch.setattr(terminal, '_release_physx_replicator_cache', lambda report: calls.append(report))
    app = SimpleNamespace(close=lambda **kwargs: closes.append(kwargs))
    runtime.lifecycle.close(app, make_world(runtime)[0])
    assert calls == []
    assert closes[0]['exit_code'] == 1
    assert not runtime.report['cleanup_succeeded']
    assert any('context' in error.lower() for error in runtime.report['cleanup_errors'])


def test_terminal_registration_preserves_frozen_manifest_and_proves_inverse():
    import json
    from CrowdSim.humanoid.runtime import grouped_source_migration as grouped
    from CrowdSim.humanoid.runtime import terminal_cache_source_migration as migration
    before = grouped.MANIFEST.read_bytes()
    current = grouped.manifest()
    original = json.loads(before)
    assert original['dependencies'][str(migration.LIFECYCLE)]['sha256'] == migration.HISTORICAL_SHA
    assert current['terminal_cache_migration']['whole_source_inverse_verified']
    assert not current['terminal_cache_migration']['physical_execution_verified']
    assert grouped.MANIFEST.read_bytes() == before


@pytest.mark.parametrize('target', ['seam', 'other_byte', 'helper'])
def test_terminal_registration_rejects_changes_outside_exact_fix(target, monkeypatch):
    import json
    from CrowdSim.humanoid.runtime import grouped_source_migration as grouped
    from CrowdSim.humanoid.runtime import terminal_cache_source_migration as migration
    inventory = json.loads(grouped.MANIFEST.read_text())
    read_bytes = type(migration.LIFECYCLE).read_bytes

    def changed(path):
        data = read_bytes(path)
        if path == migration.LIFECYCLE:
            if target == 'seam':
                data = data.replace(migration.TERMINAL_ADDITION.encode(), b'')
            elif target == 'other_byte':
                data += b'\n'
        if path == migration.HELPER and target == 'helper':
            data += b'\n'
        return data

    monkeypatch.setattr(type(migration.LIFECYCLE), 'read_bytes', changed)
    with pytest.raises(ValueError):
        migration.admit_dependency(inventory)
