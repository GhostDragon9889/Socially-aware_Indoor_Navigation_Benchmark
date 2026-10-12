"""Execute production environment cleanup without importing the native SDK."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from test_sdk61_lifecycle import runtime  # noqa: F401
from test_sdk61_owned_close import owned  # noqa: F401
from protomotions.simulator.isaaclab.utils.sdk61_owned_close import close_owned_simulator


@pytest.fixture
def close_env():
    path = Path(__file__).resolve().parents[1] / 'src/protomotions/envs/base_env/env.py'
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'BaseEnv')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'close')
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['close']


def exception_chain(error):
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        pending.extend(value for value in (current.__cause__, current.__context__) if value is not None)
    return seen


@pytest.mark.parametrize('failures', [(), ('first',), ('second',), ('ui',), ('simulator',),
                                    ('first', 'second', 'ui', 'simulator')])
def test_cleanup_attempts_every_owner_in_order_and_preserves_errors(close_env, failures):
    events, active = [], {}
    errors = {name: RuntimeError(name + ' failed') for name in failures}

    def cleanup(name):
        events.append(name)
        active[name] = sys.exception()
        if name in errors:
            raise errors[name]

    env = SimpleNamespace(control_manager=SimpleNamespace(components={
        name: SimpleNamespace(close=lambda name=name: cleanup(name)) for name in ('first', 'second')}),
        _key_bindings=SimpleNamespace(unregister_all=lambda: cleanup('ui')),
        simulator=SimpleNamespace(close=lambda: cleanup('simulator')))
    if failures:
        with pytest.raises(RuntimeError) as caught:
            close_env(env)
        assert caught.value is errors[failures[-1]]
        assert {id(error) for error in errors.values()} <= exception_chain(caught.value)
    else:
        close_env(env)
    assert events == ['first', 'second', 'ui', 'simulator']
    assert env._key_bindings is None
    preceding = [name for name in ('first', 'second', 'ui') if name in errors]
    assert active['simulator'] is (errors[preceding[-1]] if preceding else None)


def test_cleanup_preserves_active_original_exception_and_interrupt(close_env):
    original, interrupted = ValueError('original operation failed'), KeyboardInterrupt('cleanup interrupted')
    seen = []

    def component_close():
        raise interrupted

    env = SimpleNamespace(control_manager=SimpleNamespace(components={
        'first': SimpleNamespace(close=component_close),
        'second': SimpleNamespace(close=lambda: seen.append('second'))}),
        _key_bindings=SimpleNamespace(unregister_all=lambda: seen.append('ui')),
        simulator=SimpleNamespace(close=lambda: seen.append(sys.exception())))
    with pytest.raises(KeyboardInterrupt) as caught:
        try:
            raise original
        finally:
            close_env(env)
    assert caught.value is interrupted
    assert id(original) in exception_chain(caught.value)
    assert seen == ['second', 'ui', interrupted]
    assert env._key_bindings is None


def test_cleanup_accepts_missing_optional_owners(close_env):
    close_env(SimpleNamespace())


@pytest.mark.parametrize('failure_stage', ['component', 'ui', 'both'])
def test_cleanup_failures_reach_real_sdk_lifecycle_without_retaining_native_frames(close_env, owned, failure_stage):  # noqa: F811 - imported pytest fixture
    events = []

    def cleanup(name):
        events.append(name)
        if failure_stage in (name, 'both'):
            native_alias = owned.runtime.backend.simulation_view
            assert native_alias is not None
            raise RuntimeError(name + ' native cleanup failed')

    env = SimpleNamespace(control_manager=SimpleNamespace(components={
        'failing': SimpleNamespace(close=lambda: cleanup('component')),
        'remaining': SimpleNamespace(close=lambda: events.append('remaining'))}),
        _key_bindings=SimpleNamespace(unregister_all=lambda: cleanup('ui')),
        simulator=SimpleNamespace(close=lambda: close_owned_simulator(owned.simulator)))
    with pytest.raises(RuntimeError, match='native cleanup failed'):
        close_env(env)
    report = owned.simulator._sdk61_close_report
    assert events == ['component', 'remaining', 'ui']
    assert env._key_bindings is None
    assert report['cleanup_succeeded'] and report['app_exit_code'] == 1
    assert 'native cleanup failed' in report['active_exception_traceback']
    assert owned.runtime.backend.simulation_view is None
    for asset in (owned.robot, owned.car, owned.box):
        assert asset._physics_sim_view is asset._data._physics_sim_view is None
