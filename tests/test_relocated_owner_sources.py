"""Exact module origin gates must resolve package links after the src move."""
import importlib
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


@pytest.mark.parametrize('module,class_name', [
    ('CrowdSim.humanoid.runtime.trace_pace_measured_lifecycle', 'TracePaceMeasuredLifecycle'),
    ('CrowdSim.humanoid.runtime.social_navigation_lifecycle', 'FourActorNavigationLifecycle'),
])
def test_class_origin_gate_accepts_only_the_exact_pinned_files(module, class_name, monkeypatch, tmp_path):
    implementation = importlib.import_module(module)
    names = ('PedestrianContactMonitor', 'FootSupportMonitor', 'FootCaptureAdapter', 'ContactSensor')
    for (name, (path, _sha)), cls_name in zip(implementation.PINS.items(), names):
        cls = type(cls_name, (), {'__module__': name})
        monkeypatch.setitem(sys.modules, name, SimpleNamespace(__file__=str(path), **{cls_name: cls}))
    owner = getattr(implementation, class_name)
    assert set(owner._classes()) == set(names)
    name, (path, _sha) = next(iter(implementation.PINS.items()))
    copied = tmp_path / 'different_module.py'
    copied.write_bytes(Path(path).read_bytes())
    monkeypatch.setattr(sys.modules[name], '__file__', str(copied))
    with pytest.raises(RuntimeError, match='ownership source differs'):
        owner._classes()
