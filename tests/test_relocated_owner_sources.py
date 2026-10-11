"""Exact module origin gates must resolve package links after the src move."""
import importlib
import hashlib
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
    # This is the explicitly sealed Sim6.0 class-origin contract. Sim6.1
    # uses SDK61Lifecycle and must never satisfy these old source hashes.
    legacy_contact = Path("/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab_physx/isaaclab_physx/sensors/contact_sensor/contact_sensor.py")
    if not legacy_contact.is_file():
        pytest.skip("The sealed Sim6.0 SDK source is not installed")
    pins = dict(implementation.PINS)
    # The current monitor now records its world/robot owner for SDK61 cleanup.
    # This synthetic legacy class-origin test must use the exact old bytes;
    # prove the constructor-only inverse instead of rebasing a historical pin.
    name = 'CrowdSim.humanoid.contact_monitor'
    current, expected = pins[name]
    addition = ('        # Terminal cleanup must prove the monitor belongs to this exact world\n'
                '        # and articulation, including when no floor-capture wrapper is used.\n'
                '        self.world, self.robot = world, world.simulator._robot\n')
    source = current.read_text()
    assert source.count(addition) == 1
    legacy_source = source.replace(addition, '', 1)
    assert hashlib.sha256(legacy_source.encode()).hexdigest() == expected
    fixture = tmp_path / 'sealed_contact_monitor.py'
    fixture.write_text(legacy_source)
    pins[name] = (fixture, expected)
    contact_name = "isaaclab_physx.sensors.contact_sensor.contact_sensor"
    pins[contact_name] = (legacy_contact, pins[contact_name][1])
    monkeypatch.setattr(implementation, "PINS", pins)
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
