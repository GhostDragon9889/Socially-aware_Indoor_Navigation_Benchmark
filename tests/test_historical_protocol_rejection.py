"""Real source/version mismatch rejection; no SDK launch or fake native PASS."""
import hashlib
from pathlib import Path

import pytest


def test_native_initial_core_accepts_only_the_reversible_location_migration(monkeypatch):
    from CrowdSim.humanoid.analysis import native_paired_initial as legacy
    actual = hashlib.sha256(legacy.PARENT.read_bytes()).hexdigest()
    assert actual != legacy.PARENT_SHA
    assert hashlib.sha256(legacy.restore_core_location(legacy.PARENT.read_text()).encode()).hexdigest() == legacy.PARENT_SHA
    bindings = legacy.source_bindings()
    assert bindings["source_version"] == "native_initial.current_scene_sources.v2"
    assert bindings["core"]["sha256"] == actual
    with pytest.raises(ValueError, match="exact fixed N4"):
        legacy.restore_core_location(legacy.PARENT.read_text() + "\n# unregistered change\n")
    monkeypatch.setattr(legacy, "PARENT_SHA", "0" * 64)
    with pytest.raises(ValueError, match="Fixed initial-state core changed"):
        legacy.source_bindings()


def test_current_CPU_reader_validates_unchanged_record_and_rejects_unregistered_source(monkeypatch):
    from scripts import audit_shared_social_high_level as legacy
    from CrowdSim.humanoid.analysis import cpu_dependency_migration as migration
    path, expected = next(row for row in legacy.DEPENDENCIES if row[0].endswith("social_evaluation_lifecycle.py"))
    assert hashlib.sha256((legacy.ROOT / path).read_bytes()).hexdigest() != expected
    proof = migration.verify_dependency(legacy.ROOT / path, expected)
    assert proof['exact_function_bytes_unchanged']
    assert proof['SDK_execution_verified'] is False
    assert len(legacy.reader_dependencies()) == 5
    monkeypatch.setattr(migration, 'LIFECYCLE_CURRENT_SHA', '0' * 64)
    with pytest.raises(ValueError, match="Pinned transitive CPU reader dependency changed"):
        legacy.reader_dependencies()


def test_sim61_contact_source_cannot_satisfy_the_sealed_sim60_hash():
    from CrowdSim.humanoid.runtime import trace_pace_measured_lifecycle as legacy
    from social_navigation.sdk_paths import lab_root
    name = "isaaclab_physx.sensors.contact_sensor.contact_sensor"
    _, expected = legacy.PINS[name]
    current = lab_root() / "source/isaaclab_physx/isaaclab_physx/sensors/contact_sensor/contact_sensor.py"
    assert hashlib.sha256(current.read_bytes()).hexdigest() != expected
    previous = Path("/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source/isaaclab_physx/isaaclab_physx/sensors/contact_sensor/contact_sensor.py")
    if previous.is_file():
        assert hashlib.sha256(previous.read_bytes()).hexdigest() == expected


def test_registered_current_sfm_accepts_live_source_and_rejects_unregistered_edit():
    from CrowdSim.humanoid.trajectory import native_social_force_runtime as current
    bindings = current.source_bindings()
    assert bindings["source_version"] == "native_sfm.current_scene_sources.v2"
    source = (current.ROOT / current.PINS["fixture_bridge"][0]).read_text()
    _, proof = current.derive_actor_methods(source)
    assert proof["historical_actor_method_bytes_equal"] and proof["fixture_location_exact_inverse"]
    with pytest.raises(ValueError, match="exact source-bound"):
        current.derive_actor_methods(source + "\n# unregistered change\n")
