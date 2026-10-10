"""Source and owned-instance alias fixtures; no native SDK smoke evidence."""
import ast
import gc
import json
import subprocess
import sys
from types import SimpleNamespace
import weakref

import pytest

from CrowdSim.humanoid.runtime.native_social_navigation_lifecycle import (
    NativeNavigationLifecycle, PARENT, derive_native_release,
)
from CrowdSim.humanoid.runtime import native_social_observation_producer as producer_migration
from scripts.evaluate_shared_social_native import ROOT, prepare_entry


@pytest.mark.parametrize("actors,planner", [(3, "diffusion"), (15, "diffusion"), (3, "sfm"), (15, "sfm")])
def test_source_composition_has_explicit_native_actor_and_owner_contract(actors, planner):
    namespace, binding = prepare_entry(actors=actors, planner=planner)
    assert namespace._migration_observation_producer is producer_migration
    assert hasattr(namespace._migration_observation_producer, "RawSocialNativeEvaluationObservations")
    sources = namespace._sources()
    assert sources["scripted_origin_adapter"]["path"] == producer_migration.__file__
    namespace._profile(namespace.HERE / "evaluation_profile.json")
    derived, proof = namespace._derive(namespace.DRIVER.read_text())
    compile(derived, "prepared_native_driver", "exec")
    assert binding["native_entry_derivation"]["whole_source_reverse_equal"]
    assert binding["canonical_entry_migration"]["exact_inverse"]
    assert binding["native_lifecycle"]["navigation_actor_count"] == actors
    assert binding["native_lifecycle"]["expected_owned_contact_views"] == actors + 1
    assert binding["native_lifecycle"]["expected_SDK_sensor_objects"] == 24
    assert not binding["actual_native_SDK_execution_verified"]
    if planner == "sfm":
        assert "planner = NativeSocialForcePlanner" in derived
        assert proof["shared_control_reference_measurement_math_changed"] is False
    else:
        assert "PaperDiffusionPlanner" in derived


@pytest.mark.parametrize("actors", [2, 16, True])
def test_actor_contract_cannot_be_extended_implicitly(actors):
    with pytest.raises(ValueError):
        prepare_entry(actors=actors, planner="diffusion")


def test_native_producer_changes_only_origin_hook():
    source = producer_migration.PARENT.read_text()
    derived, proof = producer_migration.derive_native_origin(source)
    assert proof["exact_inverse"] and proof["other_methods_AST_equal"]
    assert proof["observation_query_clock_reset_math_changed"] is False
    assert "_migration_origin_deriver(native_source)" in derived
    with pytest.raises(ValueError, match="Ambiguous"):
        producer_migration.derive_native_origin(source.replace("derived, instance.derivation", "changed"))


def test_native_release_preserves_exact_one_asset_call_and_24_sensor_contract():
    original = PARENT.read_text()
    derived, proof = derive_native_release(original)
    assert proof["exact_inverse"] and proof["replacement_count"] == 11
    def calls(value):
        return [ast.dump(node, include_attributes=False) for node in ast.walk(ast.parse(value))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "release_asset"]
    assert calls(original) == calls(derived)
    assert "len(sensors) != 24" in derived
    assert "self._release_native_aliases(world, producer, simulator._sim)" in derived
    with pytest.raises(ValueError, match="exact current N4"):
        derive_native_release(original + "\n")


def alias_fixture():
    class Context:
        def get_physics_step_count(self):
            return 0
    class Core:
        def describe(self):
            return {"fixture_only": True}
    class Facade:
        def describe(self):
            return self._base_describe()
    class World:
        pass
    world, core, facade, context = World(), Core(), Facade(), Context()
    provider = SimpleNamespace(_query=object(), _counter=context.get_physics_step_count)
    world.scene_provider, world.validation_scene = provider, object()
    core.world, core.sim, core.sampler = world, context, object()
    facade.world, facade._core, facade._base_describe = world, core, core.describe
    facade._provider, facade._scene = provider, world.validation_scene
    core.describe = facade.describe
    return world, core, facade, context, provider


def test_terminal_instance_alias_release_breaks_facade_core_context_cycle():
    world, core, facade, context, provider = alias_fixture()
    lifecycle = NativeNavigationLifecycle.__new__(NativeNavigationLifecycle)
    lifecycle.report = {}
    refs = weakref.ref(core), weakref.ref(facade)
    lifecycle._release_native_aliases(world, facade, context)
    assert provider._query is provider._counter is None
    assert world.scene_provider is world.validation_scene is None
    assert core.world is core.sim is core.sampler is None
    assert "describe" not in core.__dict__
    assert facade.world is facade._core is facade._base_describe is facade._provider is facade._scene is None
    assert lifecycle.report["native_observation_facade_core_cycle_released"]
    del core, facade
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_foreign_context_alias_is_rejected_before_release():
    world, core, facade, context, provider = alias_fixture()
    lifecycle = NativeNavigationLifecycle.__new__(NativeNavigationLifecycle)
    lifecycle.report = {}
    original_query, original_counter = provider._query, provider._counter
    with pytest.raises(RuntimeError, match="changed before"):
        lifecycle._release_native_aliases(world, facade, object())
    assert core.sim is context and facade._core is core and world.scene_provider is provider
    assert provider._query is original_query and provider._counter is original_counter


@pytest.mark.parametrize("planner,count", [("diffusion", 3), ("sfm", 15)])
def test_source_cli_in_fresh_process_has_no_torch_or_sdk(planner, count):
    result = subprocess.run([sys.executable, str(ROOT / "scripts/evaluate_shared_social_native.py"),
        "--actors", str(count), "--planner", planner, "--check-sources"], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "native_source_checks_passed"
    assert report["Torch_imported"] is False and report["SDK_started"] is False
    assert report["migration"]["native_lifecycle"]["native_SDK_smoke_verified"] is False
