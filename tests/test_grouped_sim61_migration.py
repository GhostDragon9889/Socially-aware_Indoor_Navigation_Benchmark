"""Source rejection and cleanup failure tests; CPU fixtures are not physics evidence."""
from pathlib import Path
import sys
from types import SimpleNamespace
import weakref

import pytest

from CrowdSim.humanoid.runtime import grouped_source_migration as migration
from CrowdSim.humanoid.runtime.grouped_sdk61_lifecycle import GroupedSDK61Lifecycle
from test_sdk61_lifecycle import runtime as runtime_fixture

runtime = runtime_fixture


@pytest.fixture
def grouped(runtime):
    runtime.Context.current = runtime.Sim()
    lifecycle = GroupedSDK61Lifecycle(runtime.report)
    assets = tuple(SimpleNamespace(_is_initialized=False, _root_view=None,
        _physics_sim_view=runtime.backend.simulation_view,
        _data=SimpleNamespace(_physics_sim_view=runtime.backend.simulation_view)) for _ in range(4))
    scene = SimpleNamespace(cfg=SimpleNamespace(num_envs=2), sensors={},
                            articulations={f"robot_{i}": a for i, a in enumerate(assets)})
    class View:
        pass
    class World:
        def __init__(self):
            self.sim, self.scene, self.assets = runtime.Context.current, scene, assets
            self.device, self.views = "cpu", [View() for _ in range(8)]

        def release_owned_native(self, callback):
            try:
                callback(self)
            finally:
                self.sim = self.scene = None
                self.assets = ()
    world = World()
    receipts, tracker = [], list(assets)
    def release(owner):
        runtime.events.append("contacts")
        receipts.append({"native_weakrefs": tuple(weakref.ref(v) for v in owner.views),
                         "native_view_count": 8, "module_owned_aliases_released": True})
        owner.views.clear()
    collector, expert = SimpleNamespace(world=world), SimpleNamespace(_world=world)
    lifecycle.configure(release, receipts, tracker, 8, collector, expert)
    return SimpleNamespace(runtime=runtime, lifecycle=lifecycle, world=world, assets=assets,
                           collector=collector, expert=expert, receipts=receipts, tracker=tracker)


def test_four_articulations_and_contacts_release_in_order(grouped):
    g, r = grouped, grouped.runtime
    g.lifecycle.close(r.App(), g.world)
    assert r.events[:3] == ["contacts", "stop", "clear"]
    assert r.events[-1][0] == "app"
    assert r.report["cleanup_succeeded"] is True
    assert r.report["owned_contact_views_live_after_release"] == 0
    assert r.report["owned_asset_simulation_view_release_counts"] == {"articulation_fields": 4, "data_fields": 4}
    assert all(a._physics_sim_view is a._data._physics_sim_view is None for a in g.assets)
    assert g.collector.world is g.expert._world is None
    assert g.lifecycle.scene is None and g.lifecycle.assets == () and g.tracker == []


def test_partial_constructor_without_receipt_never_claims_success(grouped):
    g, r = grouped, grouped.runtime
    for a in g.assets:
        del a._physics_sim_view
        del a._data._physics_sim_view
    g.lifecycle.close(r.App(), None)
    assert r.report["app_close_returned"] is True
    assert r.report["cleanup_succeeded"] is False
    assert any("partial contact release" in e for e in r.report["cleanup_errors"])
    assert any("eight borrowed" in e for e in r.report["cleanup_errors"])


def test_last_foreign_data_alias_rejects_before_any_of_eight_fields_change(grouped):
    g, r = grouped, grouped.runtime
    foreign = object()
    g.assets[-1]._data._physics_sim_view = foreign
    before = [(a._physics_sim_view, a._data._physics_sim_view) for a in g.assets]
    g.lifecycle.close(r.App(), g.world)
    assert r.report["cleanup_succeeded"] is False
    assert any("foreign backend" in e for e in r.report["cleanup_errors"])
    assert [(a._physics_sim_view, a._data._physics_sim_view) for a in g.assets] == before
    assert g.lifecycle.scene is None and not g.lifecycle.assets and not g.tracker


def test_contact_external_reference_blocks_success_but_app_closes(grouped):
    g, r = grouped, grouped.runtime
    external = g.world.views[0]
    g.lifecycle.close(r.App(), g.world)
    assert external is not None
    assert r.report["owned_contact_views_live_after_release"] == 1
    assert r.report["cleanup_succeeded"] is False
    assert r.report["app_close_returned"] is True


def test_callback_failure_drops_consumers_and_lifecycle_owners(grouped):
    g, r = grouped, grouped.runtime
    def fail(owner):
        owner.views.clear()
        raise RuntimeError("fixture release failed")
    g.lifecycle.callback = fail
    g.lifecycle.close(r.App(), g.world)
    assert r.report["cleanup_succeeded"] is False
    assert any("fixture release failed" in e for e in r.report["cleanup_errors"])
    assert g.collector.world is g.expert._world is None
    assert g.lifecycle.callback is g.lifecycle.collector is g.lifecycle.expert is None
    assert not g.lifecycle.assets and not g.tracker
    assert r.report["app_close_returned"] is True


def test_foreign_singleton_has_no_stop_clear_or_close_side_effect(grouped):
    g, r = grouped, grouped.runtime
    g.world.sim = r.Sim()
    with pytest.raises(RuntimeError, match="foreign grouped SimulationContext"):
        g.lifecycle.close(r.App(), g.world)
    assert r.events == []


def test_wrong_scene_structure_preserves_failure_and_closes_owned_app(grouped):
    g, r = grouped, grouped.runtime
    g.world.scene.cfg.num_envs = 3
    g.lifecycle.close(r.App(), g.world)
    assert r.report["cleanup_succeeded"] is False
    assert r.report["app_close_returned"] is True
    assert any("four-articulation" in e for e in r.report["cleanup_errors"])


def test_pin_tuple_cannot_drop_readonly_declaration():
    from scripts import train_shared_social_episode_reference as entry
    spec = migration.manifest()["modules"][str(entry.WORLD.relative_to(migration.ROOT))]
    pins = {k: (Path(v[0]), *v[1:]) for k, v in spec["pins"].items()}
    pins["factory"] = pins["factory"][:3]
    with pytest.raises(ValueError, match="Unreviewed grouped dependency"):
        migration.pins_for(entry.WORLD, pins)


def test_writer_parent_source_is_bound_without_sdk_import():
    described = migration.describe()
    row = migration.manifest()["dependencies"][
        "/project/EXP/IsaacLab/source/isaaclab/isaaclab/assets/articulation/base_articulation.py"]
    assert row["bytes"] > 0 and len(row["sha256"]) == 64
    assert described["runtime_executed"] is False
    assert described["old_lifecycle_certificate_reused"] is False


def test_changed_compile_bytes_rejected_and_owned_namespace_not_published(monkeypatch):
    from scripts import train_shared_social_episode_reference as entry
    identity = migration.record(entry.WORLD)
    original_read = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda p: b"raise AssertionError('must not execute')\n"
                        if p == entry.WORLD else original_read(p))
    name = "_grouped_changed_compile_fixture"
    with pytest.raises(ValueError, match="source changed before execution"):
        migration.load(entry.WORLD, name, identity)
    assert name not in sys.modules


@pytest.mark.parametrize("role", ("factory", "observations", "SDK_scene"))
def test_unknown_dependency_bytes_never_auto_refresh(role):
    from scripts import train_shared_social_episode_reference as entry
    spec = migration.manifest()["modules"][str(entry.WORLD.relative_to(migration.ROOT))]
    pins = {k: (Path(v[0]), *v[1:]) for k, v in spec["pins"].items()}
    changed = list(pins[role])
    changed[2] = "0" * 64
    pins[role] = tuple(changed)
    with pytest.raises(ValueError, match="Unreviewed grouped dependency"):
        migration.pins_for(entry.WORLD, pins)
