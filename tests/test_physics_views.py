"""Reject foreign/uninitialized managers without creating or stepping physics."""

from types import SimpleNamespace

import pytest

from CrowdSim.humanoid.physics_views import resolve_native_contact_factory


class ContactFactory:
    def create_rigid_contact_view(self, *args, **kwargs):
        raise AssertionError("Lookup must not create a contact view")


def context(view, *, owner_view=None):
    return SimpleNamespace(
        physics_sim_view=view,
        physics_manager=SimpleNamespace(get_physics_sim_view=lambda: view if owner_view is None else owner_view),
        step=lambda: pytest.fail("Lookup must not advance physics"),
    )


def test_current_registered_view_is_returned_without_creating_or_stepping():
    view = ContactFactory()
    actual, record = resolve_native_contact_factory(context(view))
    assert actual is view
    assert record["active_manager_getter_same_view"]
    assert not record["legacy_isaacsim_SimulationManager_used"]
    assert not record["new_simulation_view_created"]
    assert not record["private_view_lookup_used"]


def test_registered_class_manager_and_property_context_are_supported():
    view = ContactFactory()

    class Manager:
        @classmethod
        def get_physics_sim_view(cls):
            return view

    class SimulationContext:
        physics_manager = Manager

        @property
        def physics_sim_view(self):
            return self.physics_manager.get_physics_sim_view()

    actual, record = resolve_native_contact_factory(SimulationContext())
    assert actual is view
    assert record["owner"].endswith("Manager")


@pytest.mark.parametrize("manager", [None, SimpleNamespace(), SimpleNamespace(get_physics_sim_view=None)])
def test_missing_active_manager_does_not_use_private_or_legacy_view(manager):
    fake = SimpleNamespace(physics_manager=manager, physics_sim_view=ContactFactory(),
                           _physics_sim_view=ContactFactory())
    with pytest.raises(RuntimeError, match="public simulation-view getter"):
        resolve_native_contact_factory(fake)


def test_uninitialized_view_fails_even_with_a_valid_private_view():
    fake = context(None)
    fake._physics_sim_view = ContactFactory()
    with pytest.raises(RuntimeError, match="unavailable"):
        resolve_native_contact_factory(fake)


def test_foreign_view_cannot_sense_the_current_world():
    with pytest.raises(RuntimeError, match="differs"):
        resolve_native_contact_factory(context(ContactFactory(), owner_view=ContactFactory()))


@pytest.mark.parametrize("view", [object(), SimpleNamespace(create_rigid_contact_view=None)])
def test_backend_without_native_contact_factory_is_explicitly_unsupported(view):
    with pytest.raises(RuntimeError, match="does not support"):
        resolve_native_contact_factory(context(view))
