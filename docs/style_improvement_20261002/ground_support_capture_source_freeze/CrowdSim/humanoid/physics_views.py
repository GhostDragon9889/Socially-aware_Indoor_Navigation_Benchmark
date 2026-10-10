"""Read the initialized Lab physics view without importing or starting Kit.

Isaac Lab 3 registers its own physics manager. The Isaac Sim legacy manager
can be uninitialized while the Lab articulation is live. Contact sensors must
belong to the same native simulation as that articulation.
"""

from __future__ import annotations


def resolve_native_contact_factory(simulation_context):
    """Return the existing registered contact factory and its ownership record.

    This performs read-only lookups. It does not create a SimulationView,
    advance physics, or search private/legacy attributes for a fallback.
    A backend without native rigid-contact support is rejected explicitly.
    """
    manager = getattr(simulation_context, "physics_manager", None)
    getter = getattr(manager, "get_physics_sim_view", None)
    if not callable(getter):
        raise RuntimeError("Active Lab physics manager has no public simulation-view getter")
    view = getattr(simulation_context, "physics_sim_view", None)
    if view is None:
        raise RuntimeError("Lab physics_sim_view is unavailable after world initialization")
    if getter() is not view:
        raise RuntimeError("Lab simulation view differs from its active physics manager view")
    if not callable(getattr(view, "create_rigid_contact_view", None)):
        raise RuntimeError("Active Lab physics view does not support native rigid contacts")
    manager_type = manager if isinstance(manager, type) else type(manager)
    return view, {
        "source": "world.simulator._sim.physics_sim_view",
        "owner": f"{manager_type.__module__}.{manager_type.__qualname__}",
        "active_manager_getter_same_view": True,
        "legacy_isaacsim_SimulationManager_used": False,
        "new_simulation_view_created": False,
        "private_view_lookup_used": False,
    }
