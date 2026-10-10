"""Read configured gravity in the existing Office world; this module launches nothing.

An optional per-provider adapter adds this observation to the existing live
floor/source report. It preserves the original verifier and creates no SDK
context, view, physics step or USD edit. Configuration is not a native-engine
gravity/timestamp measurement or proof of human stability.
"""
from __future__ import annotations

import math


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _vector(value, name):
    try:
        values = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(name + " requires a finite three-vector") from error
    _require(len(values) == 3 and all(math.isfinite(item) for item in values),
             name + " requires a finite three-vector")
    return values


def observe_configured_gravity(simulation_context, *, _bindings=None):
    """Observe the existing singleton/stage/config without advancing its clock.

    ``_bindings`` is solely a dependency injection seam for CPU analytic tests.
    Production reads the installed SDK's public singleton/stage and USD APIs.
    """
    if _bindings is None:
        import isaaclab.sim as sim_utils
        from isaaclab.sim.utils import get_current_stage
        from pxr import UsdPhysics
        context_class = sim_utils.SimulationContext
    else:
        context_class, get_current_stage, UsdPhysics = _bindings
    _require(context_class.instance() is simulation_context,
             "Configured gravity must belong to the existing SimulationContext singleton")
    stage = get_current_stage()
    _require(stage is not None and simulation_context.stage is stage,
             "Configured gravity must use the active singleton stage")
    cfg = simulation_context.cfg
    getter = getattr(simulation_context, "get_physics_step_count", None)
    _require(callable(getter), "Existing submitted physics-step counter required")
    before = getter()
    _require(type(before) is int and before >= 0, "Nonnegative submitted-step counter required")
    configured = _vector(cfg.gravity, "SimulationCfg gravity")
    _require(all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-6)
                 for a, b in zip(configured, (0.0, 0.0, -9.81))),
             "Office probe requires configured Earth gravity [0,0,-9.81]")
    _require(math.isclose(float(cfg.dt), 1.0 / 120.0, rel_tol=0.0, abs_tol=1e-12),
             "Office probe requires configured 120Hz physics")
    scenes = [prim for prim in stage.Traverse()
              if prim.IsActive() and prim.IsA(UsdPhysics.Scene)]
    _require(len(scenes) == 1, "Exactly one active USD PhysicsScene required")
    prim = scenes[0]
    _require(str(prim.GetPath()) == cfg.physics_prim_path,
             "Active USD PhysicsScene must match SimulationCfg path")
    scene = UsdPhysics.Scene(prim)
    direction = _vector(scene.GetGravityDirectionAttr().Get(), "USD gravity direction")
    magnitude = float(scene.GetGravityMagnitudeAttr().Get())
    _require(math.isfinite(magnitude) and magnitude > 0.0
             and math.isclose(math.sqrt(sum(item * item for item in direction)), 1.0,
                              rel_tol=0.0, abs_tol=1e-6),
             "Finite positive USD gravity with unit direction required")
    authored = tuple(item * magnitude for item in direction)
    _require(all(math.isclose(a, b, rel_tol=0.0, abs_tol=1e-5)
                 for a, b in zip(authored, configured)),
             "Active USD gravity differs from the singleton SimulationCfg")
    after = getter()
    _require(type(after) is int and after == before,
             "Configured-gravity observation must not advance submitted physics steps")
    return {
        "schema": "local_Office.configured_gravity_observation.v1",
        "status": "observed_consistent_singleton_config_and_active_USD_gravity",
        "configured_gravity_m_s2": list(configured),
        "configured_physics_dt_s": float(cfg.dt),
        "active_physics_scene_path": str(prim.GetPath()),
        "USD_gravity_direction": list(direction),
        "USD_gravity_magnitude_m_s2": magnitude,
        "USD_gravity_vector_m_s2": list(authored),
        "observed_current_singleton_context_identity": True,
        "observed_active_stage_matches_context": True,
        "submitted_step_counter_before": before,
        "submitted_step_counter_after": after,
        "physics_steps_advanced_by_observer": 0,
        "new_App_or_Context_created_by_observer": False,
        "USD_properties_written_by_observer": False,
        "native_engine_gravity_measured": False,
        "native_engine_timestamp_observed": False,
        "human_stability_or_navigation_verified": False,
    }


def wrap_native_scene_factory(factory, *, observer=observe_configured_gravity):
    """Attach one Office-only read callback to each future provider instance.

    The existing factory owns all provider creation/admission. The sole AMPWorld
    still calls the original live floor/source verifier before its first reset.
    Only the returned observation dictionary gains a new gravity record.
    """
    _require(callable(factory) and callable(observer), "Existing factory and observer required")

    def prepare(description, *, device):
        scene = factory(description, device=device)
        provider = scene.provider
        _require(provider.scene_name == "Office", "This observation adapter is registered only for local Office")
        _require(not getattr(provider, "_local_Office_gravity_observer_attached", False),
                 "Office provider already has a configured-gravity observer")
        original = provider.verify_live_runtime

        def verify(*, origins_xy, simulation_context, free_cell_limit=0):
            report = original(origins_xy=origins_xy, simulation_context=simulation_context,
                              free_cell_limit=free_cell_limit)
            gravity = observer(simulation_context)
            _require(report.get("scene") == "Office"
                     and report.get("submitted_step_counter_after") == gravity["submitted_step_counter_before"],
                     "Office floor and configured-gravity observations must share the same submitted clock")
            return {**report, "local_configured_gravity_observation": gravity}

        provider.verify_live_runtime = verify
        provider._local_Office_gravity_observer_attached = True
        return scene

    return prepare
