"""Pairwise PhysX contact measurements for the shared pedestrian fixtures.

Import after AppLauncher. Every sensing body is filtered against exact obstacle
and other-actor paths. Floor and self contacts are excluded. Measurements are
normal contact forces [N], not pelvis-distance or occupancy-grid proxies.
"""
from __future__ import annotations

import math

import torch


def resolve_static_collision_paths(stage, obstacle_paths):
    """Resolve the actual CollisionAPI descendants, including nested meshes.

    A spawn root can be an Xform with no physical shape. Missing colliders must
    fail explicitly rather than producing an apparently collision-free run.
    """
    from pxr import Usd, UsdPhysics

    result = []
    for path in obstacle_paths:
        root = stage.GetPrimAtPath(path)
        if not root:
            raise ValueError(f"Static contact obstacle does not exist: {path}")
        colliders = [str(prim.GetPath()) for prim in Usd.PrimRange(root)
                     if prim.HasAPI(UsdPhysics.CollisionAPI)
                     and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False]
        if not colliders:
            raise ValueError(f"Static contact obstacle has no enabled CollisionAPI shape: {path}")
        result.extend(colliders)
    return result


def contact_filter_layout(actor_body_paths, static_paths):
    """Plan exact partner columns without importing SDK or guessing body order.

    The fixture's ground namespace is explicitly excluded. All actors must use
    the same named rig, and no rigid-body path can belong to two actors.
    """
    bodies = [tuple(paths) for paths in actor_body_paths]
    static = tuple(static_paths)
    if not bodies or not bodies[0] or any(len(paths) != len(bodies[0]) for paths in bodies):
        raise ValueError("Contact reporting requires an equal nonzero body count per actor")
    flat = [path for paths in bodies for path in paths]
    for path in flat + list(static):
        if not isinstance(path, str) or not path.startswith("/") or any(value in path for value in ("*", "[", "]", "?")):
            raise ValueError("Contact reporting requires exact absolute prim paths")
        if path == "/World/ground" or path.startswith("/World/ground/"):
            raise ValueError("Floor support must be excluded from map/peer collision filters")
    if len(set(flat)) != len(flat) or len(set(static)) != len(static) or set(flat) & set(static):
        raise ValueError("Contact actor/static paths must be unique and disjoint")
    result = []
    for actor, sources in enumerate(bodies):
        peers = tuple(peer for peer in range(len(bodies)) if peer != actor)
        filters = static + tuple(path for peer in peers for path in bodies[peer])
        partner_ids = (-1,) * len(static) + tuple(peer for peer in peers for _ in bodies[peer])
        result.append({"actor": actor, "sources": sources, "filters": filters,
                       "static_count": len(static), "peers": peers,
                       "body_count": len(sources), "partner_ids": partner_ids})
    return result


class PedestrianContactMonitor:
    """Preserve transient contacts across all physics substeps of a control.

    ``peer_max_force_n[i,j]`` is the largest body-pair normal-force magnitude,
    not the sum of forces over the actor. A contact is strictly above the
    declared threshold. This local definition is not an author metric protocol.
    """

    def __init__(self, world, *, threshold_n=1., max_contacts_per_view=512,
                 batched_sensor_bodies=True):
        if not math.isfinite(threshold_n) or threshold_n <= 0:
            raise ValueError("Contact force threshold must be positive and finite")
        if type(max_contacts_per_view) is not int or max_contacts_per_view < 1:
            raise ValueError("Contact buffer capacity must be a positive integer")
        if type(batched_sensor_bodies) is not bool:
            raise ValueError("Contact body batching must be an explicit boolean")
        if not world.shared_world:
            raise ValueError("Pairwise fixture contact reporting requires shared physics")
        from isaaclab.sim.utils import get_current_stage
        from protomotions.simulator.isaaclab.utils.usd_body_paths import resolve_body_prim_paths
        from CrowdSim.humanoid.physics_views import resolve_native_contact_factory

        self.num_envs = world.num_envs
        self.device = world.device
        self.threshold_n = float(threshold_n)
        body_names = list(world.mapping.body_names)
        relative = resolve_body_prim_paths(world.simulator._robot.cfg.spawn.usd_path, body_names)
        bodies = [[f"/World/envs/env_{actor}/Robot/{relative[name]}" for name in body_names]
                  for actor in range(self.num_envs)]
        provider = getattr(world, "scene_provider", None)
        if provider is None:
            cuboids = (world.validation_scene.description if world.validation_scene is not None else {}).get("cuboids", [])
            static_roots = [f"/World/PaperValidation/Obstacle_{i}" for i in range(len(cuboids))]
        else:
            static_roots = list(provider.static_collision_paths)
        static = resolve_static_collision_paths(get_current_stage(), static_roots)
        if provider is not None and set(static) != set(static_roots):
            raise RuntimeError("Native scene static collision paths differ after USD composition/cooking")
        self.views = []
        self.manifest = {"source": "PhysX exact-path rigid_contact_view.get_contact_force_matrix",
            "units": "N", "components": "normal forces only", "threshold_n": self.threshold_n,
            "substep_reduction": "maximum body-pair magnitude over every physics substep; strict > threshold",
            "excluded": ["floor", "self articulation"], "author_metric_equivalence": False,
            "actor_body_paths": bodies, "static_collider_paths": static,
            "static_obstacle_root_paths": static_roots,
            "physics_substeps_per_control": int(world.simulator.decimation),
            "physics_dt_s": float(world.simulator._sim.get_physics_dt()),
            "control_dt_s": float(world.dt),
            "sensor_view_grouping": "exact bodies per actor" if batched_sensor_bodies else "one exact body per view",
            "max_contacts_per_view": max_contacts_per_view}
        physics_view, ownership = resolve_native_contact_factory(world.simulator._sim)
        self.manifest["physics_view_ownership"] = ownership
        for layout in contact_filter_layout(bodies, static):
            actor, sources = layout["actor"], layout["sources"]
            peers, filters = list(layout["peers"]), list(layout["filters"])
            partner_ids = list(layout["partner_ids"])
            if not filters:
                continue
            source_groups = [list(sources)] if batched_sensor_bodies else [[source] for source in sources]
            for source_group in source_groups:
                view = physics_view.create_rigid_contact_view(source_group, filter_patterns=[filters] * len(source_group),
                    max_contact_data_count=max_contacts_per_view)
                if view.sensor_count != len(source_group) or view.filter_count != len(filters):
                    raise RuntimeError(f"Contact path resolution changed: {source_group}: {view.sensor_count=}, {view.filter_count=}, expected {len(source_group)} sensors and {len(filters)} filters")
                self.views.append((actor, view, len(static), peers, len(body_names), partner_ids, len(source_group)))
        self.manifest["sensor_view_count"] = len(self.views)
        self.manifest["force_matrix_shapes"] = [
            [sensors, view.filter_count, 3] for _, view, _, _, _, _, sensors in self.views
        ]
        self.static_max_force_n = torch.zeros(self.num_envs, device=self.device)
        self.peer_max_force_n = torch.zeros(self.num_envs, self.num_envs, device=self.device)
        self.substep_count = 0

    def begin_control_step(self):
        self.static_max_force_n.zero_()
        self.peer_max_force_n.zero_()
        self.substep_count = 0

    def sample(self, dt):
        """Read the just-completed physics step with its real duration [s]."""
        if type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Contact measurement duration must be positive and finite")
        import warp as wp
        for actor, view, static_count, peers, body_count, _, sensor_count in self.views:
            force = wp.to_torch(view.get_contact_force_matrix(dt=dt))
            expected_filters = static_count + len(peers) * body_count
            expected_shape = (sensor_count, expected_filters, 3)
            if tuple(force.shape) != expected_shape:
                raise RuntimeError(f"Contact force shape changed: {tuple(force.shape)}, expected {expected_shape}")
            if not torch.isfinite(force).all():
                raise FloatingPointError("Nonfinite pairwise physical contact force")
            magnitude = torch.linalg.vector_norm(force, dim=-1).amax(0)
            if static_count:
                self.static_max_force_n[actor] = torch.maximum(self.static_max_force_n[actor], magnitude[:static_count].max())
            if peers:
                peer_force = magnitude[static_count:].reshape(len(peers), body_count).amax(-1)
                self.peer_max_force_n[actor, peers] = torch.maximum(self.peer_max_force_n[actor, peers], peer_force)
        self.substep_count += 1

    def measurements(self):
        if not self.substep_count:
            raise RuntimeError("Contact reporting requires a completed physics substep")
        return {"static_contact": self.static_max_force_n > self.threshold_n,
                "peer_contact": self.peer_max_force_n > self.threshold_n,
                "static_contact_max_force_n": self.static_max_force_n.clone(),
                "peer_contact_max_force_n": self.peer_max_force_n.clone()}
