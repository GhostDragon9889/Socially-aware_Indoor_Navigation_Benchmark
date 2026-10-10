"""Pairwise PhysX contact measurements for the shared pedestrian fixtures.

Import after AppLauncher. Every sensing body is filtered against exact obstacle
and other-actor paths. Floor and self contacts are excluded. Measurements are
normal contact forces [N], not pelvis-distance or occupancy-grid proxies.
"""
from __future__ import annotations

import math

import torch


class PedestrianContactMonitor:
    """Preserve transient contacts across all physics substeps of a control.

    ``peer_max_force_n[i,j]`` is the largest body-pair normal-force magnitude,
    not the sum of forces over the actor. A contact is strictly above the
    declared threshold. This local definition is not an author metric protocol.
    """

    def __init__(self, world, *, threshold_n=1., max_contacts_per_view=512):
        if not math.isfinite(threshold_n) or threshold_n <= 0:
            raise ValueError("Contact force threshold must be positive and finite")
        if type(max_contacts_per_view) is not int or max_contacts_per_view < 1:
            raise ValueError("Contact buffer capacity must be a positive integer")
        if not world.shared_world:
            raise ValueError("Pairwise fixture contact reporting requires shared physics")
        from isaacsim.core.simulation_manager import SimulationManager
        from protomotions.simulator.isaaclab.utils.usd_body_paths import resolve_body_prim_paths

        self.num_envs = world.num_envs
        self.device = world.device
        self.threshold_n = float(threshold_n)
        body_names = list(world.mapping.body_names)
        relative = resolve_body_prim_paths(world.simulator._robot.cfg.spawn.usd_path, body_names)
        bodies = [[f"/World/envs/env_{actor}/Robot/{relative[name]}" for name in body_names]
                  for actor in range(self.num_envs)]
        cuboids = (world.validation_scene.description if world.validation_scene is not None else {}).get("cuboids", [])
        static = [f"/World/PaperValidation/Obstacle_{i}" for i in range(len(cuboids))]
        self.views = []
        self.manifest = {"source": "PhysX exact-path rigid_contact_view.get_contact_force_matrix",
            "units": "N", "components": "normal forces only", "threshold_n": self.threshold_n,
            "substep_reduction": "maximum body-pair magnitude over every physics substep; strict > threshold",
            "excluded": ["floor", "self articulation"], "author_metric_equivalence": False,
            "actor_body_paths": bodies, "static_collider_paths": static,
            "max_contacts_per_view": max_contacts_per_view}
        physics_view = SimulationManager.get_physics_sim_view()
        for actor, sources in enumerate(bodies):
            peers = [j for j in range(self.num_envs) if j != actor]
            filters = static + [path for peer in peers for path in bodies[peer]]
            partner_ids = [-1]*len(static) + [peer for peer in peers for _ in bodies[peer]]
            if not filters:
                continue
            for source in sources:
                view = physics_view.create_rigid_contact_view([source], filter_patterns=[filters],
                    max_contact_data_count=max_contacts_per_view)
                if view.sensor_count != 1 or view.filter_count != len(filters):
                    raise RuntimeError(f"Contact path resolution changed: {source}: {view.sensor_count=}, {view.filter_count=}, expected {len(filters)} filters")
                self.views.append((actor, view, len(static), peers, len(body_names), partner_ids))
        self.static_max_force_n = torch.zeros(self.num_envs, device=self.device)
        self.peer_max_force_n = torch.zeros(self.num_envs, self.num_envs, device=self.device)
        self.substep_count = 0

    def begin_control_step(self):
        self.static_max_force_n.zero_()
        self.peer_max_force_n.zero_()
        self.substep_count = 0

    def sample(self, dt):
        """Read the just-completed physics step with its real duration [s]."""
        import warp as wp
        for actor, view, static_count, peers, body_count, _ in self.views:
            force = wp.to_torch(view.get_contact_force_matrix(dt=dt)).reshape(-1, 3)
            if not torch.isfinite(force).all():
                raise FloatingPointError("Nonfinite pairwise physical contact force")
            magnitude = torch.linalg.vector_norm(force, dim=-1)
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
