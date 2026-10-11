"""Four-articulation ownership for the common, complete Sim 6.1 shutdown."""
from __future__ import annotations

import gc
import weakref

from CrowdSim.humanoid.runtime.sdk61_lifecycle import SDK61Lifecycle

OVERLAY_KIT_ARGS = ""  # No Sim 6.0 native overlay enters this runtime.
_lifecycle = None


def _verify_overlay_inputs(report):
    global _lifecycle
    _lifecycle = GroupedSDK61Lifecycle(report)
    report["grouped_lifecycle"] = {"schema": "shared_social.grouped_cleanup.sim61.v2",
                                   "historical_overlay_used": False}


def _verify_loaded_overlay(report):
    if _lifecycle is None or _lifecycle.report is not report:
        raise RuntimeError("Grouped lifecycle belongs to another invocation")
    _lifecycle.verify_loaded()


class GroupedSDK61Lifecycle(SDK61Lifecycle):
    def configure(self, callback, receipts, tracker, expected, collector, expert):
        self.callback, self.receipts, self.tracker = callback, receipts, tracker
        self.expected, self.collector, self.expert = expected, collector, expert
        self.scene, self.assets = None, ()

    def _owned_context(self, world):
        from isaaclab.sim import SimulationContext
        sim = SimulationContext.instance() if world is None else world.sim
        self.scene = None if world is None else world.scene
        self.assets = tuple(self.tracker) if world is None else world.assets
        if sim is not SimulationContext.instance():
            raise RuntimeError("Refuse a foreign grouped SimulationContext")
        return sim

    def _validate_owners(self, world):
        if self.scene is not None:
            if (set(self.scene.articulations) != {f"robot_{i}" for i in range(4)}
                    or len(self.assets) != 4 or len({id(a) for a in self.assets}) != 4
                    or any(self.scene.articulations[f"robot_{i}"] is not a for i, a in enumerate(self.assets))
                    or self.scene.cfg.num_envs * 4 != self.expected or self.scene.sensors):
                raise RuntimeError("Expected the owned four-articulation grouped scene")
        for owner, field in ((self.collector, "world"), (self.expert, "_world")):
            if owner is not None and getattr(owner, field) is not world:
                raise RuntimeError("Grouped training consumer belongs to another World")

    def _release_contacts(self, world):
        import torch
        if torch.cuda.is_initialized():
            torch.cuda.synchronize(world.device)
        try:
            self._validate_owners(world)
            world.release_owned_native(self.callback)
        finally:
            for owner, field in ((self.collector, "world"), (self.expert, "_world")):
                if owner is not None and getattr(owner, field) is world:
                    setattr(owner, field, None)
            self.collector = self.expert = self.callback = None
        self._verify_receipts()
        self.report.update(contacts_released_before_timeline_stop=True, owned_world_native_release_returned=True)

    def _verify_receipts(self):
        gc.collect()
        if (type(self.expected) is not int or self.expected < 4 or type(self.receipts) is not list
                or len(self.receipts) != 1):
            raise RuntimeError("Require one full grouped contact release receipt")
        receipt = self.receipts[0]
        refs = receipt.get("native_weakrefs")
        if (set(receipt) != {"native_weakrefs", "native_view_count", "module_owned_aliases_released"}
                or type(receipt["native_view_count"]) is not int
                or receipt["native_view_count"] != self.expected
                or receipt["module_owned_aliases_released"] is not True
                or type(refs) is not tuple or len(refs) != self.expected
                or any(type(ref) is not weakref.ReferenceType for ref in refs)):
            raise RuntimeError("Grouped contact receipt does not cover all actors")
        live = sum(ref() is not None for ref in refs)
        self.report.update(owned_contact_view_count=len(refs), owned_contact_views_live_after_release=live)
        if live:
            raise RuntimeError("Grouped native contacts remain live before stop")

    def _detach_world(self, world, producer):
        try:
            from isaaclab_physx.physics import PhysxManager
            backend = PhysxManager.get_scene_data_backend()
            view = None if backend is None else backend.simulation_view
            owners = []
            if len(self.assets) != 4:
                raise RuntimeError("Incomplete four-articulation ownership after construction")
            for asset in self.assets:
                if getattr(asset, "_is_initialized", False) or asset.__dict__.get("_root_view") is not None:
                    raise RuntimeError("Grouped articulation survived SDK stop")
                data = asset.__dict__.get("_data")
                if data is None:
                    raise RuntimeError("Grouped articulation data ownership is missing")
                owners.extend((asset, data))
            # Preflight every alias before mutating any of the eight owners.
            if any(owner.__dict__.get("_physics_sim_view") is not None
                   and owner.__dict__["_physics_sim_view"] is not view for owner in owners):
                raise RuntimeError("Grouped asset/data borrows a foreign backend view")
            counts = {"articulation_fields": 0, "data_fields": 0}
            for index, owner in enumerate(owners):
                if "_physics_sim_view" in owner.__dict__:
                    owner._physics_sim_view = None
                    counts["data_fields" if index % 2 else "articulation_fields"] += 1
            if counts != {"articulation_fields": 4, "data_fields": 4}:
                raise RuntimeError("Grouped teardown lacks all eight borrowed view fields")
            self.report.update(owned_asset_simulation_views_release_returned=True,
                               owned_asset_simulation_view_release_counts=counts)
        finally:
            if self.scene is not None:
                for field in ("articulations", "rigid_objects", "rigid_object_collections", "deformable_objects"):
                    getattr(self.scene, field, {}).clear()
            self.tracker.clear()
            self.scene, self.assets = None, ()
            self.collector = self.expert = self.callback = None
        self.report["evaluation_world_detached"] = True

    def close(self, app, world, producer=None, **unused):
        if world is None:
            try:
                self._verify_receipts()
            except Exception as error:
                self.report["cleanup_errors"].append("partial contact release: " + str(error))
        return super().close(app, world, producer, **unused)


def cleanup_owned_runtime(app, world, sim_type, creation_attempted, release_callback, receipts, tracked_assets,
                          expected_actors, terminal, report, *, collector=None, expert=None):
    lifecycle = terminal._lifecycle
    if lifecycle is None or lifecycle.report is not report:
        raise RuntimeError("Grouped cleanup lacks its owned lifecycle")
    lifecycle.configure(release_callback, receipts, tracked_assets, expected_actors, collector, expert)
    if not report.get("training_loop_completed", False):
        report["status"] = "failed_preserved"
    lifecycle.close(app, world)
    if report.get("public_backend_view_live_before_App_close") is False:
        report["scene_backend_views_live_after_post_context_gc"] = 0
