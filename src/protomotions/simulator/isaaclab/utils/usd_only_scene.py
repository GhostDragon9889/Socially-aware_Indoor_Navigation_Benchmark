"""N1/N4 USD scene cloning without PhysX's retained replication callback cache.

Sim 6.1 / current Lab 3 EA implements replicate_physics=False officially and
uses a cfg-only replication queue. That path delegates directly to the SDK.
The older Sim 6.0 checkout queues (cfg, backend) pairs unconditionally; only
that revision needs the project-owned owner filter before its session drains.
No SDK method or file is replaced, and actual bodies use ordinary PhysX.
"""
from functools import partial

from isaaclab.scene import InteractiveScene
from protomotions.simulator.isaaclab.utils.runtime_compat import (
    supports_native_usd_only_replication,
)

def call_with_preserved_edit_target(function, *args, **kwargs):
    """Scope a scene-provider spawn that authors session-layer overrides.

    The SDK USD cloner copies root-layer specs. A global provider may author
    only its own scene in the session layer, but must not leave that layer as
    the edit target for the following robot spawn. The context restores the
    caller's full EditTarget on success and exception; assets stay in their
    original layers and the SDK performs ordinary USD replication.
    """
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Usd

    stage = get_current_stage()
    with Usd.EditContext(stage, stage.GetEditTarget()):
        return function(*args, **kwargs)


class USDOnlyInteractiveScene(InteractiveScene):
    def __init__(self,cfg):
        if cfg.num_envs not in (1,4) or cfg.replicate_physics is not False:
            raise ValueError("Explicit N1/N4 non-replicating scene required")
        from isaaclab.cloner.replicate_session import REPLICATION_QUEUE, replicate
        self._official_usd_only = supports_native_usd_only_replication(replicate)
        if REPLICATION_QUEUE:
            raise RuntimeError("Another replication session has pending owners")
        self.physics_replication_report={"mode":"official USD copy and direct PhysX parsing",
            "physics_replicator_used":False,"SDK_method_or_file_patched":False,
            "official_replicate_physics_flag": self._official_usd_only}
        super().__init__(cfg)
        template=self.stage.GetPrimAtPath('/World/template')
        prototype_present=template.IsValid()
        for root in self.env_prim_paths:
            robot=self.stage.GetPrimAtPath(root+'/Robot')
            if not robot.IsValid():raise RuntimeError("Complete robot USD copy required")
            from pxr import Usd
            for prim in Usd.PrimRange(robot):
                for relation in prim.GetRelationships():
                    if any(str(v).startswith('/World/template') for v in relation.GetTargets()):
                        raise RuntimeError("Copied robot still targets the prototype")
                for spec in prim.GetPrimStack():
                    for key in ('references','inheritPaths','specializes'):
                        if not spec.HasInfo(key):continue
                        for item in spec.GetInfo(key).GetAppliedItems():
                            path=str(getattr(item,'primPath',item));asset=getattr(item,'assetPath','')
                            if not asset and path.startswith('/World/template'):
                                raise RuntimeError("Copied robot still composes its prototype")
        if prototype_present and not self.stage.RemovePrim('/World/template'):
            raise RuntimeError("Prototype must be removed before ordinary PhysX parsing")
        self.physics_replication_report.update(prototype_present_before_cleanup=prototype_present,prototype_absent_before_PhysX=True,complete_robot_copies=cfg.num_envs,
            robot_roots=[root+'/Robot' for root in self.env_prim_paths],
            physics_initialized_after_copy=True)

    def _add_entities_from_cfg(self):
        if self._official_usd_only:
            # The current SDK owns cfg-only queue dispatch and honors the flag.
            # Do not unpack or mutate its queue using the legacy tuple protocol.
            provider = getattr(self.cfg, "dynbench_source", None)
            original_spawn = None if provider is None else provider.spawn.func
            try:
                if original_spawn is not None:
                    provider.spawn.func = partial(call_with_preserved_edit_target, original_spawn)
                super()._add_entities_from_cfg()
            finally:
                if original_spawn is not None:
                    provider.spawn.func = original_spawn
            self.physics_replication_report.update(
                owned_Physx_queue_entries_removed=0,
                provider_edit_target_scoped=original_spawn is not None,
            )
            return
        from isaaclab.cloner.replicate_session import REPLICATION_QUEUE
        from isaaclab_physx.cloner.replicate import PhysxReplicateContext
        owned={id(cfg) for cfg in self._collect_asset_cfgs()}
        super()._add_entities_from_cfg()
        removed=[(cfg,backend) for cfg,backend in REPLICATION_QUEUE
                 if id(cfg) in owned and backend is PhysxReplicateContext]
        if len(removed)!=1:raise RuntimeError("Require exactly one vectorized robot physics queue owner")
        kept=[entry for entry in REPLICATION_QUEUE if not(id(entry[0]) in owned and entry[1] is PhysxReplicateContext)]
        if not kept:raise RuntimeError("Official USD replication must remain queued")
        REPLICATION_QUEUE[:]=kept
        self.physics_replication_report.update(owned_Physx_queue_entries_removed=len(removed),
                                               preserved_backend_entries=len(kept))
