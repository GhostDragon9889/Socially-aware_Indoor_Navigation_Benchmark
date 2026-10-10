"""N1/N4 USD scene cloning without PhysX's retained replication callback cache.

Current pinned IsaacLab's InteractiveScene ignores its replicate_physics flag
and unconditionally queues PhysxReplicateContext. The project-owned subclass
keeps the official USD backend/ClonePlan and drops only its OWN PhysX entries
before the SDK session drains. It then removes the fully copied prototype.
No SDK method or file is replaced, and all actual bodies use ordinary PhysX.
"""
from isaaclab.scene import InteractiveScene

class USDOnlyInteractiveScene(InteractiveScene):
    def __init__(self,cfg):
        if cfg.num_envs not in (1,4) or cfg.replicate_physics is not False:
            raise ValueError("Explicit N1/N4 non-replicating scene required")
        from isaaclab.cloner.replicate_session import REPLICATION_QUEUE
        if REPLICATION_QUEUE:
            raise RuntimeError("Another replication session has pending owners")
        self.physics_replication_report={"mode":"official USD copy and direct PhysX parsing",
            "physics_replicator_used":False,"SDK_method_or_file_patched":False}
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
