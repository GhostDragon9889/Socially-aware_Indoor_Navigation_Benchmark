"""Independent nonfloor/nonpeer normal force residual at every actual120Hz tick.

GPU filtering of imported static meshes is unsupported. Subtract supported
floor and peer vectors from unfiltered body net forces. Self contact is separately measured and subtracted. Opposing static forces
can cancel; geometry clearance remains an independent gate.
Raw arrays are archived for independent recomputation; no physics is advanced.
"""
import math

def read_static_report(provider, *, simulator, monitor, dt, physics_step):
    import numpy as np
    import torch
    import warp as wp
    if not math.isclose(dt,1/120,abs_tol=1e-12,rel_tol=0):raise ValueError('Actual120Hz dt required')
    stats=getattr(provider,'_native_contact_stats',None)
    if stats is None:
        stats={'source':'PhysX unfiltered net minus exact floor, peer and own-articulation normal force vectors',
            'body_identity_verified_each_tick':True,'reads':0,'floor_positive_substeps':0,'maximum_static_force_n':0.,'maximum_floor_force_n':0.,
            'first_physics_step':physics_step,'last_physics_step':physics_step-1,'floor_witnesses':[],
            'filtered_GPU_static_forces_are_insufficient':True,'self_contact_separately_subtracted':True,
            'force_cancellation_is_possible':True,'body_names':list(monitor.world.mapping.body_names)}
        provider._native_contact_stats=stats;provider._native_residual_rows=[]
    if physics_step!=stats['last_physics_step']+1:raise RuntimeError('Skipped/repeated physical contact tick')
    actors=monitor.world.num_envs;names=stats['body_names'];net=[];floor=[];peer=[]
    for actor,view,static_count,peers,body_count,_,sensor_count in monitor.peer.views:
        if actor!=len(peer) or body_count!=24 or sensor_count!=24:raise ValueError('One exact24-body native contact view per actor required')
        sensor_paths=list(view.sensor_paths)
        expected_paths=monitor.peer.manifest["actor_body_paths"][actor]
        if len(sensor_paths)!=24 or set(sensor_paths)!=set(expected_paths):raise ValueError("Native contact body identities differ")
        order=[sensor_paths.index(path) for path in expected_paths]
        matrix=wp.to_torch(view.get_contact_force_matrix(dt=dt)).detach()
        peer.append(matrix[order,static_count:].sum(1).clone())
    matrices=[wp.to_torch(simulator._contact_sensor_map[name].data.force_matrix_w).detach().clone()[:,0] for name in names]
    if any(tuple(v.shape)!=(actors,25,3) for v in matrices):raise ValueError("Exact floor plus24 own articulation force filters required")
    floor=torch.stack([v[:,0] for v in matrices],1)
    own=torch.stack([v[:,1:].sum(1) for v in matrices],1)
    net=torch.stack([wp.to_torch(simulator._contact_sensor_map[name].data.net_forces_w).detach().clone()[:,0] for name in names],1)
    peer=torch.stack(peer)
    if any(tuple(x.shape)!=(actors,24,3) or not bool(torch.isfinite(x).all()) for x in (net,floor,peer,own)):
        raise ValueError('All24 body normal force vectors must be finite and aligned')
    residual=net-floor-peer-own;magnitude=residual.norm(dim=-1).amax(1);floor_total=float(floor.norm(dim=-1).sum())
    provider._native_residual_rows.append((physics_step,net.cpu().numpy(),floor.cpu().numpy(),peer.cpu().numpy(),own.cpu().numpy()))
    if floor_total>1 and len(stats['floor_witnesses'])<3:stats['floor_witnesses'].append({'physics_step':physics_step,'force_n':floor_total})
    stats.update(reads=stats['reads']+1,last_physics_step=physics_step,
        floor_positive_substeps=stats['floor_positive_substeps']+int(floor_total>1),
        maximum_floor_force_n=max(stats['maximum_floor_force_n'],floor_total),
        maximum_static_force_n=max(stats['maximum_static_force_n'],float(magnitude.max())))
    return magnitude


def write_static_report_archive(provider, output):
    """Archive every actually read substep, including a failed run's prefix."""
    import hashlib
    from pathlib import Path
    import numpy as np
    rows=getattr(provider,'_native_residual_rows',[])
    stats=getattr(provider,'_native_contact_stats',None)
    if not rows:
        return stats
    if stats is None or stats['reads']!=len(rows) or len(stats['body_names'])!=24:
        raise ValueError('Raw contact rows must match verified actual body/tick reads')
    steps=np.array([r[0] for r in rows],dtype=np.int64)
    if not np.array_equal(steps,np.arange(stats['first_physics_step'],stats['last_physics_step']+1)):
        raise ValueError('Missing or repeated native contact substep')
    arrays={k:np.stack([r[i] for r in rows]) for i,k in enumerate(('net','floor','peer','own'),1)}
    shape=arrays['net'].shape
    if len(shape)!=4 or shape[0]!=len(rows) or shape[-2:]!=(24,3) or any(a.shape!=shape or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Exact finite all-body native normal forces required')
    path=Path(output)
    if path.exists():raise ValueError('Raw contact archive must be fresh')
    np.savez_compressed(path,physics_step=steps,body_names=np.array(stats['body_names']),**arrays)
    return {**stats,'raw_artifact':{'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},'all_read_substeps_archived':True}
