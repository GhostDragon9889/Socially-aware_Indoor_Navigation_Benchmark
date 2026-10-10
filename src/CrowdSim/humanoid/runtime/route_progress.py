"""Actual spatial progress from every captured frame, including closed routes."""
import numpy as np

def physical_progress(positions,start,*,cyclic_route):
    positions=np.asarray(positions);start=np.asarray(start)
    if (positions.ndim!=3 or positions.shape[-1]!=2 or start.shape!=positions.shape[1:]
            or not len(positions) or not np.isfinite(positions).all() or not np.isfinite(start).all()
            or type(cyclic_route) is not bool):
        raise ValueError('Finite all-frame XY and exact spawn XY required')
    displacement=np.linalg.norm(positions[-1]-start,axis=-1)
    excursion=np.linalg.norm(positions-start[None],axis=-1).max(0)
    distance=np.linalg.norm(np.diff(np.concatenate((start[None],positions),0),axis=0),axis=-1).sum(0)
    progress=excursion if cyclic_route else displacement
    return {'displacement_m':displacement.tolist(),'maximum_excursion_from_spawn_m':excursion.tolist(),
            'actual_path_length_m':distance.tolist(),'progress_distance_m':progress.tolist(),
            'progress_distance_metric':'maximum_excursion_from_spawn' if cyclic_route else 'final_displacement_from_spawn',
            'spatial_progress_passed':bool((progress>.3).all())}
