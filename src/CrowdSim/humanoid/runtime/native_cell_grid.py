"""Re-admit explicit full-grid PhysX evidence without an omap dependency.

This protocol is selected by name, never as a fallback for an omap failure.
Each 10cm cell must have recorded native full-height overlap hits and each
exported free cell must also have declared, measured ground support.
"""
import json
from social_navigation.resource_bindings import read_json as read_resource_json, resolve_path
from pathlib import Path
import numpy as np


def verify_cell_arrays(arrays, *, bounds, dimensions, ground, floor_paths, static_paths):
    """Validate full-grid centers, CSR identities and labels from actual hits."""
    def require(condition,message):
        if not bool(condition):raise ValueError('Native cell-grid '+message)
    dims=np.asarray(dimensions,dtype=np.int64)
    require(dims.shape==(3,) and (dims>0).all() and dims[2]==1,'needs finite 2D dimensions')
    low,high=np.asarray(bounds,dtype=np.float64)
    require(low.shape==(3,) and high.shape==(3,) and np.isfinite([low,high]).all()
            and np.allclose(high[:2]-low[:2],dims[:2]*.1,atol=2e-4,rtol=0)
            and np.allclose([low[2],high[2]],[ground+.15,ground+1.8],atol=1e-8,rtol=0),'bounds/clock band mismatch')
    require(np.array_equal(arrays['dimensions'],dims) and np.array_equal(arrays['min_bound'],low)
            and np.array_equal(arrays['max_bound'],high),'archive bounds/dimensions mismatch')
    count=int(dims[0]*dims[1]);linear=np.arange(count);row,east=np.divmod(linear,dims[0])
    centers=np.column_stack((low[0]+(dims[0]-east-.5)*.1,low[1]+(row+.5)*.1,np.full(count,ground+.975)))
    require(np.array_equal(arrays['native_free_index'],linear)
            and arrays['free_world_positions'].shape==(count,3)
            and np.allclose(arrays['free_world_positions'],centers,atol=1e-8,rtol=0),'all grid cells need exact fixed world centers')
    offsets=arrays['hit_offsets'];paths=arrays['hit_collision_paths']
    require(offsets.shape==(count+1,) and offsets.dtype.kind=='i' and offsets[0]==0
            and np.all(np.diff(offsets)>=0) and offsets[-1]==len(paths)
            and paths.ndim==1 and paths.dtype.kind in 'US'
            and arrays['hit_rigid_body_paths'].shape==paths.shape
            and arrays['hit_classification'].shape==paths.shape,'all-hit CSR invalid')
    floor=np.isin(paths,floor_paths);static=np.isin(paths,static_paths)
    require((floor|static).all(),'foreign hit path')
    require(np.array_equal(arrays['hit_classification'],np.where(floor,'selected_floor_excluded','authored_obstacle')),'hit classes differ from actual paths')
    hit_row=np.repeat(linear,np.diff(offsets));obstacle=np.zeros(count,bool);obstacle[hit_row[static]]=True
    for key,expected in [('full_height_clear',~obstacle),('authored_obstacle_hit',obstacle),('all_hit_paths_classified',np.ones(count,bool))]:
        require(arrays[key].dtype==np.bool_ and np.array_equal(arrays[key],expected),'row mask contradicts native hits: '+key)
    require(np.array_equal(arrays['selected_floor_hit_count'],np.bincount(hit_row[floor],minlength=count)),'floor hit counts mismatch')
    require(arrays['buffer'].shape==(count,) and np.array_equal(arrays['buffer'],np.where(obstacle,4,5)),'grid labels contradict native hits')
    return centers,obstacle


def verify_physx_cell_grid(provider,data,*,admitted=False):
    from .dynbench_scene import _check_record
    from PIL import Image
    grid=data['native_query_grid']
    if (data['mapping_backend']!='physx_cell_queries.v1' or grid['omap_plugin_loaded'] is not False
            or grid['projection_labels_used_to_claim_native_free'] is not False
            or grid['resolution_m']!=.1 or grid['layout']!='row-major x-fastest; row0=minY, column0=maxX'
            or grid['clock_before']!=grid['clock_after']):
        raise ValueError('Explicit actual full-cell PhysX mapping protocol required')
    _check_record(grid['artifact']);_check_record(grid['clock_artifact'])
    clock=read_resource_json(grid['clock_artifact']['path'])
    if (clock['clock_before']!=grid['clock_before'] or clock['clock_after']!=grid['clock_after']
            or clock['query_time_unchanged'] is not True or clock['foreign_hit_paths']!=[]
            or type(grid['clock_before']['submitted_physics_step']) is not int
            or grid['clock_before']['submitted_physics_step']<0
            or grid['clock_before']['timeline_s']!=0.):
        raise ValueError('Native cell-grid needs unchanged actual query clock')
    with np.load(grid['artifact']['path'],allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in z.files}
    centers,obstacle=verify_cell_arrays(arrays,bounds=data['actual_bounds_world_m'],dimensions=data['actual_dimensions'],
            ground=provider.ground_height_m,floor_paths=provider.floor_paths,static_paths=provider.static_paths)
    if grid['all_grid_cells_queried']!=len(centers):raise ValueError('Every grid cell must be queried')
    image=np.asarray(Image.open(data['exports']['occupancy.png']['path']))
    linear_export=image[::-1,::-1].reshape(-1)
    if len(linear_export)!=len(centers) or np.any(linear_export[obstacle]==254):
        raise ValueError('Native occupied cells cannot be exported walkable')
    with np.load(data['floor_support']['artifact']['path'],allow_pickle=False) as rays:
        free=centers[~obstacle]
        if (not np.array_equal(rays['free_world_positions'],free)
                or not rays['supported'].all() or data['floor_support']['tested']!=len(free)):
            raise ValueError('Every native clear cell needs real declared floor support')
    # Reconstruct every export from actual hits and the full support rows.
    native_image=np.where(obstacle.reshape(image.shape)[::-1,::-1],0,254).astype(np.uint8)
    if not np.array_equal(np.asarray(Image.open(data['exports']['native_reconstructed.png']['path'])),native_image):
        raise ValueError('Cell-grid reconstruction differs from raw native hits')
    with np.load(data['floor_support']['artifact']['path'],allow_pickle=False) as rays:
        low_obstacle=rays['known_static_nonfloor_near_ground_obstacle_hit'].copy()
    clear_ids=np.flatnonzero(~obstacle);native_rows,native_cols=np.divmod(clear_ids,int(arrays['dimensions'][0]))
    refined=native_image.copy()
    refined[image.shape[0]-1-native_rows[low_obstacle],image.shape[1]-1-native_cols[low_obstacle]]=0
    if not np.array_equal(np.asarray(Image.open(data['exports']['height_band_refined.png']['path'])),refined):
        raise ValueError('Refined image differs from actual native support and obstacles')
    projection=np.asarray(Image.open(provider._data['input_projection']['occupancy.png']['path']))
    if projection.shape!=image.shape:raise ValueError('Source support grid shape mismatch')
    refined[(projection==205)&(refined!=0)]=205
    if not np.array_equal(image,refined) or (admitted and not np.array_equal(provider._map,refined)):
        raise ValueError('Admitted grid differs from raw native evidence and source support domain')
    coverage=data['native_collider_coverage'];_check_record(coverage['artifact'])
    observed=read_resource_json(coverage['artifact']['path']);rows=observed['rows']
    expected=set((*provider.floor_paths,*provider.static_paths))
    if (coverage['all_enabled_colliders_registered'] is not True or coverage['count']!=len(rows)
            or observed['all_enabled_colliders_checked'] is not True
            or observed['physics_counter_before']!=observed['physics_counter_after']
            or len(rows)!=len(expected) or {r['path'] for r in rows}!=expected
            or not all(r['native_exact_path_observed'] is True and r['path'] in r['native_collision_paths'] for r in rows)):
        raise ValueError('Every enabled source collider must register in native queries')

    return {'native_evidence':provider._native['record'] if admitted else None,
            'canonical_known5_cells':len(centers)-int(obstacle.sum()),
            'height_blocked_known5_cells':0,'source_bound_artifacts_checked':True,
            'fixed_native_layout':grid['layout'],'query_clock':grid['clock_before'],
            'full_height_band_above_floor_m':[.15,1.8],'new_physical_execution_verified':False,
            'mapping_backend':'physx_cell_queries.v1','registered_colliders':len(rows)}
