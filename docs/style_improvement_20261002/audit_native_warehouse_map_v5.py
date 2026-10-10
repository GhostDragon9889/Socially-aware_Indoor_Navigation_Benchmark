"""Read only CPU admission for the actual v5 Warehouse native map artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_SHA = '16f845af4d315cd9328d7e1ebdf68317b64192947b5f49df07b3942b4f2f9c0d'


def fingerprint(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh report only')
    folder = args.directory.resolve()
    result = json.loads((folder/'result.json').read_text())
    preflight = json.loads((folder/'preflight.json').read_text())
    checks = []
    artifacts = {}

    def require(ok, label):
        checks.append({'check': label, 'passed': bool(ok)})
        if not ok:
            raise ValueError(label)

    def records(value):
        if isinstance(value, dict):
            if {'path', 'bytes', 'sha256'} <= value.keys():
                actual = fingerprint(value['path'])
                require(actual == {k: value[k] for k in actual}, 'full SHA/size: '+value['path'])
                artifacts[value['path']] = actual
            for child in value.values():
                records(child)
        elif isinstance(value, list):
            for child in value:
                records(child)

    report = {'schema': 'dynbench.native_warehouse_map_v5.independent_audit.v1',
              'CPU_only': True, 'SDK_started': False, 'GPU_used': False,
              'result': fingerprint(folder/'result.json'), 'audit_script': fingerprint(__file__),
              'checks': checks, 'artifacts': artifacts, 'scene_resource_admission': False,
              'navigation_quality_validated': False, 'author_map_claim': False}
    try:
        require(result['scene'] == 'Warehouse' and result['scene_resource_gate_passed'] is True
                and result['native_started'] is True and result['fallback_used'] is False, 'actual native v5 Warehouse full execution')
        require(result['script']['sha256'] == SCRIPT_SHA == preflight['script']['sha256'], 'fixed v5 executing script SHA')
        records(result)
        records(preflight)
        require(preflight['status'] == 'passed' and preflight['original_composition_unchanged'] is True
                and result['original_source_layers_unchanged'] is True, 'all original layers rehashed; unchanged')
        floor = result['floor_collider_paths']
        require(floor == ['/__PaperMapProbeReferenceGround'], 'one explicit finite reference floor')
        owner = result['floor_ownership']
        require(owner['schema'] == 'dynbench.native_floor_owner.v1' and owner['all_support_hit_paths_declared']
                and owner['single_declared_floor_owner_verified'] and owner['floor_ownership_verified']
                and owner['full_grid_floor_rays'] and owner['all_ray_hits_saved'], 'complete single-owner declarations')
        require(owner['actual_enabled_flat_ground_paths'] == owner['declared_floor_paths'] == floor
                and not owner['undeclared_near_floor_collision_paths'] and not owner['foreign_ray_collision_paths'], 'no extra near-floor/foreign paths')
        disabled = result['reference_ground']['disabled_flat_ground_collision_paths']
        require(disabled == preflight['reference_ground']['disabled_flat_ground_collision_paths']
                and '/Root/GroundPlane/CollisionPlane' in disabled and len(disabled) == 12, 'original plane and11 decals explicitly disabled')
        original_inventory = preflight['flat_ground_collider_inventory']
        plane = [r for r in original_inventory if r['path'] == '/Root/GroundPlane/CollisionPlane']
        require(len(plane) == 1 and plane[0]['type_name'] == 'Plane' and plane[0]['near_ground_flat_surface']
                and plane[0]['static'], 'real original Plane axis/height certificate')
        body = result['dynamic_activity']
        require(body['static_snapshot_gate'] and not body['time_sampled_xform_paths']
                and not any(r['enabled'] and not r['kinematic'] for r in body['authored_rigid_bodies']), 'static snapshot scope only')
        physics = result['physics']
        require(physics['device'] == 'cpu' and physics['dt_s'] == 1/120 and physics['gravity'] == [0., 0., 0.]
                and physics['scene_query_support'] and physics['counter_before_snapshot'] == 0
                and physics['counter_at_query'] == physics['snapshot_steps'] == 2
                and physics['snapshot_physics_time_increment_s'] == 2/120
                and physics['timeline_at_query_s'] == physics['timeline_after_queries_s'] == 0.
                and physics['no_query_time_advance'] and not physics['automatic_timeline_update']
                and not physics['app_player_playSimulations'], 'two120Hz CPU snapshot submits; zero query timeline advance')
        raw = dict(np.load(folder/'native_omap.npz', allow_pickle=False))
        canonical = dict(np.load(folder/'canonical_omap_world_positions.npz', allow_pickle=False))
        rays = dict(np.load(folder/'floor_support_rays.npz', allow_pickle=False))
        band = dict(np.load(folder/'full_height_band_queries.npz', allow_pickle=False))
        dims = raw['dimensions']
        require(dims.tolist() == [251, 370, 1] and raw['buffer'].shape == (92870,)
                and set(np.unique(raw['buffer'])) == {4., 5., 6.}, 'actual native2D dimensions/labels')
        w, h = map(int, dims[:2])
        counts = {str(int(v)): int((raw['buffer'] == v).sum()) for v in (4, 5, 6)}
        require(counts == {'4': 3343, '5': 56239, '6': 33288}, 'full actual buffer label counts')
        free = canonical['canonical_free_world_positions']
        require(free.shape == (56239, 3) and raw['free_positions'].shape == (89025, 3)
                and canonical['API_free_unknown_world_positions'].shape == (32786, 3)
                and canonical['unreported_unknown_buffer_indices'].shape == (502,), 'knownfree vsAPIunknown exact counts')
        minimum, maximum = raw['min_bound'], raw['max_bound']
        require(np.allclose(maximum[:2]-minimum[:2], dims[:2]*.1, atol=1e-6), '.1m XY grid bounds')
        xy = np.rint((free[:, :2]-minimum[:2])/.1-.5).astype(int)
        require(np.max(np.abs(free[:, :2]-(minimum[:2]+(xy+.5)*.1))) < 2e-6
                and len(np.unique(xy, axis=0)) == len(free), 'all worldcenters exact/unique within native float precision')
        native_indices = xy[:, 1]*w+(w-1-xy[:, 0])
        require(np.array_equal(np.sort(native_indices), np.flatnonzero(raw['buffer'] == 5)), 'official native xfastest east-to-west mapping matches ALL buffer5')
        occxy = np.rint((raw['occupied_positions'][:, :2]-minimum[:2])/.1-.5).astype(int)
        occindices = occxy[:, 1]*w+(w-1-occxy[:, 0])
        require(np.array_equal(np.sort(occindices), np.flatnonzero(raw['buffer'] == 4)), 'ALL3343 occupied worldcenters match buffer4')
        require(np.array_equal(free, raw['free_positions'][canonical['canonical_free_API_row']])
                and np.all(canonical['unreported_unknown_buffer_labels'] == 6)
                and np.all(raw['buffer'][canonical['unreported_unknown_buffer_indices']] == 6), 'canonical API rowbind and omitted502 remain unknown')
        for name, value in [('floor', rays), ('height', band)]:
            require(np.array_equal(value['native_free_index'], np.arange(len(free)))
                    and np.array_equal(value['free_world_positions'], free), name+' EVERY canonical known5 in identical order')
        offsets = rays['all_hit_offsets']
        require(offsets.shape == (56240,) and offsets[0] == 0 and offsets[-1] == len(rays['all_hit_collision_paths'])
                and np.array_equal(np.diff(offsets), rays['all_hit_count']), 'floor complete CSR aligned rowcounts')
        require(np.array_equal(rays['all_hit_count'], np.ones(56239)) and len(rays['all_hit_collision_paths']) == 56239, 'one native owner hit per ALL56239 rays')
        for key in ['collision', 'all_hit_collision_paths', 'all_hit_rigid_body_paths']:
            require(np.unique(rays[key]).tolist() == floor, 'native rays exact only reference owner: '+key)
        require(np.unique(rays['all_hit_classification']).tolist() == ['selected_floor_support']
                and rays['supported'].dtype == np.bool_ and rays['supported'].all()
                and not rays['undeclared_near_floor_hit'].any() and not rays['foreign_hit'].any(), 'raw normal support/classification all declared; noextra plane')
        positions, normals, distances = rays['all_hit_positions'], rays['all_hit_normals'], rays['all_hit_distances']
        require(positions.shape == normals.shape == (56239, 3) and distances.shape == (56239,)
                and np.isfinite(positions).all() and np.isfinite(normals).all() and np.isfinite(distances).all(), 'finite complete ray geometry')
        require(np.max(np.abs(positions[:, :2]-free[:, :2])) < 1e-4 and np.max(np.abs(positions[:, 2])) < 1e-6
                and np.min(normals[:, 2]) >= .9 and np.max(np.abs(np.linalg.norm(normals, axis=1)-1.)) < 1e-3
                and np.max(np.abs(distances-(.3-positions[:, 2]))) < 1e-6, 'native floorhit XY/height/normal/raydistance geometry')
        require(np.max(np.abs(rays['hit_position']-positions)) < 1e-6
                and np.max(np.abs(rays['hit_normal']-normals)) < 1e-6, 'nearest declared support agrees full raw rows')
        origin = result['origin_verification']
        require(not origin['overlap_hits'] and len(origin['floor_ray_all_hits']) == 1
                and origin['floor_ray_all_hits'][0]['collision'] == floor[0]
                and origin['floor_ownership']['all_support_hit_paths_declared'], 'origin clear and only reference floor ray')
        offsets = band['hit_offsets']
        require(offsets.shape == (56240,) and offsets[0] == 0 and offsets[-1] == len(band['hit_collision_paths'])
                and np.all(np.diff(offsets) >= 0), 'height complete hit CSR')
        enabled = {r['path'] for r in preflight['authored_colliders'] if r['enabled']}-set(disabled)
        enabled.add(floor[0])
        require(set(band['hit_collision_paths']) <= enabled-set(floor)
                and np.unique(band['hit_classification']).tolist() == ['authored_obstacle'], 'ALL native bandhits exact authored obstacles, noforeign/floor')
        blocked = np.diff(offsets) > 0
        require(int(blocked.sum()) == 520 and len(band['hit_collision_paths']) == 708
                and np.array_equal(band['authored_obstacle_hit'], blocked)
                and band['all_hit_paths_classified'].all() and not band['selected_floor_hit_count'].any()
                and np.array_equal(band['full_height_clear'], ~blocked), '520blocked/55719clear;708native hits, masks derived from all CSR')
        clock = json.loads((folder/'full_height_band_query_clock.json').read_text())
        require(clock['clock_before'] == clock['clock_after'] and clock['query_time_unchanged']
                and not clock['foreign_hit_paths'], 'native fullheight unchanged observed submittedcounter/timeline')
        native_image = np.choose(raw['buffer'].reshape(h, w)[::-1, ::-1].astype(int)-4, [0, 254, 205]).astype(np.uint8)
        require(np.array_equal(native_image, np.asarray(Image.open(folder/'native_reconstructed.png'))), 'fixed official layout rawbuffer full image roundtrip')
        refined = native_image.copy()
        refined[h-1-xy[:, 1], xy[:, 0]] = np.where(blocked, 0, 254)
        require(np.array_equal(refined, np.asarray(Image.open(folder/'height_band_refined.png')))
                and np.all(refined[native_image == 205] == 205), 'fullheight refinement exactly blocked520; native6 never upgraded')
        projection = np.asarray(Image.open(preflight['input_projection']['occupancy.png']['path']))
        exported = refined.copy()
        exported[(projection == 205) & (exported != 0)] = 205
        require(np.array_equal(exported, np.asarray(Image.open(folder/'occupancy.png'))), 'final map retains originalprojection unknown')
        require(result['requested_height_band_world_m'] == [.15, 1.8]
                and result['height_band_refinement']['all_native_known5_cells_tested']
                and not result['height_band_refinement']['pure_omap2D_full_height_claim']
                and result['floor_support']['full_grid'], 'explicit native2D+fullheight engineering scope')
        # Read-only standalone USD child verifies exported session disabledcollisions.
        code = """import json,sys\nfrom pxr import Sdf,Usd,UsdPhysics\nr=json.load(open(sys.argv[1]));session=Sdf.Layer.FindOrOpen(sys.argv[2]);root=Sdf.Layer.FindOrOpen(r['source']['path']);s=Usd.Stage.Open(root,session,load=Usd.Stage.LoadAll);disabled=r['reference_ground']['disabled_flat_ground_collision_paths'];v={p:bool(UsdPhysics.CollisionAPI(s.GetPrimAtPath(p)).GetCollisionEnabledAttr().Get()) for p in disabled};print(json.dumps({'disabled_enabled':v,'reference_enabled':bool(UsdPhysics.CollisionAPI(s.GetPrimAtPath(r['floor_collider_paths'][0])).GetCollisionEnabledAttr().Get()),'composition_errors':len(s.GetCompositionErrors())}))"""
        child = subprocess.run([sys.executable, '-c', code, str(folder/'result.json'), str(folder/'runtime_session.usda')],
                               capture_output=True, text=True, check=True)
        session = json.loads(child.stdout)
        require(not any(session['disabled_enabled'].values()) and session['reference_enabled']
                and session['composition_errors'] == 0, 'exported session actualschema disables plane/11decals; source open errorfree')
        report.update(status='passed', scene_resource_admission=True,
            counts={'native_cells': 92870, 'known_free': 56239, 'native_occupied': 3343, 'native_unknown': 33288,
                    'API_free_unknown': 32786, 'omitted_unknown': 502, 'all_floor_hits': 56239,
                    'height_blocked_cells': 520, 'height_clear_cells': 55719, 'height_obstacle_hits': 708,
                    'disabled_original_colliders': 12, 'final_free': int((exported == 254).sum()),
                    'final_occupied': int((exported == 0).sum()), 'final_unknown': int((exported == 205).sum())},
            submitted_physics_steps=2, query_timeline_advance_s=0., source_layers_checked=len(preflight['layers']),
            runtime_session_verification=session,
            limits=['CPU auditor did not start Kit; it reads actual root-run native measurements.',
                    'Submitted counter/timeline recorded; native engine-time API not claimed.',
                    'Explicit finite generated reference ground and static snapshot; not authored floor/private paper map.',
                    'Cell0.1m full-height clearance is not body footprint or eight-scene navigation quality.',
                    'v4 failed single-floor evidence remains unchanged and unadmitted.'])
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
    report['check_count'] = len(checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status': report['status'], 'checks': len(checks), 'error': report.get('error')}))
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
