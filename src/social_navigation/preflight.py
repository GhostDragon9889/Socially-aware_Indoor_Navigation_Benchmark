"""Verify source-bound scene maps on CPU; importing this module starts no SDK."""
import hashlib
from pathlib import Path


def verify_native_scene_config(path):
    import yaml
    from CrowdSim.humanoid.runtime.dynbench_scene import DynBenchSceneProvider
    from CrowdSim.humanoid.runtime.native_grid_scene import NativeGridScene

    scene = yaml.safe_load(Path(path).read_text())
    if scene.get('schema') != 'crowdsim.native_pedestrian_scene.v1':
        raise ValueError('Expected an explicit native pedestrian scene')
    snapshot = scene.get('probe_script_snapshot')
    if snapshot is not None:
        with Path(snapshot['path']).open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != snapshot['sha256']:
                raise ValueError('Native probe snapshot changed')
    provider = DynBenchSceneProvider.from_preflight(scene['preflight']['path'],
        expected_sha256=scene['preflight']['sha256'],
        probe_script_snapshot=None if snapshot is None else snapshot['path'])
    provider.admit_native_map(scene['native_map']['path'], expected_sha256=scene['native_map']['sha256'])
    provider.configure_policy_height_feature(scene.get('amp_policy_height_mode', 'native_obstacle_cap'))
    NativeGridScene.from_provider(provider, origins_xy=scene['origins_xy'], routes_xy=scene['routes_xy'],
        clearance_m=scene['clearance_m'], arrival_radius_m=scene.get('arrival_radius_m', .75),
        goal_lookahead_m=scene.get('goal_lookahead_m', 5.), device='cpu',
        cyclic_route_lookahead=scene.get('cyclic_route_lookahead', False),
        start_recovery_radius_m=scene.get('start_recovery_radius_m', 0.),
        start_recovery_clearance_m=scene.get('start_recovery_clearance_m', .4),
        route_clearance_weight=scene.get('route_clearance_weight', 0.),
        route_comfort_clearance_m=scene.get('route_comfort_clearance_m', .6))
    return {'status': 'passed', 'scene': scene['scene'], 'actor_count': len(scene['origins_xy']),
            'source_bound_native_map': True, 'route_configuration_verified': True,
            'sdk_started': False, 'live_physx_execution_verified': False}
