"""Independent CPU recount of the closed15 capture after the JSON-proof revision."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path('/project/EXP/SangforAndSaut/ProtoMotions')
HERE = Path(__file__).resolve().parent
CORE = ROOT / 'docs/style_improvement_20261003/state_prior_native_reader_preparation_v1'
REV = ROOT / 'docs/style_improvement_20261003/state_prior_native_reader_json_revision_v2'
ACTUAL = ROOT / 'docs/style_improvement_20261003/state_prior_native_actual_review_v1'


def record(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(8388608), b''):
            h.update(b)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def main():
    checks = []

    def check(name, value):
        if not bool(value):
            raise ValueError(name)
        checks.append(name)

    check('CUDA hidden', os.environ.get('CUDA_VISIBLE_DEVICES') == '')
    audit_path = ACTUAL / 'native15_audit.json'
    audit = read(audit_path)
    check('Exact actual15 report66f', record(audit_path)['sha256'] ==
          '66f16e6dd9d36e517b95197aae83829d1db1b19ae378128016f40c4ac0d47176')
    check('Exact v2freeze837 and all9 readonly sources', record(REV / 'source_freeze_v2.json')['sha256'] ==
          '837fe71a8f7574e10b8c68b707363c05bb6dfae9d6b608528c955984c8162866' and
          all(record(x['path']) == x and Path(x['path']).stat().st_mode & 0o222 == 0
              for x in read(REV / 'source_freeze_v2.json')['files']))
    outer = read(REV / 'native15_revision_registration.json')
    check('Explicit new wrapper and immutable numerical body identities',
          audit['representation_revision_registration'] == outer and
          audit['representation_adapter'] == outer['representation_adapter'] ==
          record(REV / 'audit_native15_json_revision.py') and
          audit['execution_reader'] == record(CORE / 'audit_state_prior_native.py') and
          audit['proof_tuple_list_difference_only'] is True and
          audit['compiled_v1_v2_routes_math_changed'] is False)
    check('Actual15 CPU completion/sourcefreeze closed0 exact',
          all(record(x['path']) == x for x in read(ACTUAL / 'native15_source_freeze_v2.json')['files']) and
          read(ACTUAL / 'native15_CPU_completion_v2.json')['actual_closed_exec_exit_code'] == 0)
    sys.path.insert(0, str(CORE))
    import native_evidence as evidence
    registered = read(CORE / 'native15_registration.json')
    fresh = evidence.prepare_evidence(15)
    check('Fresh True terminal full evidence equals original core registration', fresh == registered['evidence'])
    check('Canonical complete denominators strict type', registered['expected_counts'] == evidence.expected_counts(15)
          and all(type(registered['expected_counts'][k]) is type(v)
                  for k, v in evidence.expected_counts(15).items()))
    admitted = evidence.admit_closed_native(15, registered)
    check('Actual source183 closed15v2 observedchild and owner inputs exact',
          admitted == audit['actual_source_admission'])
    artifacts = {k: record(v['path']) for k, v in audit['artifacts'].items()}
    check('All five actual capture files exact before NumPy', artifacts == audit['artifacts'])
    import numpy as np
    run = evidence.run_path(15)
    result = read(run / 'result.json')
    with np.load(run / 'physical_states.npz', allow_pickle=False) as archive:
        names = ('root_pos', 'root_rot', 'terminated', 'truncated', 'reset_mask', 'episode_ids',
                 'static_contact', 'static_contact_max_force_n', 'peer_contact', 'peer_contact_max_force_n',
                 'route_vertex_reached', 'route_indices', 'route_vertex_xy')
        physical = {name: archive[name] for name in names}
    with np.load(run / 'ground_support.npz', allow_pickle=False) as archive:
        names = ('control_step', 'physics_step_before', 'physics_step', 'physics_step_end', 'valid',
                 'substep_valid_count', 'actor_substep_valid_count', 'foot_floor_normal_force_w',
                 'foot_contact_substeps', 'terminated', 'truncated', 'reset_mask', 'episode_ids')
        ground = {name: archive[name] for name in names}
    check('All6000-by15 selected raw count channels', physical['root_pos'].shape == (6000, 15, 3)
          and all(len(v) == 6000 for v in physical.values()))
    check('All6000 controls/four actual substeps valid',
          np.array_equal(ground['control_step'], np.arange(6000)) and ground['valid'].all()
          and np.all(ground['substep_valid_count'] == 4) and np.all(ground['actor_substep_valid_count'] == 4))
    check('Exact submitted0-to24000 consecutive physical counters',
          int(ground['physics_step_before'][0]) == 0 and int(ground['physics_step_end'][-1]) == 24000
          and np.array_equal(ground['physics_step'], ground['physics_step_before'][:, None] + np.arange(1, 5))
          and np.array_equal(ground['physics_step_end'], ground['physics_step_before'] + 4)
          and np.array_equal(ground['physics_step_before'][1:], ground['physics_step_end'][:-1]))
    for name in ('terminated', 'truncated', 'reset_mask', 'episode_ids'):
        check('Ground/physical exact ' + name, np.array_equal(ground[name], physical[name]))
    check('Reset/episode transitions retain failures/time limits',
          np.array_equal(np.diff(physical['episode_ids'], axis=0) == 1, physical['reset_mask'][1:])
          and np.array_equal(physical['reset_mask'][1:], (physical['terminated'] | physical['truncated'])[:-1]))
    force = ground['foot_floor_normal_force_w']
    check('All1.44M actual foot observations', force.shape == (6000, 4, 15, 4, 1, 3)
          and int(np.prod(force.shape[:4])) == audit['counts']['feet_observations'] == 1440000)
    height = physical['root_pos'][..., 2] - result['native_scene_provider']['ground_height_m']
    quat = physical['root_rot']
    definition = result['physics']['fall_definition']
    falls = (height < definition['pelvis_ground_height_less_than_m']) | (
        1 - 2 * (quat[..., 0] ** 2 + quat[..., 1] ** 2) < definition['root_up_z_less_than'])
    check('All actual falls retained and zero', np.array_equal(falls, physical['terminated'])
          and int(falls.sum()) == audit['counts']['falls'] == result['fall_resets'] == 0)
    check('All actual time limits retained and zero', int(physical['truncated'].sum()) ==
          audit['counts']['time_limit_resets'] == result['time_limit_resets'] == 0)
    threshold = result['physical_contact_measurement']['threshold_n']
    check('Strict1N full static/peer masks', threshold == 1
          and np.array_equal(physical['static_contact'], physical['static_contact_max_force_n'] > threshold)
          and np.array_equal(physical['peer_contact'], physical['peer_contact_max_force_n'] > threshold))
    i, j = np.triu_indices(15, 1)
    check('Complete90000 static/630000 peer denominators and zero counts',
          int(physical['static_contact'].sum()) == audit['counts']['static_actor_controls'] == 0
          and int(physical['peer_contact'][:, i, j].sum()) == audit['counts']['peer_pair_controls'] == 0
          and audit['counts']['static_denominator_actor_controls'] == 90000
          and audit['counts']['peer_denominator_unique_pair_controls'] == 630000)
    check('All foot contacts independently strict1N recomputed',
          np.array_equal(np.linalg.norm(force.sum(-2), axis=-1) > 1, ground['foot_contact_substeps']))
    vertices = physical['route_vertex_reached'].sum(0).tolist()
    check('All160 raw reached vertices match source/result/report', vertices == audit['counts']['route_vertices']
          == result['route_vertices_reached'] and sum(vertices) == 160)
    cycles = []
    routes = np.asarray(result['evaluation_scene']['routes_xy'], dtype=physical['route_vertex_xy'].dtype)
    for actor in range(15):
        frames = np.flatnonzero(physical['route_vertex_reached'][:, actor])
        indices = physical['route_indices'][frames, actor].tolist()
        check('Ordered sameepisode route progression ' + str(actor),
              indices == [step % 4 for step in range(1, len(frames) + 1)]
              and len(np.unique(physical['episode_ids'][frames, actor])) == 1)
        for frame in frames:
            vertex = (int(physical['route_indices'][frame, actor]) - 1) % 4
            target = routes[actor, vertex]
            check('Reached geometry ' + str(frame) + '/' + str(actor),
                  np.array_equal(physical['route_vertex_xy'][frame, actor], target)
                  and np.linalg.norm(physical['root_pos'][frame, actor, :2] - target) <=
                  result['physics']['scene_geometry']['route_arrival_radius_m'] + 1e-6)
        cycles.append(len(frames) // 4)
    check('All32 ordered cycles match original route report', cycles == audit['counts']['complete_cycles']
          and sum(cycles) == audit['counts']['complete_cycles_total'] == 32)
    check('Five captures unchanged after own read', all(record(x['path']) == x for x in artifacts.values()))
    check('Actual source/terminal/fresh registry still exact', evidence.prepare_evidence(15) == fresh)
    check('No model or SDK imports', not any(n.split('.')[0] in
          {'torch', 'isaacsim', 'isaaclab', 'omni', 'pxr', 'warp'} for n in sys.modules))
    output = HERE / 'actual15_QC.json'
    report = {'schema': 'public_state_prior.native15_actual_independent_peer_QC.v2', 'status': 'passed',
              'CPU_only': True, 'GPU_query_called': False, 'GPU_started': False, 'Popen_called': False,
              'SDK_started': False, 'actual_native15_arrays_read': True,
              'source': record(__file__), 'actual_report': record(audit_path),
              'representation_source_freeze': record(REV / 'source_freeze_v2.json'),
              'representation_registration': record(REV / 'native15_revision_registration.json'),
              'closed_owner_launch': admitted['closed_owner_launch'], 'production_source': fresh['production_source'],
              'checkpoint': fresh['checkpoint'], 'actual_capture_records': artifacts,
              'counts': audit['counts'], 'checks_passed': len(checks), 'checks': checks,
              'COM_error_from_author_owned_full_math_audit': audit['support']['COM_recomposition_max_error_m'],
              'COM_math_rerun_by_peer': False, 'overall_style_gain_claimed': False, 'default_promotion': False,
              'limitations': ['Independent actual count/route/contact/foot QC supplements the author1241 full raw checks.',
                              'Source183 is not a complete import ledger; this does not establish causal/general style gain.']}
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    output.chmod(0o444)
    print(json.dumps({'status': 'passed', 'report': record(output), 'checks_passed': len(checks), 'counts': report['counts']}))


if __name__ == '__main__':
    main()
