"""Read only the closed native4 inputs and independently recount retained arrays."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path('/project/EXP/SangforAndSaut/ProtoMotions')
HERE = Path(__file__).resolve().parent
PREP = ROOT / 'docs/style_improvement_20261003/state_prior_native_reader_preparation_v1'
ACTUAL = ROOT / 'docs/style_improvement_20261003/state_prior_native_actual_review_v1'


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def main():
    checks = []

    def check(name, condition):
        if not bool(condition):
            raise ValueError(name)
        checks.append(name)

    check('CUDA hidden for CPU QC', os.environ.get('CUDA_VISIBLE_DEVICES') == '')
    audit_path = ACTUAL / 'native4_audit.json'
    audit = read(audit_path)
    check('Exact actual4 audit5655', record(audit_path)['sha256'] ==
          '5655f818094a360a50ad7cc3ecb3100965cbf13e5e248e85ccc70edff9f75862')
    check('Final e5ece reader and ce11 helper exact',
          record(PREP / 'audit_state_prior_native.py')['sha256'] ==
          'e5ece5e523083e13aaab4d906defdad21240983551e55b40e1b4911ec1a6f9e5' and
          record(PREP / 'native_evidence.py')['sha256'] ==
          'ce11fd5c73d2c50207e7aa032f82db7fb97150eddd3dcb9d73988ca24c610d03')
    check('Final preparation154c frozen records exact',
          record(PREP / 'source_freeze_v1.json')['sha256'] ==
          '154c4eb9debc41a83a487e4549c73009dcc3c4f15ab3c36e85e0990129a2ac50' and
          all(record(row['path']) == row for row in read(PREP / 'source_freeze_v1.json')['files']))
    actual_freeze = read(ACTUAL / 'native4_source_freeze_v1.json')
    check('Author actual4 completion/report/log freeze exact',
          all(record(row['path']) == row for row in actual_freeze['files']))
    completion = read(ACTUAL / 'native4_CPU_completion_v1.json')
    check('Author actually closed CPU raw read0/no future15',
          completion['status'] == 'passed' and completion['actual_closed_exec_exit_code'] == 0
          and completion['future15_capture_read'] is False and completion['GPU_used'] is False)
    sys.path.insert(0, str(PREP))
    import native_evidence as evidence
    registered = read(PREP / 'native4_registration.json')
    fresh = evidence.prepare_evidence(4)
    check('Fresh source183/True full terminal/config/entry equals registration',
          fresh == registered['evidence'] and fresh['production_source']['source_count'] == 183
          and fresh['production_source']['complete_runtime_import_ledger'] is False)
    check('Canonical strictly typed full denominators',
          registered['expected_counts'] == evidence.expected_counts(4) and
          all(type(registered['expected_counts'][key]) is type(value)
              for key, value in evidence.expected_counts(4).items()))
    admitted = evidence.admit_closed_native(4, registered)
    check('Actual owner4 closed/exactsource before-after/input identity',
          admitted == audit['actual_source_admission'])
    check('Actual report pending registration remains same original registration',
          audit['registration'] == registered and audit['execution_reader'] ==
          record(PREP / 'audit_state_prior_native.py'))
    artifact_before = {name: record(row['path']) for name, row in audit['artifacts'].items()}
    check('All actual4 capture file records exact before NumPy', artifact_before == audit['artifacts'])
    run = evidence.run_path(4)
    result = read(run / 'result.json')
    import numpy as np

    def arrays(name):
        with np.load(run / name, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}

    physical = arrays('physical_states.npz')
    ground = arrays('ground_support.npz')
    check('All6000 rows4 actors retained', physical['root_pos'].shape == (6000, 4, 3)
          and all(len(value) == 6000 for value in physical.values()))
    check('All6000 four-substep controls complete',
          np.array_equal(ground['control_step'], np.arange(6000)) and
          ground['valid'].all() and np.all(ground['substep_valid_count'] == 4) and
          np.all(ground['actor_substep_valid_count'] == 4))
    check('Exact submitted0-to24000 counter chain',
          int(ground['physics_step_before'][0]) == 0 and int(ground['physics_step_end'][-1]) == 24000 and
          np.array_equal(ground['physics_step'], ground['physics_step_before'][:, None] + np.arange(1, 5)) and
          np.array_equal(ground['physics_step_end'], ground['physics_step_before'] + 4) and
          np.array_equal(ground['physics_step_before'][1:], ground['physics_step_end'][:-1]))
    for name in ('terminated', 'truncated', 'reset_mask', 'episode_ids'):
        check('Physical/ground exact ' + name, np.array_equal(physical[name], ground[name]))
    check('No omitted episode transitions',
          np.array_equal(np.diff(physical['episode_ids'], axis=0) == 1, physical['reset_mask'][1:]) and
          np.array_equal(physical['reset_mask'][1:], (physical['terminated'] | physical['truncated'])[:-1]))
    force = ground['foot_floor_normal_force_w']
    check('All384000 actual foot observations', force.shape == (6000, 4, 4, 4, 1, 3) and
          int(np.prod(force.shape[:4])) == audit['counts']['feet_observations'] == 384000)
    height = physical['root_pos'][..., 2] - result['native_scene_provider']['ground_height_m']
    quat = physical['root_rot']
    definition = result['physics']['fall_definition']
    falls = (height < definition['pelvis_ground_height_less_than_m']) | (
        1 - 2 * (quat[..., 0] ** 2 + quat[..., 1] ** 2) < definition['root_up_z_less_than'])
    check('Raw actual fall definitions and complete zero falls',
          np.array_equal(falls, physical['terminated']) and int(falls.sum()) ==
          audit['counts']['falls'] == result['fall_resets'] == 0)
    truncated = int(physical['truncated'].sum())
    check('Raw time limits retained and actual0', truncated == audit['counts']['time_limit_resets'] ==
          audit['counts']['all_raw_time_limit_actor_controls_retained'] == result['time_limit_resets'] == 0)
    threshold = result['physical_contact_measurement']['threshold_n']
    check('Actual strict1N static and peer masks', threshold == 1 and
          np.array_equal(physical['static_contact'], physical['static_contact_max_force_n'] > threshold) and
          np.array_equal(physical['peer_contact'], physical['peer_contact_max_force_n'] > threshold))
    i, j = np.triu_indices(4, 1)
    static = int(physical['static_contact'].sum())
    peer = int(physical['peer_contact'][:, i, j].sum())
    check('All static/peer denominators and actual zero counts', static == audit['counts']['static_actor_controls'] == 0
          and peer == audit['counts']['peer_pair_controls'] == 0 and
          audit['counts']['static_denominator_actor_controls'] == 24000 and
          audit['counts']['peer_denominator_unique_pair_controls'] == 36000)
    vertices = physical['route_vertex_reached'].sum(0).tolist()
    check('All44 actual reached vertices retained', vertices == [11, 11, 11, 11] ==
          audit['counts']['route_vertices'] == result['route_vertices_reached'])
    cycles = []
    routes = np.asarray(result['evaluation_scene']['routes_xy'])
    for actor in range(4):
        frames = np.flatnonzero(physical['route_vertex_reached'][:, actor])
        indices = physical['route_indices'][frames, actor].tolist()
        check('Actor route ordering complete in one episode ' + str(actor),
              indices == [step % 4 for step in range(1, 12)] and
              len(np.unique(physical['episode_ids'][frames, actor])) == 1)
        for frame in frames:
            index = (int(physical['route_indices'][frame, actor]) - 1) % 4
            check('Actual reached route geometry ' + str(frame) + '/' + str(actor),
                  np.array_equal(physical['route_vertex_xy'][frame, actor],
                                 np.asarray(routes[actor, index], dtype=physical['route_vertex_xy'].dtype)) and
                  np.linalg.norm(physical['root_pos'][frame, actor, :2] - routes[actor, index]) <=
                  result['physics']['scene_geometry']['route_arrival_radius_m'] + 1e-6)
        cycles.append(len(frames) // 4)
    check('Eight ordered within-episode cycles', cycles == [2, 2, 2, 2] ==
          audit['counts']['complete_cycles'] and sum(cycles) == audit['counts']['complete_cycles_total'] == 8)
    load = np.linalg.norm(force.sum(-2), axis=-1)
    foot = load > 1
    check('Actual foot support strict1N recomputed', np.array_equal(foot, ground['foot_contact_substeps']))
    side = foot.reshape(6000, 4, 4, 2, 2).any(-1)
    check('All side/no-support means from four substeps',
          np.array_equal(side, ground['side_contact_substeps']) and
          side.mean((0, 1)).tolist() == audit['support']['side_supported_fraction_per_actor'] and
          (~side.any(-1)).mean((0, 1)).tolist() == audit['support']['no_support_fraction_per_actor'])
    pos = ground['body_link_pos_w']
    quat = ground['body_link_quat_w']
    offset = ground['local_com_pose_b'][..., :3]
    cross = 2 * np.cross(quat[..., :3], offset)
    composed = pos + offset + quat[..., 3:] * cross + np.cross(quat[..., :3], cross)
    com_error = float(np.abs(composed - ground['body_com_pos_w']).max())
    check('True COM independent FP32 recomposition exact', com_error ==
          audit['support']['COM_recomposition_max_error_m'] == 1.9073486328125e-6)
    check('Capture file records unchanged after own NumPy QC',
          all(record(row['path']) == row for row in artifact_before.values()))
    check('Source183/terminal/registration unchanged after own QC', evidence.prepare_evidence(4) == fresh)
    check('No model or SDK imports', not any(name.split('.')[0] in
          {'torch', 'isaacsim', 'isaaclab', 'omni', 'pxr', 'warp'} for name in sys.modules))
    output = HERE / 'actual4_QC.json'
    report = {'schema': 'public_state_prior.native4_actual_independent_peer_QC.v1', 'status': 'passed',
              'checked_utc': datetime.now(timezone.utc).isoformat(), 'CPU_only': True,
              'GPU_query_called': False, 'GPU_started': False, 'Popen_called': False, 'SDK_started': False,
              'actual_native4_arrays_read': True, 'future15_capture_read': False,
              'source': record(__file__), 'actual_audit': record(audit_path),
              'reader_preparation_freeze': record(PREP / 'source_freeze_v1.json'),
              'author_completion': record(ACTUAL / 'native4_CPU_completion_v1.json'),
              'closed_owner_launch': admitted['closed_owner_launch'], 'source_boundary': fresh['production_source'],
              'registration': record(PREP / 'native4_registration.json'), 'checkpoint': fresh['checkpoint'],
              'counts': audit['counts'], 'independent_COM_error_m': com_error,
              'review_harness_correction': record(HERE / 'initial_route_assumption_failure.json'),
              'actual_input_records': artifact_before, 'checks_passed': len(checks), 'checks': checks,
              'overall_style_gain_claimed': False, 'default_promotion': False,
              'limitations': ['Independent recount/support/COM QC supplements the author-owned987-check complete raw audit.',
                              'Source183 is not a complete import ledger. No causal style improvement or future15 quality is inferred.']}
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    output.chmod(0o444)
    print(json.dumps({'status': 'passed', 'report': record(output), 'checks_passed': len(checks),
                      'counts': report['counts'], 'COM_error_m': com_error}))


if __name__ == '__main__':
    main()
