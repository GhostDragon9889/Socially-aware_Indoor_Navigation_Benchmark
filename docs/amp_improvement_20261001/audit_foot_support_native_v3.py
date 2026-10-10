from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path('/project/EXP/SangforAndSaut/ProtoMotions')
DIRECTORY = ROOT / 'output/diffusion_amp/foot_support_native_qa_v3'
REPORT = json.loads((DIRECTORY / 'result.json').read_text())
PHASES = ('standing', 'airborne', 'airborne_wall')
FEET = ('L_Ankle', 'L_Toe', 'R_Ankle', 'R_Toe')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rotate(q, v):
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return v + 2 * q[..., 3:] * np.cross(q[..., :3], v) + 2 * np.cross(q[..., :3], np.cross(q[..., :3], v))

result = {'schema': 'navisaaclab.foot_support_native_validation.v3',
          'scope': 'Independent CPU/NumPy recomputation of completed actual native v3 captures; no new SDK/GPU run, no business changes',
          'overall_original_native_probe_passed': False,
          'sensor_channel_criteria_passed': None,
          'standing_no_fall_criterion_passed': None,
          'author_gait_metric_equivalence': False,
          'input_result': {'path': str(DIRECTORY / 'result.json'), 'sha256': sha(DIRECTORY / 'result.json')},
          'input_log': {'path': '/tmp/foot_support_native_qa_v3.log', 'sha256': sha('/tmp/foot_support_native_qa_v3.log')},
          'reported_status': REPORT['status'],
          'reported_native_physical_probe_passed': REPORT['native_physical_probe_passed'],
          'native_contact_factory_lookup': REPORT['native_contact_factory_lookup'],
          'exact_foot_rigid_paths': REPORT['exact_foot_rigid_paths'],
          'exact_floor_collider_paths': REPORT['exact_floor_collider_paths'],
          'exact_wall_collider_paths': REPORT['exact_wall_collider_paths'],
          'captured_source_sha256': REPORT['selected_source_sha256'],
          'sources_match_current_at_review': {name: sha(ROOT / name) == digest for name, digest in REPORT['selected_source_sha256'].items()},
          'phases': [], 'checks': {}}
checks = result['checks']
checks['reported_all_independent_phases_completed'] = REPORT['all_independent_phases_completed']
checks['reported_phase_order_exact'] = tuple(p['phase'] for p in REPORT['phases']) == PHASES
sensors = [REPORT['exact_foot_rigid_paths'][0][name] for name in FEET]
manifest = REPORT['floor_monitor_contract']
checks['floor_sensor_metadata_exact_bijection'] = (len(manifest['native_sensor_paths']) == 4 and set(manifest['native_sensor_paths']) == set(sensors))
checks['floor_filter_metadata_exact_excludes_wall'] = (manifest['native_filter_paths'] == [[REPORT['exact_floor_collider_paths'][0]]] * 4
                                                      and not set(REPORT['exact_floor_collider_paths']) & set(REPORT['exact_wall_collider_paths']))
checks['wall_sensor_metadata_exact_bijection'] = set(REPORT['substep_adapter_contract']['wall_sensor_paths']) == set(sensors)
checks['wall_filter_exact_disjoint_from_floor'] = REPORT['substep_adapter_contract']['wall_filter_path'] == REPORT['exact_wall_collider_paths'][0]
checks['public_view_active_manager_identity_observed'] = REPORT['native_contact_factory_lookup']['active_manager_getter_same_view']
checks['no_legacy_private_or_new_view_fallback'] = not any(REPORT['native_contact_factory_lookup'][name] for name in ('legacy_isaacsim_SimulationManager_used','new_simulation_view_created','private_view_lookup_used'))

prior_physics, prior_control = 0, 0
for phase, phase_report, controls in zip(PHASES, REPORT['phases'], (60,4,4)):
    path = DIRECTORY / f'{phase}.npz'
    with np.load(path, allow_pickle=False) as f:
        d = {key: f[key] for key in f.files}
    names = tuple(d['common_body_names'][0].tolist())
    foot_indices = [names.index(name) for name in FEET]
    floor = d['foot_floor_normal_force_w']
    wall = d['foot_wall_normal_force_w']
    # Magnitude of vector sum per foot/filter; no scalar patch-force or friction claim.
    loads = np.linalg.norm(floor.astype(np.float64).sum(-2), axis=-1)
    contacts = loads > REPORT['arguments']['threshold_n']
    sides = np.stack((contacts[..., :2].any(-1), contacts[..., 2:].any(-1)), -1)
    side_any = sides.any(1)
    side_all = sides.all(1)
    pos = d['body_link_pos_w'].astype(np.float64)
    q = d['body_link_quat_xyzw'].astype(np.float64)
    local = d['local_com_pos_b'].astype(np.float64)
    composed = pos + rotate(q, local)
    com_error = float(np.abs(composed - d['body_com_pos_w']).max())
    masses = d['body_masses_kg'].astype(np.float64)
    weighted = (d['body_com_pos_w'].astype(np.float64) * masses[..., None]).sum(-2) / masses.sum(-1)[..., None]
    weighted_error = float(np.abs(weighted - d['mass_weighted_world_com']).max())
    up = 1 - 2 * (d['root_rot_xyzw'][:,0,:2].astype(np.float64)**2).sum(-1)
    expected_fall = (d['root_pos'][:,0,2] < .45) | (up < .1)
    floor_max = float(np.abs(floor).max())
    wall_max = float(np.linalg.norm(wall.astype(np.float64),axis=-1).max())
    min_foot_z = float(pos[..., foot_indices,2].min())
    pc = {}
    pc['declared_capture_hash_matches_actual'] = sha(path) == phase_report['capture']['sha256']
    pc['shape_controls_substeps_actor_feet_filter_vector'] = floor.shape == wall.shape == (controls,4,1,4,1,3)
    pc['all_numeric_physical_and_force_arrays_finite'] = all(np.isfinite(v).all() for v in d.values() if v.dtype.kind in 'fci')
    pc['common_body_names_constant_exact24'] = len(names) == 24 and len(set(names)) == 24 and all(tuple(row)==names for row in d['common_body_names'])
    pc['actual_positive_masses'] = bool((masses > 0).all())
    pc['quaternions_unit_within1e_5'] = float(np.abs(np.linalg.norm(q,axis=-1)-1).max()) < 1e-5
    pc['consecutive_substeps_without_gap_or_duplicate'] = np.array_equal(d['physics_step_before_read'].ravel(), np.arange(prior_physics+1,prior_physics+controls*4+1))
    pc['physical_contact_reads_do_not_advance_counter'] = np.array_equal(d['physics_step_before_read'],d['physics_step_after_read'])
    pc['control_clock_consecutive_without_extra_steps'] = np.array_equal(d['control_step'], np.repeat(np.arange(prior_control,prior_control+controls)[:,None],4,axis=1))
    pc['physics_step_end_matches_fourth_sample'] = np.array_equal(d['physics_step_end'],d['physics_step_after_read'][:,-1])
    pc['each_control_four_valid_substeps'] = np.array_equal(d['substep_valid_count'],np.full(controls,4)) and np.array_equal(d['actor_substep_valid_count'],np.full((controls,1),4))
    pc['phase_reset_zero_physics_advance'] = phase_report['reset']['counter_before_reset'] == phase_report['reset']['counter_after_reset'] == prior_physics
    pc['phase_episode_identity_constant'] = bool((d['episode_id']==phase_report['reset']['episode_id']).all() and (d['episode_ids']==phase_report['reset']['episode_id']).all())
    pc['exactly_one_initial_native_reset_declared'] = phase_report['reset']['native_reset_count_this_phase'] == 1
    pc['fall_mask_matches_unchanged_height_or_tilt_predicate'] = np.array_equal(expected_fall,d['fall_mask'])
    pc['no_time_limit_termination'] = not bool(d['time_limit_mask'].any())
    pc['foot_contact_mask_matches_strict_actual_floor_threshold'] = np.array_equal(contacts,d['foot_contact_substeps'])
    pc['side_substep_any_all_contact_masks_match'] = (np.array_equal(sides,d['side_contact_substeps']) and np.array_equal(side_any,d['side_contact_any']) and np.array_equal(side_all,d['side_contact_all']))
    pc['mean_and_peak_foot_loads_match_raw_vectors'] = np.allclose(loads.mean(1),d['foot_normal_load_mean_n'],atol=1e-4,rtol=1e-6) and np.allclose(loads.max(1),d['foot_normal_load_peak_n'],atol=1e-4,rtol=1e-6)
    pc['native_floor_rows_map_to_common_exact_feet'] = np.array_equal(d['native_floor_normal_force_w'][:,:, [manifest['native_sensor_paths'].index(path) for path in sensors]],floor[:,:,0])
    pc['live_COM_matches_link_plus_rotated_local_COM'] = com_error <= REPORT['arguments']['com_tolerance_m']
    pc['massweighted_COM_matches_actual_native_masses'] = weighted_error <= 1e-5
    pc['source_summary_floor_wall_footz_and_falls_match_raw'] = (np.isclose(floor_max,phase_report['summary']['maximum_abs_floor_force_n']) and np.isclose(wall_max,phase_report['summary']['maximum_wall_foot_normal_force_n']) and np.isclose(min_foot_z,phase_report['summary']['minimum_foot_body_origin_world_z_m']) and int(expected_fall.sum())==phase_report['summary']['observed_fall_controls'])
    stable_fraction = side_any[-20:].astype(float).mean(0)[0].tolist() if phase=='standing' else None
    com_valid = com_error <= REPORT['arguments']['com_tolerance_m']
    if phase=='standing':
        sensor_ok = com_valid and controls>=20 and all(v>=REPORT['arguments']['stable_support_fraction'] for v in stable_fraction)
        no_fall = not expected_fall.any()
        criterion = sensor_ok and no_fall
        result['standing_no_fall_criterion_passed'] = bool(no_fall)
    else:
        sensor_ok = com_valid and controls==4 and floor_max <= REPORT['arguments']['zero_force_tolerance_n'] and min_foot_z > 1.
        if phase=='airborne_wall': sensor_ok = sensor_ok and wall_max > REPORT['arguments']['threshold_n']
        criterion = sensor_ok
    pc['source_phase_sensor_and_original_criteria_match_independent_recomputation'] = bool(sensor_ok)==phase_report['summary']['support_sensor_criterion_passed'] and bool(criterion)==phase_report['summary']['criteria_passed']
    details = {'phase':phase,'path':str(path),'sha256':sha(path),'controls':controls,'substeps':controls*4,
               'first_physics_step':int(d['physics_step_before_read'].ravel()[0]),'last_physics_step':int(d['physics_step_before_read'].ravel()[-1]),
               'episode_id':phase_report['reset']['episode_id'],'reported_live_COM_error_m':phase_report['summary']['maximum_com_composition_error_m'],
               'independent_float64_COM_composition_error_m':com_error,'independent_massweighted_COM_error_m':weighted_error,
               'maximum_abs_floor_normal_force_n':floor_max,'maximum_wall_foot_normal_force_n':wall_max,
               'maximum_sum_of_four_foot_normal_vector_loads_n':float(loads.sum(-1).max()),
               'minimum_foot_body_origin_z_m':min_foot_z,'stable_last20_control_side_support_fraction':stable_fraction,
               'sensor_channel_criterion_passed':bool(sensor_ok),'original_full_phase_criterion_passed':bool(criterion),
               'fall_control_numbers_1based':(np.flatnonzero(expected_fall)+1).tolist(),
               'last_root_position_m':d['root_pos'][-1,0].tolist(),'last_up_z':float(up[-1]),
               'last_tilt_degrees':float(np.degrees(np.arccos(np.clip(up[-1],-1,1)))),'checks':pc}
    if phase=='standing':
        body_offsets = np.asarray(REPORT['fixed_PD_anchor']['stance_geometry']['geometry']['body_pos_relative_pelvis_m'])
        vertices = REPORT['fixed_PD_anchor']['stance_geometry']['geometry']['foot_collision_vertices_relative_pelvis_m']
        snapshots=[]
        for index in (0,19,39,49,59):
            footprints=[]
            for name,i in zip(FEET,foot_indices):
                local_points=np.asarray(vertices[name])-body_offsets[i]
                footprints.extend((rotate(q[index,-1,0,i],local_points)+pos[index,-1,0,i])[:,:2].tolist())
            footprints=np.asarray(footprints);w=d['mass_weighted_world_com'][index,-1,0]
            snapshots.append({'control_number':index+1,'COM_world_m':w.tolist(),
                              'foot_projected_all_collider_AABB_xy':np.r_[footprints.min(0),footprints.max(0)].tolist(),
                              'COM_outside_all_foot_projected_AABB':bool(((w[:2]<footprints.min(0))|(w[:2]>footprints.max(0))).any()),
                              'root_pos_m':d['root_pos'][index,0].tolist(),'foot_load_n':loads[index,-1,0].tolist()})
        details['standing_COM_and_footprint_snapshots']=snapshots
        details['footprint_limitation']='projected collision AABB proxy; not measured active support polygon or pressure COP'
    result['phases'].append(details)
    prior_physics += controls*4
    prior_control += controls

result['sensor_channel_criteria_passed'] = all(phase['sensor_channel_criterion_passed'] for phase in result['phases'])
result['overall_original_native_probe_passed'] = all(phase['original_full_phase_criterion_passed'] for phase in result['phases'])
checks['reported_sensor_summary_matches_raw'] = REPORT['sensor_channel_criteria_passed'] == result['sensor_channel_criteria_passed']
checks['reported_stance_and_overall_failures_match_raw'] = (REPORT['standing_no_fall_criterion_passed'] == result['standing_no_fall_criterion_passed'] and REPORT['native_physical_probe_passed'] == result['overall_original_native_probe_passed'] and REPORT['status']=='failed')
checks['runtime_protocol_zero_extra_steps_or_perstep_pose_writes'] = REPORT['protocol']['runtime_pose_writes_inside_phases']==REPORT['protocol']['extra_physics_steps']==0
all_checks = list(checks.values()) + [v for phase in result['phases'] for v in phase['checks'].values()]
result['independent_integrity_audit_passed']=all(all_checks)
result['independent_check_count']=len(all_checks)
result['independent_failed_checks']=[key for key,value in checks.items() if not value]+[phase['phase']+':'+key for phase in result['phases'] for key,value in phase['checks'].items() if not value]
result['scope_limits']=['Native submitted-step counter observed; PhysX engine simulation time not observed','Finite QA does not establish long learned-AMP gait quality, standing stability, paper locomotion success, or bitwise PhysX/Warp determinism','Floor forces are filtered normal-vector sums and exclude friction; velocity slip remains body-origin tangential-speed proxy','v1/v2 failures and v3 original all-criteria false remain unchanged']
out=ROOT/'docs/amp_improvement_20261001/foot_support_native_v3_validation.json'
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'path':str(out),'sha256':sha(out),'check_count':len(all_checks),'audit_passed':all(all_checks),'failed_checks':result['independent_failed_checks'],'sensor_channel_criteria_passed':result['sensor_channel_criteria_passed'],'standing_no_fall_criterion_passed':result['standing_no_fall_criterion_passed'],'original_native_allpass':result['overall_original_native_probe_passed'],'phases':[{k:v for k,v in phase.items() if k not in ('checks','standing_COM_and_footprint_snapshots')} for phase in result['phases']]},indent=2))
assert result['independent_integrity_audit_passed'], result['independent_failed_checks']
