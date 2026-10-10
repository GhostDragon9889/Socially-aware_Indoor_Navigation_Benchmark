"""SDK-free real-bank contract proof; does not simulate physics or rewrite data."""
from copy import copy
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from CrowdSim.humanoid.amass_corpus import AMASSCorpus, file_sha256
from CrowdSim.humanoid.training.mixed_trajectories import MixedTrajectoryCurriculum
from CrowdSim.humanoid.training.trajectories import PacerTrajectoryConfig, PacerTrajectoryCurriculum


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
MANIFEST = ROOT/'data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json'
checks = []


def check(name, condition):
    if not bool(condition):
        raise AssertionError(name)
    checks.append(name)


def eq_reference(a, b):
    for part in (0, 1):
        for name in vars(a[part]):
            check('restored_'+str(part)+'_'+name, torch.equal(getattr(a[part], name), getattr(b[part], name)))
    for part in (2, 3, 4):
        check('restored_reference_part_'+str(part), torch.equal(a[part], b[part]))


def main():
    torch.set_num_threads(4)
    global_rng = torch.get_rng_state().clone()
    corpus = AMASSCorpus(MANIFEST, device='cpu', seed=7, dof_velocity_frame='child_joint')
    config = PacerTrajectoryConfig(speed_max_mps=2, scalar_accel_max_mps2=1.5, vector_accel_max_mps2=1.5)
    mixed = MixedTrajectoryCurriculum(corpus, 256, bounds_xy=(-96, -96, 96, 96), procedural_config=config, seed=7)
    check('all_2848_selected_unchanged', len(mixed.amass.ids) == 2848 and torch.equal(mixed.amass.ids, corpus.ids))
    check('shared_readonly_bank', mixed.private_corpus.data is corpus.data)
    check('private_reset_rng', mixed.private_corpus.generator is not corpus.generator)
    ids = torch.randperm(256, generator=torch.Generator().manual_seed(912))
    actual = mixed.sample_reference(ids)
    expected_real, expected_proc = copy(corpus), copy(corpus)
    expected_real.generator = torch.Generator().manual_seed(8)
    expected_proc.generator = torch.Generator().manual_seed(7)
    for real, source, future in ((True, expected_real, 31), (False, expected_proc, 1)):
        selected = torch.nonzero((ids < 128) == real).flatten()
        expected = source.sample_reference(128, future_frames=future)
        for i in (3, 4):
            check(('real' if real else 'proc')+'_ordered_reference_'+str(i), torch.equal(actual[i][selected], expected[i]))
        check(('real' if real else 'proc')+'_ordered_root', torch.equal(actual[0].root_pos[selected], expected[0].root_pos))
        check(('real' if real else 'proc')+'_newest_first_history', torch.equal(actual[1].root_pos.reshape(256, 10, 3)[selected], expected[1].root_pos.reshape(128, 10, 3)))
        check(('real' if real else 'proc')+'_next_dof', torch.equal(actual[2][selected], expected[2]))
        check(('real' if real else 'proc')+'_actual_remaining_source', ((corpus.data['motion_num_frames'][actual[3][selected]]-1-actual[4][selected]) >= future).all())
    origins = torch.stack(((torch.arange(256)%16).float()*4-30, (torch.arange(256)//16).float()*4-30), -1)
    episodes = torch.ones(256, dtype=torch.long)
    mixed.bind_reset(ids, actual[3], actual[4], actual[0].root_pos[:, :2], origins[ids], episodes[ids])
    state = SimpleNamespace(root_pos=torch.cat((origins, torch.ones(256, 1)), -1), root_vel=torch.zeros(256, 3), root_rot=torch.tensor([[0., 0., 0., 1.]]).repeat(256, 1))
    ages = torch.zeros(256, dtype=torch.long)
    plans = mixed.build_plan(state, ages, episode_ids=episodes)
    plan_age = torch.zeros(256, dtype=torch.long)
    proc = PacerTrajectoryCurriculum(config, seed=7).build_plan(state, actor_ids=torch.arange(128, 256))
    check('procedural_canonical51_old_generator_exact', torch.equal(plans.xy[128:], proc.xy))
    check('real51_is_native151_every_third', torch.equal(plans.xy[:128], plans.native_xy[:128, ::3]))
    real_xy, _, _ = mixed.amass._sample_xy(mixed.amass.motion_ids.cpu(), mixed.amass.initial_frames.cpu()[:, None]+torch.arange(151), mixed.amass.offset_xy, mixed.amass.end_velocity_xy)
    check('real_native151_actual_source_exact', torch.equal(real_xy, plans.native_xy[:128]))
    for step in range(1, 16):
        ages += 1
        plan_age += 1
        clip_end = mixed.completed_step(ages, torch.zeros(256, dtype=torch.bool), torch.zeros(256, dtype=torch.bool))
        check('real_only_clip_end_'+str(step), not clip_end[128:].any())
        if step == 7:
            subset = torch.tensor([255, 0, 200, 61])
            sampled = mixed.sample_reference(subset)
            episodes[subset] += 1
            mixed.bind_reset(subset, sampled[3], sampled[4], sampled[0].root_pos[:, :2], origins[subset], episodes[subset])
            ages[subset] = 0
            plan_age[subset] = 15
            refs = torch.cat((mixed.amass.motion_ids, mixed.proc_reference_ids))
            mixed.validate_runner_state(ages, plan_age, plans.native_xy, refs, episodes, replan_steps=15, origins_xy=origins)
            updated = mixed.build_plan(state, ages, actor_ids=subset, episode_ids=episodes)
            plans.native_xy[subset] = updated.native_xy
            plans.xy[subset] = updated.xy
            plan_age[subset] = 0
        target = mixed.future_targets(ages, plan_age, plans.native_xy, episodes)
        check('actual_source_task_'+str(step), torch.equal(target[:128], mixed.amass.future_targets(ages[:128], episode_ids=episodes[:128])))
        seconds = plan_age[128:, None]/30+torch.arange(1, 11)[None]*.5
        f = (seconds*10).clamp(0, 50)
        lo = f.floor().long().clamp_max(49)
        row = torch.arange(128)[:, None]
        expected = torch.lerp(plans.xy[128:][row, lo], plans.xy[128:][row, lo+1], (f-lo)[..., None])
        check('old_proc_task_lerp_exact_'+str(step), torch.equal(target[128:], expected))
        query = torch.ones(256, 1)/30
        executed = mixed.interpolate_reference(query, plans.native_xy)
        check('old_proc_next30hz_reward_lerp_exact_'+str(step), torch.equal(executed[128:], torch.lerp(plans.xy[128:, 0], plans.xy[128:, 1], torch.full((128, 1), 1/30*10))[:, None]))
    check('group_actual_transition_equality', mixed.statistics['amass_actor_transitions'] == mixed.statistics['procedural_actor_transitions'] == 15*128)
    saved, rng = mixed.state_dict(), corpus.generator.get_state().clone()
    restored_corpus = copy(corpus)
    restored_corpus.generator = torch.Generator().manual_seed(987)
    restored = MixedTrajectoryCurriculum(restored_corpus, 256, bounds_xy=(-96, -96, 96, 96), procedural_config=config, seed=7)
    untouched = restored_corpus.generator.get_state().clone()
    restored.load_state_dict(saved)
    check('curriculum_restore_does_not_change_original_corpus_rng', torch.equal(untouched, restored_corpus.generator.get_state()))
    restored_corpus.generator.set_state(rng)
    refs = torch.cat((mixed.amass.motion_ids, mixed.proc_reference_ids))
    restored.validate_runner_state(ages, plan_age, plans.native_xy, refs, episodes, replan_steps=15, origins_xy=origins)
    eq_reference(mixed.sample_reference(torch.tensor([240, 31, 200, 8])), restored.sample_reference(torch.tensor([240, 31, 200, 8])))
    # Expert/reference original-stream draw remains the exact existing API.
    eq_reference(corpus.sample_reference(16), restored_corpus.sample_reference(16))
    next_plan, restored_plan = mixed.build_plan(state, ages, episode_ids=episodes), restored.build_plan(state, ages, episode_ids=episodes)
    check('all_group_restored_next_plan_exact', torch.equal(next_plan.xy, restored_plan.xy) and torch.equal(next_plan.native_xy, restored_plan.native_xy))
    check('global_torch_rng_unchanged', torch.equal(global_rng, torch.get_rng_state()))
    report = {'status': 'passed', 'checks': len(checks), 'check_names': checks, 'cuda_available_used': False,
              'num_envs': 256, 'amass_actors': 128, 'procedural_actors': 128, 'actual_control_steps': 15,
              'transitions_per_group': 1920, 'manifest_path': str(MANIFEST.relative_to(ROOT)),
              'manifest_sha256': file_sha256(MANIFEST), 'bank_sha256_declared_not_rehashed': corpus.entry['sha256'],
              'bank_bytes_actual': Path(corpus.entry['path']).stat().st_size,
              'module_sha256': file_sha256(ROOT/'CrowdSim/humanoid/training/mixed_trajectories.py'),
              'tests_sha256': file_sha256(ROOT/'tests/test_mixed_trajectories.py'),
              'statistics': mixed.statistics, 'config': mixed.config,
              'scope': 'CPU command/source/RNG/resume proof over actual mmap v7; no GPU/Kit/physics or gait/closedloop quality evidence; no 34 test outputs read'}
    (OUT/'actual_v7.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'status': report['status'], 'checks': report['checks'], 'module_sha256': report['module_sha256']}))


if __name__ == '__main__':
    main()
