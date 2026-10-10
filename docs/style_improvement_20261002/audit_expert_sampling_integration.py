"""Independent CPU contract review; no Isaac SDK imports or simulation."""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
import tempfile
import time

import torch

from CrowdSim.humanoid.amass_corpus import AMASSCorpus
from CrowdSim.humanoid.amp import action_to_pd, build_amp_observation
from CrowdSim.humanoid.amp.observations import PacerSMPLMapping
from CrowdSim.humanoid.application.settings import TrainingConfig
from CrowdSim.humanoid.training.expert_sampling import ExpertReferenceSampler, STRATA
from CrowdSim.humanoid.training.trainer import PaperAMPTrainer


ROOT = Path('/project/EXP/SangforAndSaut/ProtoMotions')
checks = []


def check(name, passed):
    passed = bool(passed)
    checks.append({'name': name, 'passed': passed})
    if not passed:
        raise AssertionError(name)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu_world_class():
    source = ast.parse((ROOT/'CrowdSim/humanoid/amp_world.py').read_text())
    world = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == 'AMPWorld')
    names = {'expert_batch', 'configure_expert_sampling', 'expert_sampling_state_dict', 'load_expert_sampling_state_dict'}
    methods = [deepcopy(node) for node in world.body if isinstance(node, ast.FunctionDef) and node.name in names]
    module = ast.Module(body=[ast.ClassDef(name='CPUExpertWorld', bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    namespace = {'torch': torch, 'build_amp_observation': build_amp_observation, 'action_to_pd': action_to_pd}
    exec(compile(ast.fix_missing_locations(module), str(ROOT/'CrowdSim/humanoid/amp_world.py'), 'exec'), namespace)
    return namespace['CPUExpertWorld']


def make_world(cls, corpus):
    world = cls()
    world.corpus = corpus
    world.device = torch.device('cpu')
    world.mapping = PacerSMPLMapping()
    world.history_frames = 10
    world.lower = torch.full((69,), -1.)
    world.upper = torch.ones(69)
    return world


def expected_legacy(corpus, world, batch):
    _, history, next_position, _, _ = corpus.sample_reference(batch, 10)
    features = build_amp_observation(history, mapping=world.mapping).reshape(batch, 10, 195)
    scale = action_to_pd(torch.ones(batch, 69), world.lower, world.upper, dof_names=world.mapping.dof_names)
    scale = scale[:, world.mapping.dof_indices]
    next_position = next_position[:, world.mapping.dof_indices]
    raw = torch.where(scale.abs() > 1e-8, next_position / scale.clamp_min(1e-8), torch.zeros_like(next_position)).clamp(-1, 1)
    return features, raw


def compare_reference(left, right, name):
    for position in (0, 1):
        for field in ('rigid_body_pos', 'rigid_body_rot', 'rigid_body_vel', 'rigid_body_ang_vel', 'dof_pos', 'dof_vel'):
            check(f'{name}: state{position}.{field}', torch.equal(getattr(left[position], field), getattr(right[position], field)))
    for position in (2, 3, 4):
        check(f'{name}: tuple{position}', torch.equal(left[position], right[position]))


def run(output):
    started = time.monotonic()
    fixture = runpy.run_path(str(ROOT/'tests/test_expert_window_sampling.py'))['make_corpus']
    with tempfile.TemporaryDirectory(prefix='expert-sampler-independent-review-') as temporary:
        directory = Path(temporary)
        raw = fixture(directory, [{'speed': 0.}, {'speed': 2.3},
            {'speed': 1., 'movement_angle': torch.pi/2}, {'speed': 1., 'yaw_rate': 1.}, {'speed': 1.}])
        data = raw.data
        count = len(data['gvs'])
        data['gts'] = torch.zeros(count, 24, 3)
        data['gts'][..., 2] = 1.
        data['gts'][:, :, 0] = torch.arange(count)[:, None]*.01
        data['gavs'] = torch.zeros(count, 24, 3)
        data['dps'] = torch.arange(count)[:, None]*.001+torch.arange(69)[None]*.0001
        data['dvs'] = torch.zeros(count, 69)
        data['dof_velocity_frame'] = 'child_joint'
        bank_path = directory/'bank.pt'
        torch.save(data, bank_path)
        raw.entry.update(bytes=bank_path.stat().st_size, sha256=digest(bank_path), dof_velocity_frame='child_joint')
        for row in raw.entry['clips']:
            row['diffusion_eligible'] = False
        raw.manifest_path.write_text(json.dumps(raw.manifest))
        config = {'mode': 'stratified_windows', 'seed': 29,
                  'target_weights': {name: float(name == 'fast') for name in STRATA},
                  'minimum_parents_per_bucket': 1, 'minimum_windows_per_bucket': 1}
        cls = cpu_world_class()
        baseline = AMASSCorpus(raw.manifest_path, seed=71, dof_velocity_frame='child_joint')
        enabled_corpus = AMASSCorpus(raw.manifest_path, seed=71, dof_velocity_frame='child_joint')
        base_world = make_world(cls, baseline)
        enabled = make_world(cls, enabled_corpus)
        baseline_before = baseline.generator.get_state().clone()
        global_before = torch.get_rng_state().clone()
        enabled.configure_expert_sampling(config)
        check('construction preserves shared corpus RNG', torch.equal(enabled_corpus.generator.get_state(), baseline_before))
        check('construction preserves global Torch CPU RNG', torch.equal(torch.get_rng_state(), global_before))
        check('private generator is independent', enabled.expert_reference_sampler.generator is not enabled_corpus.generator)
        for index, (reset_batch, expert_batch) in enumerate(((4, 64), (2, 17), (7, 91), (1, 32))):
            compare_reference(baseline.sample_reference(reset_batch, 10), enabled_corpus.sample_reference(reset_batch, 10), f'paired reset {index}')
            legacy_features, _ = base_world.expert_batch(expert_batch)
            expert_features, _ = enabled.expert_batch(expert_batch)
            check(f'paired expert {index} shared RNG state', torch.equal(baseline.generator.get_state(), enabled_corpus.generator.get_state()))
            check(f'replacement {index} actually uses fast windows', torch.allclose(expert_features[..., 6], torch.full((expert_batch, 10), 2.3), atol=1e-6, rtol=0))
            check(f'replacement {index} differs from discarded legacy', not torch.equal(legacy_features, expert_features))
        compare_reference(baseline.sample_reference(64, 10), enabled_corpus.sample_reference(64, 10), 'paired following reset')
        check('all paired draws preserve global Torch CPU RNG', torch.equal(torch.get_rng_state(), global_before))
        direct = AMASSCorpus(raw.manifest_path, seed=303)
        disabled_corpus = AMASSCorpus(raw.manifest_path, seed=303)
        disabled = make_world(cls, disabled_corpus)
        check('disabled configuration returns None', disabled.configure_expert_sampling(None) is None)
        expected = expected_legacy(direct, disabled, 43)
        actual = disabled.expert_batch(43)
        check('disabled features exactly equal direct legacy method', torch.equal(expected[0], actual[0]))
        check('disabled actions exactly equal direct legacy method', torch.equal(expected[1], actual[1]))
        check('disabled shared RNG exactly equals direct legacy method', torch.equal(direct.generator.get_state(), disabled_corpus.generator.get_state()))
        check('disabled omits sampler state', disabled.expert_sampling_state_dict() is None)
        disabled.load_expert_sampling_state_dict(None)
        try:
            disabled.load_expert_sampling_state_dict(enabled.expert_sampling_state_dict())
        except ValueError:
            check('disabled rejects enabled resume state', True)
        else:
            check('disabled rejects enabled resume state', False)
        state = enabled.expert_sampling_state_dict()
        clone = ExpertReferenceSampler(enabled_corpus, config)
        clone.load_state_dict(state)
        draw = clone.sampler.sample(37)
        current, history, next_position, motion_ids, frames = enabled.expert_reference_sampler.sample_reference(37, 10)
        check('history is exact newest-first sampled data', torch.equal(history.dof_pos.reshape(37, 10, 69), data['dps'][draw.history_indices]))
        check('next pose is exact current+1 sampled data', torch.equal(next_position, data['dps'][draw.next_indices]))
        check('current is exact selected center', torch.equal(current.dof_pos, data['dps'][draw.absolute_indices]))
        check('sample IDs and relative frames remain exact', torch.equal(motion_ids, draw.motion_ids) and torch.equal(frames, draw.frame_indices))
        check('history chronological markers decrease', torch.all(history.dof_pos.reshape(37, 10, 69)[:, :-1, 0] > history.dof_pos.reshape(37, 10, 69)[:, 1:, 0]))
        try:
            enabled.load_expert_sampling_state_dict(None)
        except ValueError:
            check('enabled rejects incomplete resume rather than reseed', True)
        else:
            check('enabled rejects incomplete resume rather than reseed', False)
        trainer = PaperAMPTrainer(TrainingConfig(task_layers=(8,), policy_layers=(8,), discriminator_layers=(8,), expert_sampling=config), device='cpu')
        saved_runner = {'corpus_rng': enabled_corpus.generator.get_state().clone(), 'expert_sampling': enabled.expert_sampling_state_dict()}
        checkpoint = directory/'trainer.pt'
        trainer.save(checkpoint, runner_state=saved_runner)
        saved_global = torch.get_rng_state().clone()
        expected_reset = enabled_corpus.sample_reference(13, 10)
        expected_expert = enabled.expert_batch(71)
        expected_following = enabled_corpus.sample_reference(9, 10)
        expected_shared_rng = enabled_corpus.generator.get_state().clone()
        expected_private_rng = enabled.expert_reference_sampler.generator.get_state().clone()
        loaded_trainer, loaded_runner = PaperAMPTrainer.load(checkpoint, device='cpu', restore_rng=False)
        restored_corpus = AMASSCorpus(raw.manifest_path, seed=999, dof_velocity_frame='child_joint')
        restored = make_world(cls, restored_corpus)
        restored.configure_expert_sampling(loaded_trainer.config.expert_sampling)
        restored_corpus.generator.set_state(loaded_runner['corpus_rng'].cpu())
        restored.load_expert_sampling_state_dict(loaded_runner['expert_sampling'])
        loaded_trainer.restore_rng()
        check('real checkpoint restores global RNG after setup', torch.equal(torch.get_rng_state(), saved_global))
        compare_reference(expected_reset, restored_corpus.sample_reference(13, 10), 'checkpoint first reset')
        actual_expert = restored.expert_batch(71)
        check('real checkpoint next expert features exact', torch.equal(expected_expert[0], actual_expert[0]))
        check('real checkpoint next expert actions exact', torch.equal(expected_expert[1], actual_expert[1]))
        compare_reference(expected_following, restored_corpus.sample_reference(9, 10), 'checkpoint following reset')
        check('real checkpoint shared corpus RNG continuation exact', torch.equal(restored_corpus.generator.get_state(), expected_shared_rng))
        check('real checkpoint private expert RNG continuation exact', torch.equal(restored.expert_reference_sampler.generator.get_state(), expected_private_rng))
        check('real checkpoint private counters continuation exact', restored.expert_sampling_state_dict()['sampler']['sampled_windows'] == enabled.expert_sampling_state_dict()['sampler']['sampled_windows'])
        result = {'status': 'passed', 'validation': 'actual CPU AMASSCorpus and PaperAMPTrainer save/load; current AMPWorld expert methods AST-extracted to avoid SDK imports',
                  'sdk_started': False, 'physical_simulation_validated': False, 'fixture': 'synthetic finite small bank, five known root-kinematic classes',
                  'checks': checks, 'check_count': len(checks), 'elapsed_s': time.monotonic()-started,
                  'sources': {str(path.relative_to(ROOT)): digest(path) for path in [ROOT/'CrowdSim/humanoid/amp_world.py', ROOT/'CrowdSim/humanoid/training/expert_sampling.py', ROOT/'CrowdSim/humanoid/application/settings.py', ROOT/'CrowdSim/humanoid/amass_corpus.py', ROOT/'CrowdSim/humanoid/training/trainer.py', ROOT/'scripts/train_paper_amp.py']}}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({'status': result['status'], 'checks': len(checks), 'elapsed_s': result['elapsed_s'], 'output': str(output)}))


if __name__ == '__main__':
    import sys
    run(Path(sys.argv[1]))
