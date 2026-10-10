"""Actual fixed2848 CPU interface probe, not a PhysX or training result."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path('/project/EXP/SangforAndSaut/ProtoMotions')
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
from CrowdSim.humanoid.amass_corpus import AMASSCorpus  # noqa: E402
from CrowdSim.humanoid.training.mixed_trajectories import MixedTrajectoryCurriculum  # noqa: E402
from CrowdSim.humanoid.training.minimum_horizon_curriculum import MinimumHorizonMixedCurriculum  # noqa: E402
from CrowdSim.humanoid.training.trajectories import PacerTrajectoryConfig  # noqa: E402
from scripts.diagnose_shared_social_training import identity  # noqa: E402


def check(value, message):
    if not value:
        raise ValueError(message)


check(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'Explicit CPU-only probe required')
output = Path(__file__).with_name('actual_corpus_probe.json')
check(not output.exists(), 'Preserve previous probe output')
horizon_path = ROOT / 'docs/style_improvement_20261006/shared_social_episode_horizon_CPU_v1/actual_result.json'
horizon_record = identity(horizon_path)
check(horizon_record['sha256'] == '37c225599e365ef4cdd8f197046009787a1de6ce99fc251c1da2926cb9b2850a', 'Actual horizon evidence changed')
horizon = json.loads(horizon_path.read_text())
manifest = ROOT / 'data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json'
inputs = [horizon_record] + [row for row in horizon['inputs'] if row['path'] == str(manifest) or row['sha256'] == '46e0241108e200409c08d1115a270205fe0b6ce827e366a06e8a09f6a2cc82c0']
check(len(inputs) == 3 and all(identity(row['path']) == row for row in inputs), 'Fixed2848 inputs changed')
sources = [identity(path) for path in (Path(__file__), ROOT / 'CrowdSim/humanoid/training/minimum_horizon_curriculum.py',
    ROOT / 'CrowdSim/humanoid/training/mixed_trajectories.py', ROOT / 'CrowdSim/humanoid/training/amass_trajectories.py',
    ROOT / 'CrowdSim/humanoid/training/trajectories.py', ROOT / 'CrowdSim/humanoid/amass_corpus.py')]
rng = torch.random.get_rng_state().clone()
before = AMASSCorpus(manifest, device='cpu', seed=7, dof_velocity_frame='child_joint')
after = AMASSCorpus(manifest, device='cpu', seed=7, dof_velocity_frame='child_joint')
baseline = MixedTrajectoryCurriculum(before, 256, procedural_config=PacerTrajectoryConfig(), seed=7)
candidate = MinimumHorizonMixedCurriculum(after, 256, procedural_config=PacerTrajectoryConfig(), seed=7)
check(torch.equal(before.ids, after.ids) and before.ids.tolist() == list(range(2848)), 'No source filtering allowed')
actors = torch.arange(256)
old, new = baseline.sample_reference(actors), candidate.sample_reference(actors)
check(torch.equal(old[3], new[3]) and torch.equal(old[4][128:], new[4][128:]), 'ID/procedural per-call draw order changed')
expected_rng = torch.Generator().manual_seed(8)
expected_ids = before.ids[torch.randint(2848, (128,), generator=expected_rng)]
unit = torch.rand(128, generator=expected_rng)
count = before.data['motion_num_frames'][expected_ids]
check(torch.equal(old[3][:128], expected_ids) and torch.equal(new[3][:128], expected_ids), 'Actual AMASS reset IDs differ')
for future, sampled in ((31, old), (81, new)):
    check(torch.equal(sampled[4][:128], 9 + (unit * (count - 9 - future)).long()), 'Actual reset frame formula changed')
check(torch.equal(baseline.private_corpus.generator.get_state(), candidate.private_corpus.generator.get_state())
      and torch.equal(before.generator.get_state(), after.generator.get_state()), 'Per-call source RNG consumption changed')
remaining = count - 1 - new[4][:128]
check(bool((remaining >= 81).all()), 'Actual reset lacks81 real future controls')
# Use the same legal source phase for both instances to isolate command logic.
# These are source tensors; no physical states or contacts are simulated.
root = new[0].root_pos[:, :2]
episodes, ages = torch.ones(256, dtype=torch.long), torch.zeros(256, dtype=torch.long)
for mixed in (baseline, candidate):
    mixed.bind_reset(actors, new[3], new[4], root, root, episodes)
state = SimpleNamespace(root_pos=new[0].root_pos, root_vel=new[0].root_vel, root_rot=new[0].root_rot)
plans = [mixed.build_plan(state, ages, episode_ids=episodes) for mixed in (baseline, candidate)]
check(torch.equal(plans[0].xy, plans[1].xy) and torch.equal(plans[0].native_xy, plans[1].native_xy), 'Same-binding native plans changed')
targets = [mixed.future_targets(ages, ages, plans[i].native_xy, episode_ids=episodes)
           for i, mixed in enumerate((baseline, candidate))]
check(torch.equal(*targets) and targets[0].shape == (256, 10, 2), 'Same-binding native target points changed')
expert = [c.sample_reference(4096, future_frames=1) for c in (before, after)]
for slot in (0, 1):
    for field, value in vars(expert[0][slot]).items():
        check(torch.equal(value, getattr(expert[1][slot], field)), 'Per-call expert source tensor changed: ' + field)
for slot in (2, 3, 4):
    check(torch.equal(expert[0][slot], expert[1][slot]), 'Per-call expert source IDs/frames/next marker changed')
check(torch.equal(before.generator.get_state(), after.generator.get_state()), 'Per-call expert source RNG differs')
saved = deepcopy(candidate.state_dict())
recovered = MinimumHorizonMixedCurriculum(AMASSCorpus(manifest, device='cpu', seed=7, dof_velocity_frame='child_joint'),
                                        256, procedural_config=PacerTrajectoryConfig(), seed=7)
recovered.load_state_dict(saved)
check(torch.equal(recovered.amass.remaining_steps, candidate.amass.remaining_steps)
      and recovered.describe() == candidate.describe(), 'CPU candidate holder roundtrip changed')
check(torch.equal(torch.random.get_rng_state(), rng) and not torch.cuda.is_initialized(), 'CPU probe consumed global RNG or CUDA')
check(not any(name.split('.')[0] in {'isaacsim', 'isaaclab', 'omni', 'pxr', 'warp'} for name in sys.modules), 'SDK imported')
check(all(identity(row['path']) == row for row in inputs + sources), 'Input/source changed during actual probe')
report = {'schema': 'shared_social.actual_fixed2848_future81_CPU_interface_probe.v1', 'status': 'passed_CPU_interfaces',
    'inputs': inputs, 'sources': sources, 'clips_retained': len(after.ids), 'actors': 256, 'seed': 7,
    'AMASS_reference_rows': 128, 'procedural_reference_rows': 128,
    'actual_AMASS_reset_ID_draws_same': True, 'actual_procedural_reset_frames_same': True,
    'minimum_actual_remaining_frames': int(remaining.min()), 'mean_actual_remaining_frames': float(remaining.double().mean()),
    'AMASS_reset_frames_changed_rows': int((old[4][:128] != new[4][:128]).sum()),
    'actual_frame_formula_exact': True, 'per_call_source_RNG_consumption_same': True,
    'same_binding_native_plan_and_ten_point_targets_exact': True,
    'per_call_expert_source_windows_exact': 4096,
    'native_target_bytes_sha256': hashlib.sha256(targets[0].numpy().tobytes()).hexdigest(),
    'CPU_holder_state_roundtrip_passed': True, 'global_RNG_consumed': False, 'CUDA_initialized': False,
    'NN_constructed': False, 'SDK_started': False, 'actual_training_executed': False,
    'original_World_admits_this_new_type': False,
    'scope': 'CPU source ABI and per-call rules only. Actual episode/reset frequencies and expert RNG sequence may change after physics; not a full training RNG or exposure match.',
    'paper_effect_achieved': False, 'default_model_promoted': False}
with output.open('x') as stream:
    stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps({'result': identity(output), 'clips_retained': report['clips_retained'],
    'minimum_actual_remaining_frames': report['minimum_actual_remaining_frames'],
    'AMASS_reset_frames_changed_rows': report['AMASS_reset_frames_changed_rows'],
    'per_call_expert_source_windows_exact': 4096, 'SDK_started': False}))
