"""Independent CPU admission of actual training and its deployment export."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from CrowdSim.humanoid.amp.controller import PacerAMPController, NO_SHAPE_SHA256
from CrowdSim.humanoid.training.pacer_finetune import AnchoredPacerTrainer


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    parser.error('Fresh CPU audit output required')
torch.set_num_threads(4)
receipt = json.loads(args.run.with_name(args.run.name+'_execution.json').read_text())
result_path = args.run/'result.json'
result = json.loads(result_path.read_text())
if (receipt['actual_returncode'] != 0 or receipt['all_sources_unchanged'] is not True
        or receipt['result_sha256'] != digest(result_path) or result['status'] != 'passed'
        or result['cleanup_succeeded'] is not True):
    raise ValueError('Actual completed training with ordinary exit0 is required')
for name in ('last.pt', 'Humanoid_AMASS.pth', 'metrics.jsonl'):
    if receipt['files'][name]['sha256'] != digest(args.run/name):
        raise ValueError('Completed training artifact changed: '+name)
rows = [json.loads(line) for line in (args.run/'metrics.jsonl').read_text().splitlines()]
if len(rows) != result['updates'] or [r['update_count'] for r in rows] != list(range(1, len(rows)+1)):
    raise ValueError('Training log must include every committed optimizer update')
manifest = ROOT/'data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json'
ids = {row['motion_id'] for row in json.loads(manifest.read_text())['splits']['train']['clips']}
if (set(result['expert_unique_motion_ids']) != ids or len(ids) != 2848
        or result['manifest_sha256'] != digest(manifest)
        or rows[-1]['physical_transitions'] != result['physical_transitions']
        or rows[-1]['falls'] != result['falls']):
    raise ValueError('Fixed AMASS population or final training clock differs')
limits = torch.ones(69)
reference = PacerAMPController.from_checkpoint(ROOT/'data/pretrained_models/paper_references/pacer/output/release/pacer_no_shape/Humanoid.pth', -limits, limits)
trained = PacerAMPController.from_checkpoint(args.run/'Humanoid_AMASS.pth', -limits, limits,
    expected_sha256=receipt['files']['Humanoid_AMASS.pth']['sha256'])
trainer, _ = AnchoredPacerTrainer.load(args.run/'last.pt', device='cpu', restore_rng=False)
for name in ('observation_normalizer', 'value_normalizer', 'discriminator_normalizer'):
    if any(not torch.equal(v, getattr(trained, name).state_dict()[k])
           for k, v in getattr(reference, name).state_dict().items()):
        raise ValueError('Public observation/value/AMP moments changed: '+name)
torch.manual_seed(814)
raw = (reference.observation_normalizer.running_mean.float()[None]
       + torch.randn(16, 1401)*reference.observation_normalizer.running_var.float().sqrt()[None])
normalized = trained.observation_normalizer(raw)
with torch.no_grad():
    own, task = trainer._observations(raw[:, :357], raw[:, 357:])
    distribution, values = trainer.model.policy(own, task)
    mean, log_std = trained.network.eval_actor(normalized)
    critic = trained.network.eval_critic(normalized).squeeze(-1)
    history = torch.randn(16, 10, 195)
    student_d = trainer.model.disc_logits(history, torch.zeros(16, 69))
    exported_d = trained.network.eval_disc(history.flatten(1)).squeeze(-1)
if not all((torch.equal(distribution.mean, mean), torch.equal(values, critic),
            torch.equal(student_d, exported_d), torch.equal(normalized, torch.cat((own, task), -1)))):
    raise ValueError('Full actor/critic/discriminator/normalization export equivalence failed')
changes = {group: max(float((v-reference.network.state_dict()[k]).abs().max())
                     for k, v in trained.network.state_dict().items() if k.startswith(group))
           for group in ('_task_mlp', 'actor_mlp', 'critic_mlp', 'mu', 'value', '_disc_mlp', '_disc_logits')}
if not all(value > 0 for value in changes.values()):
    raise ValueError('A declared fine tuned network did not actually change')
report = {'schema': 'crowdsim.amass_finetune_CPU_audit.v1', 'status': 'passed',
          'training_result_sha256': digest(result_path), 'parent_sha256': NO_SHAPE_SHA256,
          'checkpoint_sha256': trained.checkpoint_sha256, 'manifest_sha256': digest(manifest),
          'updates': len(rows), 'physical_transitions': result['physical_transitions'],
          'all2848_expert_ids_verified': True, 'all_four_normalizers_exactly_frozen': True,
          'export_actor_critic_discriminator_and_normalization_exact': True,
          'weight_max_abs_deltas': changes, 'actual_training_falls': result['falls'],
          'teacher_not_used_at_inference': True, 'physical_quality_evaluation_required': True}
with args.output.open('x') as stream:
    stream.write(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
