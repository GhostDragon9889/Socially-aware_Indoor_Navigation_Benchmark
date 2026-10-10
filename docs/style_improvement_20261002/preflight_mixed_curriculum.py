"""Pure resource/profile gate; prints the exact authorized GPU training argv."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='output/diffusion_amp/amp_style_mixed_curriculum_v1')
    parser.add_argument('--updates', type=int, default=2000)
    parser.add_argument('--num-envs', type=int, default=256)
    parser.add_argument('--resume')
    parser.add_argument('--report', type=Path, required=True)
    request = parser.parse_args()
    from CrowdSim.humanoid.application.configuration import load_amp_profile, parse_amp_args
    from CrowdSim.humanoid.application.resources import preflight_amp_resources
    from CrowdSim.humanoid.application.settings import TrainingConfig
    argv = ['--config', 'configs/diffusion_amp/amp_style_mixed_amass_procedural.yaml',
            '--manifest', 'data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json',
            '--diffusion-checkpoint', 'output/diffusion_amp/social_diffusion_training_extended_v2/step_5000.ckpt',
            '--output', request.output, '--trajectory-source', 'mixed_amass_procedural',
            '--num-envs', str(request.num_envs), '--updates', str(request.updates), '--rollout-steps', '32',
            '--episode-steps', '300', '--replan-steps', '15', '--synchronized-replanning',
            '--terrain-extent-m', '192', '--spawn-margin-m', '48', '--curriculum-mode', 'smooth',
            '--curriculum-speed-max-mps', '2', '--curriculum-accel-max-mps2', '1.5',
            '--power-source', 'projected_joint_effort_substep_mean', '--physics-only', '--seed', '7']
    if request.resume:
        argv += ['--resume', request.resume]
    args = parse_amp_args(argv)
    profile = load_amp_profile(args.config)
    resources = preflight_amp_resources(args, profile)
    baseline = load_amp_profile('configs/diffusion_amp/amp_style_amass_backward_penalty.yaml')
    if asdict(TrainingConfig(**profile['training'])) != asdict(TrainingConfig(**baseline['training'])):
        raise ValueError('Mixed core learning config differs from negative baseline')
    if any(name == 'torch' or name.startswith(('isaacsim', 'isaaclab', 'omni', 'pxr')) for name in sys.modules):
        raise RuntimeError('Pure preflight unexpectedly imported Torch/SDK')
    digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = {'schema': 'fixed_slots_mixed.resource_preflight.v1', 'status': 'passed',
              'Torch_imported': False, 'Kit_started': False, 'GPU_used': False,
              'script_sha256': digest, 'argv': argv,
              'command': shlex.join(['../IsaacLab-Protomotions/.venv/bin/python', 'scripts/train_paper_amp.py', *argv]),
              'resources': resources.as_dict(), 'canonical_baseline_training_config_equal': True,
              'planned_this_stage_total_transitions': args.num_envs*args.rollout_steps*args.updates,
              'planned_this_stage_each_group_transitions': args.num_envs//2*args.rollout_steps*args.updates,
              'counts_are_planned_not_claimed_executed': True,
              'no_test_metrics_used_for_recipe': True,
              'restore_tensor_schema_and_native_physics_validation_still_required': True}
    with request.report.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'passed', 'command': result['command'],
                      'planned_total_transitions': result['planned_this_stage_total_transitions']}, indent=2))


if __name__ == '__main__':
    main()
