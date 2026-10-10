"""Unified source-checkout commands; native execution retains its owning launcher."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .paths import repository_root

SCENES = ('YOHO', 'NewMetro', 'SecondFloor', 'ThirdFloor')
TOOLS = {
    'train-amp': 'finetune_pacer_amass_verified.py',
    'record': 'record_pedestrian_coverage.py',
    'evaluate-speed': 'evaluate_trace_pace_speed_coverage.py',
    'evaluate-posture': 'evaluate_trace_pace_posture.py',
    'evaluate-physics': 'evaluate_trace_pace_reliability.py',
    'evaluate-stasis': 'evaluate_trace_pace_spatial_stasis.py',
    'audit-speed': 'audit_trace_pace_substep_speed.py',
    'audit-feet': 'audit_trace_pace_foot_surface.py',
    'audit-camera': 'audit_trace_pace_camera_skin.py',
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('status', help='Show project locations and the latest completed release')
    for command in ('preflight', 'simulate'):
        item = sub.add_parser(command, help='Resource-only preflight' if command == 'preflight' else 'Owned native physical execution')
        item.add_argument('--scene', choices=SCENES, default='YOHO')
        item.add_argument('--output', type=Path)
        item.add_argument('--steps', type=int, default=1800)
        item.add_argument('--seed', type=int, default=7)
    for command, filename in TOOLS.items():
        sub.add_parser(command, add_help=False, help=f'Forward arguments to {filename}')
    for command in ('baseline-train', 'baseline-evaluate'):
        sub.add_parser(command, add_help=False, help='Simulator-free PointGoal PPO')
    args, forwarded = parser.parse_known_args(argv)
    root = repository_root()
    if forwarded and forwarded[0] == '--':
        forwarded = forwarded[1:]
    if args.command == 'status':
        if forwarded:
            parser.error('status takes no additional arguments')
        release = root / 'docs/four_scene_speed_final_delivery_20261010.json'
        print(json.dumps({'project': 'Socially-aware_Indoor_Navigation_Benchmark',
                          'root': str(root), 'source_root': str(root / 'src'),
                          'python': sys.executable, 'latest_completed_release': str(release),
                          'release_available': release.is_file()}, ensure_ascii=False, indent=2))
        return 0
    if args.command in ('preflight', 'simulate'):
        if forwarded:
            parser.error('Unexpected arguments: ' + ' '.join(forwarded))
        config = root / 'configs/diffusion_amp/four_scene_speed_20261010'
        output = args.output or root / 'output/interactive' / (
            f'{args.command}_{args.scene}_{datetime.now():%Y%m%d_%H%M%S}_{time.time_ns()}'
        )
        native = ['--config', str(config / (args.scene + '_profile.json')),
                  '--scene', str(config / (args.scene + '_scene.yaml')),
                  '--steps', str(args.steps), '--measure-contacts', '--seed', str(args.seed)]
        if args.command == 'preflight':
            command = [sys.executable, str(root / 'scripts/run_trace_pace.py'),
                       '--output', str(output), *native, '--preflight-only']
        else:
            command = [sys.executable, str(root / 'scripts/run_trace_pace_verified.py'),
                       '--output', str(output), '--', *native]
        print(json.dumps({'output': str(output), 'sdk_requested': args.command == 'simulate'}), flush=True)
    elif args.command.startswith('baseline-'):
        module = 'crowdsim_ppo.train' if args.command == 'baseline-train' else 'crowdsim_ppo.evaluate'
        command = [sys.executable, '-m', module, *forwarded]
    else:
        command = [sys.executable, str(root / 'scripts' / TOOLS[args.command]), *forwarded]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = os.pathsep.join([str(root / 'src'), str(root), env.get('PYTHONPATH', '')])
    child = subprocess.Popen(command, cwd=root, env=env)
    code = child.wait()
    if code == 0 and args.command == "preflight":
        from .preflight import verify_native_scene_config
        scene_report = verify_native_scene_config(config / (args.scene + "_scene.yaml"))
        (output / "scene_resources.json").write_text(json.dumps(scene_report, indent=2) + "\n")
        print(json.dumps(scene_report), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
