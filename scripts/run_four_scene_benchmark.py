#!/usr/bin/env python3
"""Run the repository's four-scene physical, quality and optional video pipeline.

Every job uses the active interpreter, checkout sources and fresh outputs.
A failed quality gate stops recording and preserves the physical evidence.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenes', nargs='+', choices=('YOHO','NewMetro','SecondFloor','ThirdFloor'),
                   default=['YOHO','NewMetro','SecondFloor','ThirdFloor'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--record', action='store_true', help='Record 1800 actual frames only after every quality gate passes')
    p.add_argument('--preflight-only', action='store_true')
    a = p.parse_args()
    if a.record and a.preflight_only:
        p.error('--record and --preflight-only cannot be combined')
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, SOCIAL_NAVIGATION_ROOT=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH'] = os.pathsep.join((str(ROOT/'src'), str(ROOT)))
    cfg = ROOT/'configs/reproduction/four_scene'
    actors = json.loads((cfg/'camera_actors.json').read_text())
    receipt = {'schema':'social_navigation.four_scene_pipeline.v1','status':'running',
               'root':str(ROOT),'scenes':a.scenes,'seed':a.seed,'jobs':[],
               'author_protocol':False,'full_scene_coverage_claimed':False}
    def save():
        (a.output/'pipeline.json').write_text(json.dumps(receipt, indent=2)+'\n')
    def run(command):
        receipt['jobs'].append({'command':list(map(str,command)),'actual_returncode':None})
        save()
        code = subprocess.run(receipt['jobs'][-1]['command'], cwd=ROOT, env=env).returncode
        receipt['jobs'][-1]['actual_returncode'] = code
        save()
        if code:
            raise RuntimeError('Failed job; evidence retained: '+str(command[1]))
    def tool(name, *args):
        run([sys.executable, ROOT/'scripts'/name, *args])
    try:
        for scene in a.scenes:
            base = a.output/scene
            base.mkdir()
            physical = base/'physical'
            profile, geometry, camera = (cfg/(scene+s) for s in ('_profile.json','_scene.yaml','_cameras.json'))
            controls = ['--config',profile,'--scene',geometry,'--steps','1800',
                        '--measure-contacts','--seed',str(a.seed)]
            if a.preflight_only:
                tool('run_trace_pace.py','--output',physical,*controls,'--preflight-only')
                run([sys.executable,'-m','social_navigation','preflight','--scene',scene,
                     '--output',base/'scene_preflight'])
                continue
            tool('run_trace_pace_verified.py','--output',physical,'--',*controls)
            quality, posture, speed, stasis, foot, substep, framing = (
                base/(n+'.json') for n in ('quality','posture','speed','stasis','foot','substep','framing'))
            tool('evaluate_trace_pace_reliability.py','--run',physical,
                 '--execution',base/'physical_execution.json','--output',quality)
            tool('evaluate_trace_pace_posture.py','--run',physical,'--output',posture,'--require-walking-screen')
            tool('evaluate_trace_pace_speed_coverage.py','--run',physical,'--scene',geometry,'--output',speed)
            tool('evaluate_trace_pace_spatial_stasis.py','--run',physical,'--output',stasis)
            tool('audit_trace_pace_foot_surface.py','--run',physical,'--output',foot,'--steps','1800')
            tool('audit_trace_pace_substep_speed.py','--run',physical,'--output',substep)
            if a.record:
                tool('audit_trace_pace_camera_skin.py','--run',physical,'--camera-plan',camera,
                     '--actors',json.dumps(actors[scene]),'--output',framing)
                tool('record_pedestrian_coverage.py','--run',physical,'--quality',quality,
                     '--posture',posture,'--speed',speed,'--spatial-stasis',stasis,
                     '--spatial-stasis10',stasis,'--foot-surface',foot,'--substep-speed',substep,
                     '--camera-skin',framing,'--camera-plan',camera,'--output-dir',base/'videos','--tag',scene)
        receipt['status'] = 'passed'
        save()
        return 0
    except BaseException as error:
        receipt.update(status='failed_preserved',error=str(error))
        save()
        raise

if __name__ == '__main__':
    raise SystemExit(main())
