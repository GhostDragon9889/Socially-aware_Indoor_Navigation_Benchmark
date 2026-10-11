#!/usr/bin/env python3
"""Verify committed source/config closure and report separately fetched resources.

Run after cloning. This does not execute physics or certify historical trials.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    p.add_argument('--require-resources', action='store_true')
    a = p.parse_args()
    tracked = set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines())
    failures, files, aliases = [], [], []
    directories = ('src','scripts','configs','env','data/scripts','examples')
    suffixes = ('.py','.json','.yaml','.yml','.toml','.lock','.kit')
    for rel in sorted(tracked):
        part = Path(rel)
        if part.suffix in suffixes and any(rel.startswith(d+'/') for d in directories):
            if not (ROOT/rel).is_file():
                failures.append('Missing committed code/config: '+rel)
    for directory in directories:
        for file in (ROOT/directory).rglob('*'):
            if not file.is_file() or file.suffix not in suffixes:
                continue
            rel = file.relative_to(ROOT).as_posix()
            if '__pycache__' in file.parts or '.egg-info' in str(file) or directory=='env' and 'native_shutdown' in file.parts:
                continue
            if rel not in tracked:
                failures.append('Uncommitted code/config: '+rel)
                continue
            if file.suffix=='.py':
                try:
                    ast.parse(file.read_bytes(), filename=rel)
                except SyntaxError as error:
                    failures.append('Invalid Python: '+str(error))
            if file.suffix=='.json':
                try:
                    json.loads(file.read_bytes())
                except (ValueError, UnicodeError) as error:
                    failures.append('Invalid JSON: '+rel+': '+str(error))
            # Kit accepts settings namespace reopening; its grammar is checked by Kit.
            if file.name == 'requirements.overlay.lock':
                from social_navigation.runtime_environment import read_overlay_lock
                try:
                    read_overlay_lock(file)
                except RuntimeError as error:
                    failures.append('Invalid package lock: '+rel+': '+str(error))
            elif file.suffix in ('.toml','.lock'):
                try:
                    tomllib.loads(file.read_text())
                except (ValueError, UnicodeError) as error:
                    failures.append('Invalid TOML: '+rel+': '+str(error))
            files.append(rel)
    changed = subprocess.check_output(['git','diff','--name-only','HEAD','--',*files],cwd=ROOT,text=True).splitlines()
    for rel in changed:
        if rel in files:
            failures.append('Code/config differs from committed HEAD: '+rel)
    required = ['src/CrowdSim/humanoid/runtime/sealed/sdk6_terminal.py',
        'env/isaaclab3_sim6/native_shutdown/libisaacsim.core.simulation_manager.plugin.so',
        'env/isaaclab3_sim6/native_shutdown/original_build_manifest.json',
        'configs/reproduction/resource_bindings.json','resources/pedestrian_visuals/manifest.json']
    required += ['configs/reproduction/amass_containers/'+s+'.json' for s in ('train','validation','test')]
    for rel in required:
        if rel not in tracked or not (ROOT/rel).is_file():
            failures.append('Missing committed runtime input: '+rel)
    for rel in ('CrowdSim','protomotions','deployment','social_navigation','src/CrowdSim/config','src/CrowdSim/maps'):
        file = ROOT/rel
        if not file.resolve().is_relative_to(ROOT):
            failures.append('External source alias: '+rel)
        aliases.append({'path':rel,'target':str(file.resolve())})
    from social_navigation.resource_bindings import read_json, resolve_path
    resources = []
    for scene in ('YOHO','NewMetro','SecondFloor','ThirdFloor'):
        profile = json.loads((ROOT/'configs/reproduction/four_scene'/ (scene+'_profile.json')).read_text())
        for name, row in profile['resources'].items():
            file = resolve_path(row['path'], root=ROOT)
            present = file.is_file()
            valid = present and hashlib.file_digest(file.open('rb'),'sha256').hexdigest()==row['sha256']
            resources.append({'scene':scene,'name':name,'path':str(file),'available':valid})
        manifest = resolve_path(profile['resources']['amass_manifest']['path'],root=ROOT)
        if manifest.is_file():
            for split, row in read_json(manifest)['splits'].items():
                file = resolve_path(row['path'],root=ROOT)
                resources.append({'scene':scene,'name':'AMASS_'+split,'path':str(file),'available':file.is_file()})
    if a.require_resources and not all(row['available'] for row in resources):
        failures.append('Required weights/motion banks are not restored; follow CLONE_REPRODUCTION.md')
    result = {'schema':'social_navigation.checkout_audit.v1','status':'passed' if not failures else 'failed',
        'root':str(ROOT),'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'committed_source_config_count':len(files),'aliases':aliases,'failures':failures,
        'resources':resources,'physics_executed':False,
        'scope':'Current four-scene TRACE/PACER flow, training tools and committed source/config tree. Historical experiment receipts are not fresh-run inputs.'}
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return int(bool(failures))

if __name__ == '__main__':
    raise SystemExit(main())
