"""Read-only inventory; no Kit startup, downloads, pickle loads or SDK mutation."""
from __future__ import annotations

import ast
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path('/project/EXP/SangforAndSaut/Socially-aware_Indoor_Navigation_Benchmark')
OUT = Path(__file__).parent
LAB = Path('/project/EXP/SangforAndSaut/IsaacLab-Protomotions')
TARGET = Path('/project/EXP/IsaacLab')
TASK_WORKSPACE = Path('/home/lfm/Documents/ChatGPT/Social Navigation')
SCENES = ('YOHO', 'NewMetro', 'SecondFloor', 'ThirdFloor')
hash_cache = {}


def command(args, cwd=None, timeout=30, env=None):
    try:
        p = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=timeout, env=env)
        return {'exit_code': p.returncode, 'stdout': p.stdout.strip(), 'stderr': p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'error': str(exc)}


def digest(path):
    real = str(path.resolve())
    if real not in hash_cache:
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                h.update(block)
        hash_cache[real] = h.hexdigest()
    return hash_cache[real]


def file_record(value, expected=None, hash_file=True):
    p = Path(value)
    if not p.is_absolute():
        p = ROOT / p
    d = {'path': str(p), 'resolved_path': str(p.resolve()), 'exists': p.is_file()}
    if d['exists']:
        d['bytes'] = p.stat().st_size
        if hash_file:
            d['sha256'] = digest(p)
            if expected:
                d['expected_sha256'] = expected
                d['sha256_matches'] = d['sha256'] == expected
    return d


def tree_record(p):
    if not p.is_dir():
        return {'path': str(p), 'exists': False}
    count = size = symlinks = 0
    extensions = Counter()
    for base, dirs, files in os.walk(p, followlinks=False):
        dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.venv', 'wandb', '.aws', '.codex', '.agents'} and not Path(base, d).is_symlink()]
        for name in files:
            f = Path(base, name)
            if f.is_symlink():
                symlinks += 1
                continue
            count += 1
            size += f.stat().st_size
            extensions[f.suffix or '<none>'] += 1
    return {'path': str(p), 'exists': True, 'regular_files': count, 'logical_bytes': size,
            'file_symlinks_excluded': symlinks, 'extensions': dict(extensions.most_common())}


def manifest_record(relative):
    p = ROOT / relative
    d = json.loads(p.read_text())
    result = {'manifest': file_record(p), 'schema': d.get('schema'), 'author_clip_list': d.get('author_clip_list'),
              'selection': d.get('selection'), 'splits': {}}
    if d.get('curation'):
        fields=('selected_filename_labeled','selected_official_description_labeled','selected_unknown_proxy_labeled',
                'selected_counts_by_style','selected_resampled_clips','heldout_fingerprints_checked','limitations')
        result['curation_summary']={k:d['curation'][k] for k in fields if k in d['curation']}
    identities = {}
    for name, split in d['splits'].items():
        clips = split['clips']
        keys = []
        datasets = Counter()
        for c in clips:
            s = c.get('parent_source_path', c['source_path']).split('#')[0]
            keys.append(s)
            dataset = c.get('dataset') or (s.split('/processed/smpl/')[-1].split('/')[0])
            datasets[dataset] += 1
        identities[name] = set(keys)
        result['splits'][name] = {'selected_clip_count': len(clips), 'declared_clip_count': split.get('clip_count'),
            'diffusion_clip_count': split.get('diffusion_clip_count'), 'unique_parent_files': len(set(keys)),
            'unique_motion_ids': len({c['motion_id'] for c in clips}), 'datasets': dict(datasets),
            'source_files_missing': [s for s in sorted(set(keys)) if not Path(s).is_file()],
            'bank': file_record(split['path'], split.get('sha256'))}
    result['parent_file_overlaps'] = {a + '__' + b: len(identities[a] & identities[b])
        for i, a in enumerate(identities) for b in list(identities)[i+1:]}
    return result


def source_scan():
    roots = defaultdict(lambda: {'count': 0, 'examples': [], 'scopes': Counter()})
    totals = {}
    errors = []
    for folder in ('src', 'scripts', 'tools', 'tests', 'examples', 'data/scripts'):
        count = lines = tests = 0
        for p in sorted((ROOT / folder).rglob('*.py')):
            if any(x in p.parts for x in ('__pycache__', '.venv', '.git')):
                continue
            text = p.read_text(encoding='utf-8')
            count += 1
            lines += len(text.splitlines())
            is_test = '/tests/' in str(p) or p.name.startswith('test_') or p.name == 'conftest.py'
            tests += int(is_test)
            try:
                t = ast.parse(text, filename=str(p))
            except SyntaxError as exc:
                errors.append({'file': str(p.relative_to(ROOT)), 'error': str(exc)})
                continue
            for node in ast.walk(t):
                names = []
                if isinstance(node, ast.Import):
                    names = [x.name for x in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module]
                for n in names:
                    k = n.split('.')[0]
                    if k in sys.stdlib_module_names:
                        continue
                    rec = roots[k]
                    rec['count'] += 1
                    rec['scopes'][('tests' if is_test else folder)] += 1
                    if len(rec['examples']) < 5:
                        rec['examples'].append({'file': str(p.relative_to(ROOT)), 'line': node.lineno, 'module': n})
        totals[folder] = {'python_files': count, 'lines': lines, 'test_files': tests}
    return {'totals': totals, 'syntax_errors': errors, 'non_stdlib_import_roots': {
        k: {**v, 'scopes': dict(v['scopes'])} for k, v in sorted(roots.items())},
        'limitations': 'Static imports include tests, conditional backends and tools; not all imports are mandatory for one workflow. No imports executed.'}


def corpus_record(path):
    data=json.loads(path.read_text())
    splits={}
    for name, split in data['splits'].items():
        refs=0
        missing=[]
        mismatches=[]
        for scene in split['scenes']:
            folder=Path(scene['folder'])
            if not folder.is_absolute():
                folder=path.parent/folder
            records=[]
            if scene.get('map'):
                records.append(scene['map'])
            for episode in scene.get('episodes',[]):
                records.append(episode)
                if episode.get('window_index'):
                    records.append(episode['window_index'])
            for record in records:
                file=Path(record['path'])
                if not file.is_absolute():
                    file=folder/file
                refs+=1
                if not file.is_file():
                    missing.append(str(file))
                elif file.stat().st_size!=record['bytes'] or digest(file)!=record['sha256']:
                    mismatches.append(str(file))
        splits[name]={'scene_count':len(split['scenes']),'episode_count':sum(len(s.get('episodes',[])) for s in split['scenes']),
            'declared_window_count':split.get('window_count',data.get('conversion_counts',{}).get(name+'_windows')),
            'file_references_checked':refs,'missing':missing,'size_or_sha_mismatches':mismatches}
    return {'manifest':file_record(path),'schema':data['schema'],'author_dataset':data.get('author_dataset'),
            'state_order':data.get('state_order'),'history_frames':data.get('history_frames'),
            'future_frames':data.get('future_frames'),'data_hz':data.get('data_hz'),'splits':splits,
            'scope':'Manifest identity and referenced map/episode/window bytes+SHA; no semantic tensor validation, trajectory simulation or generator rerun.'}


METADATA_SCRIPT = '''import importlib.metadata as m, json, platform, sys
items = []
for d in m.distributions():
 name=d.metadata.get("Name", "")
 direct=d.read_text("direct_url.json")
 items.append({"name":name,"version":d.version,"requires":d.requires or [],"direct_url":json.loads(direct) if direct else None,"location":str(d.locate_file(""))})
print(json.dumps({"python":sys.version,"executable":sys.executable,"platform":platform.platform(),"sys_path":sys.path,"packages":sorted(items,key=lambda x:x["name"].lower())}))
'''


def env_record(root):
    py = root / '.venv/bin/python'
    d = {'root': str(root), 'python_exists': py.is_file(), 'git_commit': command(['git','rev-parse','HEAD'], root)}
    v = root / 'VERSION'
    if v.is_file():
        d['product_VERSION'] = v.read_text().strip()
    if py.is_file():
        clean_env = dict(os.environ)
        d['inherited_pythonpath'] = clean_env.pop('PYTHONPATH', None)
        clean_env['PYTHONDONTWRITEBYTECODE'] = '1'
        res = command([str(py), '-c', METADATA_SCRIPT], env=clean_env)
        if res.get('exit_code') == 0:
            d['installed'] = json.loads(res['stdout'])
        else:
            d['metadata_error'] = res
    return d


def repo_record(path):
    p = Path(path)
    d = {'path': str(p), 'exists': p.is_dir(), 'resolved_path': str(p.resolve())}
    if p.is_dir():
        d['git_root'] = command(['git','rev-parse','--show-toplevel'], p)
        d['commit'] = command(['git','rev-parse','HEAD'], p)
        # Only URL configs; no auth config, tokens or helper files are accessed.
        urls = command(['git','config','--get-regexp',r'^remote\..*\.url$'], p)
        if 'stdout' in urls:
            urls['stdout'] = re.sub(r'(https?://)[^/@\s]+:[^/@\s]+@', r'\1<redacted>@', urls['stdout'])
        d['remotes'] = urls
        d['license_files'] = sorted(str(x.relative_to(p)) for x in p.glob('*LICENSE*') if x.is_file())
        d['dependency_manifests'] = sorted(x.name for x in p.iterdir() if x.is_file() and
            (x.name in {'pyproject.toml', 'setup.py', 'setup.cfg', 'environment.yml', 'environment.yaml'} or x.name.startswith('requirements')))
    return d


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='A fresh directory for audit results')
    args = parser.parse_args()
    OUT = args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'social_navigation.resource_dependency_inventory.v1', 'audit_date': '2026-10-10',
        'root': str(ROOT), 'scope': 'Read-only filesystem, SHA256, AST and installed metadata; no downloads, package install, pickle loads, Kit, physics or training.',
        'repository_commit': command(['git','rev-parse','HEAD'], ROOT)}
    manifests = {}
    for key, relative in [('project','pyproject.toml'), ('minimal_baseline','packages/crowdsim-ppo/pyproject.toml'),
                          ('sdk_snapshot','env/isaaclab3_sim6/pyproject.toml')]:
        p = ROOT / relative
        manifests[key] = {'file': file_record(p), 'contents': tomllib.loads(p.read_text())}
    lockp = ROOT/'env/isaaclab3_sim6/uv.lock'
    lock = tomllib.loads(lockp.read_text())
    report['manifests'] = manifests
    report['sdk_lock'] = {'file':file_record(lockp), 'requires_python':lock['requires-python'],
                         'package_count':len(lock['package']), 'packages':[
                             {k:v for k,v in p.items() if k not in {'wheels','sdist'}} for p in lock['package']],
                         'artifact_hashes_source_file':str(lockp)}
    report['sdk_profile'] = json.loads((ROOT/'env/isaaclab3_sim6/profile.json').read_text())
    report['sdk_snapshot_matches_working_sdk'] = {
        name: {'snapshot':digest(ROOT/'env/isaaclab3_sim6'/name), 'actual':digest(LAB/name),
               'matches':digest(ROOT/'env/isaaclab3_sim6'/name)==digest(LAB/name)} for name in ('pyproject.toml','uv.lock')}
    report['source'] = source_scan()
    with ThreadPoolExecutor(max_workers=2) as pool:
        report['environments'] = dict(zip(('working', 'target'), pool.map(env_record, (LAB, TARGET))))
    with tempfile.TemporaryDirectory(prefix='resource-audit-uv-') as cache:
        report['uv_pip_check'] = command([shutil.which('uv') or 'uv','pip','check','--python',str(LAB/'.venv/bin/python')],
                                       env=dict(os.environ, UV_CACHE_DIR=cache))
    report['system_tools'] = {tool: command(args) for tool,args in {
        'uv':['uv','--version'], 'git':['git','--version'], 'git_lfs':['git','lfs','version'],
        'ffmpeg':['ffmpeg','-version'], 'gpu_driver':['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],
        'disk':['df','-h',str(ROOT),str(OUT)]}.items()}
    report['links'] = {n:{'symlink':(ROOT/n).is_symlink(), 'resolved':str((ROOT/n).resolve()), 'exists':(ROOT/n).exists()}
        for n in ('CrowdSim','protomotions','deployment','FLUX','NavDP','pyroki','usd_convert','.venv')}
    repositories = [ROOT/'third_party'/n for n in ('trace','pacer','flux','navdp')]
    repositories += [Path('/project/EXP/NavIsaaclab2.0'),Path('/project/EXP/NavIsaaclab2.0/protomotions'),
                     Path('/project/EXP/SangforAndSaut/TraceAndPace/trace'),Path('/project/EXP/SangforAndSaut/TraceAndPace/pacer')]
    report['external_repositories'] = [repo_record(p) for p in repositories]
    report['amass'] = {key:manifest_record(relative) for key,relative in {
        'fixed_v7':'data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json',
        'general_manifest':'data/diffusion_amp/amass_manifest.json'}.items()}
    report['amass_preparation'] = json.loads(Path('/project/EXP/SangforAndSaut/Assets/motion/preparation/summary.json').read_text())
    report['trees'] = {str(p):tree_record(p) for p in [ROOT/'data',ROOT/'resources',ROOT/'output',
        Path('/project/EXP/SangforAndSaut/Assets/motion'),ROOT/'src/protomotions/data/assets', ROOT/'third_party']}
    report['body_model_files'] = [file_record(p,hash_file=False) for folder in [ROOT/'data/smpl',ROOT/'data/soma',
        Path('/project/EXP/SangforAndSaut/Assets/motion/raw')] for p in folder.rglob('*')
        if p.is_file() and (p.suffix in {'.pkl','.npz'} and (folder != Path('/project/EXP/SangforAndSaut/Assets/motion/raw') or 'smpl' in p.name.lower()))]
    report['scenes'] = {}
    for name in SCENES:
        config_dir = ROOT/'configs/diffusion_amp/four_scene_speed_20261010'
        profile_path = config_dir/(name+'_profile.json')
        scene_path = config_dir/(name+'_scene.yaml')
        profile = json.loads(profile_path.read_text())
        scene = json.loads(scene_path.read_text())
        report['scenes'][name] = {'profile':file_record(profile_path), 'scene_config':file_record(scene_path),
            'actor_count':len(scene['origins_xy']), 'resources':{k:file_record(v['path'],v.get('sha256')) for k,v in profile['resources'].items()},
            'scene_evidence':{k:file_record(scene[k]['path'],scene[k].get('sha256')) for k in ('preflight','native_map','probe_script_snapshot')}}
    release=ROOT/'resources/releases/pedestrians_20261010'
    m=json.loads((release/'manifest.json').read_text())
    report['release']={k:v for k,v in m.items() if k!='files'}
    report['release']['manifest']=file_record(release/'manifest.json')
    report['release']['catalog']=json.loads((release/'catalog.json').read_text())
    report['release']['publication_receipt']=json.loads((ROOT/'docs/ASSET_PUBLICATION_20261010.json').read_text())
    report['extra_reference_models']=[file_record(p,hash_file=False) for p in (ROOT/'data/pretrained_models').rglob('*')
        if p.is_file() and p.suffix in {'.pth','.pt','.ckpt','.onnx','.zip'}]
    video_dir=TASK_WORKSPACE/'trace_pace_reproduction/speed_coverage_optimization/videos_final_20261010_v1'
    report['videos']={name:{'file':file_record(video_dir/(name+'_60s.mp4')),
        'ffprobe':command(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,nb_frames,duration','-of','json',str(video_dir/(name+'_60s.mp4'))])} for name in SCENES}
    extra_checks(report)
    (OUT/'inventory.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'output':str(OUT/'inventory.json'),'syntax_errors':len(report['source']['syntax_errors']),
        'sdk_lock_count':report['sdk_lock']['package_count'],'fixed_splits':{k:v['selected_clip_count'] for k,v in report['amass']['fixed_v7']['splits'].items()},
        'pip_check':report['uv_pip_check'],'trees':{k:{'files':v.get('regular_files'),'bytes':v.get('logical_bytes')} for k,v in report['trees'].items()}},ensure_ascii=False,indent=2))


def extra_checks(report):
    """Read-only checks separately executed and recorded in the original audit."""
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    for e in report['environments'].values():
        if 'installed' not in e:
            continue
        pkgs = {canonicalize_name(x['name']):x for x in e['installed']['packages']}
        for group, reqs in [('project_core_requirement_checks', report['manifests']['project']['contents']['project']['dependencies']),
                            ('sdk_extra_requirement_checks', report['manifests']['project']['contents']['project']['optional-dependencies']['isaaclab'])]:
            records=[]
            for req in reqs:
                rr=Requirement(req)
                if rr.marker and not rr.marker.evaluate():
                    continue
                package=pkgs.get(canonicalize_name(rr.name))
                version=package['version'] if package else None
                records.append({'requirement':req,'installed':version,'satisfied':bool(version and rr.specifier.contains(version,prereleases=True))})
            e[group]=records
        e['lock_version_checks']=[]
        for item in report['sdk_lock']['packages']:
            if 'virtual' in item.get('source',{}):
                continue
            package=pkgs.get(canonicalize_name(item['name']))
            version=package['version'] if package else None
            e['lock_version_checks'].append({'name':item['name'],'locked_version':item['version'],'installed':version,
                'matches':bool(version and Requirement(item['name']+'=='+item['version']).specifier.contains(version,prereleases=True))})
    report['snapshot_lock_all_nonvirtual_versions_match_current']=all(x['matches'] for x in report['environments']['working']['lock_version_checks'])
    report['working_sdk_source_difference_from_profile_commit']=command(['git','diff','--stat',
        report['sdk_profile']['isaaclab_commit']+'..HEAD','--','source'],cwd=LAB)
    report['sdk_status']=command(['git','status','--short'],cwd=LAB)
    report['data_subtrees']={p.name:tree_record(p) for p in (ROOT/'data').iterdir() if p.is_dir()}
    report['body_model_files']=[dict(x,symlink=Path(x['path']).is_symlink()) for x in report['body_model_files']]
    check_file=ROOT/'scripts/isaaclab3_environment.py'
    spec=importlib.util.spec_from_file_location('resource_audit_envcheck',check_file)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        mod.require_checkout(LAB)
        report['profile_checkout_gate']={'status':'passed'}
    except RuntimeError as exc:
        report['profile_checkout_gate']={'status':'failed','error':str(exc),'scope':'Read-only; apply/sync not executed.'}
    res=command([str(LAB/'.venv/bin/python'),'-B','-I',str(check_file),'check'])
    report['environment_check']=json.loads(res['stdout']) if res.get('exit_code')==0 else res
    # This subprocess never imports the target modules or starts the SDK.
    modules=['torch','torchvision','einops','numpy','scipy','smplx','chumpy','cv2','PIL','rvo2','shapely','jax','jaxls','pyroki','yourdfpy','viser','bvh','soma','gdown','pandas','rl_games','ruff','pytest','sphinx']
    code='import importlib.util,json; print(json.dumps({m:importlib.util.find_spec(m) is not None for m in '+repr(modules)+'}))'
    res=command([str(LAB/'.venv/bin/python'),'-B','-I','-c',code])
    report['optional_module_availability']=json.loads(res['stdout']) if res.get('exit_code')==0 else res
    sys.path[:0]=[str(ROOT/'src'),str(ROOT)]
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from social_navigation.preflight import verify_native_scene_config
    from social_navigation.asset_release import verify
    checks={'native_scenes':{},'kit_started':False,'physics_rerun':False}
    for name in SCENES:
        try:
            checks['native_scenes'][name]=verify_native_scene_config(ROOT/'configs/diffusion_amp/four_scene_speed_20261010'/f'{name}_scene.yaml')
        except Exception as exc:
            checks['native_scenes'][name]={'status':'failed','error':f'{type(exc).__name__}: {exc}'}
    try:
        checks['release']=verify(ROOT/'resources/releases/pedestrians_20261010')
    except Exception as exc:
        checks['release']={'status':'failed','error':f'{type(exc).__name__}: {exc}'}
    report['resource_checks']=checks
    report['orca_corpora']={p.parent.name:corpus_record(p) for p in sorted((ROOT/'data/diffusion_amp').glob('orca*/manifest.json'))}
    report['other_scene_resources']={str(p):tree_record(p) for p in (
        ROOT.parent/'Assets/DynBench',ROOT.parent/'Assets/NVIDIA',ROOT.parent/'Assets/Sence')}
    report['projection_maps']={n:{'file':file_record(ROOT/'data/diffusion_amp'/n/'generation_manifest.json'),
        'contents':json.loads((ROOT/'data/diffusion_amp'/n/'generation_manifest.json').read_text())}
        for n in ('maps','maps_resource_complete')}
    report['reference_data']={'trace_datasets_entries':[p.name for p in (ROOT/'third_party/trace/datasets').iterdir()],
                              'pacer_sample_data':[]}
    for p in (ROOT/'third_party/pacer/sample_data').glob('*.pkl'):
        rec=file_record(p,hash_file=False)
        with p.open('rb') as stream:
            rec['is_lfs_pointer']=stream.read(120).startswith(b'version https://git-lfs.github.com/spec/v1')
        report['reference_data']['pacer_sample_data'].append(rec)
    report['auxiliary_asset_links']={str(p):file_record(p) for p in (ROOT.parent/'Assets/NVIDIA').rglob('*') if p.is_symlink()}
    hf_status=TASK_WORKSPACE/'hf_publish_20261010/publication_status.json'
    if hf_status.is_file():
        hf=json.loads(hf_status.read_text())
        report['hugging_face']={'local_status_record':str(hf_status),
            **{k:hf.get(k) for k in ('repo_id','status','remote_repository_created_by_this_task','files_uploaded_by_this_task')},
            'remote_lookup_performed_by_collector':False}
    (OUT/'resource_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    env={k:report[k] for k in ('environment_check','uv_pip_check','profile_checkout_gate',
        'sdk_snapshot_matches_working_sdk','working_sdk_source_difference_from_profile_commit','system_tools')}
    (OUT/'environment_checks.json').write_text(json.dumps(env,indent=2)+'\n')


if __name__ == '__main__':
    main()
