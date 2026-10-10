"""Fresh copied q936 bundle; one runner replacement and one added timing profile."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
STAGED = HERE.parent / 'state_only_q_poststep_timing_preparation_v1'
OLD = ROOT / 'output/isolated_training/mixed_state_terrain_state_prior_stratified_matched_v1'
NEW = ROOT / 'output/isolated_training/mixed_state_terrain_state_prior_stratified_poststep_v1'
RUNNER = 'code/scripts/train_paper_amp.py'
PROFILE = 'code/configs/diffusion_amp/isolated_state_only_q_poststep_absolute.yaml'


def record(path):
    path = Path(path).absolute()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def write(path, document):
    with Path(path).open('x') as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    assert record(OLD / 'source_manifest.json')['sha256'] == '5c2a22098bf8df60530b5f53963c041a221cc17e62759284d7586f6ea6a05f50'
    assert record(STAGED / 'source_freeze_v1.json')['sha256'] == 'd248efbdd451d90d4d672a316147eb5d57851373180c8e19165f5090153b239c'
    baseline = json.loads((OLD / 'source_manifest.json').read_text())
    delta = json.loads((STAGED / 'source_derivation.json').read_text())
    assert record(OLD / RUNNER) == delta['parent_runner']
    assert record(STAGED / 'train_paper_amp.py') == delta['staged_runner']
    assert record(STAGED / 'state_only_q_poststep_absolute.yaml') == delta['profile_staged']
    plan = json.loads((STAGED / 'future_commands_source_plan.json').read_text())
    outputs = [Path(command[command.index('--output') + 1]) for command in plan['commands'].values()]
    snapshots = Path(plan['watch_command'][plan['watch_command'].index('--snapshot-dir') + 1])
    assert not any(p.exists() or p.is_symlink() for p in [NEW, *outputs, snapshots])
    NEW.mkdir()
    for row in baseline['files']:
        source, destination = OLD / row['path'], NEW / row['path']
        identity = record(source)
        assert (identity['bytes'], identity['sha256']) == (row['bytes'], row['sha256'])
        assert not source.is_symlink()
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        assert record(destination)['sha256'] == row['sha256']
        assert source.stat().st_ino != destination.stat().st_ino and destination.stat().st_nlink == 1
    runner = NEW / RUNNER
    runner.chmod(0o644)
    runner.write_bytes((STAGED / 'train_paper_amp.py').read_bytes())
    profile = NEW / PROFILE
    profile.write_bytes((STAGED / 'state_only_q_poststep_absolute.yaml').read_bytes())
    (NEW / 'runtime_scratch').mkdir()
    links = deepcopy(baseline['shared_resource_links'])
    for row in links:
        if row['path'] == 'code/output':
            row['target'] = str(NEW / 'runtime_scratch')
        (NEW / row['path']).symlink_to(row['target'], target_is_directory=True)
    for name in ('launch_isolated.py', 'watch_snapshots.py'):
        shutil.copy2(OLD / name, NEW / name)
        assert record(NEW / name)['sha256'] == record(OLD / name)['sha256']
    actual = sorted(str(p.relative_to(NEW)) for p in (NEW / 'code').rglob('*') if p.is_file() and not p.is_symlink())
    assert set(actual) == {row['path'] for row in baseline['files']} | {PROFILE}
    manifest = deepcopy(baseline)
    manifest.update(schema='state_only_q.poststep_isolated_training_source.v1', created_utc=datetime.now(timezone.utc).isoformat(),
        code_root=str(NEW / 'code'), source_epoch='q936_exact_parent_one_restored_poststep_runner_one_added_timing_profile',
        previous_source_manifest=record(OLD / 'source_manifest.json'), shared_resource_links=links,
        profile=record(profile), launcher=record(NEW / 'launch_isolated.py'), GPU_started=False,
        CPU_only_preparation=True, future_effect=False,
        candidate_inputs={'parent_code_count': len(baseline['files']), 'modified_parent_paths': [RUNNER],
                          'added_paths': [PROFILE], 'approved_source_freeze': record(STAGED / 'source_freeze_v1.json'),
                          'approved_three_fragment_derivation': record(STAGED / 'source_derivation.json'),
                          'mechanism': 'state-only post-step AMP reward/history/replay timing; fresh only'},
        actual_import_provenance={'status': 'pending_actual_new_bundle_CPU_and_training_imports',
                                 'complete_runtime_finalization_claimed': False})
    manifest['files'] = []
    for relative in actual:
        path = NEW / relative
        row = record(path)
        manifest['files'].append({'path': relative, 'bytes': row['bytes'], 'sha256': row['sha256'],
             'mtime_ns': path.stat().st_mtime_ns, 'device': path.stat().st_dev,
             'inode': path.stat().st_ino, 'copy_not_hardlink': True})
        if relative in {r['path'] for r in baseline['files']} and relative != RUNNER:
            assert row['sha256'] == next(r['sha256'] for r in baseline['files'] if r['path'] == relative)
        path.chmod(0o444)
    write(NEW / 'source_manifest.json', manifest)
    plan.update(schema='state_only_q.poststep_actual_bundle_root_GPU_commands.v1',
        status='prepared_actual_source_copy_not_GPU_executed', bundle_built=True,
        actual_source_count=len(actual), source_manifest_record=record(NEW / 'source_manifest.json'))
    assert plan['cwd'] == str(NEW / 'code')
    for command in plan['commands'].values():
        assert command[1] == str(NEW / 'launch_isolated.py') and '--resume' not in command
        assert command[command.index('--config') + 1] == str(NEW / PROFILE)
    write(HERE / 'root_GPU_launch_commands.json', plan)
    write(NEW / 'commands.json', plan)
    archive = HERE / 'actual_source.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for name in actual + ['source_manifest.json', 'launch_isolated.py', 'commands.json', 'watch_snapshots.py']:
            tar.add(NEW / name, arcname=name, recursive=False)
    archive.chmod(0o444)
    with tarfile.open(archive, 'r:gz') as tar:
        archive_count = sum(m.isfile() for m in tar.getmembers())
    for name in ('source_manifest.json', 'launch_isolated.py', 'watch_snapshots.py', 'commands.json'):
        (NEW / name).chmod(0o444)
    write(HERE / 'registration_prepared.json', {'schema':'state_only_q.poststep_actual_source_preparation.v1',
        'status':'prepared_source_copy_CPU_only_not_training', 'source_manifest':record(NEW / 'source_manifest.json'),
        'source_count':len(actual), 'source_archive':record(archive), 'archive_regular_members':archive_count,
        'parent_source':record(OLD / 'source_manifest.json'), 'profile':record(profile),
        'launcher':record(NEW / 'launch_isolated.py'), 'commands':record(HERE / 'root_GPU_launch_commands.json'),
        'approved_staged_source':record(STAGED / 'source_freeze_v1.json'),
        'modified_parent_paths':[RUNNER], 'added_paths':[PROFILE], 'unchanged_parent_source_count':len(baseline['files']) - 1,
        'AMP_reward_timestamp':'post-step h[t+1..] before resets', 'D_input':1950,
        'future_checkpoint':{'path':str(snapshots / 'step_2000.pt'), 'bytes':None, 'sha256':None},
        'models_deserialized':False, 'SDK_started':False, 'GPU_started':False, 'future_effect':False})
    print(json.dumps({'status':'prepared_source_copy_only', 'source_count':len(actual),
        'archive_regular_members':archive_count, 'source_manifest':record(NEW / 'source_manifest.json')}))


if __name__ == '__main__':
    main()
