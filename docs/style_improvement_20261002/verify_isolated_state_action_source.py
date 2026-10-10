"""CPU source/import guard checks for the copied state-action training bundle."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT/'output/isolated_training/mixed_state_action_v1'


def main():
    spec = importlib.util.spec_from_file_location('isolated_launch_contract', BUNDLE/'launch_isolated.py')
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    manifest = launcher.verify()
    checks = []

    def require(ok, name):
        checks.append({'check': name, 'passed': bool(ok)})
        if not ok:
            raise ValueError(name)

    require(len(manifest['files']) == 926 and all(r['copy_not_hardlink'] for r in manifest['files']), 'all926 actual copiedcode/assets; no hardlinked code')
    require(len(manifest['source_epoch']['paths']) == 146 and manifest['source_epoch']['all146_source_bytes_restored'], 'old146 authoritative byte restoration')
    base = json.loads(Path(manifest['source_epoch']['authoritative_registered146_manifest']['path']).read_text())
    for row in base['files']:
        require(hashlib.sha256((BUNDLE/'code'/row['path']).read_bytes()).hexdigest() == row['sha256'], 'old146 exact: '+row['path'])
    for name in ['CrowdSim', 'protomotions']:
        try:
            launcher.OwnedImportGuard().find_spec(name, [str(ROOT)])
        except ImportError:
            require(True, 'actual PathFinder original checkout origin rejected: '+name)
        else:
            require(False, 'actual PathFinder original checkout origin rejected: '+name)
    try:
        launcher.OwnedImportGuard().find_spec('protomotions.__isolated_nonexistent', [str(BUNDLE/'code/protomotions')])
    except ImportError as error:
        require('no editable fallback' in str(error), 'missingowned submodule rejects beforeeditable finder')
    else:
        require(False, 'missingowned submodule rejects beforeeditable finder')
    launcher.activate()
    from protomotions.assets import get_asset_root
    require(get_asset_root() == BUNDLE/'code/protomotions/data/assets', 'actual default asset resolution iscopiedtree')
    try:
        exec(compile('sentinel = 1', str(ROOT/'protomotions/assets.py'), 'exec'), {})
    except ImportError:
        require(True, 'actual audit exec eventrejects arbitrarycheckout executable path')
    else:
        require(False, 'actual audit exec eventrejects arbitrarycheckout executable path')
    exec(compile('sentinel = 1', str(BUNDLE/'code/protomotions/assets.py'), 'exec'), {})
    require(True, 'actual copiedpath execadmitted')
    modules = json.loads((BUNDLE/'CPU_imports_v1.json').read_text())
    require(modules['status'] == 'passed' and len(modules['owned_modules']) == 59
            and all(Path(r['path']).is_relative_to(BUNDLE/'code') for r in modules['owned_modules']), '59 CPU actual ownedimports isolated; CUDA not initialized')
    preflight = json.loads((BUNDLE/'preflight_imports_v1.json').read_text())
    require(preflight['status'] == 'passed' and preflight['all_owned_modules_isolated'], 'actual SDKfree preflight imports isolated')
    require(all(Path(r['path']).stat().st_mode & 0o222 == 0 for r in launcher.imported_owned()), 'actual copiedbusiness file writebits removed')
    report = {'schema': 'paper_amp.isolated_guard_actual_CPU.v1', 'status': 'passed', 'checks': checks,
              'check_count': len(checks), 'CPU_only': True, 'SDK_started': False, 'GPU_used': False,
              'source_manifest_sha256': launcher.digest(BUNDLE/'source_manifest.json'),
              'launcher_sha256': launcher.digest(BUNDLE/'launch_isolated.py'),
              'limits': 'Actual GPU/Kit import behavior is still pending root4+4; runtimefinder/execguard enforce copiedowned paths. Sharedlicensed resources have SHA/stat admission, not an OS readonly sandbox.'}
    output = ROOT/'docs/style_improvement_20261002/isolated_state_action_source_CPU_verification_v1.json'
    with output.open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps({'status': 'passed', 'checks': len(checks)}))


if __name__ == '__main__':
    main()
