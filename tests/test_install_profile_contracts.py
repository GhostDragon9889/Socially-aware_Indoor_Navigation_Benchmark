"""Only external SDK binding and pinned uv installation are supported."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib

from packaging.requirements import Requirement
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_project_extras_cannot_install_alternative_simulator_environments():
    metadata = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    assert set(metadata['project']['optional-dependencies']) == {'crowd', 'docs', 'dev'}
    assert 'conflicts' not in metadata['tool']['uv']
    assert 'sources' not in metadata['tool']['uv']
    assert not list(ROOT.glob('requirements_*.txt'))
    assert not list(ROOT.glob('Dockerfile*'))
    assert not (ROOT / 'env/isaaclab3_sim6').exists()


def test_additions_never_replace_sdk_binary_providers():
    overlay = (ROOT / 'env/isaaclab3_ea_sim61/requirements.overlay.lock').read_text()
    protected = {'newton', 'usd-core', 'usd-exchange', 'warp-lang', 'numpy', 'torch', 'isaaclab', 'isaacsim'}
    assert not any(Requirement(line).name.lower() in protected for line in overlay.splitlines()
                   if line.strip() and not line.lstrip().startswith('#'))


def test_moviepy_support_does_not_change_the_sdk_lock():
    metadata = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    moviepy = next(Requirement(text) for text in metadata['project']['dependencies']
                   if Requirement(text).name == 'moviepy')
    assert '1.0.3' in moviepy.specifier and '2.2.1' in moviepy.specifier
    assert '3.0' not in moviepy.specifier


@pytest.mark.parametrize('arguments', [
    ['prepare-runtime'], ['install', '--profile', 'isaaclab3_sim6'],
    ['configure', '--profile', 'isaaclab3_ea_sim61'],
])
def test_removed_commands_fail_before_network_or_sdk_mutation(tmp_path, arguments):
    sdk = tmp_path / "SDK's 空 格"
    result = subprocess.run([sys.executable, '-I', str(ROOT / 'scripts/bootstrap_reproduction.py'),
                             *arguments, '--lab-root', str(sdk)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert not sdk.exists()


def test_existing_sdk_install_preserves_every_byte(tmp_path):
    sdk = tmp_path / "SDK's existing 空 格"
    python = sdk / '.venv/bin/python'; python.parent.mkdir(parents=True)
    python.write_bytes(b'executable sentinel')
    (sdk / 'pyproject.toml').write_bytes(b'manifest sentinel')
    before = {p.relative_to(sdk): p.read_bytes() for p in sdk.rglob('*') if p.is_file()}
    result = subprocess.run([sys.executable, '-I', str(ROOT / 'scripts/bootstrap_reproduction.py'),
                             'install', '--lab-root', str(sdk)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert 'installed SDK already exists' in result.stderr
    assert before == {p.relative_to(sdk): p.read_bytes() for p in sdk.rglob('*') if p.is_file()}


@pytest.mark.parametrize('name,key', [('pyproject.toml', 'manifest_sha256'), ('uv.lock', 'lock_sha256')])
def test_pinned_uv_sdk_inputs_match_their_receipt(name, key):
    directory = ROOT / 'env/isaaclab3_ea_sim61'
    receipt = json.loads((directory / 'profile.json').read_text())
    assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == receipt[key]
