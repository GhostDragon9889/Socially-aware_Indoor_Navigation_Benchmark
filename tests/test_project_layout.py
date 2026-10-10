"""Contracts for importable packages, checkout paths, and the editor entry points."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

from protomotions.repository import repository_root


ROOT = repository_root(__file__) if 'src' in Path(__file__).parts else repository_root()


def test_legacy_module_paths_resolve_to_canonical_sources():
    for name in ('protomotions', 'CrowdSim', 'deployment'):
        assert (ROOT / name).is_symlink()
        assert (ROOT / name).resolve() == ROOT / 'src' / name
        assert repository_root(ROOT / name / '__init__.py') == ROOT
    assert (ROOT / 'src/CrowdSim/config').resolve() == ROOT / 'configs/crowdsim'
    assert (ROOT / 'src/CrowdSim/maps').resolve() == ROOT / 'data/navigation_maps/crowdsim'


def test_discovery_includes_diffusion_runtime_and_navigation_helpers():
    from setuptools import find_packages

    config = tomllib.loads((ROOT / 'pyproject.toml').read_text())
    assert config['project']['name'] == 'Socially-aware_Indoor_Navigation_Benchmark'
    finder = config['tool']['setuptools']['packages']['find']
    packages = set(find_packages(str(ROOT / 'src'), include=finder['include'], exclude=finder['exclude']))
    assert {'social_navigation', 'CrowdSim.humanoid.diffusion', 'CrowdSim.humanoid.runtime',
            'CrowdSim.humanoid.trajectory', 'CrowdSim.utils', 'protomotions.simulator.isaaclab',
            'crowdsim_ppo', 'deployment'} <= packages
    assert not any('.tests' in name for name in packages)


def test_cli_runs_from_an_external_working_directory(tmp_path):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT / 'src'))
    result = subprocess.run([sys.executable, '-m', 'social_navigation', 'status'], cwd=tmp_path,
                            env=env, capture_output=True, text=True, check=True)
    record = json.loads(result.stdout)
    assert Path(record['root']) == ROOT
    assert record['release_available']
    assert record['project'] == 'Socially-aware_Indoor_Navigation_Benchmark'


def test_repository_root_override(monkeypatch, tmp_path):
    monkeypatch.setenv('SOCIAL_NAVIGATION_ROOT', str(ROOT))
    assert repository_root(tmp_path / 'unrelated.py') == ROOT


def test_editor_uses_process_arguments_and_current_debugger():
    tasks = json.loads((ROOT / '.vscode/tasks.json').read_text())['tasks']
    assert all(task['type'] == 'process' and isinstance(task['args'], list) for task in tasks)
    assert all(task['command'] == '${workspaceFolder}/.venv/bin/python' for task in tasks)
    launch = json.loads((ROOT / '.vscode/launch.json').read_text())['configurations']
    assert all(item['type'] == 'debugpy' for item in launch)
    workspace = json.loads((ROOT / 'Socially-aware_Indoor_Navigation_Benchmark.code-workspace').read_text())
    assert workspace['folders'] == [{'name': 'Socially-aware_Indoor_Navigation_Benchmark', 'path': '.'}]
