"""The deployment-only priority flag cannot change training by accident."""

import importlib.util
import ast
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from CrowdSim.humanoid.application.configuration import parse_amp_args


ROOT = Path(__file__).resolve().parents[1]


def arguments(tmp_path, *flags):
    checkpoint = tmp_path/'diffusion.ckpt'
    checkpoint.write_bytes(b'CLI existence fixture, not weights')
    return ['--diffusion-checkpoint', str(checkpoint), '--output', str(tmp_path/'fresh'), *flags]


def evaluation(tmp_path, *flags):
    return arguments(tmp_path, '--evaluate-steps', '10', '--resume', 'actor.pt',
        '--scene-config', 'scene.yaml', '--route-guidance', 'timed_path',
        '--diffusion-guidance-mode', 'trace_clean', *flags)


@pytest.mark.parametrize('flags', [(), ('--evaluate-steps', '10', '--resume', 'actor.pt'),
    ('--evaluate-steps', '10', '--resume', 'actor.pt', '--measure-ground-support')])
def test_new_disabled_default_preserves_every_old_cli_field_exactly(tmp_path, flags):
    source = ROOT/'docs/style_improvement_20261002/near_horizon_waypoint_before/CrowdSim/humanoid/application/configuration.py'
    spec = importlib.util.spec_from_file_location('CrowdSim.humanoid.application.near_before', source)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    argv = arguments(tmp_path, *flags)
    previous, current = vars(old.parse_amp_args(argv)), vars(parse_amp_args(argv))
    assert current.pop('near_horizon_waypoint_mass') == 0.0
    assert current == previous


@pytest.mark.parametrize('value', ['nan', 'inf', '-.01', '1.01'])
def test_invalid_mass_is_rejected_by_pure_parser(tmp_path, value):
    with pytest.raises(SystemExit) as error:
        parse_amp_args(evaluation(tmp_path, '--near-horizon-waypoint-mass', value))
    assert error.value.code == 2


@pytest.mark.parametrize('flags', [(), ('--evaluate-steps', '10', '--resume', 'actor.pt'),
    ('--evaluate-steps', '10', '--resume', 'actor.pt', '--scene-config', 'scene.yaml', '--route-guidance', 'timed_path'),
    ('--trajectory-source', 'pacer_curriculum')])
def test_enabled_priority_requires_diffusion_evaluation_timed_path_clean_mode(tmp_path, flags):
    with pytest.raises(SystemExit) as error:
        parse_amp_args(arguments(tmp_path, *flags, '--near-horizon-waypoint-mass', '.25'))
    assert error.value.code == 2


@pytest.mark.parametrize('mode', ['trace_clean', 'trace_clean_variance'])
def test_explicit_legal_eval_priority_is_accepted_without_torch_sdk(tmp_path, mode):
    argv = evaluation(tmp_path, '--diffusion-guidance-mode', mode, '--near-horizon-waypoint-mass', '.25')
    args = parse_amp_args(argv)
    assert args.near_horizon_waypoint_mass == .25 and args.evaluate_steps == 10
    code = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
from CrowdSim.humanoid.application.configuration import parse_amp_args
assert parse_amp_args({argv!r}).near_horizon_waypoint_mass == .25
assert not any(name == 'torch' or name.startswith(('isaacsim','isaaclab','omni','pxr')) for name in sys.modules)
"""
    subprocess.run([sys.executable, '-I', '-c', code], check=True, capture_output=True, text=True)


def test_actual_runner_replan_passes_priority_and_keeps_subset_clocks(tmp_path):
    import torch

    args = parse_amp_args(evaluation(tmp_path, '--num-envs', '4', '--near-horizon-waypoint-mass', '.25'))
    tree = ast.parse((ROOT/'scripts/train_paper_amp.py').read_text())
    function = deepcopy(next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'replan'))
    # Only scope binding changes; the actual owned runner body executes intact.
    function.body = [ast.Global(names=node.names) if isinstance(node, ast.Nonlocal) else node for node in function.body]
    ids = torch.tensor([1, 3])
    plans = torch.zeros(4, 51, 2)
    ages = torch.full((4,), 15)
    seen = []

    def plan(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(xy=torch.full((len(kwargs['actor_ids']), 51, 2), 7.))

    world = SimpleNamespace(device=torch.device('cpu'), validation_scene=SimpleNamespace(),
                            planner_inputs=lambda actor_ids: {'actor_ids': actor_ids})
    namespace = dict(torch=torch, args=args, curriculum=None, reference_adapter=None, world=world,
        planner=SimpleNamespace(plan=plan), new_goals=lambda actors: None,
        plans=plans, plan_age=ages, route_waypoints=torch.zeros(4, 10, 2), goals=torch.zeros(4, 2),
        waypoint_times=torch.arange(.5, 5.1, .5), guidance_weight=20., replan_count=0, planner_forward_calls=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(ROOT/'scripts/train_paper_amp.py'), 'exec'), namespace)
    namespace['replan'](ids)
    assert len(seen) == 1 and seen[0]['near_horizon_mass'] == .25
    assert torch.equal(seen[0]['actor_ids'], ids)
    assert seen[0]['guidance_mode'] == 'trace_clean'
    assert seen[0]['waypoints'].shape == (2, 10, 2)
    assert ages.tolist() == [15, 0, 15, 0]
    assert torch.all(plans[ids] == 7.) and torch.all(plans[[0, 2]] == 0.)
    assert namespace['replan_count'] == 2 and namespace['planner_forward_calls'] == 1
