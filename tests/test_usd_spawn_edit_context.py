"""Regression for the real N4 session-layer spawn/SDK root-copy failure."""
import ast
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch

Usd = pytest.importorskip('pxr.Usd')
UsdGeom = pytest.importorskip('pxr.UsdGeom')

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT/'src').is_dir():
    ROOT = Path('/project/EXP/SangforAndSaut/Socially-aware_Indoor_Navigation_Benchmark')


def guarded_call(monkeypatch, stage):
    # The real USD EditContext is exercised; only active-stage discovery is supplied.
    tree = ast.parse((ROOT/'src/protomotions/simulator/isaaclab/utils/usd_only_scene.py').read_text())
    method = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'call_with_preserved_edit_target')
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'guarded_spawn', 'exec'), namespace)
    shim = ModuleType('isaaclab.sim.utils.stage')
    shim.get_current_stage = lambda: stage
    monkeypatch.setitem(sys.modules, 'isaaclab.sim.utils.stage', shim)
    return namespace['call_with_preserved_edit_target']


@pytest.mark.parametrize('raise_during_spawn', [False, True])
def test_provider_session_authoring_restores_full_edit_target(monkeypatch, raise_during_spawn):
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim('/World', 'Xform')
    previous = stage.GetEditTarget()
    call = guarded_call(monkeypatch, stage)
    def provider():
        stage.SetEditTarget(stage.GetSessionLayer())
        prim = stage.DefinePrim('/NativeScene', 'Xform')
        if raise_during_spawn:
            raise RuntimeError('failed native scene spawn')
        return prim
    if raise_during_spawn:
        with pytest.raises(RuntimeError, match='failed native'):
            call(provider)
    else:
        assert call(provider).GetPath() == '/NativeScene'
    assert stage.GetEditTarget() == previous
    assert stage.GetSessionLayer().GetPrimAtPath('/NativeScene') is not None
    assert stage.GetRootLayer().GetPrimAtPath('/NativeScene') is None
    stage.DefinePrim('/World/envs/env_0/Robot', 'Xform')
    assert stage.GetRootLayer().GetPrimAtPath('/World/envs/env_0/Robot') is not None


def test_sdk_root_clone_copies_all_four_robots_after_provider_scope(monkeypatch):
    from isaaclab.cloner.usd import UsdReplicateContext
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim('/World/envs/env_0', 'Xform')
    call = guarded_call(monkeypatch, stage)
    def provider():
        stage.SetEditTarget(stage.GetSessionLayer())
        stage.DefinePrim('/NativeScene', 'Xform')
    call(provider)
    source = '/World/envs/env_0/Robot'
    stage.DefinePrim(source, 'Xform')
    stage.DefinePrim(source+'/Pelvis', 'Xform')
    relation = stage.GetPrimAtPath(source).CreateRelationship('body')
    relation.SetTargets([source+'/Pelvis'])
    context = UsdReplicateContext(stage)
    context.queue('/World/envs/env_0', '/World/envs/env_{}', torch.arange(4),
                  positions=torch.tensor([[1.,-1.,0.],[1.,1.,0.],[-1.,-1.,0.],[-1.,1.,0.]]))
    context.replicate()
    for actor in range(4):
        path = f'/World/envs/env_{actor}/Robot'
        robot = stage.GetPrimAtPath(path)
        assert robot and stage.GetPrimAtPath(path+'/Pelvis')
        assert list(map(str, robot.GetRelationship('body').GetTargets())) == [path+'/Pelvis']
    assert stage.GetSessionLayer().GetPrimAtPath('/NativeScene') is not None


def test_provider_scope_restores_an_existing_session_target(monkeypatch):
    stage = Usd.Stage.CreateInMemory()
    stage.SetEditTarget(stage.GetSessionLayer())
    previous = stage.GetEditTarget()
    call = guarded_call(monkeypatch, stage)
    def spawner():
        stage.SetEditTarget(stage.GetRootLayer())
        stage.DefinePrim('/Robot', 'Xform')
    call(spawner)
    assert stage.GetEditTarget() == previous
    assert stage.GetRootLayer().GetPrimAtPath('/Robot') is not None


@pytest.mark.parametrize('actors', [3, 15])
@pytest.mark.parametrize('fails', [False, True])
def test_native_provider_restores_target_without_usd_only_scene(actors, fails):
    from isaaclab.cloner.usd import UsdReplicateContext
    from types import SimpleNamespace
    stage = Usd.Stage.CreateInMemory()
    stage.DefinePrim('/World/envs/env_0', 'Xform')
    previous = stage.GetEditTarget()
    tree = ast.parse((ROOT/'src/CrowdSim/humanoid/runtime/dynbench_scene.py').read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == '_spawn')
    scope = {'_active_spawn_stage': lambda: stage}
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'native_spawn_scope', 'exec'), scope)
    def spawn(*args, **kwargs):
        stage.SetEditTarget(stage.GetSessionLayer())
        prim = stage.DefinePrim('/Native', 'Xform')
        if fails:
            raise RuntimeError('provider failed')
        return prim
    owner = SimpleNamespace(_spawn_in_session=spawn)
    if fails:
        with pytest.raises(RuntimeError, match='provider failed'):
            scope['_spawn'](owner, '/Native', None)
    else:
        assert scope['_spawn'](owner, '/Native', None).GetPath() == '/Native'
    assert stage.GetEditTarget() == previous
    stage.DefinePrim('/World/envs/env_0/Robot', 'Xform')
    stage.DefinePrim('/World/envs/env_0/Robot/Pelvis', 'Xform')
    clone = UsdReplicateContext(stage)
    clone.queue('/World/envs/env_0', '/World/envs/env_{}', torch.arange(actors),
                positions=torch.zeros((actors, 3)))
    clone.replicate()
    assert all(stage.GetPrimAtPath(f'/World/envs/env_{i}/Robot/Pelvis') for i in range(actors))
