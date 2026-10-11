"""CPU ownership and clone-layout contracts for the project's mesh importer."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def importer(monkeypatch):
    calls = []
    modules = {name: ModuleType(name) for name in (
        "isaaclab", "isaaclab.sim", "isaaclab.terrains", "isaaclab.terrains.utils", "isaaclab.cloner",
    )}
    modules["isaaclab"].sim = modules["isaaclab.sim"]
    modules["isaaclab.sim"].SimulationContext = SimpleNamespace(instance=lambda: SimpleNamespace(device="cpu"))
    modules["isaaclab.terrains.utils"].create_prim_from_mesh = lambda *args, **kwargs: calls.append((args, kwargs))
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    source = Path(__file__).resolve().parents[1] / "src/protomotions/simulator/isaaclab/utils/usd_utils.py"
    spec = importlib.util.spec_from_file_location("isolated_mesh_importer_under_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TrimeshTerrainImporter, modules["isaaclab.cloner"], calls


def config(num_envs, spacing):
    return SimpleNamespace(validate=lambda: None, num_envs=num_envs, env_spacing=spacing,
        terrain_vertices=np.array([[10., 20., 3.], [11., 20., 3.], [10., 21., 3.]]),
        terrain_faces=np.array([[0, 1, 2]]), prim_path="/World/ground",
        visual_material=object(), physics_material=object())


@pytest.mark.parametrize("num_envs,spacing,positions", [
    (1, 2., [[0., 0., 0.]]),
    (2, 2., [[1., 0., 0.], [-1., 0., 0.]]),
    (5, 3., [[3., -1.5, 0.], [3., 1.5, 0.], [0., -1.5, 0.], [0., 1.5, 0.], [-3., -1.5, 0.]]),
])
def test_mesh_importer_exposes_the_sdk_clone_layout_without_moving_global_mesh(importer, num_envs, spacing, positions):
    cls, cloner, authored = importer
    origins = torch.tensor(positions)
    grid_calls = []
    def grid_transforms(count, separation, *, device):
        grid_calls.append((count, separation, device))
        return origins, torch.zeros(count, 4)
    cloner.grid_transforms = grid_transforms
    cfg = config(num_envs, spacing)
    terrain = cls(cfg)
    # InteractiveScene.env_origins delegates directly to this attribute. Reuse
    # the SDK's tensor and ordering, including a partial last row of the grid.
    assert terrain.env_origins is origins
    assert terrain.terrain_origins is None
    assert grid_calls == [(num_envs, spacing, "cpu")]
    assert len(authored) == 1
    (path, mesh), materials = authored[0]
    assert path == "/World/ground/terrain"
    np.testing.assert_array_equal(mesh.vertices, cfg.terrain_vertices)
    np.testing.assert_array_equal(mesh.faces, cfg.terrain_faces)
    assert materials == {"visual_material": cfg.visual_material, "physics_material": cfg.physics_material}


def test_missing_spacing_fails_before_terrain_authoring(importer):
    cls, cloner, authored = importer
    cloner.grid_transforms = lambda *args, **kwargs: pytest.fail("None spacing must not reach the cloner")
    with pytest.raises(ValueError, match="spacing"):
        cls(config(2, None))
    assert not authored
