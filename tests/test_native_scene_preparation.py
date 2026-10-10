"""Pure CPU layout/A* contracts; synthetic grids are not native SDK evidence."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from CrowdSim.humanoid.application.native_scene_preparation import (
    publish_configuration,
    route_shifts,
    select_origins,
    verify_route_paths,
)
from CrowdSim.humanoid.runtime.native_grid_scene import NativeGridScene
from scripts.prepare_native_pedestrian_scene import parse_args
from test_native_grid_scene import fixture_provider


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("actors, expected", [(3, [1, 2, 3]), (4, [1, 2, 3, 4]), (15, [1, 4, 8, 12])])
def test_known_route_protocols(actors, expected):
    assert route_shifts(actors) == expected
    assert len(set(x % actors for x in expected)) == len(expected)


@pytest.mark.parametrize("actors, shifts", [(1, None), (2, None), (16, None), (4, [1, 1]),
                                         (4, [4, 1]), (4, [1, 5]), (4, [1]), (4, [True, 2]),
                                         (4, [-1, 2]), (3, [1, 2, 3, 0])])
def test_degenerate_offsets_rejected(actors, shifts):
    with pytest.raises(ValueError):
        route_shifts(actors, shifts)


def test_largest_four_connected_component_and_unknown_are_explicit():
    labels = np.zeros((120, 240), np.uint8)
    labels[10:100, 10:100] = 254
    labels[10:110, 130:230] = 254
    # Touching corners cannot bridge the two disjoint four-neighbour components.
    labels[10:25, 145:160] = 205
    result = select_origins(labels, [-15., -6.], actors=4)
    cells = result["pixels"]
    assert np.all(cells[:, 1] >= 130)
    assert (labels[cells[:, 0], cells[:, 1]] == 254).all()
    assert np.all(np.asarray(result["selection"]["selected_spawn_clearance_m"]) >= 3.)
    assert result["selection"]["component_count"] == 2
    assert not result["walkable"][10:25, 145:160].any()
    assert np.array_equal(cells, select_origins(labels, [-15., -6.], actors=4)["pixels"])
    assert np.unique(cells, axis=0).shape == (4, 2)
    assert np.any(result["origins_xy"] < 0.)


def test_first_deepest_cell_tie_is_ros_row_major_without_rng():
    labels = np.full((100, 100), 254, np.uint8)
    rng = np.random.get_state()
    result = select_origins(labels, [-5., -5.], actors=3)
    assert result["pixels"][0].tolist() == [49, 49]
    assert np.allclose(result["origins_xy"][0], [-.05, .05], atol=1e-14)
    after = np.random.get_state()
    assert rng[0] == after[0] and np.array_equal(rng[1], after[1]) and rng[2:] == after[2:]


def test_insufficient_spawn_clearance_never_falls_back_to_smaller_component():
    labels = np.zeros((100, 220), np.uint8)
    labels[5:35, 5:205] = 254  # Largest component is too narrow to hold >=3m spawn clearance.
    labels[40:99, 5:64] = 254
    with pytest.raises(ValueError, match="eligible spawn cells"):
        select_origins(labels, [0., 0.], actors=4)


def test_many_eligible_cells_cannot_hide_overlapping_spawns():
    # An8m room contains many pixels with3m obstacle clearance, but they are
    # concentrated in a2m central region.15 actors cannot have1.6m root spacing.
    labels = np.full((80, 80), 254, np.uint8)
    with pytest.raises(ValueError, match="root spawn separation"):
        select_origins(labels, [-4., -4.], actors=15)


@pytest.mark.parametrize("kwargs", [{"clearance_m": .35}, {"spawn_clearance_m": 2.9},
                                    {"spawn_clearance_m": float("inf")}, {"resolution_m": .2},
                                    {"minimum_spawn_separation_m": 1.5}])
def test_quality_requirements_cannot_be_reduced(kwargs):
    with pytest.raises(ValueError):
        select_origins(np.full((100, 100), 254, np.uint8), [0., 0.], actors=3, **kwargs)


def test_every_initial_and_wrap_edge_requires_path(tmp_path):
    labels = np.full((100, 180), 254, np.uint8)
    labels[:, 88:92] = 205  # Unknown separates the two components.
    provider = fixture_provider(tmp_path / "fixture", labels)
    origins = np.array([[-1., 0.], [-1., 3.], [10., 0.]])
    routes = origins[(np.arange(3)[:, None] + np.array([1, 2, 3])) % 3]
    grid = NativeGridScene.from_provider(provider, origins_xy=origins, routes_xy=routes, clearance_m=.8)
    with pytest.raises(ValueError, match="disconnected"):
        verify_route_paths(grid, origins, routes)
    assert labels[:, 88:92].tolist() == [[205] * 4] * 100


def test_initial_route_and_wrap_paths_are_all_measured(tmp_path):
    provider = fixture_provider(tmp_path / "fixture", np.full((120, 140), 254, np.uint8))
    origins = np.array([[-.5, 1.], [3., 1.], [3., 5.]])
    routes = origins[(np.arange(3)[:, None] + np.array([1, 2, 3])) % 3]
    grid = NativeGridScene.from_provider(provider, origins_xy=origins, routes_xy=routes, clearance_m=.8)
    paths = verify_route_paths(grid, origins, routes)
    assert len(paths) == 12
    assert [x["kind"] for x in paths[:4]] == ["initial", "route", "route", "wrap"]
    assert all(x["length_m"] > 0. and x["vertices"] >= 2 for x in paths)


def test_output_existing_config_is_never_overwritten(tmp_path):
    target = tmp_path / "existing.yaml"
    target.write_text("original")
    with pytest.raises(ValueError, match="no overwrite"):
        publish_configuration({}, {}, target, tmp_path / "proof.json")
    assert target.read_text() == "original"
    assert not (tmp_path / "proof.json").exists()


def test_cli_rejects_degenerate_or_existing_output_before_resource_load(tmp_path):
    args = ["--preflight", "missing.json", "--native-map", "missing2.json", "--actors", "1",
            "--output-config", str(tmp_path / "scene.yaml"), "--proof", str(tmp_path / "proof.json")]
    with pytest.raises(SystemExit):
        parse_args(args)
    args[5] = "3"
    (tmp_path / "scene.yaml").write_text("original")
    with pytest.raises(SystemExit):
        parse_args(args)
    assert not (tmp_path / "proof.json").exists()


def test_cold_cli_help_imports_no_tensor_or_sdk():
    command = "import sys; from scripts.prepare_native_pedestrian_scene import parse_args; assert not any(x in sys.modules for x in ['torch','numpy','pxr','isaacsim','omni','isaaclab']); "
    command += "\ntry: parse_args(['--help'])\nexcept SystemExit as e: assert e.code == 0"
    result = subprocess.run([sys.executable, "-c", command], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--route-shifts" in result.stdout
