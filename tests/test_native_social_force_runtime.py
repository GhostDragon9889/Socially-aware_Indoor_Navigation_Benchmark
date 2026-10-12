"""Numerical/input fixtures only; no actual native SDK rollout evidence."""
import json
import ast
import importlib.util
import subprocess
import sys

import numpy as np
import pytest
import torch

from CrowdSim.humanoid.trajectory.native_social_force_runtime import (
    NativeSocialForcePlanner, PINS, ROOT, derive_actor_methods, native_distance_grid,
)
from CrowdSim.humanoid.trajectory.social_force import SocialForceTrajectoryGenerator
from CrowdSim.humanoid.trajectory.social_force_runtime import file_identity
from test_native_grid_scene import fixture_provider
from scripts.native_shared_social_derivation import derive_native_sfm_driver


@pytest.fixture(scope="module")
def synthetic_source(tmp_path_factory):
    folder = tmp_path_factory.mktemp("native_SFM_numerical_fixture")
    provider = fixture_provider(folder / "resources", np.full((200, 400), 254, np.uint8))
    origins = np.array([[-2. + 5 * column, 5. * row] for row in range(3) for column in range(5)])
    paths = {}
    for count in (3, 15):
        selected = origins[:count]
        # Generated mathematical fixture routes are not an author protocol.
        routes = np.stack((selected + [2., 0.], selected), axis=1)
        document = {"schema": "crowdsim.native_pedestrian_scene.v1", "scene": "Office",
                    "preflight": {key: provider._preflight_record[key] for key in ("path", "sha256")},
                    "native_map": {key: provider._native["record"][key] for key in ("path", "sha256")},
                    "origins_xy": selected.tolist(), "routes_xy": routes.tolist(), "clearance_m": .8}
        path = folder / f"synthetic_{count}.json"
        path.write_text(json.dumps(document))
        paths[count] = path
    return paths


def inputs(planner):
    count = planner.actor_count
    states = torch.zeros(count, 5, dtype=torch.float32)
    states[:, :2] = torch.from_numpy(planner._scene._origins.copy()).float()
    history = states[:, None].repeat(1, 31, 1)
    neighbors = torch.stack([history[[peer for peer in range(count) if peer != actor]] for actor in range(count)])
    waypoints = states[:, None, :2].repeat(1, 10, 1) + torch.tensor([2., 0.])
    return dict(states=states, history=history, neighbors=neighbors, raster=torch.zeros(count, 2, 224, 224),
                history_mask=torch.ones(count, 31, dtype=torch.bool),
                neighbor_mask=torch.ones(count, count - 1, 31, dtype=torch.bool),
                waypoints=waypoints, waypoint_times=torch.arange(1, 11, dtype=torch.float32) / 2,
                num_samples=1, guidance_weight=20., guidance_mode="trace_clean", near_horizon_mass=0.,
                scene_ids=torch.zeros(count, dtype=torch.int64), agent_collision_weight=5., agent_collision_radius=.5,
                scene_collision_weight=10., scene_collision_radius=.5, trajectory_speed_weight=0.,
                max_trajectory_speed_mps=1.5, trajectory_accel_weight=0., max_trajectory_accel_mps2=2.,
                scene_bounds_xy=None, obstacle_centers_xy=None, obstacle_sizes_xy=None,
                scene_collision_cost_fn=planner._scene.collision_cost)


def test_count_derivation_is_bound_and_reversible():
    source = (ROOT / PINS["fixture_bridge"][0]).read_text()
    derived, proof = derive_actor_methods(source)
    assert proof["exact_inverse"] and proof["replacement_count"] == 12
    assert proof["force_integration_reference_math_changed"] is False
    assert "self.actor_count - 1" in derived
    with pytest.raises(ValueError, match="exact source-bound"):
        derive_actor_methods(source.replace("neighbors.shape", "changed_neighbors.shape"))


def test_native_driver_changes_planner_only_and_keeps_control_capture_calls():
    staged = ROOT / "resources/runtime_sources/style_improvement_20261004/social_native_evaluation_staging_v1/evaluate_native_social.py"
    spec = importlib.util.spec_from_file_location("native_source_derivation_fixture", staged)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entry, proof = module._derive_parent(module.PARENT.read_text())
    assert proof["whole_source_reverse_equal"]
    namespace = {"__file__": str(staged), "__name__": "native_test_entry"}
    exec(compile(entry, str(staged), "exec"), namespace)
    driver = (ROOT / "scripts/train_paper_amp.py").read_text()
    before, _ = namespace["_derive"](driver)
    derived, derivation = derive_native_sfm_driver(driver, namespace["_derive"])
    assert derivation["exact_inverse_to_shared_SFM_driver"]
    assert derivation["shared_helpers_AST_equal"]
    assert "planner = NativeSocialForcePlanner" in derived
    assert "PaperDiffusionPlanner" not in derived
    def world_calls(value, method):
        return [ast.dump(node, include_attributes=False) for node in ast.walk(ast.parse(value))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "world" and node.func.attr == method]
    for name in ("step", "state", "reset", "planner_inputs", "enable_pairwise_contacts"):
        assert world_calls(before, name) == world_calls(derived, name)
    # The unchanged native DP call carries exactly its own grid callback.
    assert '"scene_collision_cost_fn": native_scene.collision_cost' in derived


def test_ros_EDT_keeps_unknown_occupied_exterior_and_negative_world():
    labels = np.full((8, 12), 254, np.uint8)
    labels[1, 7], labels[6, 3] = 205, 0
    original = labels.copy()
    grid = native_distance_grid(labels, origin_xy=[-2., -1.], bounds_xy=[-2., -1., -.8, -.2],
                                resolution_m=.1, source={"path": "synthetic", "sha256": "a" * 64})
    assert np.array_equal(labels, original)
    assert grid.distance_m[1, 7] == grid.distance_m[6, 3] == 0
    assert grid.distance_m[0, 0] == pytest.approx(.1)
    for row, col in ((1, 7), (6, 3)):
        xy = np.array([[-2. + (col + .5) * .1, -1. + (8 - row - .5) * .1]])
        with pytest.raises(ValueError, match="unknown|occupied"):
            grid.sample(xy)
    with pytest.raises(ValueError, match="exterior"):
        grid.sample(np.array([[-2.01, -.5]]))
    labels[0, 0] = 205
    with pytest.raises(RuntimeError, match="input/frame changed"):
        grid.verify()


@pytest.mark.parametrize("change", ["resolution", "bounds", "labels"])
def test_map_layout_is_never_refitted(change):
    args = dict(labels=np.full((8, 12), 254, np.uint8), origin_xy=[-2., -1.],
                bounds_xy=[-2., -1., -.8, -.2], resolution_m=.1,
                source={"path": "synthetic", "sha256": "a" * 64})
    if change == "resolution":
        args["resolution_m"] = .2
    elif change == "bounds":
        args["bounds_xy"] = [-2., -1., 1., 1.]
    else:
        args["labels"][0, 0] = 5  # Native SDK raw5 is not the admitted ROS free254.
    with pytest.raises(ValueError, match="exact finite native"):
        native_distance_grid(**args)


@pytest.mark.parametrize("count", [3, 15])
def test_full_scene_output_matches_original_generator_without_rng(synthetic_source, count):
    planner = NativeSocialForcePlanner.for_CPU_snapshot(scene_source=synthetic_source[count], actors=count, seed=7)
    call = inputs(planner)
    originals = {key: value.clone() for key, value in call.items() if isinstance(value, torch.Tensor)}
    rng = torch.get_rng_state().clone()
    np_rng = np.random.get_state()
    first = planner.plan(**call)
    second = planner.plan(**call)
    direct = SocialForceTrajectoryGenerator.generate(planner.generator,
        call["states"][:, :2].numpy(), call["states"][:, 2:4].numpy(), call["waypoints"][:, -1].numpy())
    assert first.xy.shape == (count, 51, 2)
    assert torch.equal(first.xy, torch.from_numpy(direct.raw_xy.copy()).float())
    assert torch.equal(first.xy, second.xy) and torch.equal(first.xy[:, 0], call["states"][:, :2])
    assert all(torch.equal(call[key], value) for key, value in originals.items())
    assert torch.equal(torch.get_rng_state(), rng)
    after = np.random.get_state()
    assert np_rng[0] == after[0] and np.array_equal(np_rng[1], after[1]) and np_rng[2:] == after[2:]
    assert not torch.cuda.is_initialized()
    description = planner.describe()
    assert description["actual_world_boundary_checked"] is False
    assert description["physical_validation_established"] is False
    assert not description["diffusion_weights_loaded"]


@pytest.mark.parametrize("change", ["subset", "peer", "objective", "callback", "AABB", "nonfinite"])
def test_invalid_joint_contract_fails(synthetic_source, change):
    planner = NativeSocialForcePlanner.for_CPU_snapshot(scene_source=synthetic_source[3], actors=3, seed=7)
    call = inputs(planner)
    if change == "subset":
        call["states"] = call["states"][:2]
    elif change == "peer":
        call["neighbors"][0, 0, -1, 0] += 1.
    elif change == "objective":
        call["undeclared_objective"] = 1.
    elif change == "callback":
        call["scene_collision_cost_fn"] = lambda *args: None
    elif change == "AABB":
        call["scene_bounds_xy"] = torch.zeros(4)
    else:
        call["states"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        planner.plan(**call)


def test_scene_mutation_and_false_live_world_are_rejected(synthetic_source):
    planner = NativeSocialForcePlanner.for_CPU_snapshot(scene_source=synthetic_source[3], actors=3, seed=7)
    planner._provider._map = planner._provider._map.copy()
    planner._provider._map[0, 0] = 205
    with pytest.raises(ValueError, match="differs from selected"):
        planner.describe()
    planner = NativeSocialForcePlanner.for_CPU_snapshot(scene_source=synthetic_source[3], actors=3, seed=7)
    planner.actual_world_boundary_checked = True
    with pytest.raises(ValueError, match="lost the live"):
        planner.describe()


def test_cpu_source_import_does_not_launch_sdk_or_import_torch():
    command = ("import sys; from CrowdSim.humanoid.trajectory.native_social_force_runtime import source_bindings; "
               "assert source_bindings(); assert not any(x in sys.modules for x in "
               "['torch', 'pxr', 'isaacsim', 'isaaclab', 'omni'])")
    result = subprocess.run([sys.executable, "-c", command], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_selected_configuration_mutation_is_rejected(synthetic_source, tmp_path):
    selected = tmp_path / "scene.json"
    selected.write_bytes(synthetic_source[3].read_bytes())
    planner = NativeSocialForcePlanner.for_CPU_snapshot(scene_source=selected, actors=3, seed=7)
    assert planner.scene_identity == file_identity(selected)
    selected.write_text(selected.read_text() + "\n")
    with pytest.raises(ValueError, match="resource changed"):
        planner.describe()
