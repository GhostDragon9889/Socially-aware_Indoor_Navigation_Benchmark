"""CPU contracts against the complete public SFM class; no SDK or weights."""
import ast
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from CrowdSim.humanoid.trajectory.social_force import (
    MAINTAINED_SFM_SHA256, PUBLIC_SFM_SHA256, VERIFIED_SFM_REVISIONS,
    SFMParameters, SocialForceTrajectoryGenerator, SourceBinding, StaticDistanceGrid,
)


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "CrowdSim/world/sfm.py"


def grid(*, wall=False):
    labels = np.full((400, 400), 254, dtype=np.uint8)
    distance = np.full(labels.shape, 20., dtype=np.float64)
    if wall:
        labels[:, 200:] = 0
        distance = np.maximum((200 - np.arange(400)) * .1, 0.)[None, :].repeat(400, axis=0)
    return StaticDistanceGrid(distance, labels, origin_xy=(-20., -20.), resolution_m=.1,
                              source_id="synthetic-test-grid", source_sha256=hashlib.sha256(labels.tobytes()).hexdigest())


def generator(environment=None, params=None):
    return SocialForceTrajectoryGenerator(
        environment or grid(), parameters=params or SFMParameters(), seed=7,
        sources=(SourceBinding.bind_sfm(PUBLIC),))


def public_controller(environment, params):
    # Execute only the unchanged class AST, with its actual NumPy dependency.
    # The public module's plotting/OpenCV imports and __main__ are not executed.
    tree = ast.parse(PUBLIC.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Social_Force")
    namespace = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(PUBLIC), "exec"), namespace)
    cfg = {"env": {"dt": params.integration_dt_s, "safe_distance": params.safe_distance_m,
                   "neighbor_radius": params.neighbor_radius_m, "reach_distance": params.reach_distance_m,
                   "prediction_horizon": params.prediction_horizon_s,
                   "ttc_threshold": params.ttc_threshold_s, "ttc_gain": params.ttc_gain},
           "agent": {"radius": .35, "max_vel": params.max_speed_mps},
           "map": {"resolution": environment.resolution_m}}
    controller = namespace["Social_Force"](environment.distance_m, cfg)
    controller.gain_k = params.attraction_gain
    controller.gain_a_static, controller.gain_b_static = params.static_gain, params.static_decay_m
    controller.gain_a_agent, controller.gain_b_agent = params.agent_gain, params.agent_decay_m
    return controller


def public_step(controller, environment, params, positions, velocities, goals):
    next_velocities, terms = [], []
    pixels = environment.pixel_indices(positions)
    for actor in range(len(positions)):
        relative = positions - positions[actor]
        distances = np.linalg.norm(relative, axis=-1)
        indices = np.flatnonzero((np.arange(len(positions)) != actor) & (distances < params.neighbor_radius_m))
        velocity, forces = controller.get_action(
            (positions[actor], tuple(pixels[actor]), velocities[actor], goals[actor]),
            (indices, distances[indices], relative[indices], velocities[indices] - velocities[actor]))
        next_velocities.append(velocity)
        terms.append(forces)
    next_velocities = np.asarray(next_velocities)
    return positions + next_velocities * params.integration_dt_s, next_velocities, np.asarray(terms)


def test_complete_force_terms_match_public_sfm_and_synchronous_snapshot():
    environment = grid(wall=True)
    params = SFMParameters()
    planner = generator(environment, params)
    positions = np.array([[-.85, 0.], [-1.8, .12], [-2.7, -.1], [-1.4, 1.1]])
    velocities = np.array([[.4, 0.], [.2, -.1], [.9, .1], [-.1, -.4]])
    goals = np.array([[-.2, 0.], [-.2, .2], [-.2, -.2], [-3., -1.]])
    expected_position, expected_velocity, terms = public_step(
        public_controller(environment, params), environment, params, positions, velocities, goals)
    actual = planner.step(positions, velocities, goals)
    np.testing.assert_allclose(actual.xy, expected_position, rtol=2e-7, atol=2e-8)
    np.testing.assert_allclose(actual.velocity, expected_velocity, rtol=2e-7, atol=2e-8)
    for actual_term, expected_term in ((actual.forces.agent, terms[:, 0]),
                                     (actual.forces.static, terms[:, 1]),
                                     (actual.forces.attraction, terms[:, 2]),
                                     (actual.forces.ttc, terms[:, 3])):
        np.testing.assert_allclose(actual_term, expected_term, rtol=3e-6, atol=2e-6)
    assert np.linalg.norm(actual.forces.static) > 0
    assert np.linalg.norm(actual.forces.agent) > 0
    assert np.linalg.norm(actual.forces.ttc) > 0


def test_full_horizon_matches_joint_public_steps_and_exact_sampling():
    planner = generator()
    positions = np.array([[-3., -.3], [3., .3], [0., 2.5]])
    velocities = np.array([[.7, 0.], [-.7, 0.], [0., -.4]])
    goals = np.array([[5., -.3], [-5., .3], [0., -5.]])
    original = positions.copy()
    controller = public_controller(planner.grid, planner.parameters)
    expected = [positions.copy()]
    current, velocity = positions.copy(), velocities.copy()
    for _ in range(150):
        current, velocity, _ = public_step(controller, planner.grid, planner.parameters, current, velocity, goals)
        expected.append(current.copy())
    result = planner.generate(positions, velocities, goals)
    assert result.raw_xy.shape == (3, 51, 2)
    assert result.future_xy.shape == (3, 10, 2)
    assert result.internal_xy.shape == (3, 151, 2)
    np.testing.assert_allclose(result.internal_xy, np.asarray(expected).transpose(1, 0, 2), atol=3e-6, rtol=2e-6)
    np.testing.assert_array_equal(result.raw_xy, result.internal_xy[:, ::3])
    np.testing.assert_array_equal(result.future_xy, result.raw_xy[:, 5::5])
    np.testing.assert_allclose(result.times_s, np.arange(51) / 10, atol=1e-15)
    np.testing.assert_array_equal(positions, original)
    contract = result.reference_adapter_kwargs()
    assert contract["control_dt_s"] == 1 / 30
    assert contract["raw_knot_dt_s"] == .1
    assert contract["max_acceleration_mps2"] == 2
    assert result.metadata["reference_constraints_applied"] is False


def test_permutation_and_opposing_symmetry_over_the_full_horizon():
    planner = generator()
    positions = np.array([[-3., 0.], [3., 0.], [0., 3.], [0., -3.]])
    velocities = np.zeros_like(positions)
    goals = -positions
    result = planner.generate(positions, velocities, goals)
    np.testing.assert_allclose(result.raw_xy[0], -result.raw_xy[1], atol=1e-10)
    np.testing.assert_allclose(result.raw_xy[2], -result.raw_xy[3], atol=1e-10)
    order = np.array([2, 0, 3, 1])
    reordered = planner.generate(positions[order], velocities[order], goals[order])
    np.testing.assert_allclose(reordered.raw_xy, result.raw_xy[order], atol=1e-10)


def test_static_wall_repels_in_world_x_and_flipped_world_y():
    environment = grid(wall=True)
    planner = generator(environment)
    state = planner.step(np.array([[-.6, 0.]]), np.zeros((1, 2)), np.array([[-.2, 0.]]))
    assert state.forces.static[0, 0] < 0
    assert state.velocity[0, 0] < 0
    distance = np.arange(80, dtype=np.float64)[:, None].repeat(80, axis=1) * .1
    labels = np.full(distance.shape, 254, dtype=np.uint8)
    labels[0] = 0
    environment = StaticDistanceGrid(distance, labels, origin_xy=(-4., -4.), resolution_m=.1,
                                     source_id="top-wall", source_sha256="a" * 64)
    state = generator(environment).step(np.array([[0., 3.35]]), np.zeros((1, 2)), np.array([[0., 3.75]]))
    assert state.forces.static[0, 1] < 0


@pytest.mark.parametrize("distance,velocity,expected", [(1., 1., True), (3., .5, False),
                                                        (1., -1., False), (1., 0., False)])
def test_ttc_threshold_receding_and_stationary_boundaries(distance, velocity, expected):
    planner = generator()
    position = np.array([[-distance / 2, 0.], [distance / 2, 0.]])
    velocities = np.array([[velocity, 0.], [-velocity, 0.]])
    result = planner.step(position, velocities, -position)
    assert bool(np.linalg.norm(result.forces.ttc) > 1e-10) == expected
    if expected:
        # Exact head-on degeneracy uses the upstream goal-relative pass-right rule.
        assert result.forces.ttc[0, 1] < 0 < result.forces.ttc[1, 1]


def test_all_neighbor_rows_are_used_beyond_five_and_coincident_agents_remain_finite():
    planner = generator()
    angles = np.arange(8) * np.pi / 4
    positions = np.vstack(([0., 0.], np.column_stack((np.cos(angles), np.sin(angles))) + [1.3, 0.]))
    goals = positions + [4., .2]
    velocities = np.zeros_like(positions)
    actual = planner.step(positions, velocities, goals)
    expected = public_step(public_controller(planner.grid, planner.parameters), planner.grid,
                           planner.parameters, positions, velocities, goals)
    np.testing.assert_allclose(actual.velocity, expected[1], rtol=1e-6, atol=1e-7)
    result = planner.generate(np.zeros((3, 2)), np.zeros((3, 2)), np.array([[5., 0.], [0., 5.], [-5., 0.]]))
    assert np.isfinite(result.raw_xy).all()
    assert np.linalg.norm(result.raw_velocity[:, 1:], axis=-1).max() <= planner.parameters.max_speed_mps + 1e-12


@pytest.mark.parametrize("name", ["positions", "velocities", "goals"])
def test_nan_inputs_fail(name):
    values = {"positions": np.zeros((2, 2)), "velocities": np.zeros((2, 2)), "goals": np.ones((2, 2))}
    values[name][0, 0] = np.nan
    with pytest.raises(FloatingPointError, match="nonfinite"):
        generator().generate(**values)


def test_unknown_exterior_and_swept_unknown_fail_without_clipping():
    environment = grid()
    environment.labels[200, 200] = 205
    with pytest.raises(RuntimeError, match="changed"):
        generator(environment).generate(np.zeros((1, 2)), np.zeros((1, 2)), np.ones((1, 2)))
    environment = grid()
    with pytest.raises(ValueError, match="exterior"):
        generator(environment).generate(np.array([[20., 0.]]), np.zeros((1, 2)), np.ones((1, 2)))
    labels = np.full((20, 20), 254, dtype=np.uint8)
    labels[:, 11] = 205
    distance = np.full((20, 20), 20.)
    distance[:, 11] = 0
    environment = StaticDistanceGrid(distance, labels, origin_xy=(-1., -1.), resolution_m=.1,
                                     source_id="unknown-stripe", source_sha256="b" * 64)
    with pytest.raises(ValueError, match="unknown"):
        generator(environment).generate(np.array([[.15, 0.]]), np.array([[1., 0.]]), np.array([[.8, 0.]]))
    with pytest.raises(ValueError, match="unknown"):
        environment.check_segments(np.array([[0., 0.]]), np.array([[.8, 0.]]))


def test_diagonal_sweep_cannot_skip_a_short_unknown_cell():
    labels = np.full((10, 10), 254, dtype=np.uint8)
    labels[4, 4] = 205
    distance = np.ones((10, 10))
    distance[4, 4] = 0
    environment = StaticDistanceGrid(distance, labels, origin_xy=(0., 0.), resolution_m=1.,
                                     source_id="diagonal-grid", source_sha256="d" * 64)
    # Crosses this cell for only .01m of Y before leaving through X=5.
    with pytest.raises(ValueError, match="unknown"):
        environment.check_segments(np.array([[3., 4.01]]), np.array([[7., 6.01]]))


def test_state_input_changed_during_prediction_faults_the_complete_result(monkeypatch):
    planner = generator()
    positions = np.array([[0., 0.]])
    original = planner.grid._check_segments
    def mutate_once(before, after):
        original(before, after)
        positions[0, 0] = .5
    monkeypatch.setattr(planner.grid, "_check_segments", mutate_once)
    with pytest.raises(RuntimeError, match="input changed"):
        planner.generate(positions, np.zeros((1, 2)), np.array([[2., 0.]]))


def test_strict_neighbor_radius_and_seed_have_no_hidden_rng_effects():
    planner = generator()
    positions = np.array([[0., 0.], [4., 0.]])
    velocities = np.zeros_like(positions)
    goals = positions + [1., 0.]
    state = planner.step(positions, velocities, goals)
    np.testing.assert_array_equal(state.forces.agent, np.zeros_like(positions))
    positions[1, 0] -= .001
    assert np.linalg.norm(planner.step(positions, velocities, goals).forces.agent) > 0
    rng_before = np.random.get_state()
    first = planner.generate(positions, velocities, goals)
    second = planner.generate(positions, velocities, goals)
    rng_after = np.random.get_state()
    np.testing.assert_array_equal(rng_before[1], rng_after[1])
    assert rng_before[0] == rng_after[0] and rng_before[2:] == rng_after[2:]
    np.testing.assert_array_equal(first.raw_xy, second.raw_xy)


def test_bound_source_and_parameters_changes_fail(tmp_path):
    source = tmp_path / "sfm.py"
    source.write_bytes(PUBLIC.read_bytes())
    planner = SocialForceTrajectoryGenerator(grid(), parameters=SFMParameters(), seed=7,
                                             sources=(SourceBinding.bind_sfm(source),))
    source.write_text("# replaced\n")
    with pytest.raises(RuntimeError, match="source.*changed"):
        planner.generate(np.zeros((1, 2)), np.zeros((1, 2)), np.ones((1, 2)))
    planner = generator()
    object.__setattr__(planner.parameters, "ttc_gain", 0.)
    with pytest.raises(RuntimeError, match="parameters.*changed"):
        planner.generate(np.zeros((1, 2)), np.zeros((1, 2)), np.ones((1, 2)))


@pytest.mark.parametrize("dt", [.04, np.nan, 0., -1.])
def test_integration_clock_requires_exact_ten_hz_sampling(dt):
    with pytest.raises(ValueError):
        generator(params=SFMParameters(integration_dt_s=dt))


def test_cpu_child_process_exits_without_loading_torch_or_sdk():
    code = (
        "import sys, numpy as np; "
        "from CrowdSim.humanoid.trajectory.social_force import *; "
        "g=StaticDistanceGrid(np.ones((200,200))*20, np.full((200,200),254,dtype=np.uint8), "
        "origin_xy=(-10,-10), resolution_m=.1, source_id='cpu-child', source_sha256='c'*64); "
        "p=SocialForceTrajectoryGenerator(g, parameters=SFMParameters(), seed=7, "
        "sources=(SourceBinding.bind_sfm('CrowdSim/world/sfm.py'),)); "
        "r=p.generate(np.array([[-2.,0],[2.,0]]),np.zeros((2,2)),np.array([[2.,0],[-2.,0]])); "
        "assert r.future_xy.shape == (2,10,2); "
        "assert not any(n=='torch' or n.startswith(('omni','isaaclab','isaacsim','pxr')) for n in sys.modules); "
        "print('CPU_ONLY_EXIT_OK')"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code], cwd=ROOT, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "CPU_ONLY_EXIT_OK"

def test_reviewed_source_preserves_original_disabled_fov_equations_and_labels_its_revision(
    tmp_path,
):
    original = subprocess.check_output(
        ["git", "show", "65d1bbe9:src/CrowdSim/world/sfm.py"], cwd=ROOT
    )
    assert hashlib.sha256(original).hexdigest() == PUBLIC_SFM_SHA256
    assert hashlib.sha256(PUBLIC.read_bytes()).hexdigest() == MAINTAINED_SFM_SHA256

    def methods(raw):
        cls = next(
            node
            for node in ast.parse(raw).body
            if isinstance(node, ast.ClassDef) and node.name == "Social_Force"
        )
        return {
            node.name: ast.dump(node, include_attributes=False)
            for node in cls.body
            if isinstance(node, ast.FunctionDef)
        }

    before, after = methods(original), methods(PUBLIC.read_bytes())
    assert set(before) == set(after)
    for name in ("__init__", "compute_gradient", "compute_forces", "get_action"):
        assert before[name] == after[name]
    assert before["fov_filter"] != after["fov_filter"]
    historical = tmp_path / "original.py"
    historical.write_bytes(original)
    bound = SourceBinding.bind_sfm(historical)
    assert bound.sha256 == PUBLIC_SFM_SHA256
    bound.verify()
    assert generator().describe()["implementation_revisions"] == [
        VERIFIED_SFM_REVISIONS[MAINTAINED_SFM_SHA256]
    ]
    historical.write_bytes(PUBLIC.read_bytes())
    with pytest.raises(RuntimeError, match="source changed"):
        bound.verify()  # Another allowed revision cannot replace the bound one.


def test_unregistered_sfm_revision_is_rejected_at_bind_time(tmp_path):
    source = tmp_path / "unreviewed.py"
    source.write_bytes(PUBLIC.read_bytes() + b"\n# unreviewed modification\n")
    with pytest.raises(ValueError, match="exact verified"):
        SourceBinding.bind_sfm(source)
