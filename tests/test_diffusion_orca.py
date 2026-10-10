"""Real CPU RVO2 experts, scene-disjoint conditional windows, and raster frames.

Generator tests use an isolated optional dependency environment; they skip
when the public RVO2 extension is absent, rather than replacing it with a stub.
"""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from CrowdSim.humanoid.orca_corpus import (
    ORCACorpus, crop_raster, generate_corpus, load_trace_generator,
    rasterize_obstacles, sample_obstacles, simulate_episode, trajectory_clearance,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def actual_orca():
    module = pytest.importorskip("rvo2", reason="Public RVO2 extension is installed only in the isolated ORCA generator environment")
    pytest.importorskip("shapely")
    return module


@pytest.fixture(scope="module")
def corpus_path(tmp_path_factory, actual_orca):
    directory = tmp_path_factory.mktemp("actual_orca") / "corpus"
    return generate_corpus(directory, ROOT / "third_party/trace", ROOT / "data/tools/orca_generator/Python-RVO2",
                           split_scenes={"train": 1, "validation": 1, "test": 1},
                           episodes_per_scene=1, seconds=10.)


def test_actual_rvo2_head_on_avoidance_and_polygon_stop(actual_orca):
    obstacle = [[(-1., -.5), (1., -.5), (1., .5), (-1., .5)]]
    sim = actual_orca.PyRVOSimulator(1 / 60, 15, 10, 3, 3, .4, 2)
    sim.addObstacle(obstacle[0])
    sim.processObstacles()
    initial = [(-3., 2.), (3., 2.), (0., -3.), (4., -3.)]
    preferred = [(1., 0.), (-1., 0.), (0., 1.), (0., 1.)]
    for position, velocity in zip(initial, preferred):
        agent = sim.addAgent(position)
        sim.setAgentPrefVelocity(agent, velocity)
    positions = []
    for _ in range(600):
        sim.doStep()
        positions.append([sim.getAgentPosition(agent) for agent in range(4)])
    actual = np.asarray(positions).transpose(1, 0, 2)
    qa = trajectory_clearance(actual, np.full(4, .4), obstacle)
    assert qa["pair_overlap_sample_count"] == qa["obstacle_overlap_sample_count"] == 0
    # Straight preferred motion would put agent2 inside the obstacle at t=3s.
    assert actual[2, 179, 1] < -.89
    assert np.linalg.norm(actual[0, -1] - actual[1, -1]) >= .7999
    assert actual[3, -1, 1] > 5  # unblocked agent actually moves


def test_actual_episode_determinism_uniform_time_and_velocity_heading(actual_orca):
    generator = load_trace_generator(ROOT / "third_party/trace")
    obstacles = sample_obstacles(generator, 19)
    first = simulate_episode(generator, obstacles, seed=7, count=8, seconds=10.)
    second = simulate_episode(generator, obstacles, seed=7, count=8, seconds=10.)
    assert np.array_equal(first["states"], second["states"])
    assert np.array_equal(first["times"], np.arange(101) / 10)
    assert first["states"].shape == (8, 101, 5)
    assert np.allclose(first["states"][..., 4], np.arctan2(first["states"][..., 3], first["states"][..., 2]))
    assert first["qa"]["checked_hz"] == 60
    assert first["qa"]["pair_overlap_sample_count"] == first["qa"]["obstacle_overlap_sample_count"] == 0
    with pytest.raises(ValueError, match="4..15"):
        simulate_episode(generator, obstacles, seed=7, count=3, seconds=10.)
    with pytest.raises(ValueError, match="exact 10Hz"):
        simulate_episode(generator, obstacles, seed=7, count=4, seconds=10.01)


def test_raster_heading_and_real_occupied_semantics(actual_orca):
    raster = rasterize_obstacles([[(1., -.5), (2., -.5), (2., .5), (1., .5)]])
    states = np.asarray([[0., 0., 1., 0., 0.], [0., 0., 0., 1., math.pi / 2]])
    cropped = crop_raster(raster, states)
    assert cropped.shape == (2, 2, 224, 224)
    assert np.array_equal(cropped.sum(axis=1), np.ones((2, 224, 224)))
    assert cropped[0, 1, 112, 74] == 1  # world(1.5,0) -> local(1.5,0)
    assert cropped[1, 1, 94, 56] == 1  # yaw pi/2 -> local(0,-1.5)
    assert cropped[0, 1, 112, 56] == 0  # ego is free
    outside = crop_raster(raster, np.asarray([[100., 100., 0., 0., 0.]]))
    assert np.all(outside == .5)


def test_conditional_windows_are_synchronized_real_neighbors_and_masks(corpus_path):
    corpus = ORCACorpus(corpus_path, seed=11)
    sample = corpus.numpy_batch(16)
    assert sample["history"].shape == (16, 31, 5)
    assert sample["future"].shape == (16, 52, 5)
    assert np.array_equal(sample["states"], sample["history"][:, -1])
    assert sample["neighbor_mask"][:, :, -1].any(axis=1).all()
    assert (sample["raster"][:, 1] == 1).any(axis=(1, 2)).all()
    for row, window in enumerate(sample["window_indices"]):
        episode_index, agent, frame = corpus.windows[window]
        episode = corpus.episodes[episode_index]
        assert np.array_equal(sample["future"][row], episode["states"][agent, frame + 1:frame + 53])
        ids = sample["neighbor_ids"][row]
        valid = ids >= 0
        assert agent not in ids[valid]
        assert np.array_equal(sample["neighbors"][row, valid], episode["states"][ids[valid], frame - 30:frame + 1])
        distance = np.linalg.norm(sample["neighbors"][row, valid, -1, :2] - sample["states"][row, :2], axis=-1)
        assert (distance <= 15).all()
        assert not sample["neighbor_mask"][row, ~valid].any()
        assert not sample["neighbors"][row, ~valid].any()
    state = copy.deepcopy(corpus.generator.bit_generator.state)
    expected = corpus.numpy_batch(8)
    corpus.generator.bit_generator.state = state
    assert np.array_equal(expected["window_indices"], corpus.numpy_batch(8)["window_indices"])


def test_manifest_scene_episode_layout_disjointness_and_tamper(corpus_path, tmp_path):
    manifest = json.loads(corpus_path.read_text())
    scenes, episodes, layouts, seeds = [], [], [], []
    for entry in manifest["splits"].values():
        for scene in entry["scenes"]:
            scenes.append(scene["scene_id"])
            layouts.append(scene["layout_sha256"])
            seeds.append(scene["scene_seed"])
            for episode in scene["episodes"]:
                episodes.append(episode["episode_id"])
                seeds.append(episode["episode_seed"])
    assert all(len(values) == len(set(values)) for values in (scenes, episodes, layouts, seeds))
    assert manifest["author_dataset"] is False
    leaked = copy.deepcopy(manifest)
    leaked["splits"]["validation"]["scenes"] = copy.deepcopy(leaked["splits"]["train"]["scenes"])
    path = corpus_path.parent / "leaked_manifest.json"
    path.write_text(json.dumps(leaked))
    with pytest.raises(ValueError, match="leakage"):
        ORCACorpus(path)
    corrupted = copy.deepcopy(manifest)
    corrupted["splits"]["train"]["scenes"][0]["map"]["sha256"] = "0" * 64
    path = corpus_path.parent / "corrupted_manifest.json"
    path.write_text(json.dumps(corrupted))
    with pytest.raises(ValueError, match="changed"):
        ORCACorpus(path)


def test_offline_metrics_use_all_synchronized_experts_and_true_polygons(corpus_path):
    from scripts.train_orca_diffusion import offline_plan_metrics, polygon_surface_gap

    corpus = ORCACorpus(corpus_path)
    sample = corpus.numpy_batch(4)
    exact = sample["future"][:, :50, :2].copy()
    metrics = offline_plan_metrics(corpus, sample, exact)
    assert metrics["ADE_m"] == metrics["FDE_m"] == 0
    assert metrics["neighbor_collision_point_fraction"] == metrics["obstacle_collision_point_fraction"] == 0
    for row, window in enumerate(sample["window_indices"]):
        episode_index, agent, frame = corpus.windows[window]
        other = (agent + 1) % len(corpus.episodes[episode_index]["states"])
        exact[row] = corpus.episodes[episode_index]["states"][other, frame + 1:frame + 51, :2]
    assert offline_plan_metrics(corpus, sample, exact)["neighbor_collision_point_fraction"] == 1
    polygon = [[(0., 0.), (2., 0.), (2., 2.), (0., 2.)]]
    points = np.asarray([[1., 1.], [3., 1.], [-.2, 1.]])
    assert np.allclose(polygon_surface_gap(points, polygon), [-1.4, .6, -.2])


def test_social_training_protects_frozen_checkpoint_and_memory_cap(tmp_path):
    from scripts.train_orca_diffusion import DEFAULT_INITIAL, parse_args

    with pytest.raises(SystemExit):
        parse_args(["--output-dir", str(DEFAULT_INITIAL.parent)])
    with pytest.raises(SystemExit):
        parse_args(["--cuda-memory-fraction", ".2"])
    with pytest.raises(SystemExit):
        parse_args(["--steps", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--init-checkpoint", "initial.ckpt", "--resume", "resume.ckpt"])
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "last.ckpt").write_bytes(b"existing")
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", str(directory)])
    args = parse_args(["--output-dir", str(directory), "--resume", str(directory / "last.ckpt")])
    assert args.cuda_memory_fraction == .15


def test_real_conditional_window_feeds_five_state_model(corpus_path):
    torch = pytest.importorskip("torch")
    from CrowdSim.humanoid.diffusion import PaperDiffusionPlanner, PaperDiffusionTrainBatch

    sample = ORCACorpus(corpus_path).diffusion_batch(2)
    planner = PaperDiffusionPlanner.__new__(PaperDiffusionPlanner)
    planner.device = torch.device("cpu")
    batch = PaperDiffusionTrainBatch.from_world(planner, sample["states"], sample["history"],
        sample["neighbors"], sample["raster"], sample["future"],
        history_mask=sample["history_mask"], neighbor_mask=sample["neighbor_mask"])
    assert batch.clean_trajectory.shape == (2, 52, 5)
    assert batch.context["neighbor_mask"].any()
    assert (batch.context["raster"][:, 1] == 1).any()
