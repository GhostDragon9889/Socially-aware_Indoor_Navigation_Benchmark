"""Synchronized ORCA expert data for five-state trajectory diffusion.

The generator calls the pinned public TRACE obstacle/goal samplers and the
actual RVO2 C++ solver. It is newly generated reconstruction data, not the
unpublished NavIsaacLab authors' data. TRACE-derived generation conventions
are covered by the NVIDIA Source Code License-NC; see the dataset LICENSES.
No Isaac Sim, CUDA, torch, or pickle is needed to generate/read numpy batches.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np


SCHEMA = "navisaaclab.orca_social_reconstruction.v1"
TRACE_COMMIT = "594b2af0824c5853e1a1884e167cd5ed097dd81c"
RVO2_COMMIT = "c2c46ba8d59556aa10faf03479293236efea154d"
HISTORY = 31
FUTURE = 52
SIM_HZ = 60
DATA_HZ = 10
MAP_PX_PER_M = 12
MAP_WIDTH_M = 90
NEIGHBOR_RADIUS_M = 15.0


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def _seed_numpy(seed):
    """Original TRACE samplers use the legacy global numpy RNG."""
    previous = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(previous)


def load_trace_generator(trace_dir):
    path = Path(trace_dir) / "orca_sim/gen_dataset.py"
    spec = importlib.util.spec_from_file_location("_pinned_trace_orca_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_obstacles(generator, seed, *, maximum=5):
    """Use official shapes/ranges, rejecting empty maps for this dataset."""
    if maximum < 1:
        raise ValueError("At least one obstacle is required for conditional training")
    with _seed_numpy(seed):
        for _ in range(100):
            obstacles = generator.samp_obstacles(maximum, generator.BOUNDARY_RADIUS)
            if obstacles:
                return obstacles
    raise RuntimeError("Official obstacle sampler repeatedly returned empty maps")


def _sample_agents(generator, obstacles, count):
    """Official property distributions with a bounded valid-spawn sampler."""
    import shapely
    from shapely.geometry import Polygon

    if not 4 <= count <= 15:
        raise ValueError("This social dataset requires 4..15 synchronized agents")
    props = {"numAgents": count}
    for name, (limits, width, kind) in generator.AGT_PROP_INFO.items():
        if kind is int:
            values = np.random.randint(limits[0], limits[1] + 1, (count, width))
        else:
            values = np.random.uniform(*limits, (count, width))
        props[name] = values
    union = shapely.union_all([Polygon(poly) for poly in obstacles])
    positions = []
    for agent in range(count):
        for _ in range(10000):
            point = np.random.uniform(-generator.BOUNDARY_RADIUS, generator.BOUNDARY_RADIUS, 2)
            # True circle clearance; do not approximate a radius with buffer segments.
            geom = shapely.Point(point)
            radius = props["radius"][agent, 0]
            clear_map = not union.intersects(geom) and union.distance(geom) > radius
            clear_agents = all(np.linalg.norm(point - prev) > radius + props["radius"][index, 0]
                               for index, prev in enumerate(positions))
            if clear_map and clear_agents and np.linalg.norm(point) > 1e-8:
                positions.append(point)
                break
        else:
            raise RuntimeError("No valid initial ORCA position after 10000 proposals")
    props["initPos"] = np.asarray(positions)
    theta = np.arctan2(-props["initPos"][:, 1:2], -props["initPos"][:, 0:1])
    theta += np.random.uniform(-math.pi / 4, math.pi / 4, theta.shape)
    props["prefVelocity"] = props["prefVelocity"] * np.concatenate((np.cos(theta), np.sin(theta)), axis=-1)
    return props


def trajectory_clearance(positions, radii, obstacles):
    """Check every 60Hz sample against true circles and polygon boundaries."""
    import shapely
    from shapely.geometry import Polygon

    positions = np.asarray(positions)
    count = positions.shape[0]
    pair = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    pair -= radii[:, None, None] + radii[None, :, None]
    pair[np.arange(count), np.arange(count)] = np.inf
    union = shapely.union_all([Polygon(poly) for poly in obstacles])
    points = shapely.points(positions)
    obstacle_gap = shapely.distance(points, union) - radii[:, None]
    inside = shapely.intersects(union, points)
    # Inside a polygon has negative signed distance even for zero-radius points.
    obstacle_gap[inside] = -shapely.distance(points[inside], union.boundary) - np.broadcast_to(radii[:, None], inside.shape)[inside]
    return {"checked_hz": SIM_HZ, "min_pair_surface_gap_m": float(pair.min()),
            "min_obstacle_surface_gap_m": float(obstacle_gap.min()),
            "pair_overlap_sample_count": int(np.count_nonzero(pair < -1e-4) // 2),
            "obstacle_overlap_sample_count": int(np.count_nonzero(obstacle_gap < -1e-4))}


def simulate_episode(generator, obstacles, *, seed, count, seconds=20.0):
    """Real RVO2, t=0 then every six physics steps, with velocity-derived yaw."""
    import rvo2

    if not math.isfinite(seconds) or seconds < (HISTORY + FUTURE) / DATA_HZ:
        raise ValueError("Episode length must support 31+52 frames and exact 10Hz sampling")
    steps = round(seconds * SIM_HZ)
    if abs(steps / SIM_HZ - seconds) > 1e-8 or steps % (SIM_HZ // DATA_HZ):
        raise ValueError("Episode length must support 31+52 frames and exact 10Hz sampling")
    with _seed_numpy(seed):
        agents = _sample_agents(generator, obstacles, count)
        sim = rvo2.PyRVOSimulator(1 / SIM_HZ, 10, 10, 2, 2, .4, 2)
        for poly in obstacles:
            # Official samples include a repeated closing point; RVO2 requires vertices.
            vertices = [tuple(point) for point in poly]
            if vertices[0] == vertices[-1]:
                vertices = vertices[:-1]
            sim.addObstacle(vertices)
        sim.processObstacles()
        for agent in range(count):
            sim.addAgent(tuple(agents["initPos"][agent]), float(agents["neighborDist"][agent, 0]),
                         int(agents["maxNeighbors"][agent, 0]), float(agents["timeHorizon"][agent, 0]),
                         float(agents["timeHorizonObst"][agent, 0]), float(agents["radius"][agent, 0]),
                         float(agents["maxSpeed"][agent, 0]), tuple(agents["initVelocity"][agent]))
            sim.setAgentPrefVelocity(agent, tuple(agents["prefVelocity"][agent]))
        positions = np.empty((count, steps + 1, 2), dtype=np.float64)
        velocities = np.empty_like(positions)
        next_resample = (np.random.uniform(*generator.RESAMPLE_GOAL_RANGE, count) * SIM_HZ).astype(int)
        for step in range(steps + 1):
            if step:
                sim.doStep()
            positions[:, step] = [sim.getAgentPosition(agent) for agent in range(count)]
            velocities[:, step] = [sim.getAgentVelocity(agent) for agent in range(count)]
            if step:
                next_resample = generator.resample_goals(sim, step, next_resample)
        radii = agents["radius"][:, 0]
        qa = trajectory_clearance(positions, radii, obstacles)
        if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
            raise RuntimeError("Actual RVO2 output is non-finite")
        if qa["pair_overlap_sample_count"] or qa["obstacle_overlap_sample_count"]:
            raise RuntimeError(f"Actual ORCA episode failed geometric clearance: {qa}")
        sample = slice(None, None, SIM_HZ // DATA_HZ)
        pos, vel = positions[:, sample], velocities[:, sample]
        yaw = np.arctan2(vel[..., 1:2], vel[..., 0:1])
        states = np.concatenate((pos, vel, yaw), axis=-1).astype(np.float32)
        qa.update(max_speed_m_s=float(np.linalg.norm(velocities, axis=-1).max()),
                  max_abs_position_m=float(np.abs(positions).max()),
                  simulation_steps=steps, frames=states.shape[1])
        return {"states": states, "times": np.arange(states.shape[1], dtype=np.float64) / DATA_HZ,
                "radii": radii.astype(np.float32), "properties": agents, "qa": qa}


def rasterize_obstacles(obstacles):
    """Two global semantic layers, TRACE ORCA positive-y row convention."""
    import shapely
    from shapely.geometry import Polygon

    size = MAP_WIDTH_M * MAP_PX_PER_M
    row, col = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    union = shapely.union_all([Polygon(poly) for poly in obstacles])
    occupied = shapely.intersects_xy(union, (col - size / 2) / MAP_PX_PER_M,
                                     (row - size / 2) / MAP_PX_PER_M)
    return np.stack((~occupied, occupied)).astype(np.uint8)


def crop_raster(global_raster, states, *, outside_value=.5):
    """[B,2,224,224], local (x,y) -> (col,row)=(12*x+56,12*y+112)."""
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[1] != 5 or not np.isfinite(states).all():
        raise ValueError("Finite world states[B,5] are required")
    if global_raster.ndim != 3 or global_raster.shape[0] != 2:
        raise ValueError("Global semantic map[2,H,W] is required")
    row, col = np.meshgrid(np.arange(224), np.arange(224), indexing="ij")
    x, y = (col - 56) / MAP_PX_PER_M, (row - 112) / MAP_PX_PER_M
    cosine = np.cos(states[:, 4, None, None])
    sine = np.sin(states[:, 4, None, None])
    world_x = states[:, 0, None, None] + cosine * x - sine * y
    world_y = states[:, 1, None, None] + sine * x + cosine * y
    # Nearest-neighbor global raster sampling, as in TRACE's ORCA map cache.
    px = np.floor(world_x * MAP_PX_PER_M + global_raster.shape[2] / 2 + .5).astype(np.int64)
    py = np.floor(world_y * MAP_PX_PER_M + global_raster.shape[1] / 2 + .5).astype(np.int64)
    valid = (px >= 0) & (py >= 0) & (px < global_raster.shape[2]) & (py < global_raster.shape[1])
    result = np.full((len(states), 2, 224, 224), outside_value, dtype=np.float32)
    for batch in range(len(states)):
        result[batch, :, valid[batch]] = global_raster[:, py[batch][valid[batch]], px[batch][valid[batch]]].T
    return result


def conditional_windows(states, obstacles):
    """Eligible windows with real current neighbors and an obstacle in the crop."""
    import shapely
    from shapely.geometry import Polygon

    count, frames, _ = states.shape
    agent, frame = np.meshgrid(np.arange(count), np.arange(HISTORY - 1, frames - FUTURE), indexing="ij")
    agent, frame = agent.ravel(), frame.ravel()
    current = states[agent, frame]
    distances = np.linalg.norm(states[:, frame, :2].transpose(1, 0, 2) - current[:, None, :2], axis=-1)
    distances[np.arange(len(agent)), agent] = np.inf
    real_neighbor = (distances <= NEIGHBOR_RADIUS_M).any(axis=-1)
    corners = np.asarray([[-56, -112], [167, -112], [167, 111], [-56, 111]]) / MAP_PX_PER_M
    cosine, sine = np.cos(current[:, 4]), np.sin(current[:, 4])
    rotation = np.stack((cosine, -sine, sine, cosine), axis=-1).reshape(-1, 2, 2)
    world_corners = np.einsum("bij,kj->bki", rotation, corners) + current[:, None, :2]
    union = shapely.union_all([Polygon(poly) for poly in obstacles])
    # A mere boundary touch can contain no occupied raster samples. Require a
    # small actual visible obstacle area (36 pixels at the declared resolution).
    sees_map = shapely.area(shapely.intersection(shapely.polygons(world_corners), union)) >= .25
    keep = real_neighbor & sees_map
    return np.stack((agent[keep], frame[keep]), axis=-1).astype(np.int32)


def _write_npz(path, **values):
    # No object arrays or unsafe pickle payloads.
    if any(np.asarray(value).dtype.hasobject for value in values.values()):
        raise ValueError("Object arrays are not allowed in the ORCA corpus")
    np.savez_compressed(path, **values)
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def generate_corpus(output, trace_dir, rvo2_source, *, seed=20261001,
                    split_scenes=None, episodes_per_scene=2, seconds=20., max_obstacles=5,
                    progress=None):
    """Atomic new dataset publication; splits have distinct maps and seeds."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing corpus: {output}")
    split_scenes = dict(split_scenes or {"train": 16, "validation": 4, "test": 4})
    if set(split_scenes) != {"train", "validation", "test"} or any(value < 1 for value in split_scenes.values()) or episodes_per_scene < 1:
        raise ValueError("Positive disjoint train/validation/test scenes and episodes are required")
    for source, expected in ((trace_dir, TRACE_COMMIT), (rvo2_source, RVO2_COMMIT)):
        source = Path(source).resolve()
        top = Path(subprocess.check_output(["git", "-C", str(source), "rev-parse", "--show-toplevel"], text=True).strip()).resolve()
        actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if top != source or actual != expected:
            raise ValueError(f"Expected initialized pinned source {expected} at {source}, got {actual}")
    generator = load_trace_generator(trace_dir)
    root_rng = np.random.default_rng(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        licenses = temporary / "LICENSES"
        licenses.mkdir()
        shutil.copyfile(Path(trace_dir) / "LICENSE", licenses / "TRACE-NVIDIA-License-NC.txt")
        shutil.copyfile(Path(rvo2_source) / "LICENSE", licenses / "RVO2-Apache-2.0.txt")
        manifest = {"schema": SCHEMA, "seed": seed, "author_dataset": False,
                    "state_order": ["world_x", "world_y", "world_vx", "world_vy", "velocity_heading_yaw"],
                    "simulation_hz": SIM_HZ, "data_hz": DATA_HZ,
                    "history_frames": HISTORY, "future_frames": FUTURE,
                    "future_offsets_seconds": [.1, 5.2], "neighbor_radius_m": NEIGHBOR_RADIUS_M,
                    "raster": {"layers": ["walkable", "obstacles"], "global_width_m": MAP_WIDTH_M,
                               "pixels_per_metre": MAP_PX_PER_M, "crop_shape": [2, 224, 224],
                               "local_to_pixel": "col=12*x+56,row=12*y+112", "positive_y_row": True,
                               "outside_value": .5},
                    "sources": {"trace_commit": TRACE_COMMIT, "rvo2_commit": RVO2_COMMIT,
                                "generator_sha256": sha256(Path(trace_dir) / "orca_sim/gen_dataset.py"),
                                "wrapper_sha256": sha256(Path(__file__)),
                                "trace_url": f"https://github.com/nv-tlabs/trace/tree/{TRACE_COMMIT}/orca_sim",
                                "rvo2_url": f"https://github.com/sybrenstuvel/Python-RVO2/tree/{RVO2_COMMIT}"},
                    "assumptions": ["New ORCA experts, not author NavIsaacLab data or the original TRACE training set",
                                    "4..15 disc agents of radius0.4m; physical heading unavailable: atan2(world_vy,world_vx), initial zero velocity yaw=0",
                                    "Official obstacle shapes/property ranges and preferred-goal perturbations; bounded sequential spawn rejection replaces unbounded simultaneous rejection",
                                    "Require a nonempty map; remove repeated closing obstacle vertices before actual RVO2",
                                    "Exact t=0 then six physics steps per0.1s replaces original TRACE first1/60s and irregular final interval",
                                    "Scene and all its episodes belong to one split; sample only windows with a real15m neighbor and at least0.25m2 obstacle intersection in the crop",
                                    "No 3D body poses/torques; this corpus trains planar diffusion, not the AMP body discriminator"],
                    "splits": {}, "rejected_episodes": []}
        seen_layouts, seen_seeds = set(), set()

        def new_seed():
            while True:
                value = int(root_rng.integers(0, 2**32 - 1))
                if value not in seen_seeds:
                    seen_seeds.add(value)
                    return value

        scene_index = 0
        for split, number in split_scenes.items():
            split_entry = {"scene_count": number, "scenes": [], "episode_count": 0, "window_count": 0}
            manifest["splits"][split] = split_entry
            for _ in range(number):
                scene_id = f"scene_{scene_index:06d}"
                folder = temporary / split / scene_id
                folder.mkdir(parents=True)
                for _ in range(100):
                    scene_seed = new_seed()
                    obstacles = sample_obstacles(generator, scene_seed, maximum=max_obstacles)
                    layout = json.dumps(obstacles, separators=(",", ":"))
                    layout_hash = hashlib.sha256(layout.encode()).hexdigest()
                    if layout_hash not in seen_layouts:
                        seen_layouts.add(layout_hash)
                        break
                else:
                    raise RuntimeError("Failed to generate a distinct obstacle layout")
                (folder / "obstacles.json").write_text(layout + "\n")
                raster = rasterize_obstacles(obstacles)
                raster_file = _write_npz(folder / "map.npz", raster=raster)
                scene = {"scene_id": scene_id, "scene_seed": scene_seed, "layout_sha256": layout_hash,
                         "folder": str(folder.relative_to(temporary)), "obstacle_count": len(obstacles),
                         "obstacles_sha256": sha256(folder / "obstacles.json"), "map": raster_file,
                         "episodes": []}
                split_entry["scenes"].append(scene)
                for episode_index in range(episodes_per_scene):
                    count = 4 + (scene_index + episode_index) % 12
                    for attempt in range(10):
                        episode_seed = new_seed()
                        try:
                            result = simulate_episode(generator, obstacles, seed=episode_seed, count=count, seconds=seconds)
                            windows = conditional_windows(result["states"], obstacles)
                            if not len(windows):
                                raise RuntimeError("No real neighbor plus map-conditioned training windows")
                            break
                        except RuntimeError as error:
                            manifest["rejected_episodes"].append({"scene_id": scene_id,
                                "episode_index": episode_index, "seed": episode_seed, "reason": str(error)})
                    else:
                        raise RuntimeError(f"No accepted episode in scene {scene_id}")
                    episode_id = f"{scene_id}_episode_{episode_index:03d}"
                    arrays = {"states": result["states"], "times": result["times"], "radii": result["radii"],
                              "conditional_windows": windows}
                    arrays.update({f"prop_{key}": value for key, value in result["properties"].items()})
                    file = _write_npz(folder / f"episode_{episode_index:03d}.npz", **arrays)
                    episode = {"episode_id": episode_id, "episode_seed": episode_seed, "agents": count,
                               "frames": result["states"].shape[1], "conditional_window_count": len(windows),
                               "all_window_count": count * (result["states"].shape[1] - HISTORY + 1 - FUTURE),
                               "qa": result["qa"], **file}
                    scene["episodes"].append(episode)
                    split_entry["episode_count"] += 1
                    split_entry["window_count"] += len(windows)
                    if progress:
                        progress(split, scene_id, episode)
                scene_index += 1
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.rename(output)
        return output / "manifest.json"
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


class ORCACorpus:
    """Real synchronized neighbor windows; numpy CPU generation, optional torch."""

    def __init__(self, manifest, split="train", *, device="cpu", seed=7, verify_hashes=True):
        self.manifest_path = Path(manifest).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get("schema") != SCHEMA:
            raise ValueError("Unknown ORCA reconstruction manifest")
        if split not in self.manifest["splits"]:
            raise ValueError(f"Unknown split: {split}")
        self.split, self.device = split, device
        self.generator = np.random.default_rng(seed)
        self.episodes, self.maps, self.windows = [], {}, []
        layouts, scene_ids, episode_ids = set(), set(), set()
        for split_name, entry in self.manifest["splits"].items():
            for scene in entry["scenes"]:
                if scene["layout_sha256"] in layouts or scene["scene_id"] in scene_ids:
                    raise ValueError("Scene/layout leakage between dataset splits")
                layouts.add(scene["layout_sha256"])
                scene_ids.add(scene["scene_id"])
                for episode in scene["episodes"]:
                    if episode["episode_id"] in episode_ids:
                        raise ValueError("Episode leakage between dataset splits")
                    episode_ids.add(episode["episode_id"])
                if split_name != split:
                    continue
                folder = self.manifest_path.parent / scene["folder"]
                map_path = folder / scene["map"]["path"]
                self._verify(map_path, scene["map"], verify_hashes)
                with np.load(map_path, allow_pickle=False) as archive:
                    self.maps[scene["scene_id"]] = archive["raster"]
                for episode in scene["episodes"]:
                    path = folder / episode["path"]
                    self._verify(path, episode, verify_hashes)
                    with np.load(path, allow_pickle=False) as archive:
                        states, times, windows = archive["states"], archive["times"], archive["conditional_windows"]
                    if states.shape != (episode["agents"], episode["frames"], 5) or not np.isfinite(states).all() or not np.allclose(np.diff(times), .1, atol=1e-9):
                        raise ValueError("Invalid synchronized finite 10Hz five-state trajectory")
                    if len(windows) != episode["conditional_window_count"] or windows.ndim != 2 or windows.shape[1] != 2:
                        raise ValueError("Invalid conditional window index")
                    if ((windows[:, 0] < 0) | (windows[:, 0] >= len(states)) | (windows[:, 1] < HISTORY - 1) | (windows[:, 1] + FUTURE >= states.shape[1])).any():
                        raise ValueError("Conditional window crosses an episode boundary")
                    index = len(self.episodes)
                    self.episodes.append({"states": states, "scene_id": scene["scene_id"], **episode})
                    self.windows.append(np.column_stack((np.full(len(windows), index), windows)))
        if not self.windows:
            raise ValueError("Split contains no valid conditional training windows")
        self.windows = np.concatenate(self.windows).astype(np.int32)

    @staticmethod
    def _verify(path, entry, enabled):
        if path.stat().st_size != entry["bytes"] or (enabled and sha256(path) != entry["sha256"]):
            raise ValueError(f"ORCA source changed since generation: {path}")

    def numpy_batch(self, batch, *, indices=None):
        if not isinstance(batch, int) or batch < 1:
            raise ValueError("Batch size must be a positive integer")
        selection = self.generator.integers(len(self.windows), size=batch) if indices is None else np.asarray(indices)
        if selection.shape != (batch,) or (selection < 0).any() or (selection >= len(self.windows)).any():
            raise ValueError("Invalid explicit window indices")
        current, histories, futures, maps, scene_ids, episode_ids, agents, frames = [], [], [], [], [], [], [], []
        neighbors = np.zeros((batch, 14, HISTORY, 5), dtype=np.float32)
        masks = np.zeros((batch, 14, HISTORY), dtype=bool)
        neighbor_ids = np.full((batch, 14), -1, dtype=np.int32)
        for row, (episode_index, agent, frame) in enumerate(self.windows[selection]):
            entry = self.episodes[episode_index]
            states = entry["states"]
            state = states[agent, frame]
            history = states[agent, frame - HISTORY + 1:frame + 1]
            future = states[agent, frame + 1:frame + FUTURE + 1]
            distances = np.linalg.norm(states[:, frame, :2] - state[:2], axis=-1)
            distances[agent] = np.inf
            ids = np.argsort(distances)
            ids = ids[distances[ids] <= NEIGHBOR_RADIUS_M]
            if not len(ids):
                raise ValueError("Stored conditional window has no real neighbor")
            neighbors[row, :len(ids)] = states[ids, frame - HISTORY + 1:frame + 1]
            masks[row, :len(ids)] = True
            neighbor_ids[row, :len(ids)] = ids
            current.append(state)
            histories.append(history)
            futures.append(future)
            maps.append(crop_raster(self.maps[entry["scene_id"]], state[None])[0])
            scene_ids.append(entry["scene_id"])
            episode_ids.append(entry["episode_id"])
            agents.append(agent)
            frames.append(frame)
        return {"states": np.asarray(current), "history": np.asarray(histories), "future": np.asarray(futures),
                "neighbors": neighbors, "neighbor_mask": masks, "history_mask": np.ones((batch, HISTORY), bool),
                "raster": np.asarray(maps), "neighbor_ids": neighbor_ids,
                "scene_ids": scene_ids, "episode_ids": episode_ids, "agent_ids": np.asarray(agents),
                "frame_indices": np.asarray(frames), "window_indices": selection}

    def diffusion_batch(self, batch, *, indices=None):
        import torch

        sample = self.numpy_batch(batch, indices=indices)
        return {name: torch.as_tensor(value, device=self.device) if isinstance(value, np.ndarray) else value
                for name, value in sample.items()}

    def provenance(self):
        return {"manifest": str(self.manifest_path), "manifest_sha256": sha256(self.manifest_path),
                "split": self.split, "scenes": len(self.maps), "episodes": len(self.episodes),
                "conditional_windows": len(self.windows), "author_dataset": False,
                "state_order": self.manifest["state_order"], "sources": self.manifest["sources"]}
