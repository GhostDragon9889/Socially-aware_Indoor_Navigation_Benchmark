"""No-map Interact loader, reusing the production ORCACorpus tensor boundary."""
import json
from pathlib import Path

import numpy as np

from CrowdSim.humanoid.orca_corpus import ORCACorpus, sha256
from convert_official_raw import EMPTY_LAYOUT_SHA, FUTURE, HISTORY, MAX_NEIGHBORS, SCHEMA


class InteractionORCACorpus(ORCACorpus):
    def __init__(self, manifest, split="train", *, device="cpu", seed=7, verify_hashes=True):
        self.manifest_path = Path(manifest).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get("schema") != SCHEMA or self.manifest.get("condition_mode") != "interaction_only":
            raise ValueError("Expected explicit Interact adapter manifest")
        if set(self.manifest["splits"]) != {"train", "validation", "test"} or split not in self.manifest["splits"]:
            raise ValueError("Expected preserved scene-level split")
        self.split, self.device = split, device
        self.generator = np.random.default_rng(seed)
        self.episodes, self.maps, self.windows = [], {}, []
        scene_ids, episode_ids, raw_paths = set(), set(), set()
        for split_name, entry in self.manifest["splits"].items():
            for scene in entry["scenes"]:
                if scene["scene_id"] in scene_ids or scene["source_raw"]["path"] in raw_paths:
                    raise ValueError("Scene/raw identity leakage between splits")
                scene_ids.add(scene["scene_id"])
                raw_paths.add(scene["source_raw"]["path"])
                if (scene["map"] is not None or scene["obstacles"] != [] or scene["map_condition_mask"] is not False
                        or scene["layout_sha256"] != EMPTY_LAYOUT_SHA):
                    raise ValueError("Interact has no map; repeated true empty geometry is legal")
                for episode in scene["episodes"]:
                    if episode["episode_id"] in episode_ids:
                        raise ValueError("Episode identity leakage between splits")
                    episode_ids.add(episode["episode_id"])
                if split_name != split:
                    continue
                self.maps[scene["scene_id"]] = None  # scene count; no synthetic map artifact
                folder = self.manifest_path.parent / scene["folder"]
                for episode in scene["episodes"]:
                    path = folder / episode["path"]
                    self._verify(path, episode, verify_hashes)
                    with np.load(path, allow_pickle=False) as archive:
                        states, times, windows = archive["states"], archive["times"], archive["conditional_windows"]
                    if (not 1 <= episode["agents"] <= 20 or states.shape != (episode["agents"], 100, 5)
                            or not np.isfinite(states).all() or times.shape != (100,)
                            or not np.array_equal(times, np.arange(100, dtype=np.float64) / 10)):
                        raise ValueError("Exact t0..9.9 uniform finite five-state Interact data required")
                    if (len(windows) != episode["conditional_window_count"] or windows.ndim != 2 or windows.shape[1] != 2
                            or windows.dtype.kind not in "iu"):
                        raise ValueError("Invalid interaction-only index")
                    if ((windows[:, 0] < 0) | (windows[:, 0] >= len(states)) | (windows[:, 1] < HISTORY - 1)
                            | (windows[:, 1] + FUTURE >= states.shape[1])).any():
                        raise ValueError("Interaction window crosses its episode")
                    index = len(self.episodes)
                    self.episodes.append({"states": states, "scene_id": scene["scene_id"], **episode})
                    self.windows.append(np.column_stack((np.full(len(windows), index), windows)))
        if not self.windows or not sum(len(value) for value in self.windows):
            raise ValueError("No real-neighbor Interact windows in split")
        self.windows = np.concatenate(self.windows).astype(np.int32)

    def numpy_batch(self, batch, *, indices=None):
        if not isinstance(batch, int) or batch < 1:
            raise ValueError("Batch size must be positive")
        selection = self.generator.integers(len(self.windows), size=batch) if indices is None else np.asarray(indices)
        if selection.shape != (batch,) or selection.dtype.kind not in "iu" or (selection < 0).any() or (selection >= len(self.windows)).any():
            raise ValueError("Invalid explicit window index")
        current, histories, futures, scene_ids, episode_ids, agents, frames = [], [], [], [], [], [], []
        neighbors = np.zeros((batch, MAX_NEIGHBORS, HISTORY, 5), dtype=np.float32)
        masks = np.zeros((batch, MAX_NEIGHBORS, HISTORY), dtype=bool)
        neighbor_ids = np.full((batch, MAX_NEIGHBORS), -1, dtype=np.int32)
        for row, (episode_index, agent, frame) in enumerate(self.windows[selection]):
            entry = self.episodes[episode_index]
            states = entry["states"]
            state = states[agent, frame]
            history = states[agent, frame - HISTORY + 1:frame + 1]
            future = states[agent, frame + 1:frame + FUTURE + 1]
            distances = np.linalg.norm(states[:, frame, :2] - state[:2], axis=-1)
            distances[agent] = np.inf
            ids = np.argsort(distances)
            ids = ids[distances[ids] <= 15.]
            if not len(ids) or len(ids) > MAX_NEIGHBORS:
                raise ValueError("Stored window must have1..19 real current neighbors")
            neighbors[row, :len(ids)] = states[ids, frame - HISTORY + 1:frame + 1]
            masks[row, :len(ids)] = True
            neighbor_ids[row, :len(ids)] = ids
            current.append(state)
            histories.append(history)
            futures.append(future)
            scene_ids.append(entry["scene_id"])
            episode_ids.append(entry["episode_id"])
            agents.append(agent)
            frames.append(frame)
        return {"states": np.asarray(current), "history": np.asarray(histories), "future": np.asarray(futures),
                "neighbors": neighbors, "neighbor_mask": masks, "history_mask": np.ones((batch, HISTORY), bool),
                "raster": np.full((batch, 2, 224, 224), .5, dtype=np.float32), "map_condition_mask": np.zeros(batch, bool),
                "neighbor_ids": neighbor_ids, "scene_ids": scene_ids, "episode_ids": episode_ids,
                "agent_ids": np.asarray(agents), "frame_indices": np.asarray(frames), "window_indices": selection}

    def provenance(self):
        return {**super().provenance(), "condition_mode": "interaction_only", "map_condition_mask": False,
                "max_neighbors": MAX_NEIGHBORS, "uniform_t0_resampled": True,
                "empty_geometry_layout_unique_claim": False, "manifest_sha256": sha256(self.manifest_path)}
