"""Official Maps loader with real maps, including legal empty geometry."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
INTERACT_STAGE = ROOT / "docs/style_improvement_20261003/orca_interact_usable_staging_v1"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(INTERACT_STAGE) not in sys.path:
    sys.path.insert(0, str(INTERACT_STAGE))

from CrowdSim.humanoid.orca_corpus import ORCACorpus, crop_raster  # noqa: E402
from interaction_corpus import InteractionORCACorpus  # noqa: E402

SCHEMA = "navisaaclab.orca_official_maps_adapter.v1"
HISTORY, FUTURE, MAX_NEIGHBORS = 31, 52, 19


class OfficialMapsORCACorpus(InteractionORCACorpus):
    """Reuse the original Interact temporal/neighbor batch math, replacing maps."""
    def __init__(self, manifest, split="train", *, device="cpu", seed=7, verify_hashes=True):
        self.manifest_path = Path(manifest).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get("schema") != SCHEMA or self.manifest.get("condition_mode") != "official_maps":
            raise ValueError("Expected explicit official Maps adapter manifest")
        if set(self.manifest["splits"]) != {"train", "validation", "test"} or split not in self.manifest["splits"]:
            raise ValueError("Expected preserved scene-level split")
        self.split, self.device = split, device
        self.generator = np.random.default_rng(seed)
        self.episodes, self.maps, self.windows = [], {}, []
        scene_ids, episode_ids, raw_paths, map_paths = set(), set(), set(), set()
        for split_name, entry in self.manifest["splits"].items():
            for scene in entry["scenes"]:
                scene_id, raw_path, map_path = scene["scene_id"], scene["source_raw"]["path"], scene["source_map_pkl"]["path"]
                if scene_id in scene_ids or raw_path in raw_paths or map_path in map_paths:
                    raise ValueError("Scene/raw/map identity leakage between splits")
                scene_ids.add(scene_id)
                raw_paths.add(raw_path)
                map_paths.add(map_path)
                if (scene["map"] is None or scene["map_present"] is not True
                        or scene["map_condition_mask"] is not True):
                    raise ValueError("Official Maps annotation remains present even for empty geometry")
                # No geometry uniqueness/visibility rejection: repeated empty
                # layouts and known maps with no visible obstacle remain legal.
                for episode in scene["episodes"]:
                    if episode["episode_id"] in episode_ids:
                        raise ValueError("Episode identity leakage between splits")
                    episode_ids.add(episode["episode_id"])
                if split_name != split:
                    continue
                folder = self.manifest_path.parent / scene["folder"]
                artifact = folder / scene["map"]["path"]
                self._verify(artifact, scene["map"], verify_hashes)
                with np.load(artifact, allow_pickle=False) as archive:
                    raster = archive["raster"]
                if raster.dtype != np.uint8 or raster.shape != (2, 1080, 1080):
                    raise ValueError("Official semantic map requires uint8[2,1080,1080]")
                self.maps[scene_id] = raster
                for episode in scene["episodes"]:
                    path = folder / episode["path"]
                    self._verify(path, episode, verify_hashes)
                    with np.load(path, allow_pickle=False) as archive:
                        states, times, windows = archive["states"], archive["times"], archive["conditional_windows"]
                    if (not 1 <= episode["agents"] <= 10 or states.shape != (episode["agents"], 100, 5)
                            or not np.isfinite(states).all() or times.shape != (100,)
                            or not np.array_equal(times, np.arange(100, dtype=np.float64) / 10)):
                        raise ValueError("Exact finite t0..9.9 official Maps five-state data required")
                    if (len(windows) != episode["conditional_window_count"] or windows.ndim != 2
                            or windows.shape[1] != 2 or windows.dtype.kind not in "iu"):
                        raise ValueError("Invalid neighbor-only Maps window index")
                    if ((windows[:, 0] < 0) | (windows[:, 0] >= len(states)) | (windows[:, 1] < HISTORY - 1)
                            | (windows[:, 1] + FUTURE >= states.shape[1])).any():
                        raise ValueError("Maps window crosses its episode")
                    index = len(self.episodes)
                    self.episodes.append({"states": states, "scene_id": scene_id, **episode})
                    self.windows.append(np.column_stack((np.full(len(windows), index), windows)))
        if not self.windows or not sum(len(value) for value in self.windows):
            raise ValueError("No real-neighbor official Maps windows in split")
        self.windows = np.concatenate(self.windows).astype(np.int32)

    def numpy_batch(self, batch, *, indices=None):
        sample = super().numpy_batch(batch, indices=indices)
        sample["raster"] = np.stack([crop_raster(self.maps[scene_id], state[None])[0]
                                    for scene_id, state in zip(sample["scene_ids"], sample["states"], strict=True)])
        sample["map_condition_mask"] = np.ones(batch, dtype=bool)
        return sample

    def provenance(self):
        return {**ORCACorpus.provenance(self), "condition_mode": "official_maps", "map_condition_mask": True,
                "max_neighbors": MAX_NEIGHBORS, "source_max_agents": 10, "uniform_t0_resampled": True,
                "geometry_uniqueness_or_visible_obstacle_filter": False}

