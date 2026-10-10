"""Root-authorized NumPy-only official Maps batch ABI, not a second dataset QC."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mixed_corpus import MixedORCACorpus  # noqa: E402

started = time.monotonic()
rows = []
for split in ("train", "validation", "test"):
    union = MixedORCACorpus(split=split, device="cpu", seed=7)
    corpus = union.domains["maps"]
    scenes = {scene["scene_id"]: scene for scene in corpus.manifest["splits"][split]["scenes"]}
    empty_ids = {key for key, scene in scenes.items() if not scene["obstacles"]}
    window_scenes = np.asarray([corpus.episodes[int(index)]["scene_id"] for index in corpus.windows[:, 0]])
    empty_windows = np.flatnonzero(np.isin(window_scenes, list(empty_ids)))
    nonempty_windows = np.flatnonzero(~np.isin(window_scenes, list(empty_ids)))
    if not len(empty_windows) or not len(nonempty_windows):
        raise ValueError("Expected real eligible empty and nonempty Maps windows in each split")
    indices = np.array([empty_windows[0], nonempty_windows[0]], dtype=np.int64)
    batch = corpus.numpy_batch(2, indices=indices)
    assert batch["map_condition_mask"].tolist() == [True, True]
    assert batch["neighbors"].shape == (2, 19, 31, 5)
    assert batch["history"].shape == (2, 31, 5) and batch["future"].shape == (2, 52, 5)
    assert batch["raster"].shape == (2, 2, 224, 224)
    assert np.isfinite(batch["raster"]).all() and np.isfinite(batch["states"]).all()
    assert (batch["raster"][0, 0] == 1).all() and (batch["raster"][0, 1] == 0).all()
    assert (batch["raster"][1, 1] == 1).any()
    assert np.array_equal(batch["history"][:, -1], batch["states"])
    # One mixed call exercises the new Maps source together with unchanged Interact.
    mixed = union.numpy_batch(2, indices=np.array([empty_windows[0], union.offsets[1]], dtype=np.int64))
    assert mixed["domain_names"] == ["maps", "interact"]
    assert mixed["map_condition_mask"].tolist() == [True, False]
    rows.append({"split": split, "scene_count": len(corpus.maps), "episodes": len(corpus.episodes),
                 "empty_scene_count_retained": len(empty_ids), "Maps_window_count": len(corpus.windows),
                 "union_window_counts": dict(zip(union.domains, union.counts, strict=True)),
                 "maps_batch_local_indices": indices.tolist(), "maps_batch_scenes": batch["scene_ids"],
                 "maps_batch_true_neighbor_counts": batch["neighbor_mask"][:, :, -1].sum(1).tolist(),
                 "maps_batch_map_presence": batch["map_condition_mask"].tolist(),
                 "maps_batch_shapes": {key: list(value.shape) for key, value in batch.items() if isinstance(value, np.ndarray)},
                 "empty_known_free_semantics_verified": True, "nonempty_visible_occupied_pixels_verified": True,
                 "mixed_batch_domain_names": mixed["domain_names"],
                 "mixed_batch_map_presence": mixed["map_condition_mask"].tolist()})
    del union, corpus, batch, mixed


def record(path):
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


assert not any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules)
result = {"schema": "local_orca.official_maps_actual_numpy_batch_ABI.v1", "status": "passed",
          "source_files": [record(HERE / name) for name in ("official_maps_corpus.py", "mixed_corpus.py", "check_numpy_interface.py")],
          "actual_batches": rows, "Maps_observed_batch_windows": 6, "additional_mixed_batch_windows": 6,
          "NN_weights_SDK_GPU_or_DDPM_used": False,
          "not_full_dataset_QC_or_training": True, "elapsed_seconds": time.monotonic() - started}
(HERE / "actual_numpy_interface.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps({"status": "passed", "report": record(HERE / "actual_numpy_interface.json"),
                  "elapsed_seconds": result["elapsed_seconds"]}))

