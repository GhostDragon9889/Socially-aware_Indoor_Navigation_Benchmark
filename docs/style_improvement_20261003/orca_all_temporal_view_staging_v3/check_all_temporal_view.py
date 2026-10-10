"""Full view identity/count QC plus necessary empty-neighbor NumPy batch checks."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np

from all_temporal_corpus import (HERE, INTERACT, FORMAL, MixedORCACorpus, VIEW_SCHEMA,
                                 DEFAULT_MAPS, DEFAULT_INTERACT, DERIVATION, record,
                                 verified_readonly, manifest_window_counts)
from build_all_temporal_view import write_json


def original_class(domain):
    path = INTERACT / "interaction_corpus.py" if domain == "interact" else FORMAL / "official_maps_corpus.py"
    spec = importlib.util.spec_from_file_location("_all_temporal_QC_original_" + domain, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.InteractionORCACorpus if domain == "interact" else module.OfficialMapsORCACorpus


def same_batch(before, after, exclude=()):
    if set(before) != set(after):
        raise ValueError("Batch keys changed")
    for key in before:
        if key in exclude:
            continue
        if isinstance(before[key], np.ndarray):
            if (before[key].dtype != after[key].dtype or before[key].shape != after[key].shape
                    or not np.array_equal(before[key], after[key])):
                raise ValueError(f"Batch math changed: {key}")
        elif before[key] != after[key]:
            raise ValueError(f"Batch metadata changed: {key}")


def check_split(split, build):
    union = MixedORCACorpus(DEFAULT_MAPS, DEFAULT_INTERACT, split, seed=7)
    if union.resume_identity().get("view_schema") != VIEW_SCHEMA:
        raise ValueError("New resume view identity missing")
    rows = []
    for name in ("maps", "interact"):
        domain = union.domains[name]
        original = json.loads(verified_readonly(domain.manifest["source_manifest"]).read_text())
        original_scenes = original["splits"][split]["scenes"]
        new_scenes = domain.manifest["splits"][split]["scenes"]
        old = original_class(name).__new__(original_class(name))
        old.__dict__.update(domain.corpus.__dict__)
        old_rows, positions, offset = [], [], 0
        zero_positions, known_empty_map, known_nonempty_map = [], [], []
        first_anchor = domain.manifest["history_frames"] - 1
        frames_per_agent = domain.episodes[0]["frames"] - domain.manifest["future_frames"] - first_anchor
        for episode_index, (original_scene, scene) in enumerate(zip(original_scenes, new_scenes, strict=True)):
            expected_scene = copy.deepcopy(original_scene)
            expected_scene["folder"] = str((Path(domain.manifest["source_manifest"]["path"]).parent / expected_scene["folder"]).resolve())
            for old_episode, new_episode in zip(expected_scene["episodes"], scene["episodes"], strict=True):
                old_episode["conditional_window_count"] = new_episode["conditional_window_count"]
                old_episode["window_index"] = new_episode["window_index"]
            if expected_scene != scene:
                raise ValueError("Scene/state/map/radius/split identity changed beyond the explicit view delta")
            # Both frozen corpora have exactly one episode per scene.
            if len(scene["episodes"]) != 1:
                raise ValueError("Unexpected existing episode layout")
            ep = domain.episodes[episode_index]
            with np.load(Path(scene["folder"]) / ep["path"], allow_pickle=False) as data:
                old_windows = data["conditional_windows"]
            if len(old_windows) and len(positions) < 16:
                old_rows.append([episode_index, *old_windows[0]])
                positions.append(offset + int(old_windows[0, 0]) * frames_per_agent + int(old_windows[0, 1]) - first_anchor)
            if ep["agents"] == 1 and not zero_positions:
                zero_positions.append(offset)
            if name == "maps" and not scene["obstacles"] and not known_empty_map:
                known_empty_map.append(offset)
            if name == "maps" and scene["obstacles"] and not known_nonempty_map:
                known_nonempty_map.append(offset)
            offset += ep["conditional_window_count"]
        if offset != len(domain.windows) or offset != build["domains"][name]["split_counts"][split]["windows"]:
            raise ValueError("Complete loaded view count differs from actual build")
        if not positions or not zero_positions:
            raise ValueError("Expected retained original and real single-agent empty-neighbor windows")
        old.windows = np.asarray(old_rows, dtype=np.int32)
        expected = old.numpy_batch(len(old_rows), indices=np.arange(len(old_rows), dtype=np.int64))
        actual = domain.corpus.numpy_batch(len(positions), indices=np.asarray(positions, dtype=np.int64))
        same_batch(expected, actual, exclude=("window_indices",))
        selection = np.asarray(zero_positions + known_empty_map + known_nonempty_map, dtype=np.int64)
        batch = domain.numpy_batch(len(selection), indices=selection)
        if (batch["neighbors"].shape != (len(selection), 19, 31, 5)
                or batch["neighbor_mask"][0].any() or batch["neighbors"][0].any()
                or not (batch["neighbor_ids"][0] == -1).all()
                or not np.array_equal(batch["history"][:, -1], batch["states"])
                or not np.isfinite(batch["raster"]).all()
                or not (batch["map_condition_mask"] == (name == "maps")).all()):
            raise ValueError("Real zero-neighbor/domain-map batch ABI differs")
        if name == "maps":
            if (not known_empty_map or not known_nonempty_map
                    or not (batch["raster"][1, 0] == 1).all() or not (batch["raster"][1, 1] == 0).all()):
                raise ValueError("Empty real map semantics changed")
        elif not (batch["raster"] == .5).all():
            raise ValueError("Absent Interact map must retain unknown raster")
        rows.append({"domain": name, "split": split, "all_windows": len(domain.windows),
                     "retained_original_batch_exact_windows": len(positions),
                     "real_empty_neighbor_batch_indices": zero_positions,
                     "explicit_batch_indices": selection.tolist(), "map_mask": batch["map_condition_mask"].tolist(),
                     "empty_neighbor_all_false_mask_zero_values_ids_minus1": True})
    rng = copy.deepcopy(union.generator.bit_generator.state)
    first = union.numpy_batch(8)
    union.generator.bit_generator.state = rng
    second = union.numpy_batch(8)
    same_batch(first, second)
    before = copy.deepcopy(union.generator.bit_generator.state)
    explicit = union.numpy_batch(2, indices=np.asarray([0, union.offsets[1]], dtype=np.int64))
    if union.generator.bit_generator.state != before or explicit["domain_names"] != ["maps", "interact"]:
        raise ValueError("Explicit union selection must preserve the private union RNG")
    return {"split": split, "domain_batches": rows, "resume_identity": union.resume_identity(),
            "union_rng_roundtrip_batch_exact": True, "explicit_indices_private_union_RNG_unchanged": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-report", type=Path, default=HERE / "actual_build_QC.json")
    parser.add_argument("--output", type=Path, default=HERE / "actual_QC.json")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Fresh QC output required")
    started = time.monotonic()
    build_record = record(args.build_report)
    build = json.loads(verified_readonly(build_record).read_text())
    if build.get("status") != "passed":
        raise ValueError("Actual complete build QC required")
    for row in build["domains"].values():
        verified_readonly(row["manifest"])
    splits = [check_split(split, build) for split in ("train", "validation", "test")]
    assert not any(name.split(".")[0] in {"torch", "isaaclab", "isaacsim", "omni", "pxr", "warp"} for name in sys.modules)
    document = {"schema": "local_orca.all_temporal_view_actual_CPU_QC.v1", "status": "passed",
                "build_report": build_record, "actual_split_batches": splits,
                "split_window_counts": manifest_window_counts(DEFAULT_MAPS, DEFAULT_INTERACT),
                "source_derivation": DERIVATION, "source_files": [record(HERE / filename) for filename in
                    ("all_temporal_corpus.py", "build_all_temporal_view.py", "check_all_temporal_view.py")],
                "NN_SDK_GPU_used": False, "original_states_maps_rewritten": False,
                "elapsed_seconds": time.monotonic() - started}
    write_json(args.output, document)
    print(json.dumps({"status": "passed", "report": record(args.output), "elapsed_seconds": document["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
