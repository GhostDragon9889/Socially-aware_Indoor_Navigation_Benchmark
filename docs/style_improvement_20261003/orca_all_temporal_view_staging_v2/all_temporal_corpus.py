"""Independent all-legal-anchor view; private exact deltas of frozen loaders."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
INTERACT = HERE.parent / "orca_interact_usable_staging_v1"
FORMAL = HERE.parent / "orca_official_mixed_training_staging_v2"
VIEW_SCHEMA = "local_orca.all_temporal_anchor_view.v1"
SELECTION = {"history_frames": 31, "future_frames": 52, "data_hz": 10,
             "criterion": "all_ego_temporally_valid_agent_time_anchors",
             "current_neighbor_radius_m": 15.0, "minimum_real_neighbors": 0,
             "maximum_real_neighbors": 19, "geometry_or_velocity_filter": False,
             "index_order": "scene_episode_agent_then_ascending_anchor"}
DEFAULT_MAPS = ROOT / "data/diffusion_amp/orca_maps_all_temporal_1000_v1/manifest.json"
DEFAULT_INTERACT = ROOT / "data/diffusion_amp/orca_interact_all_temporal_1000_v1/manifest.json"
SOURCE_SHA = {
    "interaction": "8156b4d1af13f0debb46ffa46046f322b0ea3f3f667361e35d5fdc2e98ce9865",
    "maps": "680f65024839332508a368424c400d5e6c5f2e4060e779a0e7186f501a0089e6",
    "mixed": "1095656a672f9bce9604506b476c49a4637d7c4bacc5275366a98444e45ba739"}
for path in (ROOT, INTERACT, FORMAL):
    if str(path) not in sys.path:
        sys.path.append(str(path))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def verified_readonly(entry):
    path = Path(entry["path"])
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222 or record(path) != entry:
        raise ValueError(f"Changed/non-regular/non-readonly view source: {path}")
    return path


def validate_view(path, manifest):
    if (manifest.get("view_schema") != VIEW_SCHEMA or manifest.get("window_selection") != SELECTION
            or manifest.get("history_frames") != SELECTION["history_frames"]
            or manifest.get("future_frames") != SELECTION["future_frames"]
            or manifest.get("data_hz") != SELECTION["data_hz"]):
        raise ValueError("Explicit all-temporal view and original temporal contract required")
    if Path(path).is_symlink() or Path(path).stat().st_mode & 0o222:
        raise ValueError("Readonly real view manifest required")
    verified_readonly(manifest["source_manifest"])


def load_windows(episode, verify_hashes):
    # Window identity is always verified; verify_hashes keeps the old NPZ API.
    windows = np.load(verified_readonly(episode["window_index"]), allow_pickle=False)
    agents, frames = np.meshgrid(np.arange(episode["agents"]),
                                np.arange(SELECTION["history_frames"] - 1,
                                          episode["frames"] - SELECTION["future_frames"]), indexing="ij")
    expected = np.column_stack((agents.ravel(), frames.ravel())).astype(np.int32)
    if windows.dtype != np.int32 or not np.array_equal(windows, expected):
        raise ValueError("Window artifact must contain every legal anchor exactly once in canonical order")
    return windows


DERIVATION = []


def derived_module(name, path, replacements, supplied=None):
    before = path.read_text()
    if hashlib.sha256(before.encode()).hexdigest() != SOURCE_SHA[name]:
        raise ValueError(f"Frozen loader source changed: {path}")
    after = before
    for old, new in replacements:
        if after.count(old) != 1:
            raise ValueError(f"Exact loader delta missing/ambiguous: {name}: {old}")
        after = after.replace(old, new, 1)
    reverse = after
    for old, new in reversed(replacements):
        if reverse.count(new) != 1:
            raise ValueError("Reverse source delta is ambiguous")
        reverse = reverse.replace(new, old, 1)
    if reverse != before:
        raise ValueError("Reverse loader bytes differ")
    DERIVATION.append({"source": record(path), "derived_sha256": hashlib.sha256(after.encode()).hexdigest(),
                       "reverse_entire_bytes_exact": True,
                       "replacements": [{"before": old, "after": new} for old, new in replacements]})
    namespace = {"__file__": str(path), "__name__": "_all_temporal_" + name,
                 "validate_view": validate_view, "load_windows": load_windows, **(supplied or {})}
    exec(compile(after, str(path) + ":all_temporal_private_delta", "exec"), namespace)
    return namespace


VALIDATE = ('        self.manifest = json.loads(self.manifest_path.read_text())',
            '        self.manifest = json.loads(self.manifest_path.read_text())\n        validate_view(self.manifest_path, self.manifest)')
WINDOWS = ('states, times, windows = archive["states"], archive["times"], archive["conditional_windows"]',
           'states, times = archive["states"], archive["times"]\n                    windows = load_windows(episode, verify_hashes)')
interaction = derived_module("interaction", INTERACT / "interaction_corpus.py", [VALIDATE, WINDOWS,
    ('if not len(ids) or len(ids) > MAX_NEIGHBORS:', 'if len(ids) > MAX_NEIGHBORS:'),
    ('Stored window must have1..19 real current neighbors', 'Stored window must have0..19 real current neighbors'),
    ('No real-neighbor Interact windows in split', 'No temporally valid Interact windows in split')])
InteractionORCACorpus = interaction["InteractionORCACorpus"]
maps = derived_module("maps", FORMAL / "official_maps_corpus.py", [VALIDATE, WINDOWS,
    ('from interaction_corpus import InteractionORCACorpus  # noqa: E402',
     '# InteractionORCACorpus supplied by the exact private all-temporal delta'),
    ('No real-neighbor official Maps windows in split', 'No temporally valid official Maps windows in split')],
    {"InteractionORCACorpus": InteractionORCACorpus})
OfficialMapsORCACorpus = maps["OfficialMapsORCACorpus"]
mixed = derived_module("mixed", FORMAL / "mixed_corpus.py", [
    ('from interaction_corpus import InteractionORCACorpus  # noqa: E402',
     '# InteractionORCACorpus supplied by the exact private all-temporal delta'),
    ('from official_maps_corpus import OfficialMapsORCACorpus  # noqa: E402',
     '# OfficialMapsORCACorpus supplied by the exact private all-temporal delta'),
    ('orca_maps_official_usable_1000_v1', 'orca_maps_all_temporal_1000_v1'),
    ('orca_interact_usable_1000_v1', 'orca_interact_all_temporal_1000_v1')],
    {"InteractionORCACorpus": InteractionORCACorpus, "OfficialMapsORCACorpus": OfficialMapsORCACorpus})
manifest_window_counts = mixed["manifest_window_counts"]


class MixedORCACorpus(mixed["MixedORCACorpus"]):
    def provenance(self):
        return {**super().provenance(), "schema": "local_orca.official_all_temporal_union.v1",
                "view_schema": VIEW_SCHEMA, "window_selection": SELECTION,
                "author_shuffle_order_bitwise_claimed": False}

    def resume_identity(self):
        identity = super().resume_identity()
        return {**identity, "view_schema": VIEW_SCHEMA, "window_selection": SELECTION,
                "window_index_storage": "independent_canonical_int32_NPY_per_episode",
                "source_manifests": {name: self.domains[name].manifest["source_manifest"]
                                     for name in identity["domain_order"]}}
