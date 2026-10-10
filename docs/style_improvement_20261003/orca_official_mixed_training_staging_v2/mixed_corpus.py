"""Independent Maps + Interact window union; no models, SDK or data writes."""
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

from CrowdSim.humanoid.orca_corpus import ORCACorpus, sha256  # noqa: E402
from interaction_corpus import InteractionORCACorpus  # noqa: E402
from official_maps_corpus import OfficialMapsORCACorpus  # noqa: E402

DOMAINS = ("maps", "interact")
MAX_NEIGHBORS = 19
SAMPLING = "uniform_over_union_valid_windows_with_replacement"
DEFAULT_MAPS = ROOT / "data/diffusion_amp/orca_maps_official_usable_1000_v1/manifest.json"
DEFAULT_INTERACT = ROOT / "data/diffusion_amp/orca_interact_usable_1000_v1/manifest.json"


class DomainCorpus(ORCACorpus):
    """Keep a domain's original metric/source identities, adding batch padding."""
    def __init__(self, corpus, domain):
        if domain not in DOMAINS:
            raise ValueError("Unknown domain")
        self.corpus, self.domain = corpus, domain

    def __getattr__(self, name):
        return getattr(self.corpus, name)

    @property
    def generator(self):
        return self.corpus.generator

    @generator.setter
    def generator(self, value):
        self.corpus.generator = value

    def numpy_batch(self, batch, *, indices=None):
        sample = self.corpus.numpy_batch(batch, indices=indices)
        if self.domain == "maps":
            padding = MAX_NEIGHBORS - sample["neighbors"].shape[1]
            if padding < 0:
                raise ValueError("Too many Maps neighbors")
            sample["neighbors"] = np.pad(sample["neighbors"], ((0, 0), (0, padding), (0, 0), (0, 0)))
            sample["neighbor_mask"] = np.pad(sample["neighbor_mask"], ((0, 0), (0, padding), (0, 0)))
            sample["neighbor_ids"] = np.pad(sample["neighbor_ids"], ((0, 0), (0, padding)), constant_values=-1)
            sample["map_condition_mask"] = np.ones(batch, dtype=bool)
        if sample["neighbors"].shape[1] != MAX_NEIGHBORS:
            raise ValueError("Both domains must expose 19 neighbor slots")
        sample["domain_names"] = [self.domain] * batch
        return sample

    def provenance(self):
        return {**self.corpus.provenance(), "domain": self.domain, "max_neighbors": MAX_NEIGHBORS}


class MixedORCACorpus(ORCACorpus):
    """Uniform union sampling with one RNG and deterministic child indices.

    This retains the existing trainer's with-replacement update API. TRACE's
    official loader shuffles the combined dataset without an explicit domain
    weight; this adapter does not claim identical per-epoch draw order.
    """
    def __init__(self, maps_manifest=DEFAULT_MAPS, interact_manifest=DEFAULT_INTERACT, split="train",
                 *, device="cpu", seed=7, verify_hashes=True):
        self.split, self.device = split, device
        self.generator = np.random.default_rng(seed)
        self.domains = {
            "maps": DomainCorpus(OfficialMapsORCACorpus(maps_manifest, split, device=device, seed=seed,
                                            verify_hashes=verify_hashes), "maps"),
            "interact": DomainCorpus(InteractionORCACorpus(interact_manifest, split, device=device, seed=seed,
                                                          verify_hashes=verify_hashes), "interact"),
        }
        # Namespaced domain identities are retained; true empty Interact layouts
        # are legal and never subjected to Maps' layout uniqueness condition.
        self.counts = [len(self.domains[name].windows) for name in DOMAINS]
        if any(count < 1 for count in self.counts):
            raise ValueError("Both domains need valid windows in the requested split")
        self.offsets = np.cumsum([0, *self.counts])
        self.windows = np.arange(int(self.offsets[-1]), dtype=np.int64)

    def numpy_batch(self, batch, *, indices=None):
        if type(batch) is not int or batch < 1:
            raise ValueError("Batch size must be a positive integer")
        selection = self.generator.integers(len(self.windows), size=batch) if indices is None else np.asarray(indices)
        if (selection.shape != (batch,) or selection.dtype.kind not in "iu"
                or (selection < 0).any() or (selection >= len(self.windows)).any()):
            raise ValueError("Invalid union window indices")
        domain_ids = np.searchsorted(self.offsets[1:], selection, side="right")
        result = {}
        for domain_id, name in enumerate(DOMAINS):
            rows = np.flatnonzero(domain_ids == domain_id)
            if not len(rows):
                continue
            local_indices = selection[rows] - self.offsets[domain_id]
            sample = self.domains[name].numpy_batch(len(rows), indices=local_indices)
            for key, value in sample.items():
                if key == "window_indices":
                    continue
                if isinstance(value, np.ndarray):
                    if key not in result:
                        result[key] = np.empty((batch, *value.shape[1:]), dtype=value.dtype)
                    result[key][rows] = value
                else:
                    if key not in result:
                        result[key] = [None] * batch
                    for row, item in zip(rows, value, strict=True):
                        result[key][int(row)] = item
        result["window_indices"] = selection.copy()
        result["domain_ids"] = domain_ids.astype(np.int32)
        result["local_window_indices"] = selection - self.offsets[domain_ids]
        return result

    def provenance(self):
        total = len(self.windows)
        return {"schema": "local_orca.official_maps_interact_union.v2", "split": self.split,
                "conditional_windows": total, "sampling": SAMPLING, "domain_order": list(DOMAINS),
                "max_neighbors": MAX_NEIGHBORS, "author_dataset": False,
                "domains": {name: {**self.domains[name].provenance(),
                                    "window_probability": self.counts[index] / total}
                            for index, name in enumerate(DOMAINS)}}

    def resume_identity(self):
        return {"sampling": SAMPLING, "domain_order": list(DOMAINS), "max_neighbors": MAX_NEIGHBORS,
                "split": self.split,
                "domains": {name: {"manifest": str(self.domains[name].manifest_path),
                                   "manifest_sha256": sha256(self.domains[name].manifest_path),
                                   "conditional_windows": self.counts[index]}
                            for index, name in enumerate(DOMAINS)}}


def manifest_window_counts(maps_manifest=DEFAULT_MAPS, interact_manifest=DEFAULT_INTERACT):
    """Metadata only: no corpus reload/QC or array access."""
    manifests = {name: json.loads(Path(path).read_text())
                 for name, path in zip(DOMAINS, (maps_manifest, interact_manifest), strict=True)}
    return {split: {name: sum(episode["conditional_window_count"]
                              for scene in manifests[name]["splits"][split]["scenes"]
                              for episode in scene["episodes"]) for name in DOMAINS}
            for split in ("train", "validation", "test")}
