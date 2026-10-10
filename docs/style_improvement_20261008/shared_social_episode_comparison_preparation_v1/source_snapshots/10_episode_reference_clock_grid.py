"""Explicit private derivation for CPU/CUDA-consistent 30Hz reference clocks.

CUDA float32 division produced 150/30 == 5.000000476837158 and the unchanged
strict five-second query guard correctly rejected it. Divide in float64, then
round to float32; keep that guard, schema, data and sampling rules intact.
Source: https://docs.pytorch.org/docs/2.11/generated/torch.arange.html
Source: https://docs.pytorch.org/docs/2.11/generated/torch.div.html
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record
from scripts import train_shared_social_episode_reference as candidate

PINS = {
    "primitive": (candidate.REFERENCE, candidate.REFERENCE_SHA),
    "curriculum": (candidate.CURRICULUM, candidate.CURRICULUM_SHA),
}
REPLACEMENTS = {
    "primitive": (("ages[ids, None].float() / 30", "(ages[ids, None].double() / 30).float()", 1),),
    "curriculum": (
        ("torch.arange(151, device=self.device).float() / 30",
         "(torch.arange(151, device=self.device, dtype=torch.float64) / 30).float()", 2),
        ("native_times_s=torch.arange(151, device=self.device) / 30",
         "native_times_s=(torch.arange(151, device=self.device, dtype=torch.float64) / 30).float()", 1)),
}


def derive(key):
    path, sha = PINS[key]
    original = candidate.pinned(path, sha)
    source = path.read_text()
    derived = source
    for before, after, count in REPLACEMENTS[key]:
        if derived.count(before) != count:
            raise ValueError("Ambiguous reference clock derivation: " + key)
        derived = derived.replace(before, after)
    inverse = derived
    for before, after, count in reversed(REPLACEMENTS[key]):
        if inverse.count(after) != count:
            raise ValueError("Ambiguous reference clock inverse: " + key)
        inverse = inverse.replace(after, before)
    if inverse != source:
        raise ValueError("Reference clock derivation must be exactly reversible")
    compile(derived, str(path) + ":explicit_float64_clock", "exec")
    return derived, {"source": original, "derived_sha256": hashlib.sha256(derived.encode()).hexdigest(),
                     "replacement_count": sum(count for _, _, count in REPLACEMENTS[key]), "exact_inverse": True}


def describe():
    return {"schema": "shared_social.episode_reference_float64_clock_derivation.v1", "adapter": record(__file__),
            "derived_modules": {key: derive(key)[1] for key in PINS},
            "change": "30Hz division in float64 followed by explicit float32 rounding",
            "five_second_query_guard_preserved": True, "offset_clamping_added": False,
            "original_source_files_modified": False, "AMASS_sampling_changed": False}


def adapt(loaded, key):
    if Path(getattr(loaded, "__file__", "")).resolve() != PINS[key][0] or hasattr(loaded, "EPISODE_CLOCK_DERIVATION"):
        raise ValueError("Require an unused actual pinned private reference module")
    source, proof = derive(key)
    exec(compile(source, str(PINS[key][0]) + ":explicit_float64_clock", "exec"), loaded.__dict__)
    loaded.EPISODE_CLOCK_DERIVATION = proof
    return loaded


def adapt_probe(probe):
    parent_load = probe._load

    def load(path, name, identity):
        loaded = parent_load(path, name, identity)
        for key, (source, _) in PINS.items():
            if Path(path) == source:
                return adapt(loaded, key)
        return loaded

    probe._load = load
    return probe
