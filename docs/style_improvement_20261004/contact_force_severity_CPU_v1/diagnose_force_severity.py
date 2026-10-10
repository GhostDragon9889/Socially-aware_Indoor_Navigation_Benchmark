"""Read-only NumPy severity summary of three admitted, sealed 200 s captures."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ADMISSION_SOURCE = ROOT / "docs/style_improvement_20261004/actual_tracking_contact_CPU_v1/diagnose_tracking_contact.py"
ADMISSION_SHA = "33dd5405f491955aff3e0dc561e0f6d627a16bd4b033b4a55530191b536f5a32"
EXPECTED = {"OLD63e_S1": (0, 0.), "NEWb563_S1": (70, 1288.4229736328125),
            "NEWb563_S10": (279, 1069.25244140625)}
PACER_URL = "https://github.com/nv-tlabs/pacer/blob/baeb6241f03fc9d78370a64e3c2dcef9d830d97f/pacer/env/tasks/humanoid_pedestrain_terrain.py#L1392"
FORBIDDEN = {"torch", "isaacsim", "isaaclab", "omni", "pxr", "warp"}


def load_admission():
    """Execute only the exact frozen CPU source bytes; its main is not called."""
    info = ADMISSION_SOURCE.stat()
    source = ADMISSION_SOURCE.read_bytes()
    if (not ADMISSION_SOURCE.is_file() or ADMISSION_SOURCE.is_symlink() or info.st_nlink != 1
            or info.st_mode & 0o222 or len(source) != 26264
            or hashlib.sha256(source).hexdigest() != ADMISSION_SHA):
        raise ValueError("Frozen original CPU admission source identity mismatch")
    module = types.ModuleType("frozen_tracking_contact_admission")
    module.__file__ = str(ADMISSION_SOURCE)
    exec(compile(source, str(ADMISSION_SOURCE), "exec"), module.__dict__)
    return module


def summarize(values):
    """Exact boundaries; conditional quantiles use NumPy's linear method."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Force values must be finite and nonnegative")
    contact = values[values > 1.]
    bands = {"le_1_n": int((values <= 1.).sum()),
             "gt_1_le_50_n": int(((values > 1.) & (values <= 50.)).sum()),
             "gt_50_n": int((values > 50.).sum())}
    if sum(bands.values()) != values.size:
        raise ValueError("Severity bands do not partition the complete denominator")
    return {"pair_controls": int(values.size), "bands": bands,
            "force_given_gt_1_n": {"count": int(contact.size),
                                   "p50": float(np.quantile(contact, .5, method="linear")) if contact.size else None,
                                   "p95": float(np.quantile(contact, .95, method="linear")) if contact.size else None,
                                   "max": float(contact.max()) if contact.size else None}}


def analyze(name, original, prior):
    identities, audit, result, _ = prior.admission(name, original)  # Before any NPZ array read.
    with np.load(identities["physical_states"]["path"], allow_pickle=False) as archive:
        force, flags = archive["peer_contact_max_force_n"], archive["peer_contact"]
    if (force.shape != (6000, 4, 4) or force.dtype != np.dtype("float32")
            or not np.isfinite(force).all() or (force < 0).any()
            or flags.shape != force.shape or flags.dtype != np.bool_
            or not np.array_equal(flags, force > 1.)
            or np.any(np.diagonal(force, axis1=1, axis2=2) != 0)):
        raise ValueError("Invalid original peer force/flag shape, dtype, finite, threshold or self diagonal")
    union_force = np.maximum(force, force.swapaxes(1, 2))
    union_flags = flags | flags.swapaxes(1, 2)
    pair_force = np.stack([union_force[:, i, j] for i, j in prior.PAIRS], axis=-1)
    pair_flags = np.stack([union_flags[:, i, j] for i, j in prior.PAIRS], axis=-1)
    count, peak = int(pair_flags.sum()), float(pair_force.max())
    if (pair_force.shape != (6000, 6) or not np.array_equal(pair_flags, pair_force > 1.)
            or (count, peak) != EXPECTED[name]):
        raise ValueError("Complete unordered-pair counts or peak differ from fixed original capture")
    for published in (audit["original_obstacle_peer_channels"], original["original_contact_counts"]):
        if (published["peer_contact_unordered_pair_frames"] != count
                or published["maximum_peer_force_n"] != peak):
            raise ValueError("Original sealed audit/comparison contact counts or peak disagree")
    return {"input_identities": identities, "all_controls": 6000, "unordered_pairs": 6,
            "all_unordered_pair_controls": 36000, "all_pair_summary": summarize(pair_force),
            "per_pair": [{"actors": [i, j], **summarize(pair_force[:, index])}
                         for index, (i, j) in enumerate(prior.PAIRS)],
            "original_gt_1_pair_controls": count, "original_maximum_peer_force_n": peak,
            "contact_measurement_from_genuine_result": result["physical_contact_measurement"],
            "all_failed_contact_idle_reset_frames_retained": True,
            "frame_or_style_filter_applied": False}


def main():
    started = time.perf_counter()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Run with CUDA_VISIBLE_DEVICES='' explicitly")
    if any(module.split(".")[0] in FORBIDDEN for module in sys.modules):
        raise ValueError("CPU diagnostic must not import model or simulator modules")
    prior = load_admission()
    source, admission = prior.record(__file__), prior.record(ADMISSION_SOURCE, readonly=True)
    producer, comparison = prior.record(prior.PRODUCER), prior.record(prior.COMPARISON, readonly=True)
    if (admission["sha256"] != ADMISSION_SHA or producer["sha256"] != prior.PRODUCER_SHA
            or comparison["sha256"] != prior.COMPARISON_SHA):
        raise ValueError("Original source-bound admission, producer or actual comparison changed")
    output = HERE / "actual_diagnostic.json"
    if output.exists() or output.is_symlink():
        raise ValueError("Preserve existing diagnostic output; never overwrite")
    original = prior.read(prior.COMPARISON)
    runs = {name: analyze(name, original["runs"][name], prior) for name in EXPECTED}
    for expected in [source, admission, producer, comparison,
                     *(identity for run in runs.values() for identity in run["input_identities"].values())]:
        if prior.record(expected["path"], readonly=expected["mode"] == "0o444") != expected:
            raise ValueError("Source/input changed during actual CPU observation")
    if any(module.split(".")[0] in FORBIDDEN for module in sys.modules):
        raise ValueError("CPU diagnostic imported model or simulator modules")
    report = {"schema": "local_peer_contact.paper_inspired_force_severity.v1",
              "status": "completed_CPU_diagnostic", "execution_source": source,
              "frozen_admission_source": admission, "producer": producer, "original_comparison": comparison,
              "definition": {"units": "N", "unordered_pair": "max(Fij,Fji), six pairs per control",
                             "denominator": "all6000 controls x6 pairs=36000, including falls/resets/idle",
                             "bands": ["F<=1N", "1N<F<=50N", "F>50N"],
                             "conditional_quantiles": "all pair-controls with F>1N; NumPy linear P50/P95; empty=null",
                             "local_force": "maximum normal-force body-pair magnitude over4 physics substeps; all actor bodies including feet; floor/self excluded"},
              "paper_inspired_reference": {"source_url": PACER_URL,
                                            "threshold_n": 50., "author_metric_equivalence": False,
                                            "PACER_definition": "enableEarlyTermination; mask contactBodies (ankles/toes), norm(sum remaining body net-force vectors)>50N and progress_buf>1; other trajectory/disableCollision/timeout logic also applies"},
              "runs": runs, "CPU_elapsed_seconds": time.perf_counter()-started,
              "limits": ["50N is a paper-inspired severity threshold, not PACER termination reproduction.",
                         "Local all-body pair normal-force maxima differ from PACER masked-body net-vector sum norm.",
                         "Archived control maxima cannot recover contact location, substep force waveform, net-body vectors or impulse.",
                         "These observations establish neither causation nor model superiority."],
              "weights_read_or_models_loaded": False, "GPU_used": False, "SDK_started": False,
              "sealed34_outputs_read": False, "new_success_gate_or_promotion": False}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "summaries": {name: row["all_pair_summary"] for name, row in runs.items()}},
                     allow_nan=False))


if __name__ == "__main__":
    main()
