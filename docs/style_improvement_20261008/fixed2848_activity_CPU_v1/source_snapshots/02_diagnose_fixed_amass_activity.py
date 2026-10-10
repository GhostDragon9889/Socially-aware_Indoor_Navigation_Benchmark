#!/usr/bin/env python3
"""Describe all fixed2848 AMASS velocities on CPU; do not select or train a model."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.analysis import amass_activity as kernel  # noqa: E402
from scripts.diagnose_shared_social_training import identity  # noqa: E402

MANIFEST = ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json"
MANIFEST_SHA = "68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca"
BANK_SHA = "46e0241108e200409c08d1115a270205fe0b6ce827e366a06e8a09f6a2cc82c0"
PINNED = {
    "CrowdSim/humanoid/amass_corpus.py": "2d4e4f7549aa38f5fd6763de4d3924a9cdd969746682c649ab773f033d747cf7",
    "CrowdSim/humanoid/amp/observations.py": "a64dc2eba5c132cf68a248f2324ea20446dd264a2ea8f9f139faae4738f9aeba",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    here = args.output.resolve()
    kernel.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not here.exists(), "Fresh CPU-only output required")
    sources = [identity(ROOT / path) for path in PINNED]
    kernel.require(all(row["sha256"] == PINNED[str(Path(row["path"]).relative_to(ROOT))] for row in sources),
                   "Expert sampling or feature source changed")
    sources += [identity(Path(__file__)), identity(Path(kernel.__file__)), identity(ROOT / "scripts/diagnose_shared_social_training.py")]
    manifest_record = identity(MANIFEST)
    kernel.require(manifest_record["sha256"] == MANIFEST_SHA, "Fixed manifest changed")
    manifest = json.loads(MANIFEST.read_text())
    entry = manifest["splits"]["train"]
    kernel.require(entry["clip_count"] == len(entry["clips"]) == 2848
                   and [row["motion_id"] for row in entry["clips"]] == list(range(2848)), "Require all2848 clips")
    bank_path = Path(entry["path"])
    bank_record = identity(bank_path if bank_path.is_absolute() else MANIFEST.parent / bank_path)
    kernel.require(bank_record["sha256"] == entry["sha256"] == BANK_SHA and bank_record["bytes"] == entry["bytes"],
                   "Actual fixed source bank differs")
    import torch
    from CrowdSim.humanoid.amass_corpus import AMASSCorpus
    from CrowdSim.humanoid.amp.observations import _DISC_DOFS

    before_rng = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone())
    corpus = AMASSCorpus(MANIFEST, device="cpu", dof_velocity_frame="child_joint")
    kernel.require(len(_DISC_DOFS) == 57 and all(tuple(_DISC_DOFS[start:start + 3]) ==
                   tuple(range(_DISC_DOFS[start], _DISC_DOFS[start] + 3)) and _DISC_DOFS[start] % 3 == 0
                   for start in range(0, 57, 3)), "Require all19 complete velocity triples")
    bank = corpus.data
    rows, frame_counts, expected_start = [], [], 0
    for clip in entry["clips"]:
        mid, count = clip["motion_id"], clip["frames"]
        start = int(bank["length_starts"][mid])
        kernel.require(int(bank["motion_num_frames"][mid]) == count and bank["motion_files"][mid] == clip["source_path"]
                       and abs(float(bank["motion_dt"][mid]) - 1 / 30) < 1e-7 and start == expected_start,
                       "Counts, source, timing or contiguous clip boundaries differ")
        selected = slice(start, start + count)
        rows.append(kernel.clip_statistics(bank["gvs"][selected, 0].numpy(), bank["dvs"][selected][:, _DISC_DOFS].numpy(),
                                           bank["gavs"][selected, 0].numpy()))
        frame_counts.append(count)
        expected_start += count
    kernel.require(expected_start == len(bank["dvs"]) == len(bank["gvs"]) == len(bank["gavs"]) == 523272,
                   "Require the complete contiguous bank")
    # Complete triples have rotation-invariant norms. Check the actual corpus's
    # parent-to-child conversion on every frame, instead of assuming arbitrary banks.
    max_conversion_norm_error = 0.
    low_joint_classification_changes, joint_bin_changes = 0, 0
    for start in range(0, expected_start, 4096):
        indices = torch.arange(start, min(start + 4096, expected_start))
        converted = corpus.state_at(indices).dof_vel[:, _DISC_DOFS].double().reshape(-1, 19, 3).norm(dim=-1)
        stored = bank["dvs"][indices][:, _DISC_DOFS].double().reshape(-1, 19, 3).norm(dim=-1)
        error = (converted - stored).abs()
        max_conversion_norm_error = max(max_conversion_norm_error, float(error.max()))
        kernel.require(torch.allclose(converted, stored, atol=1e-5, rtol=1e-5), "Actual inverse rotation changed velocity norms")
        converted_rms = (converted.square().sum(axis=1) / 57).sqrt().numpy()
        stored_rms = (stored.square().sum(axis=1) / 57).sqrt().numpy()
        low_joint_classification_changes += int(np.count_nonzero((converted_rms < .5) != (stored_rms < .5)))
        joint_bin_changes += int(np.count_nonzero(np.searchsorted(kernel.JOINT_EDGES, converted_rms, side="right") !=
                                                  np.searchsorted(kernel.JOINT_EDGES, stored_rms, side="right")))
    after_rng = np.random.get_state()
    kernel.require(random.getstate() == before_rng[0] and after_rng[0] == before_rng[1][0]
                   and np.array_equal(after_rng[1], before_rng[1][1]) and after_rng[2:] == before_rng[1][2:]
                   and torch.equal(torch.random.get_rng_state(), before_rng[2]) and not torch.cuda.is_initialized(),
                   "Diagnostic consumed global RNG or initialized CUDA")
    kernel.require(not any(name.split(".")[0] in {"isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules),
                   "Diagnostic imported an SDK")
    inputs = [manifest_record, bank_record]
    kernel.require(all(identity(row["path"]) == row for row in sources + inputs), "Inputs or sources changed")
    report = {"schema": "shared_social.fixed_AMASS_activity_CPU_diagnostic.v1", "status": "passed_complete_data_accounting",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "sources": sources, "inputs": inputs,
              "clip_count": 2848, "source_frames": expected_start, "all_clips_retained": True,
              "expert_window": {"past": 9, "future": 1, "history_frames": 10, "control_Hz": 30},
              "root_speed_bin_edges_m_s": [*kernel.SPEED_EDGES[:-1], "infinity"],
              "joint_component_RMS_bin_edges_rad_s": [*kernel.JOINT_EDGES[:-1], "infinity"],
              "low_thresholds": {"planar_speed_m_s": .1, "joint_component_RMS_rad_s": .5, "root_angular_speed_rad_s": .3},
              "thresholds_origin": "Local descriptive choices, not author protocol or standing labels; strict less-than",
              "velocity_frame": corpus.provenance()["dof_velocity"],
              "all_frame_child_conversion_norm_max_abs_error_rad_s": max_conversion_norm_error,
              "all_frame_child_conversion_low_joint_classification_changes": low_joint_classification_changes,
              "all_frame_child_conversion_joint_bin_changes": joint_bin_changes,
              "analysis_precision": "float64 norms of stored finite float32 velocities; rootXY and complete joint-triple norms are frame invariant",
              "aggregates": kernel.aggregate(rows),
              "weighting_limits": "Ideal uniform clip then legal center abstracts finite-precision torch.rand; not actual training exposure. All-frames uniform-clip measure has different boundary support.",
              "interpretation_limits": "Low root speed does not imply static joints. Even all10 frames below local thresholds do not establish standing semantics, physical support or discriminator causation. No D scores or held-out model quality are measured.",
              "NN_constructed": False, "SDK_started": False, "CUDA_initialized": False, "global_RNG_consumed": False,
              "training_restarted": False, "training_sources_changed": False, "default_model_promoted": False, "paper_effect_achieved": False}
    here.mkdir(parents=True)
    np.savez_compressed(here / "per_clip_statistics.npz", motion_ids=np.arange(2848), source_frame_counts=np.asarray(frame_counts),
                        counts=np.asarray([[view["count"] for view in row["views"]] for row in rows]),
                        histograms=np.asarray([[view["histogram"] for view in row["views"]] for row in rows]),
                        metric_means=np.asarray([[view["metric_means"] for view in row["views"]] for row in rows]),
                        proxy_counts=np.asarray([[view["proxy_counts"] for view in row["views"]] for row in rows]),
                        history_proxy_counts=np.asarray([row["history_proxy_counts"] for row in rows]))
    report["per_clip_statistics"] = identity(here / "per_clip_statistics.npz")
    (here / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for path in here.iterdir():
        path.chmod(0o444)
    print(json.dumps({"result": identity(here / "result.json"), "aggregates": report["aggregates"],
                      "all_frame_child_conversion_norm_max_abs_error_rad_s": max_conversion_norm_error}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
