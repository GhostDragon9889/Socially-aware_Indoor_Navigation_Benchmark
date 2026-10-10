#!/usr/bin/env python3
"""Plot and summarize a completed CPU style comparison without SDK imports."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_pedestrian_style import record  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.evaluation.read_text())
    names = list(document["runs"])
    runs = [document["runs"][name] for name in names]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    labels = ["<0.1", "0.1–0.5", "0.5–1", "1–1.5", "1.5–2", "2–3", ">=3"]
    rows = [run["inclusive_speed_conditioned"]["speed_bins"] for run in runs]
    reference_counts = np.array([row["reference_count"] for row in rows[0]])
    axes[0, 0].plot(np.arange(7), reference_counts/reference_counts.sum(), "o-", label="Fixed validation reference")
    minimum = runs[0]["inclusive_speed_conditioned"]["minimum_metric_samples_per_bin"]
    def supported_mean(row, field, side):
        values = row["metrics"][field][side]
        return values["mean"] if values["count"] >= minimum else None
    reference_velocity = [supported_mean(row, "local_pose_velocity_rms_chain_s", "reference") for row in rows[0]]
    reference_support = [supported_mean(row, "toe_near_ground_proxy_double_support", "reference") for row in rows[0]]
    axes[0, 1].plot(np.arange(7), reference_velocity, "o-", label="Fixed validation reference")
    axes[1, 0].plot(np.arange(7), reference_support, "o-", label="Fixed validation reference")
    for name, values in zip(names, rows, strict=True):
        counts = np.array([row["generated_count"] for row in values])
        axes[0, 0].plot(np.arange(7), counts/counts.sum(), "o-", label=name)
        axes[0, 1].plot(np.arange(7), [supported_mean(row, "local_pose_velocity_rms_chain_s", "generated") for row in values], "o-", label=name)
        axes[1, 0].plot(np.arange(7), [supported_mean(row, "toe_near_ground_proxy_double_support", "generated") for row in values], "o-", label=name)
        axes[1, 1].plot(np.arange(7), [row["pose_velocity_gaussian_mmd2"]["value"] if row["pose_velocity_gaussian_mmd2"] else None for row in values], "o-", label=name)
    calibration = runs[0]["heldout_between_clip_group_calibration"]["comparison"]["speed_bins"]
    axes[1, 1].plot(np.arange(7), [row["pose_velocity_gaussian_mmd2"]["value"] if row["pose_velocity_gaussian_mmd2"] else None for row in calibration], "o--", label="Between validation clip groups")
    for axis in axes.flat:
        axis.set_xticks(np.arange(7), labels)
        axis.set_xlabel("Actual root XY speed (m/s)")
        axis.grid(alpha=.25)
        axis.legend(fontsize=8)
    axes[0, 0].set_ylabel("Fraction of legal intervals")
    axes[0, 1].set_ylabel("Local pose velocity RMS (chain lengths/s)")
    axes[1, 0].set_ylabel("Double toe near-ground proxy fraction")
    axes[1, 1].set_ylabel("Pose + velocity Gaussian MMD squared")
    axes[0, 0].set_title("Speed mixture is separate from pose quality")
    axes[0, 1].set_title("Articulation changes within physical-speed bins")
    axes[1, 0].set_title("Geometric support proxy: no measured foot contacts")
    axes[1, 1].set_title("Fixed bandwidth; unsupported bins omitted")
    figure.suptitle("Local held-out motion diagnostics - not author-equivalent style scores")
    image_path = args.output_prefix.with_suffix(".png")
    figure.savefig(image_path, dpi=160)
    plt.close(figure)
    common_aggregates = {}
    fields = runs[0]["inclusive_speed_conditioned"]["aggregate_metrics"]
    for field in fields:
        supported = [index for index in range(7) if all(values[index]["metrics"][field]["wasserstein_1d"] is not None for values in rows)]
        common_aggregates[field] = {"common_bin_indices": supported,
            "equal_common_bin_wasserstein_1d": {name: float(np.mean([values[index]["metrics"][field]["wasserstein_1d"] for index in supported])) if supported else None
                for name, values in zip(names, rows, strict=True)},
            "qualification": "Uses exactly the same supported bins across compared captures; no unsupported score treated as zero"}
    summary = {"status": "passed", "evaluation": record(args.evaluation), "plot": record(image_path),
        "reference_selection": {key: value for key, value in document["reference"]["selection"].items() if key != "windows"},
        "common_bin_aggregates": common_aggregates, "reference_sample_hashes": document["reference"]["actual_selected_array_sha256"],
        "run_summaries": {name: {"navigation": run["source_navigation_outcomes"],
            "models": run["sources"]["models_resolved_from_evaluation_arguments"],
            "inclusive": run["inclusive_speed_conditioned"]["aggregate_metrics"],
            "pose_velocity_mmd2": run["inclusive_speed_conditioned"]["equal_supported_bin_mean_pose_velocity_mmd2"],
            "actual_foot_ground_contact": None, "true_com_world": None} for name, run in zip(names, runs, strict=True)},
        "author_equivalent": False, "style_convergence_claim": False}
    args.output_prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": "passed", "png": str(image_path)}))


if __name__ == "__main__":
    main()
