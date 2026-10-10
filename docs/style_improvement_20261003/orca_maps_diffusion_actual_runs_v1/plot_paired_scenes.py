"""Plot all saved scene metrics without sampling, model inference, or new metrics."""

import csv
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "orca_maps_diffusion_actual_array_peer_v1/all100_scene_metrics.csv"
EXPECTED = "ca61f4d7e4690aaa2ff0cd329706cf164b7a0d98eaa75679b0d1307a3e488a80"


def main():
    if hashlib.sha256(CSV.read_bytes()).hexdigest() != EXPECTED:
        raise ValueError("The independently checked all-scene table has changed")
    with CSV.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 100 or len({row["scene_id"] for row in rows}) != 100:
        raise ValueError("Exactly the registered 100 unique validation scenes are required")
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout="constrained")
    for ax, key, title, lower in zip(
        axes,
        ("ADE_m", "FDE_m", "min_obstacle_surface_gap_m"),
        ("Mean trajectory error (m)", "Endpoint error (m)", "Minimum obstacle clearance (m)"),
        (True, True, False),
    ):
        old = np.array([float(row["old_" + key]) for row in rows])
        new = np.array([float(row["new_" + key]) for row in rows])
        improved = new < old if lower else new > old
        colors = np.where(improved, "#2166ac", "#b2182b")
        ax.scatter(old, new, c=colors, s=25, alpha=0.8, edgecolors="none")
        minimum = min(float(old.min()), float(new.min()), 0.0)
        maximum = max(float(old.max()), float(new.max()), 0.0)
        margin = 0.05 * (maximum - minimum)
        limits = (minimum - margin, maximum + margin)
        ax.plot(limits, limits, linestyle="--", color="0.5", linewidth=1)
        ax.set(xlim=limits, ylim=limits, title=title, xlabel="Old 63e7", ylabel="New b563")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.15)
        ax.text(0.04, 0.95, f"{improved.sum()}/100 improved", transform=ax.transAxes,
                va="top", fontsize=10)
        if not lower:
            ax.axhline(0, color="0.7", linewidth=0.7)
            ax.axvline(0, color="0.7", linewidth=0.7)
            ax.text(0.04, 0.04, "Negative = penetration\nWorst: -2.629 → -2.720 m",
                    transform=ax.transAxes, fontsize=9)
    figure.suptitle("Same 100 ORCA validation scenes · one window per scene · all predictions retained")
    figure.text(0.5, -0.035, "Blue: improvement; red: regression. Unguided open-loop predictions; not physical navigation.",
                ha="center", fontsize=10)
    output = HERE / "paired100_scene_comparison.png"
    if output.exists():
        raise FileExistsError("Preserve the original plot; choose a new artifact name")
    figure.savefig(output, dpi=180, bbox_inches="tight")
    output.chmod(0o444)
    print(output)


if __name__ == "__main__":
    main()
