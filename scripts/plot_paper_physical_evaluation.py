#!/usr/bin/env python3
"""Export measured pelvis trajectories, preserving physical reset boundaries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = (
    ("Legacy PD proxy", "shared_obstacle_evaluation_1000"),
    ("Corrected power / endpoint", "shared_obstacle_corrected_endpoint_1000"),
    ("Corrected power / timed A*", "shared_obstacle_corrected_timed_1000"),
)


def main():
    out = ROOT / "docs/diffusion_amp_20261001/physical_evaluation_comparison"
    out.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    sources = []
    for ax, (title, name) in zip(axes, RUNS):
        directory = ROOT / "output/diffusion_amp" / name
        report = json.loads((directory / "result.json").read_text())
        if report["status"] != "passed" or report.get("error"):
            raise ValueError(f"Incomplete physical run: {name}")
        state_path = directory / "physical_states.npz"
        with np.load(state_path, allow_pickle=False) as data:
            xy = data["root_pos"][..., :2]
            reset, fall = data["reset_mask"], data["terminated"]
            if xy.shape != (6000, 4, 2) or not np.isfinite(xy).all():
                raise ValueError("Expected finite 6000-frame, four-actor capture")
            for actor in range(4):
                color = f"C{actor}"
                points = xy[:, actor].copy()
                points[reset[:, actor]] = np.nan
                ax.plot(points[:, 0], points[:, 1], color=color, linewidth=.65, alpha=.8,
                        label=f"Actor {actor}")
                ax.scatter(*xy[0, actor], color=color, marker="o", s=28)
                ax.scatter(*xy[-1, actor], color=color, marker="s", s=28)
                incidents = xy[fall[:, actor], actor]
                if len(incidents):
                    ax.scatter(incidents[:, 0], incidents[:, 1], color=color, marker="x", s=35)
                goal = data["route_vertex_xy"][-1, actor]
                ax.scatter(*goal, edgecolor=color, facecolor="none", marker="*", s=90)
        scene = report["evaluation_scene"]
        for box in scene["cuboids"]:
            x, y, _ = box["center"]
            w, h, _ = box["size"]
            ax.add_patch(Rectangle((x-w/2, y-h/2), w, h, color="gray", alpha=.25))
        bounds = scene["bounds_xy"]
        ax.set(xlim=(bounds[0]-1, bounds[2]+1), ylim=(bounds[1]-1, bounds[3]+1),
               aspect="equal", xlabel="World x (m)", ylabel="World y (m)",
               title=f"{title}\nFalls {report['fall_resets']}; vertices {sum(report['route_vertices_reached'])}; A* failures {report['global_path_failures']}")
        ax.grid(alpha=.15)
        sources.append({"run": name, "states": str(state_path.relative_to(ROOT)),
                        "sha256": hashlib.sha256(state_path.read_bytes()).hexdigest()})
    axes[0].legend(loc="lower left", fontsize=7)
    figure.suptitle("Real Diffusion + AMP: 4 actors / 200 s each. Pelvis traces; resets omitted.\n"
                    "Circle=start, square=end, cross=fall, star=current route vertex. Contact collisions are not measured.", fontsize=10)
    figure.savefig(out / "pelvis_trajectories.png", dpi=160, facecolor="white")
    plt.close(figure)
    (out / "pelvis_trajectory_sources.json").write_text(json.dumps({"sources": sources,
        "qualification": "Measured pelvis positions only; geometry/contact proxies and success definitions are not author Table II metrics."}, indent=2))
    print(out / "pelvis_trajectories.png")


if __name__ == "__main__":
    main()
