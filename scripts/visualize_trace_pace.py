#!/usr/bin/env python3
"""Plot captured PhysX positions and animate measured SMPL body origins."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--scene", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--animation", action="store_true")
    parser.add_argument("--support-plot", action="store_true",
                        help="Plot measured120Hz foot loads and body-origin heights")
    parser.add_argument("--actor", type=int, default=0)
    parser.add_argument("--start-seconds", type=float, default=0.)
    parser.add_argument("--duration-seconds", type=float, default=12.)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    result = json.loads((args.run / "result.json").read_text())
    with np.load(args.run / "physical.npz", allow_pickle=False) as data:
        samples = data["samples"].copy()
        body = data["body_positions"].copy()
    dt = 1 / result["physics"]["control_hz"]
    if not (0 <= args.actor < samples.shape[1] and np.isfinite(args.start_seconds)
            and np.isfinite(args.duration_seconds) and args.start_seconds >= 0
            and args.duration_seconds > 0 and args.start_seconds < len(samples) * dt):
        parser.error("Require a captured actor and a finite positive interval")
    time = np.arange(1, len(samples) + 1) * dt
    colors = ["#2962a3", "#c8662a", "#368463", "#9467ac"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for actor in range(samples.shape[1]):
        color = colors[actor % len(colors)]
        label = f"Person {actor + 1}"
        axes[0].plot(*samples[:, actor, :2].T, color=color, label=label)
        axes[0].scatter(*samples[0, actor, :2], color=color, marker="o")
        axes[0].scatter(*samples[-1, actor, :2], color=color, marker="x")
        axes[1].plot(time, np.linalg.norm(samples[:, actor, 2:4], axis=-1), color=color)
        axes[2].plot(time, samples[:, actor, 5], color=color)
    if args.scene:
        scene = yaml.safe_load(args.scene.read_text())
        for box in scene["cuboids"]:
            center, size = np.array(box["center"]), np.array(box["size"])
            axes[0].add_patch(
                plt.Rectangle(center[:2] - size[:2] / 2, *size[:2], color="#777777")
            )
        axes[0].set_xlim(scene["bounds_xy"][0] - 1, scene["bounds_xy"][2] + 1)
        axes[0].set_ylim(scene["bounds_xy"][1] - 1, scene["bounds_xy"][3] + 1)
    axes[0].set(
        title="Actual pelvis XY",
        xlabel="World x (m)",
        ylabel="World y (m)",
        aspect="equal",
    )
    axes[0].legend(loc="best", fontsize=8)
    axes[1].set(title="Measured speed", xlabel="Physical time (s)", ylabel="m/s")
    axes[2].set(title="Measured pelvis height", xlabel="Physical time (s)", ylabel="m")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(
        f"TRACE + public PACER | actual PhysX | {result['status']} | {len(samples)} controls | {result['falls']} falls"
    )
    fig.savefig(args.output / "actual_physics.png", dpi=150)
    plt.close(fig)
    if args.support_plot:
        with np.load(args.run / "ground_support.npz", allow_pickle=False) as ground:
            if not (len(ground["control_step"]) == len(samples)
                    and (ground["substep_valid_count"] == 4).all()
                    and np.array_equal(ground["physics_step"].reshape(-1),
                                       np.arange(1, len(samples)*4+1))):
                raise ValueError("Complete native four-substep capture required")
            loads = ground["side_normal_load_substeps_n"][:, :, args.actor].reshape(-1, 2)
            feet = ground["foot_body_origin_pos_w"][:, :, args.actor].reshape(-1, 4, 3)
            foot_names = list(ground["foot_names"])
        native_time = np.arange(1, len(loads)+1) / result["physics"]["physics_hz"]
        selected = ((native_time >= args.start_seconds)
                    & (native_time < args.start_seconds + args.duration_seconds))
        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, constrained_layout=True)
        for side, label, color in ((0, "Left", colors[0]), (1, "Right", colors[1])):
            axes[0].plot(native_time[selected], loads[selected, side], label=label, color=color)
            ankle = foot_names.index("L_Ankle" if side == 0 else "R_Ankle")
            axes[1].plot(native_time[selected], feet[selected, ankle, 2], color=color)
        axes[0].set(ylabel="Measured floor normal load (N)")
        axes[0].legend()
        axes[1].set(xlabel="Physical time (s)", ylabel="Ankle body-origin world height (m)")
        for axis in axes:
            axis.grid(alpha=.2)
        fig.suptitle(f"Actual120Hz PhysX gait | person{args.actor+1} | body origins are not contact patches")
        fig.savefig(args.output / "measured_gait_support.png", dpi=150)
        plt.close(fig)
    if not args.animation:
        return
    # Correct SMPL kinematic chains; coordinates are saved physics positions,
    # not reference animation, root-normalised retargeting, or generated art.
    names = result["physics"]["body_names"]
    chains = [
        ["Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe"],
        ["Pelvis", "R_Hip", "R_Knee", "R_Ankle", "R_Toe"],
        ["Pelvis", "Torso", "Spine", "Chest", "Neck", "Head"],
        ["Chest", "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand"],
        ["Chest", "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand"],
    ]
    indices = [[names.index(name) for name in chain] for chain in chains]
    fig = plt.figure(figsize=(6, 6))
    axis = fig.add_subplot(projection="3d")
    lines = [
        axis.plot([], [], [], "o-", linewidth=2, markersize=3, color=colors[0])[0]
        for _ in chains
    ]
    axis.set(
        xlabel="World x (m)", ylabel="World y (m)", zlabel="World z (m)", zlim=(0, 2)
    )
    axis.set_box_aspect((1, 1, 1))
    first = int(args.start_seconds / dt)
    last = min(len(body), int((args.start_seconds + args.duration_seconds) / dt))
    frames = np.arange(first, last, 3)  # 10fps, exact physical time

    def draw(index):
        points = body[index, args.actor]
        root = points[0]
        axis.set_xlim(root[0] - 1, root[0] + 1)
        axis.set_ylim(root[1] - 1, root[1] + 1)
        for line, ids in zip(lines, indices):
            line.set_data_3d(*points[ids].T)
        axis.set_title(
            f"Actual PhysX body origins | t={(index + 1) * dt:.2f}s\nCamera follows person {args.actor+1}; world positions unchanged",
            fontsize=10,
        )
        return lines

    animation = FuncAnimation(fig, draw, frames=frames, interval=100, blit=False)
    animation.save(args.output / "actual_gait.gif", writer=PillowWriter(fps=10))
    plt.close(fig)


if __name__ == "__main__":
    main()
