"""Camera depth validation for CrowdSim robots.

Provides:
  - _record_headless_camera_burst(env, frames, save_envs):
      Step N times in headless mode, save depth PNGs, print timing + anomaly stats.
  - _add_camera_check_key(env):
      Bind 'C' key to capture a burst of camera frames interactively.

Automatically called by crowd_sim.py and train_ppo.py on startup
when sensors.camera.enabled: true and --headless is used.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from CrowdSim.control.drive import articulation_torch_view


def record_headless_camera_burst(env, frames: int = 100, save_envs: int = 1) -> None:
    """Step N times, save raw metric depth and PNG previews, and print timing stats.

    Args:
        env: The ProtoMotions env (must have crowdsim_robot_camera sensor).
        frames: Number of frames to capture.
        save_envs: Number of env cameras to save (default 1, use 0 for stats only).
    """
    if not hasattr(env, "crowdsim_robot_camera"):
        print("[Camera] ERROR: env.crowdsim_robot_camera not found.")
        return

    camera = env.crowdsim_robot_camera
    n_envs = env.num_envs
    n_actions = env.robot_config.number_of_actions
    zero = torch.zeros(n_envs, n_actions, device=env.device)
    sim = env.simulator
    output_dir = Path("output/renderings/camera_headless")
    output_dir.mkdir(parents=True, exist_ok=True)
    for f in output_dir.glob("*.png"):
        f.unlink()
    for f in output_dir.glob("env*_frame*.npy"):
        f.unlink()

    save_eids = list(range(min(max(save_envs, 0), n_envs)))
    print(f"[Camera] Headless burst: {frames} frames × {n_envs} env(s), "
          f"saving envs {save_eids if save_eids else 'none'}")

    depth_frames = []
    times = []
    missing = 0
    for step in range(frames):
        t0 = time.perf_counter()
        if hasattr(sim, "_sim") and sim._sim is not None:
            sim._sim.render()
        output = camera.data.output
        d = output.get("distance_to_image_plane")
        if d is None:
            missing += 1
            env.step(zero)
            continue
        # CPU tensors may reuse the sensor buffer; freeze this frame before step().
        depth = articulation_torch_view(d).detach().cpu().numpy().copy()
        t1 = time.perf_counter()
        depth_frames.append(depth)
        times.append(t1 - t0)
        env.step(zero)

    if not depth_frames:
        print("[Camera] ERROR: No frames captured!")
        return

    all_d = np.stack(depth_frames, axis=0)  # (T, N, H, W, 1)
    T = len(depth_frames)
    rt = np.array(times)
    print(f"  Captured {T}/{frames} frames ({missing} missing)")
    print(f"  Timing: render+read {rt.mean()*1000:.0f}±{rt.std()*1000:.0f} ms/frame")

    for eid in (save_eids if save_eids else range(n_envs)):
        ed = all_d[:, eid, ..., 0]  # (T, H, W)
        valid = np.isfinite(ed) & (ed > 0)
        vf = valid.mean(axis=(1, 2))
        means = np.array([ed[t][valid[t]].mean() if valid[t].any() else -1 for t in range(T)])

        anomalies = 0
        for t in range(1, T):
            if means[t] >= 0 and means[t - 1] >= 0 and abs(means[t] - means[t - 1]) > 1.0:
                anomalies += 1
            if abs(vf[t] - vf[t - 1]) > 0.05:
                anomalies += 1

        valid_mean_vals = means[means >= 0]
        detail = (f"depth={valid_mean_vals.mean():.2f}±{valid_mean_vals.std():.2f}m"
                  if valid_mean_vals.size else "no valid depth hits")
        print(f"  env{eid}: {detail} "
              f"valid={vf.mean()*100:.0f}±{vf.std()*100:.0f}% "
              f"anomalies={anomalies}")
        if anomalies:
            print(f"    ⚠ {anomalies} frame-to-frame jumps detected")

        if eid in save_eids:
            for t in range(T):
                # Keep no-hit sentinels and metric precision alongside the preview.
                np.save(output_dir / f"env{eid:02d}_frame{t:04d}.npy", ed[t])
                d_img = np.where(np.isfinite(ed[t]), ed[t], 0.0)
                d_img = np.clip(d_img / 20.0 * 255, 0, 255).astype(np.uint8)
                Image.fromarray(d_img, "L").save(
                    output_dir / f"env{eid:02d}_frame{t:04d}.png")

    if save_eids:
        print(f"[Camera] Saved {T * len(save_eids)} raw depth frames and PNG previews to {output_dir}/")


def add_camera_check_key(env) -> None:
    """Bind 'C' key to capture and save one camera frame from each env."""
    if not hasattr(env, "crowdsim_robot_camera"):
        return
    keyboard = getattr(getattr(env, "simulator", None), "keyboard_interface", None)
    if keyboard is None:
        return

    import numpy as np
    from PIL import Image
    from pathlib import Path

    camera = env.crowdsim_robot_camera
    out = Path("output/renderings/camera_check")
    out.mkdir(parents=True, exist_ok=True)

    def capture(_key=None):
        output = camera.data.output
        rgb = output.get("rgb")
        depth_raw = output.get("distance_to_image_plane")
        if depth_raw is None:
            print("[Camera] No data yet. Wait for first camera update.")
            return
        depth = articulation_torch_view(depth_raw).detach().cpu().numpy()
        print(f"[Camera] {env.num_envs} env(s), {depth.shape[1]}×{depth.shape[2]}:")
        for eid in range(env.num_envs):
            d = depth[eid, ..., 0]
            valid = np.isfinite(d) & (d > 0)
            dv = d[valid]
            if dv.size:
                detail = f"depth [{dv.min():.2f}~{dv.max():.2f}]m avg={dv.mean():.2f}m"
            else:
                detail = "no valid depth hits"
            print(f"  env{eid}: valid={valid.mean()*100:.0f}% {detail}")
            # Preserve metric values and no-hit sentinels. The PNG below is
            # only a visualization, where invalid pixels are displayed black.
            np.save(out / f"depth_env{eid}.npy", d)
            d_img = np.where(np.isfinite(d), d, 0.0)
            d_img = np.clip(d_img / 20.0 * 255, 0, 255).astype(np.uint8)
            Image.fromarray(d_img, "L").save(out / f"depth_env{eid}.png")
            if rgb is not None:
                r = articulation_torch_view(rgb)[eid].detach().cpu().numpy()
                if r.dtype != np.uint8:
                    if np.issubdtype(r.dtype, np.floating) and r.min() >= 0.0 and r.max() <= 1.0:
                        r = r * 255
                    r = np.clip(r, 0, 255).astype(np.uint8)
                Image.fromarray(r[..., :3]).save(out / f"rgb_env{eid}.png")
        print(f"[Camera] Saved to {out}/")

    keyboard.add_callback("C", capture)
    print("[CrowdSim] Press 'C' to capture one camera frame from all envs.")


