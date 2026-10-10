"""Current-source audit probes; no Isaac Sim startup or real credentials/jobs."""
import ast
import contextlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace
import uuid

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


def extracted(path, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def caught(fn):
    try:
        fn()
        return {"exception": None}
    except Exception as exc:
        return {"exception": type(exc).__name__, "message": str(exc)}


def main():
    results = {}
    from protomotions.envs.motion_manager.motion_manager import MotionManager
    from protomotions.envs.motion_manager.config import MotionManagerConfig

    lib = SimpleNamespace(motion_weights=torch.ones(2), motion_lengths=torch.tensor([0.1, 2.0]), num_motions=lambda: 2)
    manager = MotionManager(MotionManagerConfig(sample_time_truncate_s=0.5), 2, 0.1, torch.device("cpu"), lib)
    initial = manager.motion_weights.tolist()
    with contextlib.redirect_stdout(io.StringIO()):
        manager.update_sampling_weights(torch.tensor([0.5, 1.0]))
    results["short_clip_reenabled"] = {"initial": initial, "updated": manager.motion_weights.tolist()}
    subset = MotionManager(MotionManagerConfig(subset_method=[1, 1, 1]), 3, 0.1, torch.device("cpu"), lib)
    results["explicit_subset_ignored"] = {"expected": [1, 1, 1], "actual": None if subset.available_motion_ids is None else subset.available_motion_ids.tolist()}

    fn = extracted(Path("FLUX/baselines/flux/deployment_server.py"), "world_to_local", {"math": math, "np": np})
    results["flux_world_to_local"] = caught(lambda: fn([0, 0, 0], [1, 0, 0]))
    fn = extracted(Path("FLUX/baselines/flux/policy_network.py"), "predict_pointgoal_action", {"torch": torch})
    results["flux_base_pointgoal"] = caught(lambda: fn(SimpleNamespace(), np.zeros((1, 3)), None, None))
    from PIL import Image
    rgb_stream, depth_stream = io.BytesIO(), io.BytesIO()
    Image.new("RGB", (4, 3)).save(rgb_stream, format="PNG")
    Image.new("I", (4, 3), color=10000).save(depth_stream, format="PNG")
    rgb_stream.seek(0)
    depth_stream.seek(0)
    shapes = {}
    def fake_step(goal, rgb, depth):
        shapes.update(rgb=list(rgb.shape), depth=list(depth.shape))
        return (np.zeros((1, 2, 3)), np.zeros((1, 1, 2, 3)), np.zeros((1, 1)), np.zeros((3, 4, 3)), np.zeros((3, 4, 3)))
    ns = {
        "app": SimpleNamespace(route=lambda *args, **kwargs: lambda fn: fn),
        "request": SimpleNamespace(files={"image": SimpleNamespace(stream=rgb_stream), "depth": SimpleNamespace(stream=depth_stream)}, form={"json": "{}", "robot_orientation": "[0,0,0]"}),
        "agent": SimpleNamespace(step_pointgoal=fake_step), "np": np, "math": math,
        "time": time, "json": json, "Image": Image, "jsonify": lambda payload: payload,
        "step_idx": 0, "start_time": 0, "is_first_step": False, "goal_world_pos": [1, 0, 0],
        "_flux_fps_writer": None, "_flux_fps_rgbd_writer": None,
    }
    fn = extracted(Path("FLUX/baselines/flux/deployment_server.py"), "eval_dual", ns)
    with contextlib.redirect_stdout(io.StringIO()):
        response = fn()
    results["flux_eval_dual_dimension_error"] = {"decoded_shapes": shapes, "expected_rgb": [1, 3, 4, 3], "expected_depth": [1, 3, 4, 1], "response": response, "model_inference_stubbed": True}
    for folder in ("FLUX", "NavDP"):
        fn = extracted(Path(folder) / "baselines/logoplanner/deployment/visualization.py", "plot_trajectory", {})
        with contextlib.redirect_stdout(io.StringIO()):
            results[folder.lower() + "_plot_trajectory"] = caught(lambda: fn(np.zeros((2, 3)), np.zeros((2, 3)), []))
    import matplotlib.pyplot as plt
    plt.close("all")

    import protomotions.train_slurm as slurm
    # Monkeypatch only the credential provider/CLI tail: never read real credentials.
    slurm.check_wandb_credentials = lambda user: "AUDIT_ONLY_TOKEN_NOT_A_REAL_SECRET"
    slurm.build_train_agent_args = lambda args, passthrough: "--audit-only"
    args = SimpleNamespace(simulator="newton", use_wandb=True, user="audit", seed=0, torch_deterministic=False, stall_timeout_min=0)
    with tempfile.TemporaryDirectory(prefix="protomotions-audit-") as directory:
        folder = Path(directory)
        cmd = slurm.build_job_command(args, folder, "python")
        results["slurm_key_in_generated_script"] = {"synthetic_token_embedded": "export WANDB_API_KEY=AUDIT_ONLY_TOKEN_NOT_A_REAL_SECRET" in cmd, "real_credentials_read": False}
        # Run only the generated installation gate, with a fake failing pip.
        fake_pip = folder / "pip"
        fake_pip.write_text("#!/bin/sh\nexit 42\n")
        fake_pip.chmod(0o700)
        job_id = "AUDIT_" + uuid.uuid4().hex
        marker = Path("/tmp") / ("_pip_done_" + job_id)
        env = os.environ.copy()
        env.update(PATH=str(folder) + os.pathsep + env["PATH"], SLURM_LOCALID="0", SLURM_JOB_ID=job_id)
        prefix = cmd.split("export WANDB_API_KEY=", 1)[0]
        try:
            proc = subprocess.run(["bash", "-c", prefix], env=env, capture_output=True, text=True, timeout=10)
            results["slurm_failed_install_gate"] = {"fake_pip_exit": 42, "gate_exit": proc.returncode, "ready_marker_exists": marker.exists()}
        finally:
            marker.unlink(missing_ok=True)
    output = ROOT / "docs/full_audit_20261001/core_reproductions.json"
    output.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
