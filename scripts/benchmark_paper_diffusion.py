"""Bounded plain/closed-gradient/compiled 100-step FP32 DDPM comparison.

This is a standalone Torch benchmark, never an Isaac Sim/Kit entry point.
It reads a frozen checkpoint and actual ORCA conditions without training or
writing weights. Compiler errors and lack of speedup are reported explicitly.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "output/diffusion_amp/diffusion_training/step_1000.ckpt")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/diffusion_amp/orca_social_pilot_v1/manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "resources/runtime_sources/diffusion_amp_20261001/trace_compiled_forward_benchmark.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cuda-memory-fraction", type=float, default=.15)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--backend", default="inductor", choices=["inductor", "eager"])
    parser.add_argument("--max-output-difference", type=float, default=1e-3)
    args = parser.parse_args(argv)
    if min(args.batch_sizes) < 1 or max(args.batch_sizes) > 64 or args.repeats < 1 or not 0 < args.cuda_memory_fraction <= .15:
        parser.error("Use batch1..64, positive repeats, and CUDA memory fraction(0,.15]")
    sys.path.insert(0, str(ROOT))
    import torch

    from CrowdSim.humanoid.diffusion import paper
    from CrowdSim.humanoid.orca_corpus import ORCACorpus, sha256
    from protomotions.utils.checkpoint import atomic_write_text

    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda", device.index if device.index is not None else torch.cuda.current_device())
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device)
    report = {"schema": "navisaaclab.paper_diffusion_forward_benchmark.v1", "status": "running",
              "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256(args.checkpoint),
              "torch_version": str(torch.__version__), "device": str(device), "dtype": "float32",
              "backend": args.backend, "compile_options": {"dynamic": True, "mode": "default"},
              "cuda_memory_fraction": args.cuda_memory_fraction, "seed": args.seed,
              "precision": {"autocast": False, "float32_matmul_precision": torch.get_float32_matmul_precision(),
                            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32},
              "batches": [], "scope": "Same read-only weights/config, full100-step DDPM and identical seeded noise; actual ORCA conditions are benchmark fixtures, not new navigation evaluation"}

    def publish():
        atomic_write_text(json.dumps(report, indent=2) + "\n", args.output)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def reset_peak():
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

    def memory():
        return ({"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                 "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)} if device.type == "cuda" else {})

    def original_autograd(mean, target, indices, *, num_samples=1):
        with torch.enable_grad():
            guided = mean.detach().requires_grad_(True)
            residual = guided[:, indices, :2].reshape(len(target), num_samples, -1, 2) - target[:, None]
            return torch.autograd.grad(residual.square().mean(dim=(-1, -2)).sum(), guided)[0]

    @contextmanager
    def noise_capture():
        original = torch.randn
        calls = []

        def captured(*values, **kwargs):
            result = original(*values, **kwargs)
            calls.append({"shape": list(result.shape), "dtype": str(result.dtype),
                          "sha256": hashlib.sha256(result.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()})
            return result

        with patch.object(torch, "randn", captured):
            yield calls

    def difference(reference, actual):
        return {"xy_max_abs_m": float((actual.xy - reference.xy).abs().max()),
                "velocity_max_abs_m_s": float((actual.velocity - reference.velocity).abs().max()),
                "yaw_max_wrapped_abs_rad": float((torch.remainder(actual.yaw - reference.yaw + torch.pi, 2 * torch.pi) - torch.pi).abs().max()),
                "full_state_max_abs": float((actual.sample_states - reference.sample_states).abs().max()),
                "time_equal": bool(torch.equal(reference.time, actual.time))}

    try:
        planner = paper.PaperDiffusionPlanner(checkpoint_path=args.checkpoint, device=device)
        if planner.config.diffusion_steps != 100 or any(p.dtype != torch.float32 for p in planner.model.parameters()):
            raise ValueError("This benchmark requires the unchanged100-step FP32 checkpoint")
        report["model_state_keys_before"] = list(planner.model.state_dict())
        corpus = ORCACorpus(args.manifest, split="test", device=device, seed=args.seed)
        for batch_size in args.batch_sizes:
            sample = corpus.diffusion_batch(batch_size)
            # An aligned observed-expert endpoint is solely a timing fixture.
            inputs = {key: sample[key] for key in ("states", "history", "neighbors", "raster", "history_mask", "neighbor_mask")}
            inputs.update(waypoints=sample["future"][:, 49:50, :2], waypoint_times=torch.tensor([5.], device=device), seed=args.seed)
            batch_report = {"batch_size": batch_size}
            report["batches"].append(batch_report)
            planner.set_compiled_forward(False)

            def timed_calls():
                values = []
                for _ in range(args.repeats):
                    synchronize()
                    started = time.monotonic()
                    planner.plan(**inputs)
                    synchronize()
                    values.append(time.monotonic() - started)
                return {"seconds": values, "median_seconds": statistics.median(values), **memory()}

            reset_peak()
            with patch.object(paper, "waypoint_guidance_gradient", original_autograd):
                planner.plan(**inputs)  # kernel warmup
                with noise_capture() as original_noise:
                    original = planner.plan(**inputs)
                batch_report["plain_autograd"] = timed_calls()
            reset_peak()
            with noise_capture() as closed_noise:
                closed = planner.plan(**inputs)
            batch_report["plain_closed_form"] = timed_calls()
            batch_report["closed_form_difference"] = difference(original, closed)
            batch_report["original_closed_noise_equal"] = original_noise == closed_noise
            batch_report["noise_draw_count"] = len(closed_noise)
            publish()
            try:
                planner.set_compiled_forward(True, backend=args.backend)
                reset_peak()
                synchronize()
                started = time.monotonic()
                planner.plan(**inputs)  # finish lazy compile before timing/capture
                synchronize()
                batch_report["compiled_cold_100step_seconds"] = time.monotonic() - started
                with noise_capture() as compiled_noise:
                    compiled = planner.plan(**inputs)
                batch_report["compiled"] = timed_calls()
                batch_report["compiled_difference"] = difference(closed, compiled)
                batch_report["compiled_noise_equal"] = compiled_noise == closed_noise
                batch_report["compile_status"] = "passed"
                batch_report["compiled_speedup_vs_closed"] = batch_report["plain_closed_form"]["median_seconds"] / batch_report["compiled"]["median_seconds"]
                delta = batch_report["compiled_difference"]
                batch_report["eligible_for_optional_flag"] = (batch_report["compiled_noise_equal"] and delta["time_equal"]
                    and max(delta["xy_max_abs_m"], delta["yaw_max_wrapped_abs_rad"], delta["velocity_max_abs_m_s"]) <= args.max_output_difference
                    and batch_report["compiled_speedup_vs_closed"] > 1.05)
            except Exception as error:
                batch_report.update(compile_status="failed", compile_error=f"{type(error).__name__}: {error}", eligible_for_optional_flag=False)
                planner.set_compiled_forward(False)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            publish()
        report["model_state_keys_unchanged"] = report["model_state_keys_before"] == list(planner.model.state_dict())
        report["checkpoint_sha256_after"] = sha256(args.checkpoint)
        report["checkpoint_unchanged"] = report["checkpoint_sha256"] == report["checkpoint_sha256_after"]
        report["status"] = "completed"
        publish()
        print(json.dumps({key: value for key, value in report.items() if key != "model_state_keys_before"}), flush=True)
        return 0
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        publish()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
