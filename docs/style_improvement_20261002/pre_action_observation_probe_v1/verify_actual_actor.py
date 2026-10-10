"""One loaded real baseline actor, observed/unobserved CPU parity only."""

from pathlib import Path
import hashlib
import importlib.util
import json
import random

import numpy as np
import torch
from CrowdSim.humanoid.runtime.evaluation_actor import EvaluationActor

D = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "actual_pre_action_observer",
    D / "proposed/CrowdSim/humanoid/runtime/pre_action_observation.py",
)
observer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer)


def main():
    result_dir = D / "actual_CPU_final"
    result_dir.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    checkpoint = Path(
        "output/diffusion_amp/amp_style_amass2848_backward_penalty_v1/step_1000.pt"
    )
    expected = "62c4edc195d5192d6ce415f5d9825b1f07195827869ceeba7ac33c4ba9f4654b"
    actor = EvaluationActor.from_checkpoint(
        checkpoint, device="cpu", expected_sha256=expected
    )
    generator = torch.Generator().manual_seed(7)
    self_obs = torch.randn(4, 357, generator=generator)
    task_obs = torch.randn(4, 1044, generator=generator)
    inputs = [x.clone() for x in (self_obs, task_obs)]
    buffers = {
        k: v.clone() for k, v in actor._inference.task_normalizer.state_dict().items()
    }
    py, numpy, torch_rng = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
    )
    model_calls = []
    handle = actor._inference.model.register_forward_pre_hook(
        lambda *_: model_calls.append(1)
    )
    baseline = actor.act(self_obs, task_obs)
    writer = observer.PreActionObservationWriter(result_dir / "samples")
    wrapped = observer.PreActionObservedActor(actor, writer)
    observed = wrapped.act(self_obs, task_obs)
    handle.remove()
    parity = {
        key: {
            "bytewise_equal": baseline[key].numpy().tobytes()
            == observed[key].numpy().tobytes(),
            "maximum_absolute_error": float(
                (baseline[key] - observed[key]).abs().max()
            ),
        }
        for key in baseline
    }
    after = np.random.get_state()
    report = {
        "schema": "actual_baseline_pre_action_observer_CPU_parity.v1",
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "bytes": checkpoint.stat().st_size,
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
        "one_actor_loaded": True,
        "actual_model_forwards": len(model_calls),
        "baseline_forward_calls": 1,
        "observed_forward_calls": 1,
        "outputs": parity,
        "normalizer_buffers_unchanged": all(
            torch.equal(value, actor._inference.task_normalizer.state_dict()[key])
            for key, value in buffers.items()
        ),
        "input_tensors_unchanged": all(
            torch.equal(a, b) for a, b in zip(inputs, (self_obs, task_obs))
        ),
        "RNG_unchanged": {
            "Python": py == random.getstate(),
            "NumPy": numpy[0] == after[0]
            and np.array_equal(numpy[1], after[1])
            and numpy[2:] == after[2:],
            "TorchCPU": torch.equal(torch_rng, torch.get_rng_state()),
        },
        "CPU_only": True,
        "SDK_started": False,
        "GPU_used": False,
        "physical_equivalence_claim": False,
    }
    assert report["checkpoint"]["sha256"] == expected and all(
        v["bytewise_equal"] for v in parity.values()
    )
    assert (
        len(model_calls) == 2
        and report["normalizer_buffers_unchanged"]
        and report["input_tensors_unchanged"]
        and all(report["RNG_unchanged"].values())
    )
    report["executed_harness"] = {"path": str(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with (result_dir / "parity.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    print(
        json.dumps(
            {
                "status": "actual_CPU_parity_passed",
                "outputs": 3,
                "one_observed_model_call": True,
            }
        )
    )


if __name__ == "__main__":
    main()
