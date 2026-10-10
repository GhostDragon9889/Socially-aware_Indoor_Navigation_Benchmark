"""CPU-only actual checkpoint facade/RNG oracle; does not advance simulation."""
from __future__ import annotations

from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from CrowdSim.humanoid.runtime.evaluation_actor import EvaluationActor, prepare_legacy_evaluation_rng
from CrowdSim.humanoid.training.trainer import PaperAMPTrainer


CASES = [
    ("negative1000", "output/diffusion_amp/amp_style_amass2848_backward_penalty_v1/step_1000.pt",
     "62c4edc195d5192d6ce415f5d9825b1f07195827869ceeba7ac33c4ba9f4654b"),
    ("mixed500", "output/diffusion_amp/amp_style_mixed_curriculum_snapshots_v1/step_500.pt",
     "42939bf6a120e478541eb7ea7209d72158acb075fec30f86931341c8044d084b"),
]


def states():
    value = np.random.get_state()
    return random.getstate(), (value[0], value[1].copy(), *value[2:]), torch.get_rng_state().clone()


def restore(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def compare_rng(left, right):
    return {"Python": left[0] == right[0],
        "NumPy": left[1][0] == right[1][0] and left[1][2:] == right[1][2:]
            and np.array_equal(left[1][1], right[1][1]),
        "Torch_CPU": torch.equal(left[2], right[2])}


def exact(left, right):
    return torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def run():
    torch.set_num_threads(4)
    initial = states()
    records = []
    try:
        for label, path, sha in CASES:
            before = states()
            actor = EvaluationActor.from_checkpoint(path, expected_sha256=sha)
            pure_load = compare_rng(before, states())
            assert all(pure_load.values())
            oracle, _ = PaperAMPTrainer.load(path, device="cpu", restore_rng=False)
            legacy = states()
            assert actor.provenance == json.loads(json.dumps(oracle.provenance, allow_nan=False))
            assert asdict(actor.config) == asdict(oracle.config)
            assert actor.update_count == oracle.update_count
            assert actor.physical_transition_count == oracle.physical_transition_count
            oracle_normalizers = {name: {key: value.clone() for key, value in getattr(oracle, name).state_dict().items()}
                for name in ("self_normalizer", "task_normalizer")}
            batches = []
            for batch_size, seed in ((1, 0), (8, 1), (32, 50)):
                generator = torch.Generator(device="cpu").manual_seed(seed)
                own = torch.randn(batch_size, 357, generator=generator)
                task = torch.randn(batch_size, 1044, generator=generator)
                current = states()
                expected = oracle.act(own, task, deterministic=True)
                actual = actor.act(own, task)
                rng_preserved = compare_rng(current, states())
                values = {key: {"bytewise_equal": exact(actual[key], expected[key]),
                    "maximum_absolute_error": float((actual[key] - expected[key]).abs().max())}
                    for key in actual}
                assert len(values) == 3 and all(value["bytewise_equal"] for value in values.values())
                assert all(rng_preserved.values())
                batches.append({"batch_size": batch_size, "private_input_seed": seed,
                    "outputs": values, "all_global_rng_preserved": rng_preserved})
            assert all(torch.equal(value, oracle_normalizers[name][key])
                for name in oracle_normalizers for key, value in getattr(oracle, name).state_dict().items())
            helper = prepare_legacy_evaluation_rng(actor.config)
            helper_comparison = compare_rng(legacy, states())
            assert all(helper_comparison.values())
            facade_after_helper = EvaluationActor.from_checkpoint(path, expected_sha256=sha)
            pure_after_helper = compare_rng(legacy, states())
            assert all(pure_after_helper.values())
            records.append({"label": label, "checkpoint": path, "bytes": Path(path).stat().st_size,
                "sha256": sha, "updates": actor.update_count, "physical_transitions": actor.physical_transition_count,
                "complete_provenance_equal": True, "configuration_equal": True,
                "load_global_rng_unchanged": pure_load, "helper_versus_actual_legacy_load_RNG": helper_comparison,
                "facade_load_after_explicit_helper_rng_unchanged": pure_after_helper,
                "helper": helper, "batches": batches, "normalizers_unchanged": True})
            del oracle, actor, facade_after_helper
            gc.collect()
        before = states()
        composite = EvaluationActor.from_checkpoints(CASES[0][1], CASES[1][1], alpha=0.,
            base_sha256=CASES[0][2], style_sha256=CASES[1][2])
        baseline = EvaluationActor.from_checkpoint(CASES[0][1], expected_sha256=CASES[0][2])
        generator = torch.Generator().manual_seed(91)
        own, task = torch.randn(4, 357, generator=generator), torch.randn(4, 1044, generator=generator)
        expected, actual = baseline.act(own, task), composite.act(own, task)
        assert all(exact(actual[key], expected[key]) for key in actual)
        assert composite.provenance == baseline.provenance
        assert composite.update_count == baseline.update_count
        composite_rng = compare_rng(before, states())
        assert all(composite_rng.values())
        composite_report = {"alpha": 0., "one_input_batch": 4, "output_tensors": 3,
            "all_bytewise_baseline": True, "all_global_rng_unchanged": composite_rng,
            "baseline_full_provenance_counters_equal": True,
            "description": composite.describe()["experimental_arm_style"]}
    finally:
        restore(initial)
    sources = {}
    for path in ("CrowdSim/humanoid/runtime/evaluation_actor.py", "tests/test_evaluation_actor.py",
                 "CrowdSim/humanoid/runtime/amp_inference.py", "CrowdSim/humanoid/runtime/arm_style.py",
                 "CrowdSim/humanoid/training/trainer.py", "CrowdSim/humanoid/application/settings.py"):
        payload = Path(path).read_bytes()
        sources[path] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    result = {"schema": "evaluation_actor_actual_cpu_verification.v1", "status": "passed",
        "device": "cpu", "threads": 4, "single_actor_input_batches": 6, "single_actor_output_tensors": 18,
        "cases": records, "experimental_alpha0": composite_report, "source_snapshot": sources,
        "saved_checkpoint_RNG_restored": False, "CUDA_RNG_verified": False, "SDK_or_physics_started": False,
        "limitations": ["Fixed mathematical observations, not physical rollouts.",
            "CPU RNG state parity does not establish CUDA planner or physical trajectory parity.",
            "The helper deliberately mutates global RNG once; facade load and act do not.",
            "Checkpoint/source hashes identify these actual local artifacts, not author model equivalence."]}
    target = Path(__file__).with_name("actual_cpu_verification.json")
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "cases": len(records), "input_batches": 6,
        "output_tensors": 18, "actual_legacy_RNG_equal": True, "alpha0_actual_bytewise": True}))


if __name__ == "__main__":
    run()
