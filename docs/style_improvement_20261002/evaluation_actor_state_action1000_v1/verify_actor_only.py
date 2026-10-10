"""Actual 1950D/2019D bundles: before/after actor bytes and legacy CPU RNG."""
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import random
import sys
from unittest.mock import patch

import numpy as np
import torch

from verify import OUT, ROOT, SOURCES, compare_rng, file_record, forbidden_training_constructors, rng, sha256, tensor_record
from CrowdSim.humanoid.runtime.evaluation_actor import EvaluationActor, prepare_legacy_evaluation_rng
from CrowdSim.humanoid.training import trainer


def main():
    torch.set_num_threads(4)
    initial_rng = rng()
    assert not torch.cuda.is_initialized()
    source_before = {name: file_record(ROOT / name) for name in SOURCES}
    baseline = json.loads((OUT / "before_actor_outputs.json").read_text())
    baseline_path = ROOT / baseline["baseline_npz"]["path"]
    assert sha256(baseline_path) == baseline["baseline_npz"]["sha256"]
    results = []
    with np.load(baseline_path, allow_pickle=False) as data:
        for case in baseline["cases"]:
            path = ROOT / case["checkpoint"]["path"]
            sha = case["checkpoint"]["sha256"]
            before = rng()
            with forbidden_training_constructors():
                actor = EvaluationActor.from_checkpoint(path, device="cpu", expected_sha256=sha)
            assert all(compare_rng(before, rng()).values())
            model = actor._inference.model
            assert set(dict(model.named_children())) == {"actor", "task_encoder"}
            assert set(dict(actor._inference.named_children())) == {"model", "self_normalizer", "task_normalizer"}
            assert all(name.startswith(("actor.", "task_encoder.")) for name in model.state_dict())
            assert not any(hasattr(model, name) for name in ("critic", "discriminator", "log_std", "optimizer", "policy_replay"))
            stats_before = {name: tensor_record(value) for name, value in actor._inference.state_dict().items()}
            actual = []

            def forbidden_forward(*args, **kwargs):
                raise AssertionError("Actor-only inference executed critic/Normal/full policy")

            with forbidden_training_constructors(), patch.object(trainer.PaperAMPNetwork, "policy", forbidden_forward), patch.object(torch.distributions, "Normal", forbidden_forward):
                for index in range(len(baseline["input_batches"])):
                    inputs = (torch.from_numpy(data[f"self_{index}"].copy()), torch.from_numpy(data[f"task_{index}"].copy()))
                    actual.append(actor.act(*inputs, deterministic=True))
            assert all(compare_rng(before, rng()).values())
            calls = []
            seed_all = torch.cuda.manual_seed_all

            def observe_seed(seed):
                calls.append(seed)
                return seed_all(seed)

            with patch.object(torch.cuda, "manual_seed_all", observe_seed):
                oracle, runner = trainer.PaperAMPTrainer.load(path, device="cpu", restore_rng=False)
            legacy_rng = rng()
            legacy_calls = calls.copy()
            assert asdict(actor.config) == asdict(oracle.config) and actor.provenance == oracle.provenance
            assert oracle.model.discriminator[0].in_features == case["D_input"]
            records = []
            for index, output in enumerate(actual):
                inputs = (torch.from_numpy(data[f"self_{index}"].copy()), torch.from_numpy(data[f"task_{index}"].copy()))
                expected = oracle.act(*inputs, deterministic=True)
                outputs = {}
                for key, value in output.items():
                    old = torch.from_numpy(data[f"{case['label']}_{key}_{index}"].copy())
                    assert torch.equal(value.contiguous().view(torch.uint8), old.contiguous().view(torch.uint8))
                    assert torch.equal(value.contiguous().view(torch.uint8), expected[key].contiguous().view(torch.uint8))
                    outputs[key] = {**tensor_record(value), "old_facade_bytewise_equal": True,
                                    "actual_Trainer_load_False_bytewise_equal": True, "maximum_absolute_error": 0.}
                records.append({"input": baseline["input_batches"][index], "outputs": outputs})
            assert all(compare_rng(legacy_rng, rng()).values())
            assert stats_before == {name: tensor_record(value) for name, value in actor._inference.state_dict().items()}
            calls.clear()
            with forbidden_training_constructors(), patch.object(torch.cuda, "manual_seed_all", observe_seed):
                helper = prepare_legacy_evaluation_rng(actor.config)
            comparison = compare_rng(legacy_rng, rng())
            assert all(comparison.values()) and calls == legacy_calls == [actor.config.seed]
            saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            assert not torch.equal(legacy_rng[2], saved["rng"]["torch_cpu"])
            retained_bytes = sum(value.numel() * value.element_size() for value in actor._inference.state_dict().values())
            old_model_bytes = sum(value.numel() * value.element_size() for value in saved["model"].values())
            assert retained_bytes < old_model_bytes
            results.append({"label": case["label"], "checkpoint": file_record(path), "D_validated_input": case["D_input"],
                            "checkpoint_model_fully_validated": True, "no_loaded_critic_D_log_std_or_training_objects": True,
                            "no_critic_Normal_forward": True, "retained_tensor_bytes": retained_bytes,
                            "original_full_model_tensor_bytes": old_model_bytes, "output_batches": records,
                            "configuration_and_provenance_equal": True, "RMS_and_model_tensors_unchanged": True,
                            "pure_load_act_RNG_preserved": True, "legacy_CPU_rng_comparison": comparison,
                            "legacy_rng_helper": helper, "legacy_CUDA_seed_events": legacy_calls,
                            "helper_CUDA_seed_events": calls.copy(), "CUDA_state_equivalence_verified": False})
            del oracle, runner, actor, model, saved, actual, stats_before
            gc.collect()
    assert not torch.cuda.is_initialized()
    assert not any(name.split(".")[0] in {"isaaclab", "isaacsim", "omni"} for name in sys.modules)
    assert source_before == {name: file_record(ROOT / name) for name in SOURCES}
    random.setstate(initial_rng[0])
    np.random.set_state(initial_rng[1])
    torch.set_rng_state(initial_rng[2])
    report = {"schema": "evaluation_actor_actor_only_actual_cpu.v1", "status": "passed",
              "generated_utc": datetime.now(timezone.utc).isoformat(), "CPU_only": True,
              "CUDA_initialized": False, "Kit_started": False, "sealed34_read": False,
              "source_files": source_before, "auditor": file_record(Path(__file__)),
              "baseline": file_record(OUT / "before_actor_outputs.json"), "baseline_npz": file_record(baseline_path),
              "physical_capture_input": baseline["physical_source"], "cases": results,
              "input_batches": len(baseline["input_batches"]), "output_tensors": 36,
              "limits": ["Real captured physical states/commands are CPU re-encoded, not original stored actor observations.",
                  "Three additional mathematical batches per checkpoint are synthetic and do not establish GPU/physics quality.",
                  "Temporary full-network validation still checks all critic/D/log_std tensors before discarding these modules.",
                  "No public PACER loader, Trainer, optimizer or replay is constructed by the deployed facade or RNG helper.",
                  "CPU RNG states and lazy CUDA seed-call equality are verified; initialized CUDA RNG or planner/physics parity remains pending."]}
    (OUT / "actor_only_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "passed", "actual_checkpoints": 2, "D_inputs": [2019, 1950],
                      "input_batches_per_checkpoint": 6, "output_tensors": 36, "old_and_Trainer_bytewise_equal": True,
                      "legacy_CPU_rng_equal": True, "CUDA_equivalence_verified": False}))


if __name__ == "__main__":
    main()
