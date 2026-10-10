"""Read-only CPU actor parity against two actual immutable physical checkpoints.

The full trainer load is an oracle only, not the deployment implementation.
Inputs are fixed finite observation tensors, not newly simulated observations.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

from CrowdSim.humanoid.runtime.amp_inference import ReconstructionAMPInference
from CrowdSim.humanoid.training.trainer import PaperAMPTrainer


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def rng():
    numpy = np.random.get_state()
    return random.getstate(), (numpy[0], numpy[1].copy(), *numpy[2:]), torch.get_rng_state().clone()


def assert_rng(expected):
    actual = rng()
    assert actual[0] == expected[0]
    assert actual[1][0] == expected[1][0] and actual[1][2:] == expected[1][2:]
    assert np.array_equal(actual[1][1], expected[1][1])
    assert torch.equal(actual[2], expected[2])


@contextmanager
def oracle_rng_isolation():
    before = rng()
    try:
        yield
    finally:
        random.setstate(before[0])
        np.random.set_state(before[1])
        torch.set_rng_state(before[2])


def tensor_sha256(value):
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def file_record(path):
    path = Path(path)
    return {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main():
    assert not torch.cuda.is_initialized()
    cases = [
        ("backward_penalty1000", ROOT / "output/diffusion_amp/amp_style_amass2848_backward_penalty_v1/step_1000.pt",
         "62c4edc195d5192d6ce415f5d9825b1f07195827869ceeba7ac33c4ba9f4654b",
         1000, ROOT / "docs/style_improvement_20261002/backward_penalty_training_source.json"),
        ("mixed500", ROOT / "output/diffusion_amp/amp_style_mixed_curriculum_snapshots_v1/step_500.pt",
         "42939bf6a120e478541eb7ea7209d72158acb075fec30f86931341c8044d084b",
         500, ROOT / "docs/style_improvement_20261002/mixed_curriculum_training_source_v1.json"),
    ]
    results = []
    for label, path, expected_hash, updates, source in cases:
        before = rng()
        inference = ReconstructionAMPInference.from_checkpoint(path, device="cpu", expected_sha256=expected_hash)
        assert_rng(before)
        description = inference.describe()
        assert description["update_count"] == updates
        assert description["physical_transition_count"] == updates * 256 * 32
        assert set(dict(inference.named_children())) == {"model", "self_normalizer", "task_normalizer"}
        buffers = {name: value.clone() for name, value in inference.named_buffers()}
        generator = torch.Generator(device="cpu").manual_seed(20261002)
        inputs = [(torch.randn(batch, 357, generator=generator) * scale,
                   torch.randn(batch, 1044, generator=generator) * scale)
                  for batch, scale in ((1, 0.), (8, 1.), (32, 50.))]
        actual = [inference.act(own, task) for own, task in inputs]
        assert_rng(before)
        for name, value in inference.named_buffers():
            assert torch.equal(value, buffers[name]), name
        observations = []
        with oracle_rng_isolation():
            oracle, runner = PaperAMPTrainer.load(path, device="cpu", restore_rng=False)
            assert oracle.update_count == updates and runner is not None
            assert oracle.policy_replay is not None  # Real resume has replay; deployment does not.
            oracle_buffers = {name: value.clone() for name, value in oracle.model.named_buffers()}
            oracle_norms = {name: {key: value.clone() for key, value in getattr(oracle, name).state_dict().items()}
                            for name in ("self_normalizer", "task_normalizer", "amp_normalizer", "value_normalizer")}
            for (own, task), output in zip(inputs, actual):
                expected = oracle.act(own, task, deterministic=True)
                outputs = {}
                for key, value in output.items():
                    reference = expected[key]
                    assert torch.equal(value, reference)
                    # Byte comparison additionally distinguishes signed zero.
                    assert torch.equal(value.contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8))
                    outputs[key] = {"shape": list(value.shape), "dtype": str(value.dtype),
                                    "max_abs_error": float((value - reference).abs().max()),
                                    "bitwise_equal": True, "sha256": tensor_sha256(value)}
                observations.append({"batch": len(own), "self_sha256": tensor_sha256(own),
                                     "task_sha256": tensor_sha256(task), "outputs": outputs})
            for name, value in oracle.model.named_buffers():
                assert torch.equal(value, oracle_buffers[name])
            for name, state in oracle_norms.items():
                for key, value in getattr(oracle, name).state_dict().items():
                    assert torch.equal(value, state[key])
            del oracle, runner
        assert_rng(before)
        dataset = description["dataset_provenance"]
        manifest = Path(dataset["manifest"])
        assert sha256(manifest) == dataset["manifest_sha256"]
        manifest_content = json.loads(manifest.read_text())
        train = manifest_content["splits"]["train"]
        assert train["sha256"] == dataset["source_sha256"]
        bank = Path(train["path"])
        if not bank.is_absolute():
            bank = manifest.parent / bank
        assert bank.stat().st_size == train["bytes"]
        results.append({"name": label, "status": "passed", "inference": description,
                        "checkpoint_identity": {"path": str(path.relative_to(ROOT)),
                            "bytes": path.stat().st_size, "sha256": expected_hash},
                        "physical_training_prelaunch_source": file_record(source),
                        "AMASS_manifest": file_record(manifest),
                        "AMASS_bank": {"path": str(bank.resolve()), "bytes": bank.stat().st_size,
                            "sha256": train["sha256"], "sha256_recomputed_in_this_probe": False},
                        "observations": observations, "buffers_unchanged": True,
                        "python_numpy_torch_rng_unchanged": True,
                        "retained_tensor_bytes": sum(value.numel() * value.element_size() for value in inference.state_dict().values()),
                        "normalizer_buffer_sha256": {name: tensor_sha256(value) for name, value in buffers.items()
                            if name.startswith(("self_normalizer.", "task_normalizer."))}})
        del inference, actual, buffers
        gc.collect()
    assert not torch.cuda.is_initialized()
    assert not any(name.split(".")[0] in {"omni", "isaaclab", "isaacsim"} for name in sys.modules)
    report = {"schema": "navisaaclab.reconstruction_amp_cpu_parity.v1", "status": "passed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "runs": results,
        "input_protocol": "Private CPU generator seed20261002; finite float32 batch1/8/32 at scales0/1/50; tensors are mathematical observation inputs, not fresh simulated poses",
        "parity": "actions/mean_actions/applied_actions byte-identical against actual PaperAMPTrainer.load(restore_rng=False).act(deterministic=True)",
        "unused_outputs": ["critic values/value RMS", "log likelihood", "AMP reward"],
        "oracle_rng": "Full trainer load reseeds RNG; only the comparison harness explicitly restores caller states after the oracle. New inference loader itself leaves all RNG unchanged.",
        "new_inference_sources": [file_record(ROOT / name) for name in (
            "CrowdSim/humanoid/runtime/__init__.py", "CrowdSim/humanoid/runtime/checkpoint_contract.py",
            "CrowdSim/humanoid/runtime/amp_inference.py", "tests/test_reconstruction_amp_inference.py")],
        "shared_contract_sources": [file_record(ROOT / name) for name in (
            "CrowdSim/humanoid/training/trainer.py", "CrowdSim/humanoid/application/settings.py",
            "CrowdSim/humanoid/amp/observations.py")],
        "gpu_used": False, "kit_started": False, "physical_validation_performed": False,
        "runtime_deployment_changed": False, "sealed34_outputs_read": False,
        "limitations": ["No fresh physics/controller/real-USD evaluation", "Only two selected immutable checkpoints checked",
                        "AMASS train bank SHA is declared and matched to manifest; full bank hash not recomputed",
                        "Optimizer/replay/RNG/physical resume payload integrity is outside the inference loader gate"]}
    (OUT / "actual_checkpoint_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "checkpoints": len(results),
                      "observation_batches": 6, "parity_max_abs_error": 0., "bitwise_equal": True}))


if __name__ == "__main__":
    main()
