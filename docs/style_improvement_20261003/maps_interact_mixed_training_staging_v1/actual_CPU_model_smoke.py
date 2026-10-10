"""Root-authorized bounded real CPU trainstep/restore smoke, no DDPM rollout."""
from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("This smoke must run with CUDA hidden")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from mixed_corpus import MixedORCACorpus, INTERACT_STAGE  # noqa: E402
from train_mixed_diffusion import PHASE, make_training_batch, encoding_map_mask, validate_resume_metadata  # noqa: E402
sys.path.insert(0, str(INTERACT_STAGE))
import torch  # noqa: E402
from paper_interact import PaperDiffusionPlanner  # noqa: E402

torch.set_num_threads(4)
torch.set_num_interop_threads(4)
started = time.monotonic()
INITIAL = ROOT / "output/diffusion_amp/diffusion_training/step_1000.ckpt"
INITIAL_SHA = "4b6f41d0f55146bd12376db1fac0806a4f1b69ae4a12bdb1293e8baada576d7f"
OUT = ROOT / "output/diffusion_amp/maps_interact_CPU_smoke_v1"
if OUT.exists():
    raise FileExistsError("Fresh isolated smoke output required")
OUT.mkdir(parents=True)


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


initial_record = record(INITIAL)  # One opaque initial hash, one load below.
if initial_record["sha256"] != INITIAL_SHA:
    raise ValueError("Root-authorized initial checkpoint changed")
planner = PaperDiffusionPlanner(checkpoint_path=INITIAL, device="cpu")
initial_state = {key: value.clone() for key, value in planner.model.state_dict().items()}
initial_config = planner.config
print(json.dumps({"stage": "initial_loaded_once", "device": str(planner.device), "CUDA_hidden": True}), flush=True)
corpus = MixedORCACorpus(device="cpu", seed=20261001)
interact = corpus.domains["interact"]
episode = next(index for index, row in enumerate(interact.episodes)
               if row["scene_id"] == "Interact_scene_000019")
matching = np.flatnonzero((interact.windows[:, 0] == episode)
                          & (interact.windows[:, 1] == 2) & (interact.windows[:, 2] == 30))
if len(matching) != 1:
    raise ValueError("Previously observed real19-neighbor window not unique")
indices = np.array([0, int(corpus.offsets[1]) + int(matching[0])], dtype=np.int64)
state_before_explicit = copy.deepcopy(corpus.generator.bit_generator.state)
child_states = {name: copy.deepcopy(domain.generator.bit_generator.state) for name, domain in corpus.domains.items()}
sample = corpus.diffusion_batch(2, indices=indices)
assert corpus.generator.bit_generator.state == state_before_explicit
assert all(domain.generator.bit_generator.state == child_states[name] for name, domain in corpus.domains.items())
assert sample["neighbors"].shape == (2, 19, 31, 5)
assert sample["map_condition_mask"].tolist() == [True, False]
assert int(sample["neighbor_mask"][1, :, -1].sum()) == 19
assert torch.all(sample["raster"][1] == .5)
print(json.dumps({"stage": "real_union_batch", "global_indices": indices.tolist(),
                  "domain_names": sample["domain_names"], "neighbor_counts": sample["neighbor_mask"][:, :, -1].sum(1).tolist()}), flush=True)


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(equal_tree(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(equal_tree(a, b) for a, b in zip(left, right, strict=True))
    return left == right


reports = []
for mode in ("trace_unknown_raster", "skip_encoder"):
    if mode == "skip_encoder":
        planner = PaperDiffusionPlanner(config=initial_config, device="cpu")
        planner.model.load_state_dict(initial_state, strict=True)
    planner.trained_steps = 0  # Independent smoke phase, not1000 author/training updates.
    torch.manual_seed(20261003)
    np.random.seed(20261003)
    random.seed(20261003)
    optimizer = torch.optim.Adam(planner.model.parameters(), lr=1e-4)
    seen_map = []
    hook = planner.model.map_encoder.register_forward_pre_hook(
        lambda module, inputs: seen_map.append({"rows": int(inputs[0].shape[0]),
            "all_unknown_rows": (inputs[0] == .5).flatten(1).all(1).tolist()}))
    batch = make_training_batch(planner, sample, mode)
    loss_metrics = planner.train_step(batch, optimizer, drop_conditioning=False)
    hook.remove()
    expected_rows = 2 if mode == "trace_unknown_raster" else 1
    assert seen_map == [{"rows": expected_rows, "all_unknown_rows": [False, True] if expected_rows == 2 else [False]}]
    norms = {name: float(torch.sqrt(sum((parameter.grad.square().sum()
                for parameter in module.parameters() if parameter.grad is not None), torch.tensor(0.))))
             for name, module in (("map_encoder", planner.model.map_encoder), ("neighbor_encoder", planner.model.neighbor_encoder))}
    assert all(np.isfinite(value) and value > 0 for value in norms.values())
    metadata = {"phase": PHASE, "smoke_only": True, "full_trainer_main_executed": False,
                "missing_map_encoding": mode, "corpus_identity": {"train": corpus.resume_identity()},
                "sampler_rng": {"train": copy.deepcopy(corpus.generator.bit_generator.state)}}
    validate_resume_metadata(metadata, (corpus,), mode)
    rejected = []
    for mismatch in ("encoding_mode", "interact_manifest", "sampling"):
        changed = copy.deepcopy(metadata)
        requested_mode = mode
        if mismatch == "encoding_mode":
            requested_mode = "skip_encoder" if mode == "trace_unknown_raster" else "trace_unknown_raster"
        elif mismatch == "interact_manifest":
            changed["corpus_identity"]["train"]["domains"]["interact"]["manifest_sha256"] = "0" * 64
        else:
            changed["corpus_identity"]["train"]["sampling"] = "invented50:50"
        try:
            validate_resume_metadata(changed, (corpus,), requested_mode)
        except ValueError:
            rejected.append(mismatch)
        else:
            raise AssertionError("Mismatched resume identity admitted")
    snapshot = OUT / mode / "step_1.ckpt"
    snapshot.parent.mkdir()
    planner.save_training_checkpoint(snapshot, optimizer=optimizer, dataset_metadata=metadata)
    expected_sampler = corpus.generator.integers(len(corpus.windows), size=8)
    expected_torch = torch.rand(8)
    expected_numpy = np.random.rand(8)
    expected_python = [random.random() for _ in range(8)]
    restored = PaperDiffusionPlanner(config=initial_config, device="cpu")
    restored_optimizer = torch.optim.Adam(restored.model.parameters(), lr=1e-4)
    payload = restored.restore_training_checkpoint(snapshot, optimizer=restored_optimizer)
    validate_resume_metadata(payload["dataset_metadata"], (corpus,), mode)
    corpus.generator.bit_generator.state = payload["dataset_metadata"]["sampler_rng"]["train"]
    assert torch.equal(expected_torch, torch.rand(8))
    assert np.array_equal(expected_numpy, np.random.rand(8))
    assert expected_python == [random.random() for _ in range(8)]
    assert np.array_equal(expected_sampler, corpus.generator.integers(len(corpus.windows), size=8))
    corpus.generator.bit_generator.state = state_before_explicit
    assert restored.trained_steps == planner.trained_steps == 1
    assert equal_tree(planner.model.state_dict(), restored.model.state_dict())
    assert equal_tree(optimizer.state_dict(), restored_optimizer.state_dict())
    reports.append({"mode": mode, "data_map_presence": sample["map_condition_mask"].tolist(),
                    "encoder_availability": encoding_map_mask(sample, mode).tolist(),
                    "actual_map_encoder_inputs": seen_map,
                    "trainstep": loss_metrics, "gradient_norms_after_clip": norms,
                    "save_restore_model_exact": True, "save_restore_Adam_exact": True,
                    "save_restore_torch_numpy_python_union_rng_next_draw_exact": True,
                    "resume_mismatch_rejected": rejected, "smoke_checkpoint": record(snapshot),
                    "unknown_map_encoding_is_real_obstacle_evidence": False})
    del restored, restored_optimizer, payload, optimizer
    print(json.dumps({"stage": "mode_completed", "mode": mode, "loss": loss_metrics["loss"]}), flush=True)

result = {"schema": "local_orca.maps_interact_actual_CPU_model_smoke.v1", "status": "passed",
          "device": "cpu", "CUDA_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
          "torch_cuda_initialized": torch.cuda.is_initialized(), "threads": 4,
          "initial_checkpoint": initial_record, "initial_opaque_hash_calls": 1,
          "initial_checkpoint_load_calls": 1,
          "source_files": [record(HERE / name) for name in ("mixed_corpus.py", "train_mixed_diffusion.py", "actual_CPU_model_smoke.py")],
          "staged_model_source": record(INTERACT_STAGE / "paper_interact.py"),
          "dataset_manifest_records": {name: record(domain.manifest_path) for name, domain in corpus.domains.items()},
          "train_window_counts": dict(zip(corpus.domains, corpus.counts, strict=True)),
          "real_batch_global_indices": indices.tolist(), "real_batch_domain_names": sample["domain_names"],
          "real_batch_scene_ids": sample["scene_ids"], "real_batch_local_indices": sample["local_window_indices"].tolist(),
          "real_batch_shapes": {name: list(value.shape) for name, value in sample.items() if isinstance(value, torch.Tensor)},
          "real_batch_neighbor_counts": sample["neighbor_mask"][:, :, -1].sum(1).tolist(),
          "explicit_indices_preserve_union_and_child_rng": True,
          "architecture_config": asdict(initial_config), "modes": reports,
          "full_trainer_main_executed": False, "full_training_completed": False,
          "DDPM_sampling_or_physics_executed": False, "SDK_imported": False,
          "paper_effect_or_superiority_claim": False,
          "elapsed_seconds": time.monotonic() - started}
assert result["torch_cuda_initialized"] is False
(HERE / "actual_CPU_model_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"status": "passed", "elapsed_seconds": result["elapsed_seconds"],
                  "report": record(HERE / "actual_CPU_model_smoke.json")}), flush=True)

