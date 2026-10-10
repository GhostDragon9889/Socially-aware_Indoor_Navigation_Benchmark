"""One root-authorized CPU formal-state proof, no main/GPU/DDPM rollout."""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
    raise RuntimeError("CUDA must be hidden for this proof")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
INTERACT_STAGE = HERE.parent / "orca_interact_usable_staging_v1"
OLD_STAGE = HERE.parent / "maps_interact_mixed_training_staging_v1"
for path in (ROOT, HERE, INTERACT_STAGE):
    sys.path.insert(0, str(path))
sys.path.append(str(OLD_STAGE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from mixed_corpus import MixedORCACorpus  # noqa: E402
from paper_interact import PaperDiffusionPlanner, PaperDiffusionConfig  # noqa: E402
from train_mixed_diffusion import make_training_batch  # noqa: E402
from training_state import (WholeStateEMA, EMA_RECIPE, recipe_identity, optimizer_update,  # noqa: E402
                            checkpoint_payload, restore_training, validate_resume, rng_state)
from CrowdSim.humanoid.diffusion.paper import PaperDiffusionPlanner as ProductionDeployment  # noqa: E402
from protomotions.utils.checkpoint import atomic_torch_save  # noqa: E402

torch.set_num_threads(4)
torch.set_num_interop_threads(4)
started = time.monotonic()
torch.manual_seed(20261003)
np.random.seed(20261003)
random.seed(20261003)
OUTPUT = ROOT / "output/diffusion_amp/official_ORCA_formal_CPU_smoke_v2"
if OUTPUT.exists():
    raise FileExistsError("Fresh test-only output required")
OUTPUT.mkdir(parents=True)
corpora = {name: MixedORCACorpus(split=name, device="cpu", seed=20261001 + index)
           for index, name in enumerate(("train", "validation", "test"))}
corpus_identity = {name: corpus.resume_identity() for name, corpus in corpora.items()}
identity = recipe_identity(microbatch=2, accumulation=2, learning_rate=2e-4, max_grad_norm=None,
                           missing_map_encoding="trace_unknown_raster")
planner = PaperDiffusionPlanner(config=PaperDiffusionConfig(), device="cpu")
ema = WholeStateEMA(planner.model)
optimizer = torch.optim.Adam(planner.model.parameters(), lr=identity["learning_rate"])


def make_batch(sample):
    return make_training_batch(planner, sample, identity["missing_map_encoding"])


def tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(tree_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


first = optimizer_update(planner, ema, optimizer, corpora["train"], make_batch=make_batch, identity=identity, update_index=0)
artifact = checkpoint_payload(planner, ema, optimizer, corpora, identity=identity, corpus_identity=corpus_identity,
    initialization={"kind": "random_default_five_state_NN_CPU_smoke_only", "path": None, "sha256": None})
artifact["smoke_only"] = True
snapshot = OUTPUT / "test_only_step_1.ckpt"
atomic_torch_save(artifact, snapshot)
assert snapshot.stat().st_nlink == 1
validate_resume(artifact, config=planner.config, identity=identity, corpus_identity=corpus_identity)

# Clone reference only for comparisons; the actual interrupted state is serialized.
reference = optimizer_update(planner, ema, optimizer, corpora["train"], make_batch=make_batch, identity=identity, update_index=1)
expected_model = {key: value.clone() for key, value in planner.model.state_dict().items()}
expected_ema = {key: value.clone() for key, value in ema.model.state_dict().items()}
expected_optimizer = copy.deepcopy(optimizer.state_dict())
expected_rng = rng_state(planner.device)
expected_union_rng = {name: copy.deepcopy(corpus.generator.bit_generator.state) for name, corpus in corpora.items()}

loaded = torch.load(snapshot, weights_only=True, map_location="cpu")
restored = PaperDiffusionPlanner(config=planner.config, device="cpu")
restored_ema = WholeStateEMA(restored.model)
restored_optimizer = torch.optim.Adam(restored.model.parameters(), lr=identity["learning_rate"])
restore_training(loaded, restored, restored_ema, restored_optimizer, corpora,
                 identity=identity, corpus_identity=corpus_identity)
assert tree_equal(restored.model.state_dict(), loaded["training_raw_model"])
assert tree_equal(restored_ema.model.state_dict(), loaded["model"])
assert any(not torch.equal(loaded["model"][key], loaded["training_raw_model"][key])
           for key in loaded["model"])
replayed = optimizer_update(restored, restored_ema, restored_optimizer, corpora["train"],
    make_batch=lambda sample: make_training_batch(restored, sample, identity["missing_map_encoding"]),
    identity=identity, update_index=1)
assert tree_equal(expected_model, restored.model.state_dict())
assert tree_equal(expected_ema, restored_ema.model.state_dict())
assert tree_equal(expected_optimizer, restored_optimizer.state_dict())
assert tree_equal(expected_rng, rng_state(restored.device))
assert expected_union_rng == {name: corpus.generator.bit_generator.state for name, corpus in corpora.items()}
assert reference == replayed
assert restored.trained_steps == 2 and restored_ema.before_update_events == 2
assert replayed["microsteps"] == 4 and replayed["training_windows"] == 8

# Existing production deployment must loadEMA, not silently substitute raw.
deployed = ProductionDeployment(checkpoint_path=snapshot, device="cpu")
assert tree_equal(deployed.model.state_dict(), loaded["model"])
assert any(not torch.equal(deployed.model.state_dict()[key], loaded["training_raw_model"][key])
           for key in loaded["model"])

# Tiny actual nn.Module boundary probe; not1999/2000 NN training updates.
class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2., 4.]))
        self.register_buffer("float_buffer", torch.tensor([3., 5.]))
        self.register_buffer("integer_buffer", torch.tensor(7, dtype=torch.long))


tiny = Tiny()
tiny_ema = WholeStateEMA(tiny)
tiny_ema.before_update_events = 1999  # Explicitly constructed boundary state.
with torch.no_grad():
    tiny.weight.fill_(17.)
    tiny.float_buffer.fill_(29.)
    tiny.integer_buffer.fill_(1999)
tiny_ema.before_update(tiny, 1999)
assert tree_equal(tiny.state_dict(), tiny_ema.model.state_dict())
before_boundary = {key: value.clone() for key, value in tiny_ema.model.state_dict().items()}
with torch.no_grad():
    tiny.weight.fill_(19.)
    tiny.float_buffer.fill_(31.)
    tiny.integer_buffer.fill_(2000)
expected_boundary = {}
for key, value in tiny.state_dict().items():
    target = before_boundary[key].clone()
    target.copy_(EMA_RECIPE["decay"] * target + (1. - EMA_RECIPE["decay"]) * value)
    expected_boundary[key] = target
tiny_ema.before_update(tiny, 2000)
assert tree_equal(expected_boundary, tiny_ema.model.state_dict())
assert tiny_ema.before_update_events == 2001

negative_cases = []
for label in ("encoding_mode", "accumulation", "EMA_recipe", "incomplete_microstep", "EMA_counter", "missing_raw_state"):
    changed = copy.deepcopy(loaded)
    if label == "encoding_mode":
        changed["training_recipe"]["missing_map_encoding"] = "skip_encoder"
    elif label == "accumulation":
        changed["training_recipe"]["accumulation"] = 50
    elif label == "EMA_recipe":
        changed["training_recipe"]["EMA"]["start_optimizer_update"] = 0
    elif label == "incomplete_microstep":
        changed["microsteps_in_current_update"] = 1
    elif label == "EMA_counter":
        changed["EMA"]["before_update_events"] = 0
    else:
        del changed["training_raw_model"]
    try:
        validate_resume(changed, config=planner.config, identity=identity, corpus_identity=corpus_identity)
    except ValueError:
        negative_cases.append(label)
    else:
        raise AssertionError("Required mismatch was admitted")
    del changed


def record(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


assert not torch.cuda.is_initialized()
assert not any(name.split(".")[0] in {"isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules)
result = {"schema": "local_orca.formal_accum_EMA_actual_CPU_proof.v2", "status": "passed",
          "device": "cpu", "CUDA_hidden": True, "threads": 4, "random_default_NN_initialization": True,
          "old_initial_weight_files_read": False,
          "actual_microbatch": 2, "actual_accumulation": 2, "actual_effective_batch": 4,
          "formal_effective400_or_40000_training_executed": False,
          "first_optimizer_update": first, "reference_next_optimizer_update": reference,
          "restored_next_optimizer_update": replayed, "raw_model_exact": True, "EMA_model_exact": True,
          "Adam_state_exact": True, "Torch_NumPy_Python_and_all_union_RNG_exact": True,
          "production_deployment_model_key_loads_same_EMA": True,
          "EMA_and_training_raw_are_distinct": True,
          "EMA_boundary_tiny_only": {"constructed_index": 1999, "warmup_whole_state_copy_exact": True,
              "next_index": 2000, "whole_state_blend_exact": True, "integer_buffer_copy_cast_exact": True,
              "neural_training_updates_1999_or_2000_executed": False},
          "rejected_negative_cases": negative_cases, "test_only_checkpoint": record(snapshot),
          "test_only_checkpoint_nlink": snapshot.stat().st_nlink,
          "sources": [record(HERE / name) for name in ("training_state.py", "train_formal_diffusion.py",
                      "official_maps_corpus.py", "mixed_corpus.py", "check_actual_formal_CPU.py")],
          "production_deployment_source": record(ROOT / "CrowdSim/humanoid/diffusion/paper.py"),
          "staged_NN_source": record(INTERACT_STAGE / "paper_interact.py"),
          "elapsed_seconds": time.monotonic() - started,
          "full_trainer_main_executed": False, "SDK_GPU_DDPM_physics_or_sealed34_read": False,
          "paper_effect_proven": False}
(HERE / "actual_formal_CPU_proof.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps({"status": "passed", "report": record(HERE / "actual_formal_CPU_proof.json"),
                  "elapsed_seconds": result["elapsed_seconds"]}))

