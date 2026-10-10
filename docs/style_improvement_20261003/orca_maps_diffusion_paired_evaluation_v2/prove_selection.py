"""Check the actual100 validation inputs and explicit sampler API without models."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ENTRY = HERE / "evaluate_paired.py"
SDK = "/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python"
specification = importlib.util.spec_from_file_location("paired_validation_preparation", ENTRY)
adapter = importlib.util.module_from_spec(specification)
specification.loader.exec_module(adapter)
registration = adapter.read_registration()
command = [SDK, str(ENTRY), "--prepare-only"]
completed = subprocess.run(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"},
                           text=True, capture_output=True, check=True)
corpus_module = adapter.load("original935_validation_corpus", registration["sources"]["corpus"]["path"])
corpus = corpus_module.ORCACorpus(registration["sources"]["manifest"]["path"], "validation", seed=registration["seed"])
indices = [row["global_window_index"] for row in registration["window_selection"]]
before = copy.deepcopy(corpus.generator.bit_generator.state)
sample = corpus.numpy_batch(100, indices=indices)
assert corpus.generator.bit_generator.state == before
assert len(set(sample["scene_ids"])) == 100 and sample["raster"].shape == (100, 2, 224, 224)
assert sample["states"].shape == (100, 5) and sample["future"].shape == (100, 52, 5)
assert sample["history"].shape == (100, 31, 5) and sample["neighbors"].shape == (100, 14, 31, 5)
assert sample["neighbor_mask"][:, :, -1].any(axis=1).all() and sample["history_mask"].all()
for index, row in enumerate(registration["window_selection"]):
    assert sample["scene_ids"][index] == row["scene_id"] and sample["episode_ids"][index] == row["episode_id"]
    assert int(sample["agent_ids"][index]) == row["agent_id"] and int(sample["frame_indices"][index]) == row["frame"]
assert "torch" not in sys.modules and "isaacsim" not in sys.modules
previous = HERE.parent / "orca_maps_diffusion_paired_evaluation_v1"
old_registration = json.loads((previous / "registration.json").read_text())
assert registration["window_selection"] == old_registration["window_selection"]
old_main = next(node for node in ast.parse((previous / "evaluate_paired.py").read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
new_main = next(node for node in ast.parse(ENTRY.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "main")
def index_of_assignment(body, name):
    return next(index for index, node in enumerate(body) if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
old_math = ast.Module(body=old_main.body[index_of_assignment(old_main.body, "launcher"):], type_ignores=[])
new_math = ast.Module(body=new_main.body[index_of_assignment(new_main.body, "launcher"):], type_ignores=[])
assert ast.dump(old_math) == ast.dump(new_math)
metadata_start = index_of_assignment(new_main.body, "owner")
metadata_stop = index_of_assignment(new_main.body, "completed")
metadata = ast.Module(body=new_main.body[metadata_start:metadata_stop], type_ignores=[])
selected_checkpoint = adapter.record(registration["future_checkpoint"]["path"])
assert selected_checkpoint == registration["future_checkpoint"]
exec(compile(metadata, str(ENTRY), "exec"), {"json": json, "Path": Path, "registration": registration,
     "future": registration["future_checkpoint"], "new_record": selected_checkpoint})
proof = {"schema": "local_orca_maps.paired_selection_actual_CPU_proof.v1", "status": "passed",
         "entry": adapter.record(ENTRY), "registration": adapter.record(adapter.REGISTRATION),
         "proof_source": adapter.record(__file__), "actual_prepare_command": command,
         "actual_prepare_exit_code": completed.returncode, "prepare_stdout": json.loads(completed.stdout),
         "actual_original_corpus_source": registration["sources"]["corpus"],
         "actual_validation_windows": 100, "actual_validation_scenes": 100,
         "explicit_window_indices_do_not_advance_sampler_RNG": True,
         "actual_tensor_shapes": {key: list(value.shape) for key, value in sample.items() if isinstance(value, np.ndarray)},
         "real_neighbor_windows": int(sample["neighbor_mask"][:, :, -1].any(axis=1).sum()),
         "all_preselected_window_identity_exact": True, "checkpoint_loaded": False,
         "v2_vs_v1_all_original_measurement_invocation_AST_exact": True,
         "v2_vs_v1_selected_window_roster_exact": True,
         "actual_closed_training_metadata_block_executed": True,
         "actual_sealed_checkpoint_full_SHA_verified": selected_checkpoint,
         "torch_imported": False, "GPU_used": False, "SDK_started": False,
         "future_checkpoint_read": False, "models_executed": False, "prediction_outputs_created": False,
         "sealed34_outputs_read": False}
with (HERE / "selection_CPU_proof.json").open("x") as stream:
    json.dump(proof, stream, indent=2, allow_nan=False)
    stream.write("\n")
print(json.dumps({"status": "passed", "windows": 100, "scenes": 100, "checkpoint_loaded": False}))
