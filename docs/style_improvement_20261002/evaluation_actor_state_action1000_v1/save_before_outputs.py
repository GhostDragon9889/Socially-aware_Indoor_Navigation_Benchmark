"""Archive old-facade CPU outputs before removing unused networks."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from verify import CHECKPOINT, EXPECTED_SHA, OUT, ROOT, file_record, input_batches
from CrowdSim.humanoid.runtime.evaluation_actor import EvaluationActor


def main():
    torch.set_num_threads(4)
    cases = [("state_action1000", CHECKPOINT, EXPECTED_SHA),
             ("state_only1000", ROOT / "output/diffusion_amp/amp_style_amass2848_backward_penalty_v1/step_1000.pt",
              "62c4edc195d5192d6ce415f5d9825b1f07195827869ceeba7ac33c4ba9f4654b")]
    inputs, rows, physical_source = input_batches()
    arrays = {}
    case_records = []
    for i, (own, task) in enumerate(inputs):
        arrays[f"self_{i}"] = own.numpy()
        arrays[f"task_{i}"] = task.numpy()
    for label, path, sha in cases:
        actor = EvaluationActor.from_checkpoint(path, expected_sha256=sha)
        for i, (own, task) in enumerate(inputs):
            for key, value in actor.act(own, task).items():
                arrays[f"{label}_{key}_{i}"] = value.numpy()
        case_records.append({"label": label, "checkpoint": file_record(path),
                             "D_input": actor._inference.model.discriminator[0].in_features})
        del actor
    target = OUT / "before_actor_outputs.npz"
    assert not target.exists()
    np.savez_compressed(target, **arrays)
    report = {"schema": "evaluation_actor_before_outputs.v1", "CPU_only": True,
              "GPU_or_Kit_used": False, "cases": case_records, "input_batches": rows,
              "physical_source": physical_source, "baseline_npz": file_record(target),
              "old_runtime_source": file_record(ROOT / "CrowdSim/humanoid/runtime/amp_inference.py"),
              "executor": file_record(Path(__file__)),
              "arrays": {key: {"shape": list(value.shape), "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
                         for key, value in arrays.items()}}
    (OUT / "before_actor_outputs.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "saved", "cases": len(cases), "baseline_npz": report["baseline_npz"]}))


if __name__ == "__main__":
    main()
