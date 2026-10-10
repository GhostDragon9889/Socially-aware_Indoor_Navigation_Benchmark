"""One bounded metadata-only primary-source byte binding."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TRACE = ROOT / "third_party/trace"
COMMIT = "594b2af0824c5853e1a1884e167cd5ed097dd81c"


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


primary = []
for relative in ("configs/train/diffuser_orca_mlp_hist_grid_map_drop_10.json",
                 "tbsim/algos/algos.py", "tbsim/models/trace_helpers.py", "tbsim/models/trace.py",
                 "scripts/train.py", "tbsim/evaluation/policy_composers.py"):
    path = TRACE / relative
    blob = subprocess.check_output(["git", "-C", str(TRACE), "show", f"{COMMIT}:{relative}"])
    if path.read_bytes() != blob:
        raise ValueError(f"Primary source differs from fixed commit: {relative}")
    primary.append({**record(path), "official_commit_blob_bytes_exact": True})
result = {"schema": "local_trace.training_budget_primary_source_review.v1", "status": "passed",
          "primary_commit": COMMIT, "primary_sources": primary,
          "current_local_sources": [record(ROOT / relative) for relative in (
              "CrowdSim/humanoid/diffusion/paper.py", "scripts/train_orca_diffusion.py",
              "CrowdSim/humanoid/diffusion/trace_helpers.py",
              "docs/style_improvement_20261003/orca_interact_usable_staging_v1/paper_interact.py",
              "docs/style_improvement_20261003/maps_interact_mixed_training_staging_v1/train_mixed_diffusion.py")],
          "paper": record(HERE.parent / "paper_fidelity_research_v1/trace_and_pace.pdf"),
          "README": record(HERE / "README.md"),
          "NN_weights_SDK_GPU_or_future_capture_read": False,
          "paper_budget": {"iterations": 40000, "learning_rate": .0002, "paper_batch400_explicit": False},
          "official_JSON_budget": {"updates": 100000, "batch": 400, "window_exposure": 40000000},
          "candidate_budget_not_author_explicit": {"optimizer_updates": 40000, "effective_batch": 400,
              "window_exposure": 16000000, "microbatch": 8, "accumulation": 50,
              "micro_forward_backward_count": 2000000, "GPU_authorized": False},
          "EMA": {"decay": .995, "every": 1, "start_step": 2000, "pre_forward_update": True,
              "whole_state_dict_including_buffers": True, "current_five_state_trainer_has_EMA": False}}
(HERE / "source_records.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"status": "passed", "source_records": record(HERE / "source_records.json"),
                  "primary_byte_checks": len(primary)}))

