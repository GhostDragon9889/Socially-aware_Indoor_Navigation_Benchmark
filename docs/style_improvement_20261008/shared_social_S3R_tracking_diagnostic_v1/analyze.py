"""Recompute every S3-R validation frame with the frozen root/body kernel."""
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.analysis import tracking_error_decomposition as kernel  # noqa: E402
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record  # noqa: E402
from scripts import diagnose_shared_social_tracking_errors as original  # noqa: E402
from scripts import evaluate_shared_social_recovery as evaluation  # noqa: E402

HERE = Path(__file__).resolve().parent
RUN = ROOT / "output/evaluation/shared_social_S3R_budget12500_v2/tracking_style"
sources = [record(path) for path in (Path(__file__), Path(kernel.__file__), Path(original.__file__), original.FROZEN)]
kernel.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and "torch" not in sys.modules, "Require a CPU-only process")
kernel.require(sources[1]["sha256"] == "6e9f067f972f8d2fc6d146c271c589a1b9551e4b8fc795864e00ce5e88432dd9"
               and sources[2]["sha256"] == "3c754c761e5162fd5f25397e7f67a5787a24627d432e9a57943aa24c61a432a6"
               and sources[3]["sha256"] == original.FROZEN_SHA, "Frozen diagnostic or world-space scoring kernel changed")
pipeline_identity = record(RUN / "execution.json")
pipeline = json.loads((RUN / "execution.json").read_text())
kernel.require(pipeline["status"] == "passed_execution"
               and [row["name"] for row in pipeline["steps"]] == ["lifecycle_smoke", "full22", "style"]
               and all(type(row["actual_exit_code"]) is int and row["actual_exit_code"] == 0
                       and row["status"] == "passed" for row in pipeline["steps"]),
               "Require actual successful exits for the entire tracking/style pipeline")
result_identity = pipeline["steps"][1]["result"]
kernel.require(record(result_identity["path"]) == result_identity, "Actual completed tracking result changed")
report = json.loads(Path(result_identity["path"]).read_text())
module, _ = evaluation.prepare("tracking")
_, _, expected = module.admit_training(Path(report["training_execution_admission"]["receipt"]["path"]))
kernel.require(report["status"] == "passed" and report["cleanup_succeeded"] is True
               and report["cleanup_errors"] == [] and report["training_execution_admission"] == expected,
               "Actual completed tracking admission/cleanup mismatch")
rows, sequences, inputs = [], [], [pipeline_identity, result_identity]
for sequence in report["sequences"]:
    identity = record(sequence["capture"])
    kernel.require(identity["sha256"] == sequence["capture_sha256"], "Actual completed capture changed")
    inputs.append(identity)
    with np.load(identity["path"], allow_pickle=False) as arrays:
        values = kernel.verify_capture(arrays, sequence)
        metric = arrays["metric_frame"]
        observed = {key: value[metric] for key, value in values.items() if key in original.SCALAR_FIELDS}
        observed.update(motion_id=np.full(int(metric.sum()), sequence["motion_id"], np.int64),
                        capture_frame=np.flatnonzero(metric), fall_mask=arrays["fall_mask"][metric])
    rows.append(observed)
    sequences.append({"motion_id": sequence["motion_id"], "original_tracking_success": sequence["tracking_success"],
                      "original_fall_frames": len(sequence["fall_capture_frames"]),
                      "diagnostic": original.summarize(observed)})
complete = {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}
kernel.require(len(rows) == 22 and len(complete["motion_id"]) == 11982, "Require every validation22/11982 frame")
for key, metric in (("frame_weighted_mean_body_position_error_m", "mean_body_position_error_m"),
                    ("frame_weighted_mean_body_rotation_error_rad", "mean_body_rotation_error_rad")):
    kernel.require(abs(float(complete[metric].mean()) - report["aggregate"][key]) < 1e-12,
                   "Recomputed unaligned world-space aggregate differs")
kernel.require(all(record(row["path"]) == row for row in sources + inputs), "Diagnostic input/source changed")
kernel.require("torch" not in sys.modules, "CPU diagnostic imported Torch")
archive = HERE / "all_metric_frames.npz"
with archive.open("xb") as stream:
    np.savez(stream, **complete)
archive.chmod(0o444)
result = {"schema": "shared_social.recovered_complete_tracking_root_body_CPU_diagnostic.v1",
          "status": "completed_CPU_observation", "sources": sources, "inputs": inputs,
          "original_world_tracking_aggregate": report["aggregate"], "all_metric_frames": 11982,
          "fall_frames_retained": int(complete["fall_mask"].sum()), "full_population": original.summarize(complete),
          "sequences": sequences, "all_frame_archive": record(archive), "Torch_imported": False, "SDK_started": False,
          "diagnostic_alignment_is_paper_score": False, "causal_training_failure_proven": False,
          "paper_effect_achieved": False, "default_model_promoted": False}
path = HERE / "result.json"
with path.open("x") as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
    stream.write("\n")
path.chmod(0o444)
print(json.dumps({"status": result["status"], "result": record(path),
                  "means": {key: row["mean"] for key, row in result["full_population"].items()},
                  "Torch_imported": False, "SDK_started": False}))
