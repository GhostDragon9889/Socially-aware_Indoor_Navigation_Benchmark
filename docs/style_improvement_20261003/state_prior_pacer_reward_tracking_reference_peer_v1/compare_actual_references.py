"""Admit closed actual22 capture, then reuse frozen byte-pairing mathematics."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
LEGACY = ROOT / "docs/state_prior_tracking_reference_peer_v1"
ACTUAL = ROOT / "docs/style_improvement_20261003/state_prior_pacer_reward_evaluation_actual_runs_v1"
METRICS = ROOT / "docs/style_improvement_20261003/pacer_reward_metrics_v1"
NEW_RESULT = ROOT / "output/diffusion_amp/pacer_reward_tracking2000_v1/result.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).absolute()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def read(path):
    return json.loads(Path(path).read_text())


def main():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only review required")
    math_path = LEGACY / "compare_actual_references.py"
    require(record(math_path)["sha256"] == "734513ca3065adfc116ec1cb309b1f1639e79b05721ef6ad83f9388934690691", "Frozen byte-pairing math changed")
    specification = importlib.util.spec_from_file_location("unchanged_actual_reference_pairing_math", math_path)
    math = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(math)
    seal_path = ACTUAL / "tracking_root_seal.json"
    require(record(seal_path)["sha256"] == "0471a379a1518b0f8560fa75138075c21111d714a8e77d4f59d83c5a9764498f", "Root actual seal changed")
    seal = read(seal_path)
    require(seal["status"] == "sealed_after_actual_exit" and seal["actual_root_session_exit0"] is True, "Actual closed root seal required")
    for row in seal["artifacts"]:
        require(record(row["path"]) == {key: row[key] for key in ("path", "bytes", "sha256")}
                and row["readonly"] is True and not Path(row["path"]).stat().st_mode & 0o222, "Sealed actual capture/owner/import artifact differs")
    owner = read(ACTUAL / "tracking_actual_launch.json")
    require(owner["status"] == "closed" and type(owner["exit_code"]) is int and owner["exit_code"] == 0
            and owner["child_exit_observed"] is True and owner["driver_completed"] is True, "Actual GPU process closure required")
    receipt_path = ROOT / "docs/style_improvement_20261003/state_prior_pacer_reward_CPU_metrics_actual_runs_v1/tracking_decomposition_actual_launch.json"
    receipt = read(receipt_path)
    require(receipt["schema"] == "pacer_reward.actual_CPU_metric_launch.v1" and receipt["status"] == "closed"
            and receipt["metric"] == "tracking_decomposition" and type(receipt["exit_code"]) is int and receipt["exit_code"] == 0
            and receipt["child_exit_observed"] is True and receipt["driver_completed"] is True
            and not receipt_path.stat().st_mode & 0o222, "Actual closed CPU source admission required")
    for row in receipt["artifacts"].values():
        require(record(row["path"]) == row and not Path(row["path"]).stat().st_mode & 0o222, "Actual metric artifact differs")
    admission_path = METRICS / "tracking_decomposition/actual_source_admission.json"
    admission = read(admission_path)
    require(admission["status"] == "passed" and admission["metric"] == "tracking_decomposition"
            and admission["runner"]["sha256"] == "a0469add93bcf698fb74bb64ab675728c79305209c16e923073d35a09cb8a463"
            and admission["catalog"]["sha256"] == "eadad9fe03169c4d64a4938c335af22ae0831a55ee77494fd19c0e0ce1f5f37d", "Frozen actual metric admission differs")
    source = admission["actual_source_admission"]
    require(source["isolated_source_manifest"]["sha256"] == "1fd9b5e14dc377f753b04ecfe6ce93054b300d6f46c5cb341aa466f6ac887a21"
            and source["registered_protocol"]["sha256"] == "4402af3d3cfb51542f8b7a29eb74ab236648da189d7af7c3ebdc9c0a51bab72e"
            and source["resources_full_sha256_recomputed"]["controller_checkpoint"] == owner["amp_checkpoint"]
            and source["actual_terminal_admission"] == owner["terminal_admission"], "Actual capture/controller/terminal source differs")
    new_record, new = record(NEW_RESULT), read(NEW_RESULT)
    require(new["arguments"]["checkpoint_sha256"] == owner["amp_checkpoint"]["sha256"], "Actual controller differs")
    math.complete_run(new)
    historical_path = LEGACY / "actual_reference_pairing.json"
    require(record(historical_path)["sha256"] == "63284b01f538cbe45de39bc981d936b92df46fe3a6e8dee53064189cc77da955", "Original1914 proof changed")
    historical = read(historical_path)
    require(historical["status"] == "passed" and historical["total_actual_array_pairs"] == 1914
            and historical["all_required_reference_shape_dtype_C_bytes_exact"] is True, "Complete historical proof required")
    ids = [row["motion_id"] for row in new["sequences"]]
    runs, run_records, captures = {}, {"pacer_reward2000": new_record}, {"pacer_reward2000": []}
    for name, expected in historical["run_results"].items():
        require(record(expected["path"]) == expected, "Actual comparator result changed: " + name)
        document = read(expected["path"])
        math.complete_run(document)
        require([row["motion_id"] for row in document["sequences"]] == ids, "Ordered full22 motion IDs differ")
        runs[name] = {row["motion_id"]: row for row in document["sequences"]}
        run_records[name], captures[name] = expected, []
    rows = {name: [] for name in runs}
    reference_count = reset_count = 0
    for new_row in new["sequences"]:
        observed = record(new_row["capture"])
        require(observed["sha256"] == new_row["capture_sha256"], "New actual capture changed")
        captures["pacer_reward2000"].append(observed)
        with np.load(new_row["capture"], allow_pickle=False) as right:
            require(right["reference_frame"].shape == (new_row["observed_metric_steps"] + 1,), "Reset plus all metric frames required")
            for name, old_rows in runs.items():
                old_row = old_rows[new_row["motion_id"]]
                require(all(old_row[key] == new_row[key] for key in ("source_path", "tracking_window", "initial_translation_xyz_m")), "Reference source/window/translation differs")
                observed = record(old_row["capture"])
                require(observed["sha256"] == old_row["capture_sha256"], "Old actual capture changed")
                captures[name].append(observed)
                with np.load(old_row["capture"], allow_pickle=False) as left:
                    left_reference = {key for key in left.files if key.startswith("reference_")}
                    right_reference = {key for key in right.files if key.startswith("reference_")}
                    require(left_reference == right_reference, "Reference fields differ")
                    keys = sorted(left_reference | set(math.CLOCK))
                    full = {key: math.check_pair(left[key], right[key], f"{name}/{new_row['motion_id']}/{key}") for key in keys}
                    reset = {key: math.pair(left[key][0], right[key][0]) for key in math.INITIAL}
                reference_count += len(full)
                reset_count += len(reset)
                rows[name].append({"motion_id": new_row["motion_id"], "full_reference_and_clock": full,
                                   "initial_reset_diagnostic": reset})
    require(record(NEW_RESULT) == new_record, "New result changed during CPU review")
    require(all(record(row["path"]) == row for records in captures.values() for row in records), "A paired capture changed during review")
    reset_equal = all(all(check[key] for key in ("shape_equal", "dtype_equal", "C_order_bytes_equal"))
                      for group in rows.values() for row in group for check in row["initial_reset_diagnostic"].values())
    baseline = read(run_records["state_prior2000"]["path"])
    config_delta = {key: {"old": baseline["training_config"].get(key), "new": value}
                    for key, value in new["training_config"].items() if baseline["training_config"].get(key) != value}
    heavy = [name for name in sys.modules if name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni", "warp", "pxr"}]
    require(not heavy, "Reference review imported SDK/model")
    report = {"schema": "pacer_reward.actual_full22_reference_peer.v1", "status": "passed", "reader": record(__file__),
              "CPU_only": True, "GPU_used": False, "SDK_started": False, "sealed34_outputs_read": False,
              "root_actual_seal": record(seal_path), "actual_GPU_owner": record(ACTUAL / "tracking_actual_launch.json"),
              "actual_CPU_source_admission": record(admission_path), "actual_CPU_process_receipt": record(receipt_path),
              "unchanged_byte_pairing_math": record(math_path), "original1914_proof": record(historical_path),
              "run_results": run_records, "actual_capture_records": captures, "ordered_motion_ids": ids,
              "paired_actual_clips": len(ids) * len(runs), "new_metric_frames": 11982,
              "required_reference_and_clock_array_comparisons": reference_count,
              "diagnostic_initial_reset_array_comparisons": reset_count, "total_actual_array_pairs": reference_count + reset_count,
              "all_required_reference_shape_dtype_C_bytes_exact": True, "all_initial_reset_diagnostics_equal": reset_equal,
              "actual_dtype_and_C_bytes_pairings": rows, "new_aggregate": new["aggregate"],
              "baseline19f_aggregate": baseline["aggregate"], "saved_training_config_delta_vs19f": config_delta,
              "generated_post_reset_performance_array_equality_required": False,
              "actual_GPU_runtime_finalization_observed": source["actual_evaluation_imports"]["runtime_finalization_observed"],
              "scope": "Original frozen pair/check_pair/complete_run mathematics reused. All reference poses/velocities, phase/clocks/commands and initial reset diagnostics are paired against19f and its three historical comparators. No post-reset output fitting or failure filtering; no matched-training-noise or author-quality claim."}
    with (HERE / "actual_reference_pairing.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({key: report[key] for key in ("status", "paired_actual_clips", "required_reference_and_clock_array_comparisons",
                      "diagnostic_initial_reset_array_comparisons", "all_initial_reset_diagnostics_equal", "total_actual_array_pairs", "saved_training_config_delta_vs19f")}))


if __name__ == "__main__":
    main()
