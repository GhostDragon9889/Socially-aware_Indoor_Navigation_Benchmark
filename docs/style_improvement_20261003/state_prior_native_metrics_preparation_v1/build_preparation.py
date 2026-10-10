"""Bind existing mathematics and the independent native reader; no capture IO."""
from __future__ import annotations

import json
from pathlib import Path

from native_metric_sources import ROOT, record, require

HERE = Path(__file__).resolve().parent
SDK = "/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python"
OLD = ROOT / "docs/style_improvement_20261003/state_prior_CPU_metrics_preparation_v1/commands_CPU_metrics_v1.json"
READER = ROOT / "docs/style_improvement_20261003/state_prior_native_reader_preparation_v1"


def build():
    require(not (HERE / "source_freeze_v1.json").exists(), "Never modify an existing native metric freeze")
    old = json.loads(OLD.read_text())["sources"]
    catalog = {"schema": "public_state_prior.prepared_native_CPU_metric_commands.v1",
               "status": "prepared_not_executed", "future_effect": False,
               "CPU_only": True, "GPU_used": False, "Kit_started": False,
               "actual_checkpoint_read": False, "actual_capture_read": False,
               "actual_metrics_computed": False, "sealed34_outputs_read": False,
               "default_promotion": False, "python": SDK,
               "CPU_environment": {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4",
                                   "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
                                   "PYTHONDONTWRITEBYTECODE": "1"},
               "sources": {"production_V11": record(ROOT / "docs/style_improvement_20261003/native_runtime_epoch_v11/native_pedestrian_source_v11.json"),
                           "native_metric_sources": record(HERE / "native_metric_sources.py"),
                           "native_reader_freeze": record(READER / "source_freeze_v1.json"),
                           "native_evidence": record(READER / "native_evidence.py"),
                           "native_reader": record(READER / "audit_state_prior_native.py"),
                           "style90_entry": record(ROOT / "scripts/evaluate_pedestrian_style.py"),
                           "style90_math": record(ROOT / "CrowdSim/humanoid/style_evaluation.py"),
                           "style_reference": old["style90_reference"],
                           "pose45_basis": old["pose45_basis"],
                           "pose45_reference_evaluation": old["pose45_reference_evaluation"],
                           "pose45_tool": old["pose45_evaluator"]},
               "engine_extra_sources": [record(ROOT / path) for path in (
                   "CrowdSim/__init__.py", "protomotions/__init__.py",
                   "protomotions/simulator/__init__.py", "protomotions/simulator/base_simulator/__init__.py",
                   "protomotions/simulator/base_simulator/simulator_state.py", "protomotions/utils/__init__.py",
                   "scripts/evaluate_paper_pedestrians.py")],
               "metric_rules": {"original90_function_bytes_unchanged": True,
                                "original45_function_bytes_unchanged": True,
                                "all_failed_and_idle_samples_retained": True,
                                "first_frame_and_cross_reset_exclusions": "unchanged sequence_style_samples",
                                "speed_edges_m_s": [0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0, None],
                                "minimum_bin_samples": 30, "maximum_motion_samples": 256,
                                "primary90_sigma": 1.0, "pose45_sigma": 1.0,
                                "pose45_saved_K": [50, 100], "reference_refit": False,
                                "generated_fit": False, "unsupported_bins_unknown_not_zero": True,
                                "native_GPU_import_ledger_claimed": False,
                                "same_rng_bitwise_controller_comparison_claimed": False,
                                "metric_pass_is_quality_pass": False}, "metrics": {}}
    for kind in ("new", "old"):
        for actors in (4, 15):
            suffix = "" if actors == 4 else "_15"
            prefix = "native_warehouse_state_prior" if kind == "new" else "native_warehouse"
            if kind == "new":
                audit = ROOT / f"docs/style_improvement_20261003/state_prior_native_actual_review_v1/native{actors}_audit.json"
            else:
                directory = "native_scene_200s_actual_review_v1" if actors == 4 else "native15_200s_independent_review_v1"
                audit = ROOT / f"docs/style_improvement_20261002/{directory}/actual_audit.json"
            for dimension in (90, 45):
                key = f"{kind}{actors}_{dimension}"
                row = {"kind": kind, "actors": actors, "dimension": dimension,
                       "source_count": 183 if kind == "new" else 168 if actors == 4 else 174,
                       "run_name": f"{kind}_native{actors}",
                       "input": str(ROOT / f"output/diffusion_amp/{prefix}{suffix}_clearance08_200s_v1"),
                       "output": str(ROOT / f"docs/style_improvement_20261003/state_prior_native_metrics_v1/{key}"),
                       "raw_audit": str(audit), "command": [SDK, str(HERE / "run_native_CPU_metrics.py"), "--metric", key]}
                if kind == "new":
                    row["registration"] = str(READER / f"native{actors}_registration.json")
                else:
                    row["raw_audit_record"] = record(audit)
                    catalog["sources"][f"old_native{actors}_audit"] = row["raw_audit_record"]
                values = ["--run", row["run_name"] + "=" + row["input"]]
                if dimension == 90:
                    values += ["--reference-artifact", catalog["sources"]["style_reference"]["path"], "--cpu-threads", "4",
                               "--minimum-bin-samples", "30", "--maximum-motion-samples", "256"]
                row["metric_argv"] = values + ["--output", row["output"]]
                catalog["metrics"][key] = row
    catalog["prepare_only_command"] = [SDK, str(HERE / "run_native_CPU_metrics.py"), "--prepare-only"]
    catalog["comparisons"] = {str(n): {"command": [SDK, str(HERE / "compare_native_metrics.py"), "--actors", str(n)],
                                      "output": str(ROOT / f"docs/style_improvement_20261003/state_prior_native_metrics_v1/comparison{n}"),
                                      "requires_closed_metrics": [f"{kind}{n}_{d}" for kind in ("old", "new") for d in (90, 45)]}
                              for n in (4, 15)}
    path = HERE / "commands_native_CPU_metrics_v1.json"
    path.write_text(json.dumps(catalog, indent=2, allow_nan=False) + "\n")
    for filename in ("run_native_CPU_metrics.py", "compare_native_metrics.py"):
        runner = HERE / filename
        text = runner.read_text()
        require('CATALOG_SHA = "UNFROZEN"' in text, "Runner already bound; no silent rebind")
        runner.write_text(text.replace('CATALOG_SHA = "UNFROZEN"', f'CATALOG_SHA = "{record(path)["sha256"]}"'))
    print(json.dumps({"status": "prepared_catalog_bound_not_executed", "future_effect": False, "catalog": record(path)}))


if __name__ == "__main__":
    build()
