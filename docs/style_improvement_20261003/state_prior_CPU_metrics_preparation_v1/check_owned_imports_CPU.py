"""Check source relocation/owned import boundary without any actual capture."""
import json
import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location("checked_prepared_CPU_runner", HERE / "run_CPU_metrics.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU-only required")
    catalog = json.loads(runner.CATALOG.read_text())
    runner.validate_preparation(catalog)
    destination = HERE / "actual_owned_imports_CPU_v2.json"
    runner.require(not destination.exists(), "Fresh source proof only")
    launcher = runner.load("source_checked_CPU_metric_launcher", catalog["sources"]["isolated_launcher"]["path"])
    ledger = launcher.ImportLedger(HERE / "source_check_CPU_imports_v2.json")
    staging = Path(tempfile.mkdtemp(prefix="state_prior_frozen_metric_CPU_proof_", dir="/tmp"))
    files, provenance = {}, {}
    for key in ("pose45_evaluator", "raw_ground_tool", "idle_tool", "tracking_decomposition_tool", "tracking_comparison_tool"):
        files[key], provenance[key] = runner.copy_external_source(catalog, key, staging)
    for path in sorted(staging.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_dir():
            path.chmod(0o555)
    staging.chmod(0o555)
    launcher.activate(ledger)
    status = "failed"
    try:
        modules = {key: runner.load("source_checked_" + key, path) for key, path in files.items()}
        style = importlib.import_module("scripts.evaluate_pedestrian_style")
        math = importlib.import_module("CrowdSim.humanoid.style_evaluation")
        capture = importlib.import_module("scripts.evaluate_paper_pedestrians")
        runner.require(runner.record(style.__file__) == catalog["sources"]["style90_entry"], "Actual style entry escaped CODE")
        runner.require(runner.record(math.__file__) == catalog["sources"]["style90_math"], "Actual math escaped CODE")
        runner.require(list(math.StyleSpec().speed_bin_lower_edges_m_s) + [None] == catalog["metric_rules"]["speed_edges_m_s"],
                       "Prepared primary speed buckets differ from original actual feature math")
        runner.require(Path(capture.__file__).is_relative_to(launcher.CODE), "Capture reader escaped CODE")
        pose = modules["pose45_evaluator"]
        pose.ROOT = launcher.CODE
        pose.REF = Path(catalog["sources"]["style90_reference"]["path"])
        for path, sha in pose.FROZEN.items():
            runner.require(runner.record(pose.ROOT / path)["sha256"] == sha, "Relocated 45D feature source differs")
        diagnostic = modules["tracking_decomposition_tool"]
        diagnostic.ROOT = launcher.CODE
        diagnostic.check_analytic_invariance(diagnostic.parent_indices(diagnostic.ROOT / "protomotions/data/assets/mjcf/smpl_humanoid.xml"))
        runner.require(not any(name == "torch" or name == "isaacsim" or name.startswith(("omni.", "isaaclab.")) for name in sys.modules),
                       "Source-only proof imported Torch/SDK")
        proof = {"schema": "public_state_prior.prepared_CPU_metric_owned_import_proof.v1", "status": "passed_source_only",
                 "CPU_only": True, "GPU_used": False, "Kit_started": False, "Torch_imported": False,
                 "actual_checkpoint_read": False, "actual_capture_read": False, "actual_metrics_computed": False,
                 "future_effect": False, "sealed34_outputs_read": False,
                 "catalog": runner.record(runner.CATALOG), "actual_primary_speed_bucket_edges_m_s": catalog["metric_rules"]["speed_edges_m_s"],
                 "style90_entry": runner.record(style.__file__), "style90_math": runner.record(math.__file__),
                 "capture_reader": runner.record(capture.__file__), "exact_external_source_stages": provenance,
                 "fixed45D_source_path_redirect_to_newCODE_passed": True,
                 "tracking_decomposition_original_analytic_invariance_passed": True,
                 "all_actual_owned_imports_from_newCODE": True}
        runner.write(destination, proof)
        status = "passed_source_only"
    finally:
        ledger.finish(status)
    print(json.dumps({"status": status, "proof": runner.record(destination), "future_effect": False}))


if __name__ == "__main__":
    main()
