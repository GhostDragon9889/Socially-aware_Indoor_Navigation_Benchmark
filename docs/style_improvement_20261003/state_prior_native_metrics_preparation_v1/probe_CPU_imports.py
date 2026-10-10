"""Exercise guarded production mathematics imports without capture or model IO."""
import importlib
import json
import os
import sys

from native_metric_sources import CPULedger, record, require, validate_engine
from run_native_CPU_metrics import CATALOG, CATALOG_SHA, HERE, validate_catalog


def main():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU probe requires CUDA disabled")
    require(record(CATALOG)["sha256"] == CATALOG_SHA, "Frozen command catalog changed")
    catalog = json.loads(CATALOG.read_text())
    validate_catalog(catalog)
    allowed = validate_engine(catalog)
    ledger = CPULedger(HERE / "guarded_CPU_imports_v1.json", allowed, record(CATALOG))
    ledger.activate()
    status = "failed"
    try:
        for name in ("scripts.evaluate_pedestrian_style", "scripts.evaluate_paper_pedestrians",
                     "CrowdSim.humanoid.style_evaluation", "CrowdSim.humanoid.training.manifests",
                     "CrowdSim.humanoid.amass_corpus"):
            importlib.import_module(name)
        import torch
        require(not torch.cuda.is_initialized(), "Probe initialized CUDA")
        require(not any(n == "isaacsim" or n.startswith(("omni.", "isaaclab.")) for n in sys.modules), "Probe imported SDK")
        validate_engine(catalog)
        status = "passed"
    finally:
        ledger.finish(status)
    proof = {"schema": "public_state_prior.native_metric_guarded_CPU_import_proof.v1", "status": status,
             "future_effect": False, "GPU_used": False, "SDK_started": False,
             "actual_capture_read": False, "actual_checkpoint_read": False,
             "model_deserialized": False, "actual_metrics_computed": False,
             "production_source": catalog["sources"]["production_V11"],
             "supplementary_math_files": catalog["engine_extra_sources"],
             "actual_CPU_import_ledger": record(ledger.path)}
    path = HERE / "guarded_CPU_source_proof_v1.json"
    require(not path.exists(), "Keep first source probe")
    path.write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof))


if __name__ == "__main__":
    main()
