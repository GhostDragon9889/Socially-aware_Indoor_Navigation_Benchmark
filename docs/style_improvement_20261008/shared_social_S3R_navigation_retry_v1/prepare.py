"""Register one normal navigation retry from the actual failed owner prefix."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record  # noqa: E402
from scripts import resume_shared_social_recovery_evaluation as resume  # noqa: E402

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "shared_social_S3R_evaluation_v2"
DEBUG = HERE.parent / "shared_social_S3R_navigation_gdb_v1"

if (HERE / "owner_plan.json").exists():
    raise FileExistsError("Preserve the registered retry")
prefix = resume.admit_prefix(PREVIOUS / "completion.json", PREVIOUS / "failed_output_inventory.json")
prior_plan = resume.original.read(PREVIOUS / "owner_plan.json")
debug = resume.original.read(DEBUG / "completion.json")
resume.admission.require(debug.get("status") == "debugger_exited"
    and resume.admission.exact_integer(debug.get("actual_debugger_exit_code"), 0), "Require the observed GDB terminal")
sources = [record(resume.__file__), *prior_plan["sources"], record(__file__),
           record(ROOT / "tests/test_shared_social_recovery_evaluation_resume.py"),
           record(DEBUG / "completion.json"), record(DEBUG / "gdb.log")]
plan = {"schema": resume.PLAN_SCHEMA, "recorded_at_utc": resume.now(), "prefix": prefix,
        "sources": sources, "diffusion": record(resume.post.DIFFUSION),
        "evaluation_output": str(ROOT / "output/evaluation/shared_social_S3R_navigation_retry_v1"),
        "GPU_resource_policy": resume.previous_owner.POLICY,
        "retry_count": 1, "debugger_attached": False, "native_exit_root_cause_identified": False,
        "physics_or_metric_code_changed": False, "paper_effect_achieved": False,
        "default_model_promoted": False}
resume.write_json(HERE / "owner_plan.json", plan, readonly=True)
resume.admit_plan(HERE / "owner_plan.json")
print(json.dumps({"status": "normal_navigation_retry_prepared", "plan": record(HERE / "owner_plan.json"),
                  "retained_successful_prefix": resume.PREFIX_NAMES, "retained_outputs": 54,
                  "SDK_started": False, "Torch_imported": "torch" in sys.modules}))
