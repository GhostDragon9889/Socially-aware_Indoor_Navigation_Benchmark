"""Register S3-R terminal selection, run cold preflights, and bind the OS owner."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.application import recovery_evaluation_admission as admission  # noqa: E402
from CrowdSim.humanoid.runtime.social_evaluation_lifecycle import record  # noqa: E402
from scripts import evaluate_shared_social_recovery as evaluation  # noqa: E402
from scripts import own_shared_social_recovery_evaluation as owner  # noqa: E402
from scripts import run_shared_social_post_training as original  # noqa: E402
from scripts import evaluate_shared_social_style as style  # noqa: E402
from scripts import run_shared_social_budget_evaluation as budget  # noqa: E402

HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / "output/evaluation/shared_social_S3R_budget12500_v2"


def save(name, document):
    path = HERE / name
    with path.open("x") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
        stream.write("\n")
    path.chmod(0o444)
    return record(path)


receipt = HERE / "training_execution.json"
save(receipt.name, admission.receipt_document())
checkpoint, _, accepted = admission.admit_training(receipt)
registration = {
    "schema": budget.SCHEMA, "registered_at_utc": datetime.now(timezone.utc).isoformat(),
    "configuration": record(admission.TRAINING / "training.json"),
    "run_directory": str(checkpoint.parent), "preselected_checkpoint": checkpoint.name,
    "checkpoint_identity": accepted["checkpoint"], "planned_updates": 12500,
    "planned_physical_transitions": 102400000, "recovery_execution_admission": accepted,
    "evaluation_output": str(OUTPUT), "author_protocol_equivalence_claimed": False,
    "protocol": {"tracking_split": "fixed parent-disjoint validation22", "tracking_controls": 11982,
        "test34_used_for_selection": False, "failed_frames_kept": True, "style_reference_refit": False,
        "pose45_centers_refit": False, "tracking_worldspace_alignment_unchanged": True,
        "navigation_actors": 4, "navigation_seconds": 200, "navigation_controls": 6000,
        "same_Diffusion40k": original.DIFFUSION_SHA,
        "author_eight_scenes_and_rate_denominators_recovered": False},
    "wrapper_sources": [record(ROOT / path) for path in evaluation.EXTRA_SOURCES],
    "frozen_style_inputs": [record(path) for path in
                            (style.REFERENCE, style.BASIS, style.BASIS_EVALUATION)],
    "default_model_promoted": False,
}
save("registration.json", registration)
preflight = {"schema": "shared_social.recovery_evaluation_cold_preflight_execution.v1",
             "status": "running", "steps": [], "SDK_started": False,
             "Torch_imported": False, "training_receipt": record(receipt)}
environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                   OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
for stage in ("tracking", "navigation"):
    command = [sys.executable, str(ROOT / "scripts/evaluate_shared_social_recovery.py"), "--stage", stage,
               "--execution-receipt", str(receipt), "--output", str(ROOT / "output/evaluation" / (HERE.name + "_" + stage + "_unused_preflight_output")),
               "--preflight-only"]
    if stage == "navigation":
        command.extend(["--diffusion-checkpoint", str(original.DIFFUSION),
                        "--diffusion-sha256", original.DIFFUSION_SHA, "--control-steps", "6000"])
    log = HERE / (stage + "_preflight.json")
    with log.open("xb") as stream:
        completed = subprocess.run(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=stream, stderr=subprocess.STDOUT, check=False)
    step = {"stage": stage, "command": command, "actual_exit_code": completed.returncode, "log": record(log)}
    preflight["steps"].append(step)
    log.chmod(0o444)
    if completed.returncode != 0:
        preflight["status"] = "failed"
        save("preflight_execution.json", preflight)
        raise RuntimeError(stage + " cold preflight failed; output preserved")
    report = json.loads(log.read_text())
    tracking_ok = (report.get("status") == "preflight_passed"
                   and report.get("NN_constructed") is False and report.get("SDK_started") is False
                   and report.get("selected_clips") == 22 and report.get("postphysics_controls") == 11982)
    navigation_ok = (report.get("schema") == "local.neutral_smpl.social_four_actor_navigation_evaluation.v1"
                     and report.get("physical_execution_verified") is False
                     and report.get("requested_controls") == 6000
                     and report.get("requested_actor_controls") == 24000
                     and report.get("primary_200_seconds") is True)
    if not (tracking_ok if stage == "tracking" else navigation_ok):
        preflight["status"] = "failed"
        save("preflight_execution.json", preflight)
        raise ValueError("Preflight exit/result/no-SDK distinction failed")
    step["original_report_schema_admitted"] = True
preflight["status"] = "passed"
save("preflight_execution.json", preflight)

paths = [ROOT / path for path in original.SOURCE_PATHS]
paths += [ROOT / path for path in evaluation.EXTRA_SOURCES]
paths += [ROOT / "scripts/run_shared_social_budget_evaluation.py", Path(owner.__file__), Path(__file__),
          ROOT / "tests/test_shared_social_recovery_evaluation.py"]
paths += [style.TOOL, style.POSE_TOOL]
plan = {"schema": owner.SCHEMA, "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_receipt": str(receipt), "registration": str(HERE / "registration.json"),
        "preflight_execution": str(HERE / "preflight_execution.json"), "evaluation_output": str(OUTPUT),
        "sources": [record(path) for path in dict.fromkeys(paths)],
        "inputs": [record(HERE / name) for name in ("training_execution.json", "registration.json",
                                                   "preflight_execution.json", "tracking_preflight.json", "navigation_preflight.json")],
        "GPU_resource_policy": owner.POLICY, "detach_from_chat_session": True,
        "paper_effect_achieved": False, "default_model_promoted": False,
        "original_continuous_S3_run_completed": False}
save("owner_plan.json", plan)
owner.admit(HERE / "owner_plan.json")
print(json.dumps({"status": "registered_and_cold_preflights_passed", "source_records": len(plan["sources"]),
                  "input_records": len(plan["inputs"]), "SDK_started": False, "Torch_imported": "torch" in sys.modules,
                  "plan": record(HERE / "owner_plan.json"), "evaluation_output": str(OUTPUT)}))
