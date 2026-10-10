"""Observe actual CPU child exit after registering every admitted resource case."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.trajectory.social_force_runtime import file_identity  # noqa: E402


def main():
    cohort_path = HERE / "resource_cohort_CPU.json"
    cohort = json.loads(cohort_path.read_text())
    if (cohort["status"] != "CPU_resource_case_review_complete" or cohort["source_resources_unchanged"] is not True
            or cohort["sources_unchanged"] is not True or len(cohort["cases"]) != 52):
        raise ValueError("Require the complete registered52-case resource review, including all rejections")
    admitted = [row for row in cohort["cases"] if row["status"] == "CPU_resource_admitted"]
    command = [sys.executable, str(ROOT / "scripts/prepare_shared_social_native_sfm.py"), "--seed", "7",
               "--output", str(HERE / "actual_CPU_snapshots_v1")]
    for row in admitted:
        identity = row["configuration"]
        if file_identity(identity["path"]) != identity:
            raise ValueError("Registered local scene changed")
        command.extend(("--scene-config", identity["path"]))
    bindings = [file_identity(ROOT / name) for name in (
        "scripts/prepare_shared_social_native_sfm.py", "CrowdSim/humanoid/trajectory/native_social_force_runtime.py")]
    registration = {"schema": "shared_social.native_SFM_CPU_child_registration.v1", "resource_cohort": file_identity(cohort_path),
        "command": command, "sources": bindings, "accepted_resource_case_count": len(admitted),
        "resource_rejections_retained": [row for row in cohort["cases"] if row["status"] == "rejected"],
        "scope": "Synthetic spawn-state predictions on real source-bound native maps; no physical rollout",
        "physical_navigation_validated": False, "CPU_only": True}
    selected = HERE / "CPU_prediction_registration.json"
    with selected.open("x") as stream:
        stream.write(json.dumps(registration, indent=2) + "\n")
    selected.chmod(0o444)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
           "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
    log = HERE / "CPU_prediction.log"
    with log.open("x") as stream:
        process = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    unchanged = all(file_identity(row["path"]) == row for row in bindings)
    report = HERE / "actual_CPU_snapshots_v1/CPU_preparation.json"
    receipt = {"schema": "shared_social.native_SFM_actual_CPU_child_exit.v1", "registration": file_identity(selected),
               "actual_process_exit_observed": True, "actual_child_returncode": process.returncode,
               "sources_unchanged": unchanged, "child_log": file_identity(log), "reader": file_identity(__file__),
               "report": file_identity(report) if report.is_file() else None,
               "physical_navigation_validated": False, "paper_effect_achieved": False}
    with (HERE / "CPU_prediction_execution.json").open("x") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"actual_child_returncode": process.returncode, "report": str(report),
                      "sources_unchanged": unchanged, "physical_navigation_validated": False}))
    return process.returncode if unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
