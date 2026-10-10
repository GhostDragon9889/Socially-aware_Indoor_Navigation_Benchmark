"""Drop one new transfer guard in a /tmp copy and require test failure."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "docs/style_improvement_20261003/state_prior_evaluation_registration_v1"
PYTHON = "/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    temp = Path(tempfile.mkdtemp(prefix="state-prior-evaluation-peer-mutation-"))
    for source in SOURCE.iterdir():
        if source.is_file() and source.suffix in {".py", ".json", ".jsonl"}:
            shutil.copyfile(source, temp / source.name)
    executor = temp / "run_registered_evaluation.py"
    original_sha = digest(SOURCE / executor.name)
    # The isolated test copy must locate the real readonly reader/934 source.
    # This path-only harness adjustment is separately recorded, not a proposal.
    text = executor.read_text().replace("ROOT = Path(__file__).resolve().parents[3]",
                                       "ROOT = Path(" + repr(str(ROOT)) + ")")
    executor.write_text(text)
    evidence_copy = temp / "evaluation_evidence.py"
    evidence_copy.write_text(evidence_copy.read_text().replace("ROOT = Path(__file__).resolve().parents[3]",
                            "ROOT = Path(" + repr(str(ROOT)) + ")"))
    before = executor.read_bytes()
    before_sha = digest(executor)
    condition = '        or rows[0].get("public_D_or_AMP_RMS_transferred") is not True\n'
    if text.count(condition) != 1:
        raise ValueError("Mutation target is not unique")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    outcomes = []
    for label, payload, expected in (("baseline", before, 0),
                                     ("drop_row_public_transfer_guard", text.replace(condition, "").encode(), 1),
                                     ("restored", before, 0)):
        executor.write_bytes(payload)
        command = [PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest",
                   str(temp / "test_registration.py"),
                   "--junitxml=" + str(HERE / (label + ".xml"))]
        result = subprocess.run(command, cwd=temp, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        (HERE / (label + ".log")).write_text(result.stdout)
        outcomes.append({"case": label, "command": command, "return_code": result.returncode,
                         "executor_sha256": digest(executor), "expected_return_code": expected})
        if result.returncode != expected:
            executor.write_bytes(before)
            raise ValueError("Guard mutation gate failed: " + label + "\n" + result.stdout)
    if digest(executor) != before_sha or digest(SOURCE / executor.name) != original_sha:
        raise ValueError("Guard bytes did not restore or frozen source changed")
    report = {"schema": "public_state_prior.evaluation_peer_guard_mutation.v1", "status": "passed",
              "CPU_only": True, "GPU_used": False, "SDK_started": False,
              "actual_checkpoint_bytes_read": False, "temporary_copy": str(temp),
              "frozen_executor_sha256": original_sha, "path_adjusted_baseline_sha256": before_sha,
              "evidence_helper_path_adjustment": "ROOT points to production for readonly934 metadata; no guard or validation changed",
              "restored_sha256": digest(executor), "production_and_frozen_docs_unchanged": True,
              "dropped_guard": condition.strip(), "outcomes": outcomes,
              "scope": "Pure contract fixture and real frozen reader/source metadata validation. No evaluator main, checkpoint model deserialization, owned simulator entry execution, or future effect."}
    (HERE / "guard_mutation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
