"""Invert saved-reference population guard only in a /tmp runner copy."""
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
SOURCE = ROOT / "docs/style_improvement_20261003/state_prior_CPU_metrics_preparation_v1"
PYTHON = "/project/EXP/SangforAndSaut/IsaacLab-Protomotions/.venv/bin/python"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    temp = Path(tempfile.mkdtemp(prefix="state-prior-metrics-peer-mutation-"))
    for filename in ("run_CPU_metrics.py", "test_CPU_metrics_preparation.py", "commands_CPU_metrics_v1.json"):
        shutil.copyfile(SOURCE / filename, temp / filename)
    runner = temp / "run_CPU_metrics.py"
    original_sha = sha(SOURCE / runner.name)
    text = runner.read_text().replace("ROOT = Path(__file__).resolve().parents[3]",
                                    "ROOT = Path(" + repr(str(ROOT)) + ")")
    baseline = text.encode()
    condition = 'len(reference) == original_evaluation["method"]["fit"]["available_reference_poses"]'
    if text.count(condition) != 1:
        raise ValueError("Mutation guard must be unique")
    changed = text.replace(condition, condition.replace(" == ", " != ")).encode()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    rows = []
    for label, content, expected in (("baseline", baseline, 0), ("inverted_reference_population", changed, 1),
                                     ("restored", baseline, 0)):
        runner.write_bytes(content)
        command = [PYTHON, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--noconftest",
                   str(temp / "test_CPU_metrics_preparation.py"), "--junitxml=" + str(HERE / (label + ".xml"))]
        result = subprocess.run(command, cwd=temp, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, check=False)
        (HERE / (label + ".log")).write_text(result.stdout)
        rows.append({"case": label, "return_code": result.returncode, "expected_return_code": expected,
                     "runner_sha256": sha(runner), "command": command})
        if result.returncode != expected:
            runner.write_bytes(baseline)
            raise ValueError("Mutation validation failed: " + label + "\n" + result.stdout)
    if runner.read_bytes() != baseline or sha(SOURCE / runner.name) != original_sha:
        raise ValueError("Frozen/restored bytes changed")
    proof = {"schema": "public_state_prior.CPU_metric_peer_guard_mutation.v1", "status": "passed",
             "CPU_only": True, "GPU_used": False, "SDK_started": False,
             "actual_future_scores_processed": False, "actual_future_checkpoint_or_capture_read": False,
             "temporary_copy": str(temp), "frozen_runner_sha256": original_sha,
             "harness_path_override_only": "ROOT points to original readonly catalog dependencies; guard unchanged in baseline",
             "baseline_sha256": hashlib.sha256(baseline).hexdigest(), "restored_sha256": sha(runner),
             "inverted_condition": condition, "outcomes": rows,
             "production_and_frozen_source_unchanged": True}
    (HERE / "guard_mutation.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
