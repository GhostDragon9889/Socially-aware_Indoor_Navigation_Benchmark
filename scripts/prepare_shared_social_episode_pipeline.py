#!/usr/bin/env python3
"""Prepare and actually cold-probe a fresh Sim61 episode serial execution plan."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from CrowdSim.humanoid.application import episode_pipeline_plan as inventory
from CrowdSim.humanoid.application import episode_serial_release as release
from scripts import own_shared_social_episode_pipeline as owner
from scripts import train_shared_social_episode_after_recovery as training
from scripts import run_shared_social_episode_after_recovery_evaluation as evaluation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    serial, cold = output / "serial_release.json", output / "cold_preflight.json"
    owner.io.write_json(serial, release.release_document(), readonly=True)
    sources = inventory.source_inventory()
    executable = str(release.ROOT / ".venv/bin/python")
    commands = [(stage["stage"], [executable, str(Path(training.__file__)), "--serial-release", str(serial),
                                 "--stage", stage["stage"], "--check-sources"]) for stage in release.stages()]
    commands.append(("evaluation", [executable, str(Path(evaluation.__file__)),
                                    "--serial-release", str(serial), "--check-sources"]))
    preflight = {"schema": "shared_social.episode_cold_preflight.sim61.v2", "sources": sources, "steps": []}
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
    for stage, command in commands:
        process = subprocess.run(command, cwd=release.ROOT, env=environment, capture_output=True, text=True)
        stdout, stderr = output / (stage + ".stdout"), output / (stage + ".stderr")
        for path, text in ((stdout, process.stdout), (stderr, process.stderr)):
            path.write_text(text)
            path.chmod(0o444)
        row = {"stage": stage, "command": command, "actual_exit_code": process.returncode,
               "exit_observation": "subprocess.run.returncode", "stdout": release.record(stdout),
               "stderr": release.record(stderr)}
        preflight["steps"].append(row)
        if process.returncode:
            owner.io.write_json(output / "failed_preflight.json", preflight, readonly=True)
            raise RuntimeError(f"{stage} cold preflight failed; see {stderr}")
        document = json.loads(process.stdout)
        if document.get("Torch_imported") is not False or document.get("SDK_started") is not False:
            raise RuntimeError("Cold preflight imported Torch or started an SDK")
        row.update(Torch_imported=False, SDK_started=False)
    inventory.require_inventory(sources, inventory.source_inventory(), "Before/after source")
    owner.io.write_json(cold, preflight, readonly=True)
    plan = {"schema": owner.PLAN_SCHEMA, "recorded_at_utc": owner.now(), "serial_release": release.record(serial),
            "cold_preflight": release.record(cold), "sources": sources, "stages": release.stages(),
            "GPU_resource_policy": owner.POLICY, "formal_independent_evaluation": str(evaluation.admission.CURRENT_OUTPUT)}
    plan["inputs"] = [release.record(path) for path in inventory.input_paths(plan)]
    path = output / "owner_plan.json"
    owner.io.write_json(path, plan, readonly=True)
    owner.admit_plan(path)
    print(json.dumps({"status": "episode_serial_pipeline_plan_admitted", "plan": release.record(path),
                      "actual_cold_preflights": len(commands), "training_started": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
