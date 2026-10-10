"""Source-bound default62c4 Office3/10s adapter; preparation launches no SDK."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLAN = HERE / "plan.json"
PLAN_SHA = "fa49f2e9ec8b82cf160fb6a57b957f98f7d14d95b70f84c5bb1e6b95c7811488"
sys.path.insert(0, str(ROOT))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "Regular bound source/artifact required")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_sources():
    require(record(PLAN)["sha256"] == PLAN_SHA, "Registered baseline Office plan changed")
    plan = json.loads(PLAN.read_text())
    require(plan["schema"] == "local_Office.baseline62c4_prepared_probe.v1"
            and plan["status"] == "prepared_not_executed"
            and plan["new09e_reward_terminal_schema_used_for62c4"] is False,
            "Independent selected baseline Office protocol required")
    for row in plan["sources"].values():
        require(record(row["path"]) == row, "Registered baseline Office source changed: " + row["path"])
    native = load(plan["sources"]["native_admission_core"]["path"], "baseline_Office_production_roster_verifier")
    require(native.prepare_sources()["production_source"] == plan["production_source"],
            "Current full185 source bytes differ from registered baseline Office source")
    require(record(plan["selected_checkpoint"]["path"]) == plan["selected_checkpoint"],
            "Selected baseline62c4 full checkpoint bytes changed")
    return plan, native


def make_driver_callback(plan, driver_main=None):
    observer = load(plan["observer"]["path"], "baseline_Office_unchanged_configured_gravity")

    def callback(**options):
        require(callable(options.get("native_scene_factory")), "Existing native scene factory required")
        forwarded = dict(options)
        forwarded["native_scene_factory"] = observer.wrap_native_scene_factory(options["native_scene_factory"])
        target = driver_main
        if target is None:
            from CrowdSim.humanoid.application.deployment import DRIVER
            target = load(DRIVER, "baseline_Office_sole_existing_driver").main
        return target(**forwarded)

    return callback


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    from CrowdSim.humanoid.application import deployment
    deployment.require_fresh_sdk_process()
    plan, native = prepare_sources()
    if args.prepare_only:
        print(json.dumps({"status": "prepared_source_bound_not_executed", "plan": record(PLAN),
                          "adapter": record(__file__), "checkpoint": plan["selected_checkpoint"],
                          "GPU_used": False, "SDK_started": False, "actual_Office_probe_executed": False,
                          "new09e_reward_terminal_used": False}))
        return 0
    # Normal deployment preflight and EvaluationActor provide strict runtime load.
    # Historical baseline checkpoints have their original mode; no chmod is made.
    result = deployment.main(plan["deployment_command_without_observer"][2:],
                             driver_main=make_driver_callback(plan))
    require(native.prepare_sources()["production_source"] == plan["production_source"],
            "Production185 changed during baseline Office run")
    require(record(plan["selected_checkpoint"]["path"]) == plan["selected_checkpoint"],
            "Baseline62c4 checkpoint bytes changed during Office run")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
