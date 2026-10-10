"""Future root-owned Office3/10s driver adapter; --prepare-only launches nothing.

Execution reuses the existing deployment shim and sole production driver. A
per-provider callback appends configured-gravity observations to the existing
floor/source observation; no additional simulator or control loop is created.
"""
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
PLAN_SHA = "63b10ff8891e4fe2c4e215a440eb62c47b0cfd6724a8ed936b96c6a569cb5996"
OBSERVER = HERE / "observe_configured_gravity.py"
OBSERVER_SHA = "06108c64d0cc63e9dbed1b798dde9d2072b58edceb7e44a6ce3059366a7ad635"
sys.path.insert(0, str(ROOT))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record(path):
    path = Path(path).absolute()
    require(path.is_file() and not path.is_symlink(), "Regular selected source/artifact required")
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
    require(record(PLAN)["sha256"] == PLAN_SHA, "Registered local Office plan changed")
    plan = json.loads(PLAN.read_text())
    require(plan["schema"] == "local_Office.prepared_gravity_support_probe.v1"
            and plan["status"] == "prepared_source_bound_not_executed"
            and plan["actual_local_Office_human_probe_executed"] is False,
            "Prepared local Office source plan required")
    require(record(OBSERVER)["sha256"] == OBSERVER_SHA, "Read-only Office observer changed")
    for row in plan["sources"].values():
        require(record(row["path"]) == row, "Registered Office source changed: " + row["path"])
    native = load(plan["sources"]["native_admission_core"]["path"], "Office_native_terminal_admission")
    require(native.read_registration()["current_source_binding"]["production_source"] == plan["production_source"],
            "Current production185 differs from the Office source plan")
    return plan, native


def make_driver_callback(driver_main=None):
    """Preserve deployment options and delegate once to the production driver."""
    observer = load(OBSERVER, "Office_readonly_configured_gravity")

    def callback(**options):
        require(callable(options.get("native_scene_factory")), "Existing native scene factory is required")
        selected = dict(options)
        selected["native_scene_factory"] = observer.wrap_native_scene_factory(options["native_scene_factory"])
        target = driver_main
        if target is None:
            from CrowdSim.humanoid.application.deployment import DRIVER
            target = load(DRIVER, "Office_sole_existing_production_driver").main
        return target(**selected)

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
                          "observer": record(OBSERVER), "adapter": record(__file__),
                          "GPU_used": False, "SDK_started": False,
                          "future_checkpoint_read": False, "actual_Office_probe_executed": False}))
        return 0
    checkpoint_path = Path(plan["future_checkpoint"]["path"])
    checkpoint = record(checkpoint_path)
    require(checkpoint_path.stat().st_nlink == 1 and not checkpoint_path.stat().st_mode & 0o222,
            "Actual readonly single-link sealed controller required")
    terminal = json.loads(Path(plan["future_checkpoint"]["terminal_admission_path"]).read_text())
    native.validate_terminal(terminal, checkpoint)
    result = deployment.main(plan["future_deployment_command_without_observer"][2:],
                             driver_main=make_driver_callback())
    require(native.prepare_sources()["production_source"] == plan["production_source"],
            "Production source changed during the local Office run")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
