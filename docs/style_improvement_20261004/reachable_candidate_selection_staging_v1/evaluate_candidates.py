"""Independent common50 S10 selection; the original driver owns App and physics."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PARENT = ROOT / "scripts/train_paper_amp.py"
PARENT_SHA = "d8b85401ace1da7d0039dcad4e023590577944e487789e742019c363a602e3d3"
RUNNER = HERE / "evaluate_runner.py"
sys.path.insert(0, str(ROOT))


def record(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     epilog="Remaining arguments are the original simulate_paper_pedestrians CLI.")
    parser.add_argument("--selection-domain", choices=("matched_raw50", "reachable50"), required=True)
    selected, remaining = parser.parse_known_args(argv)
    from CrowdSim.humanoid.application.deployment import parse_deployment_args
    return selected.selection_domain, parse_deployment_args(remaining)


def preflight(args, selector):
    from CrowdSim.humanoid.application import deployment
    from CrowdSim.humanoid.application.configuration import load_amp_profile, parse_amp_args
    from CrowdSim.humanoid.application.resources import preflight_amp_resources

    if (not args.inference_only or args.arm_style_alpha != 0.
            or args.experimental_arm_style_checkpoint is not None or args.native_dynbench
            or args.actors != 4 or args.control_steps != 6000 or args.seed != 7):
        raise ValueError("Use the original four-person200s seed7 inference-only alpha0 fixture")
    if record(PARENT)["sha256"] != PARENT_SHA:
        raise ValueError("Reviewed source185 parent driver changed")
    report = deployment.preflight_deployment(args)
    original = report["driver_argv"]
    forwarded = list(original)
    index = forwarded.index("--diffusion-samples")+1
    if forwarded[index] != "1":
        raise ValueError("The original deployment must still forward samples1")
    forwarded[index] = "10"
    parsed = parse_amp_args(forwarded)
    expected = {**report["parsed_driver_arguments"], "diffusion_samples": 10}
    if vars(parsed) != expected:
        raise ValueError("Only samples1 to10 may change the original driver arguments")
    # Check the actual S10 request with the original SDK-free resource function.
    resources = preflight_amp_resources(parsed, load_amp_profile(parsed.config))
    report.update(schema="local_reference.common50_selection_preflight.v1",
                  entry=str(Path(__file__).resolve()), driver=str(RUNNER), driver_argv=forwarded,
                  parsed_driver_arguments=vars(parsed), resource_preflight=resources.as_dict())
    report["deployment_contract"]["candidate_selection"] = selector.describe()
    report["candidate_selection_sources"] = {
        "entry": record(__file__), "runtime": record(HERE/"selection_runtime.py"),
        "runner": record(RUNNER), "parent_driver": record(PARENT),
        "helper": record(ROOT/"CrowdSim/humanoid/training/reachable_selection.py"),
    }
    report["limitations"].append(
        "Both arms replace historical52-node ranking with common50; only the scoring domain differs between arms. No physical effect has been established by preflight.")
    return report


def main(argv=None, *, driver_main=None):
    from CrowdSim.humanoid.application import deployment

    domain, args = parse_args(argv)
    # Loading this local module is standard-library-only; callback imports are lazy.
    spec = importlib.util.spec_from_file_location("isolated_common50_selection_runtime", HERE/"selection_runtime.py")
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    selector = runtime.ReferenceSelector(domain)
    report = preflight(args, selector)
    if args.preflight_only:
        if any(name.split(".")[0] in {"torch", "numpy", "isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules):
            raise RuntimeError("SDK-free preflight requires a fresh process without numerical/SDK imports")
        print(json.dumps(report, indent=2, allow_nan=False))
        return 0
    deployment.require_fresh_sdk_process()
    for name in deployment.THREAD_ENVIRONMENT:
        os.environ[name] = "4"
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/"deployment_entry.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    if driver_main is None:
        spec = importlib.util.spec_from_file_location("isolated_common50_physical_runner", RUNNER)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        driver_main = module.main

    def invoke(**factories):
        return driver_main(reference_selector=selector, **factories)

    result = deployment.invoke_existing_driver(
        report["driver_argv"], driver_main=invoke,
        evaluation_actor_request=report["evaluation_actor_request"],
        native_scene_request=report["native_scene_request"])
    return 0 if result is None else result


if __name__ == "__main__":
    raise SystemExit(main())
