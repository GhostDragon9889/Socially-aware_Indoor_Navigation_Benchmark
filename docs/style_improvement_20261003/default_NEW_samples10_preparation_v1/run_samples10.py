"""Independent samples10 deployment; delegate to the unchanged source185 driver."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PARAMETERS = HERE / "run_parameters.json"


def record(path):
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError("Regular source required")
    raw = path.read_bytes()
    return {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def main(argv=None):
    sys.path.insert(0, str(ROOT))
    from CrowdSim.humanoid.application import deployment
    from CrowdSim.humanoid.application.configuration import parse_amp_args

    parameters = json.loads(PARAMETERS.read_text())
    if Path(deployment.__file__).absolute() != Path(parameters["sources"]["deployment"]["path"]):
        raise ValueError("Original source185 deployment import differs")
    for source in parameters["sources"].values():
        if record(source["path"]) != source:
            raise ValueError("Registered source185 method source changed")
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments != parameters["deployment_arguments"] and arguments != parameters["deployment_arguments"] + ["--preflight-only"]:
        raise ValueError("Only the registered samples10 fixture arguments are accepted")
    args = deployment.parse_deployment_args(arguments)
    report = deployment.preflight_deployment(args)
    original = list(report["driver_argv"])
    if original != parameters["parent_source_forwarded_arguments_at_new_output"]:
        raise ValueError("Parent source must still materialize the original samples1 recipe")
    index = original.index("--diffusion-samples") + 1
    forwarded = list(original)
    forwarded[index] = "10"
    if forwarded != parameters["forwarded_arguments"]:
        raise ValueError("Only samples1 to10 may modify the original forwarded arguments")
    parsed = vars(parse_amp_args(forwarded))
    if parsed != parameters["parsed_driver_arguments"]:
        raise ValueError("Actual samples10 parser disagrees with fixed parameters")
    report["driver_argv"] = forwarded
    report["parsed_driver_arguments"] = parsed
    report["deployment_contract"]["independent_samples10_extension"] = {
        "schema": "default_NEW.single_variable_samples10.v1",
        "only_method_variable": {"diffusion_samples": {"before": 1, "after": 10}},
        "entry": record(__file__), "parameters": record(PARAMETERS),
        "production_source185_unchanged": True,
        "author_exact_Nav_hyperparameter": False,
        "candidate_selection": "Existing same-scene common sample-index minimization; no new metric or selection formula",
        "physical_effect_verified_by_this_entry": False,
    }
    if args.preflight_only:
        print(json.dumps(report, indent=2))
        return 0
    deployment.require_fresh_sdk_process()
    for name in deployment.THREAD_ENVIRONMENT:
        os.environ[name] = "4"
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "deployment_entry.json").open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    result = deployment.invoke_existing_driver(
        forwarded, evaluation_actor_request=report["evaluation_actor_request"],
        native_scene_request=report["native_scene_request"])
    return 0 if result is None else result


if __name__ == "__main__":
    raise SystemExit(main())
