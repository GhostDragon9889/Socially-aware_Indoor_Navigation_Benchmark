#!/usr/bin/env python3
"""Thin observer of sole existing inference driver; never a second control loop."""

from __future__ import annotations
import argparse
import importlib.util
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
# This docs-staged script locates the real workspace without editing production.
if not (ROOT / "CrowdSim/humanoid/application/deployment.py").exists():
    ROOT = next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / "CrowdSim/humanoid/application/deployment.py").exists()
    )
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from CrowdSim.humanoid.application import deployment  # noqa: E402 -- direct-source bootstrap


def main(argv=None, *, driver_main=None, observer_module=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--observation-stride", type=int, default=30)
    extra, forwarded = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if extra.observation_stride < 1:
        parser.error("Observation stride must be positive")
    args = deployment.parse_deployment_args(forwarded)
    if (
        not args.inference_only
        or args.experimental_arm_style_checkpoint is not None
        or args.arm_style_alpha != 0
    ):
        parser.error(
            "Pre-action probe requires --inference-only and one baseline actor; no arm experiment"
        )
    if driver_main is None and not args.preflight_only:
        deployment.require_fresh_sdk_process()

    def observed_main(**options):
        module = observer_module
        if module is None:
            path = (
                Path(__file__).resolve().parents[1]
                / "CrowdSim/humanoid/runtime/pre_action_observation.py"
            )
            spec = importlib.util.spec_from_file_location(
                "pre_action_observer_module", path
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        factory = options["evaluation_actor_factory"]
        writer = module.PreActionObservationWriter(
            args.output / "pre_action_observations", stride=extra.observation_stride
        )

        def observed_factory(checkpoint, *, device):
            return module.PreActionObservedActor(
                factory(checkpoint, device=device), writer
            )

        options = {**options, "evaluation_actor_factory": observed_factory}

        def source_record(path):
            path = Path(path).resolve()
            payload = path.read_bytes()
            return {
                "path": str(path),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }

        with (args.output / "pre_action_observer.json").open("x") as stream:
            json.dump(
                {
                    "schema": "paper_pre_action_observer_delegation.v1",
                    "stride": extra.observation_stride,
                    "single_original_actor_factory_call_per_world": True,
                    "observer_loaded_source_files": [
                        source_record(__file__),
                        source_record(module.__file__),
                    ],
                    "source_scope": "These two observer files only; not full import ledger",
                    "capture_and_publish_timing": "Clone raw inputs/moments and compute z before original act; publish after its return and before driver world.step",
                    "additional_model_context_or_world_created": False,
                    "extra_normalizer_forward_only": True,
                    "native_counter_source": None,
                    "runtime_completed": False,
                    "limitations": [
                        "Additional GPU copy/CPU IO may affect wall time; no PhysX/Warp bitwise guarantee.",
                        "Samples observe pre-action inputs; they do not prove clipping causes idle or failure.",
                    ],
                },
                stream,
                indent=2,
            )
        target = driver_main
        if target is None:
            deployment.require_fresh_sdk_process()
            spec = importlib.util.spec_from_file_location(
                "observed_sole_paper_driver", deployment.DRIVER
            )
            original = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(original)
            target = original.main
        return target(**options)

    return deployment.main(forwarded, driver_main=observed_main)


if __name__ == "__main__":
    raise SystemExit(main())
