"""Unexecuted social-only evaluation derived from the frozen physical driver.

No Torch/SDK import at module import. Run only after the registered primary
Diffusion comparison and serial network/SDK gates. No training or native scene.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import random
import stat
import sys
from types import ModuleType, SimpleNamespace

from protomotions.repository import repository_root
ROOT = repository_root(__file__)
DRIVER = ROOT / "scripts/train_paper_amp.py"
HERE = Path(__file__).resolve().parent
MANIFEST = ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json"
SCENE = ROOT / "configs/diffusion_amp/shared_obstacle_timed_validation.yaml"
MANIFEST_SHA = "68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca"
SCENE_SHA = "7971206da6fd50d9981dafd068b6005f326c6e7b6a8768e5a717aeb517573e9f"
PINS = {
    "driver": ("scripts/train_paper_amp.py", 73954, "d8b85401ace1da7d0039dcad4e023590577944e487789e742019c363a602e3d3"),
    "deployment": ("CrowdSim/humanoid/application/deployment.py", 15526, "37f6f81538ae422bc8fa2505ce219c926b8252844cc1a0883c5d39620e42e8f0"),
    "configuration": ("CrowdSim/humanoid/application/configuration.py", 29717, "64dfa8f399fb4d7731ed24b66f8b5db1d96d9e879d0d72c0a03f4cd747912456"),
    "resources": ("CrowdSim/humanoid/application/resources.py", 15989, "3268a10882931cb5ad0e23506b71fd22fb21d975e9aba948e0a08fb427132a5b"),
    "world": ("CrowdSim/humanoid/amp_world.py", 30450, "1b63a4c8ac7d4bf2f9ed523edc97b41dca89888cdd6c7dcb2234826d240d7f83"),
    "loader": ("resources/runtime_sources/style_improvement_20261004/social_checkpoint_inference_staging_v1/checkpoint_inference.py", 16281, '875d9fd21af190cd0405b90d214aad71d1dffcb4678661ce7e187e61c91dc954'),
    "producer": ("resources/runtime_sources/style_improvement_20261004/social_evaluation_observation_staging_v1/observation_producer.py", 17810, 'e1c0eba40a895c4bcfe06c3fb1823ad961f29ca4e29cd922b34df021d715d4f0'),
}


def _identity(path, *, readonly=False, expected_sha=None):
    path = Path(path).absolute()
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or (readonly and before.st_mode & 0o222):
        raise ValueError("Expected an ordinary single-link" + (" readonly" if readonly else "") + " file: " + str(path))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("File identity changed before read")
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        after_open = os.fstat(stream.fileno())
    after = path.lstat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode", "st_nlink")
    if any(tuple(getattr(value, field) for field in fields) != tuple(getattr(before, field) for field in fields)
           for value in (opened, after_open, after)):
        raise ValueError("File changed during read")
    result = {"path": str(path), "bytes": before.st_size, "sha256": digest.hexdigest(),
              "device": before.st_dev, "inode": before.st_ino, "nlink": before.st_nlink,
              "mode": oct(stat.S_IMODE(before.st_mode))}
    if expected_sha is not None and result["sha256"] != expected_sha:
        raise ValueError("Selected full SHA differs: " + str(path))
    return result


def _sources():
    result = {}
    for key, (name, length, digest) in PINS.items():
        if length is None or digest is None:
            raise RuntimeError("Observation source is not yet reviewed and registered")
        record = _identity(ROOT / name, readonly=key in {"loader", "producer"}, expected_sha=digest)
        if record["bytes"] != length:
            raise ValueError("Source length changed: " + name)
        result[key] = record
    result["entry"] = _identity(Path(__file__), readonly=True)
    return result


def _load(name, record):
    module = ModuleType(name)
    module.__file__ = record["path"]
    source = Path(record["path"]).read_bytes()
    if len(source) != record["bytes"] or hashlib.sha256(source).hexdigest() != record["sha256"]:
        raise ValueError("Helper changed before execution")
    exec(compile(source, record["path"], "exec"), module.__dict__)
    return module


def _profile(path):
    record = _identity(path, readonly=True)
    if Path(path).absolute() != HERE / "evaluation_profile.json":
        raise ValueError("Use this entry's independent evaluation-only profile")
    document = json.loads(Path(path).read_text())
    expected = {"self": 357, "task": 12308, "grid_center": "root", "grid_heading": "root",
                "height_reference": "static_center_ground_z", "include_ego": False}
    if (set(document) != {"schema", "mode", "training", "policy_observations", "scope", "terrain_sampler"}
            or document["schema"] != "local.neutral_smpl.social_navigation_evaluation_profile.v1"
            or document["mode"] != "paper_method_reconstruction"
            or document["training"] != {"expert_dof_velocity_frame": "child_joint"}
            or document["policy_observations"] != expected):
        raise ValueError("Social evaluation profile contract differs")
    return document, record


def _training_run(arguments, policy, world):
    """Require the real completed shared-training run, not synthetic NN counters."""
    receipt = _identity(arguments.training_result, readonly=True, expected_sha=arguments.training_result_sha256)
    report = json.loads(Path(receipt["path"]).read_text())
    metadata = policy.describe()
    if (report.get("schema") != "local.neutral_smpl.shared_social_training_entry.v1"
            or report.get("status") != "passed_local_training"
            or report.get("training_loop_completed") is not True or report.get("cleanup_succeeded") is not True
            or report.get("source_profile_before_after_equal") is not True
            or report.get("completed_updates") != metadata["update_count"]
            or report.get("actual_training_transitions") != metadata["physical_transition_count"]
            or json.dumps(report.get("actual_configuration"), sort_keys=True, allow_nan=False)
               != json.dumps(policy.config.to_document(), sort_keys=True, allow_nan=False)
            or report.get("manifest", {}).get("sha256") != MANIFEST_SHA):
        raise ValueError("Selected policy lacks its exact completed physical training receipt")
    matching = [entry for entry in report.get("artifacts", [])
                if entry.get("path") == metadata["checkpoint"]["path"]
                and entry.get("sha256") == metadata["checkpoint"]["sha256"]
                and entry.get("bytes") == metadata["checkpoint"]["bytes"]]
    if len(matching) != 1:
        raise ValueError("The final selected model is absent from the real training artifact inventory")
    usd = _identity(world.simulator._robot.cfg.spawn.usd_path)
    training_assets = report.get("actual_native_assets", [])
    if (len(training_assets) != 4 or any(asset.get("USD", {}).get("sha256") != usd["sha256"]
            or asset.get("USD", {}).get("bytes") != usd["bytes"] for asset in training_assets)):
        raise ValueError("Actual evaluation USD differs from the four actual training prototypes")
    return {"actual_training_result": receipt, "actual_evaluation_USD": usd,
            "training_completion_attested": True, "quality_or_author_budget_attested": False}


def _partial(output, local, report):
    """Keep actual rows and lengths; never pad a half control or retry physics."""
    np = local.get("np")
    if np is None:
        return
    records = {}
    try:
        values = {}
        roots = local.get("physical_roots", [])
        if roots:
            values["root_pos"] = np.stack(roots)
        for name, rows in local.get("physical_evaluation", {}).items():
            if rows:
                values[name] = np.stack(rows)
        if values and not (output / "physical_states.npz").exists():
            target = output / "physical_states_partial.npz"
            with target.open("xb") as stream:
                np.savez_compressed(stream, **values)
            records["physical"] = {"path": str(target), "rows_by_channel": {key: len(value) for key, value in values.items()},
                                   "complete_frame_or_requested_budget_claimed": False, "padding_used": False}
        references = {key: np.concatenate(rows) for key, rows in local.get("reference_plan_records", {}).items() if rows}
        if references and not (output / "reference_plans.npz").exists():
            target = output / "reference_plans_partial.npz"
            with target.open("xb") as stream:
                np.savez_compressed(stream, **references)
            records["reference"] = {"path": str(target), "rows_by_channel": {key: len(value) for key, value in references.items()}}
    except BaseException as error:
        records["preservation_error"] = type(error).__name__ + ": " + str(error)
    report["actual_partial_arrays"] = records


def _derive(source):
    """Exact bounded replacements; original complete bytes round-trip first."""
    patches = []

    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise ValueError("Expected one source replacement, observed " + str(source.count(old)))
        source = source.replace(old, new)
        patches.append((old, new))

    original = source
    replace("    from CrowdSim.humanoid.application.configuration import load_amp_profile\n",
            "    load_amp_profile = _social_profile_reader\n")
    replace('    launcher_options = {"headless": True, "device": "cuda:0", "enable_cameras": args.capture}\n',
            '    launcher_options = {"headless": True, "device": "cuda:0", "enable_cameras": args.capture, "fast_shutdown": False}\n')
    replace("    app = launcher(launcher_options).app\n", "    app = _social_launch(launcher, launcher_options)\n")
    replace("        world = AMPWorld(app, corpus, num_envs=args.num_envs, history_frames=10,\n",
            "        world = _social_world_constructor(AMPWorld, app, corpus, num_envs=args.num_envs, history_frames=10,\n")
    tree = ast.parse(source)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    trial = next(node for node in main.body if isinstance(node, ast.Try))
    selected = [node for node in trial.body if isinstance(node, ast.If) and ast.unparse(node.test) == "args.resume"]
    if len(selected) != 1:
        raise ValueError("Expected the sole top-level old controller admission")
    node = selected[0]
    old = "".join(source.splitlines(keepends=True)[node.lineno - 1:node.end_lineno])
    replace(old, "        trainer, social_producer = _social_admission(world, report, torch, np)\n        saved = None\n")
    replace("                state = world.state()\n                self_obs, task_obs = world.observations(task_trajectory(), state)\n",
            "                social_packet = social_producer.observe(task_trajectory())\n"
            "                state = social_packet.state\n"
            "                self_obs, task_obs = social_packet.raw_self, social_packet.raw_task\n"
            "                _social_record(output, social_packet, report, control_step)\n")
    replace("                world.reset(ids)\n                if len(ids):\n",
            "                world.reset(ids)\n                if len(ids):\n                    social_producer.begin_episode(ids)\n")
    replace('        report.update(status="passed", update_count=trainer.update_count,\n                      physical_transition_count=trainer.physical_transition_count,\n',
            '        report.update(status="passed", update_count=trainer.describe()["update_count"],\n'
            '                      physical_transition_count=trainer.describe()["physical_transition_count"],\n')
    replace("    except Exception as error:\n        report.update(status=\"failed\", error=f\"{type(error).__name__}: {error}\", traceback=traceback.format_exc())\n",
            "    except BaseException as error:\n        report.update(status=\"failed\", error=f\"{type(error).__name__}: {error}\", traceback=traceback.format_exc())\n"
            "        _social_partial(output, locals(), report)\n")
    old = ('    finally:\n        (output / "result.json").write_text(json.dumps(report, indent=2, default=lambda value: getattr(value, "value", str(value)))+"\\n")\n'
           '        print(json.dumps({key:report.get(key) for key in ("status", "update_count", "physical_transition_count", "error")}), flush=True)\n'
           '        if world is not None:\n            world.simulator._shutdown_exit_code = 0 if report["status"] == "passed" else 1\n'
           '            world.close()\n        else:\n            app.close()\n')
    replace(old, "    finally:\n        _social_finish(output, report, world, app)\n")
    reversed_source = source
    for old, new in reversed(patches):
        if reversed_source.count(new) != 1:
            raise ValueError("Derived replacement is not uniquely reversible")
        reversed_source = reversed_source.replace(new, old)
    if reversed_source != original:
        raise ValueError("Derived driver did not restore the complete original bytes")
    before, after = ast.parse(original), ast.parse(source)
    before_functions = {node.name: ast.dump(node, include_attributes=False) for node in before.body
                        if isinstance(node, ast.FunctionDef) and node.name != "main"}
    after_functions = {node.name: ast.dump(node, include_attributes=False) for node in after.body
                       if isinstance(node, ast.FunctionDef) and node.name != "main"}
    if before_functions != after_functions:
        raise ValueError("An untouched driver helper changed")
    return source, {"replacement_count": len(patches), "whole_source_reverse_equal": True,
                    "untouched_function_AST_equal": True, "derived_sha256": hashlib.sha256(source.encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--amp-checkpoint", type=Path, required=True)
    parser.add_argument("--amp-sha256", required=True)
    parser.add_argument("--training-result", type=Path, required=True)
    parser.add_argument("--training-result-sha256", required=True)
    parser.add_argument("--diffusion-checkpoint", type=Path, required=True)
    parser.add_argument("--diffusion-sha256", required=True)
    parser.add_argument("--control-steps", type=int, choices=(4, 30, 6000), default=6000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    for digest in (args.amp_sha256, args.training_result_sha256, args.diffusion_sha256):
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            parser.error("Every selected artifact requires its explicit lowercase full SHA256")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    sources = _sources()
    if any(name in {"isaacsim", "isaaclab", "omni.kit.app"} or name.startswith(("isaacsim.", "isaaclab.")) for name in sys.modules):
        raise RuntimeError("Run in a fresh SDK-free process")
    output = args.output.absolute()
    if output.resolve() != output:
        parser.error("Output must have a canonical path without symlink ancestors or '..'")
    if output.exists() or output.is_symlink() or not any(output.is_relative_to(root) and output != root for root in (ROOT / "output", Path("/tmp"))):
        parser.error("Use a new output child inside this repository's output or /tmp")
    profile, profile_identity = _profile(HERE / "evaluation_profile.json")
    selected = {"policy": _identity(args.amp_checkpoint, readonly=True, expected_sha=args.amp_sha256),
                "training_result": _identity(args.training_result, readonly=True, expected_sha=args.training_result_sha256),
                "diffusion": _identity(args.diffusion_checkpoint, readonly=True, expected_sha=args.diffusion_sha256),
                "manifest": _identity(MANIFEST, readonly=True, expected_sha=MANIFEST_SHA),
                "scene": _identity(SCENE, expected_sha=SCENE_SHA)}
    from CrowdSim.humanoid.application.deployment import driver_arguments, duration_control_steps
    from CrowdSim.humanoid.application.configuration import parse_amp_args
    from CrowdSim.humanoid.application.resources import preflight_amp_resources
    request = SimpleNamespace(config=HERE / "evaluation_profile.json", manifest=MANIFEST,
        diffusion_checkpoint=args.diffusion_checkpoint, output=output, actors=4, seed=7,
        control_steps=args.control_steps, native_dynbench=False, amp_checkpoint=args.amp_checkpoint, scene_config=SCENE)
    forwarded = driver_arguments(request)
    if duration_control_steps(str(args.control_steps / 30)) != args.control_steps:
        raise ValueError("Control clock is not representable")
    driver_args = parse_amp_args(forwarded)
    preflight = preflight_amp_resources(driver_args, profile)
    source, derivation = _derive(DRIVER.read_text())
    preparation = {"schema": "local.neutral_smpl.social_four_actor_navigation_evaluation.v1",
        "sources": sources, "selected_artifacts": selected, "evaluation_profile": profile_identity,
        "derivation": derivation, "resource_preflight": preflight.as_dict(), "driver_arguments": forwarded,
        "requested_controls": args.control_steps, "requested_actor_controls": args.control_steps * 4,
        "primary_200_seconds": args.control_steps == 6000, "physical_execution_verified": False}
    if args.preflight_only:
        print(json.dumps(preparation, indent=2, default=str))
        return 0
    owned = {"app": None, "world": None, "app_launch_attempted": False,
             "world_construction_attempted": False, "sim_type": None, "finish_called": False}

    def launch(launcher, options):
        owned["app_launch_attempted"] = True
        owned["app"] = launcher(options).app
        return owned["app"]

    def construct_world(factory, *positional, **keywords):
        from isaaclab.sim import SimulationContext
        if SimulationContext.instance() is not None:
            raise RuntimeError("Refuse a pre-existing SimulationContext before the sole world constructor")
        owned["sim_type"] = SimulationContext
        owned["world_construction_attempted"] = True
        owned["world"] = factory(*positional, **keywords)
        return owned["world"]

    def admission(world, report, torch, np):
        loader = _load("_social_navigation_checkpoint", sources["loader"])
        policy = loader.FrozenSocialCheckpointInference.from_checkpoint(args.amp_checkpoint,
            expected_sha256=args.amp_sha256, device=world.device)
        report["social_training_admission"] = _training_run(args, policy, world)
        config = policy.config
        if (config.expert_dof_velocity_frame != "child_joint" or not config.freeze_toes
                or world.num_envs != 4 or not world.shared_world or world.scene_provider is not None):
            raise ValueError("Only the original four-person cuboid lifecycle and canonical child-frame neutral policy are supported")
        module = _load("_social_navigation_observations", sources["producer"])
        producer = module.RawSocialEvaluationObservations.from_world(world,
            actor_ids=torch.arange(4, dtype=torch.int64, device=world.device),
            scene_ids=torch.zeros(4, dtype=torch.int64, device=world.device),
            terrain_sampler=world.height_at, terrain_sampler_semantics=profile["terrain_sampler"],
            policy_description=policy.describe())
        producer.begin_episode()
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        report["evaluation_actor"] = policy.describe()
        report["evaluation_actor"]["runtime_deployment_changed"] = True
        report["social_evaluation_observation"] = producer.describe()
        report["evaluation_rng_preparation"] = {
            "seed": 7, "boundary": "after social actor admission, before Diffusion planner construction",
            "global_Python_NumPy_Torch_reseeded": True, "legacy_MLP_initializer_draws_performed": False,
            "training_saved_RNG_restored": False, "old_whole_run_bitwise_identity_claimed": False}
        report["evaluation_training_components"] = {"optimizer_constructed": False, "replay_constructed": False,
            "critic_evaluated": False, "discriminator_evaluated": False,
            "temporary_full_network_constructed_by_checkpoint_loader": True,
            "physical_control_loop": "original source with explicit social observation replacements"}
        report["training_counter_source"] = "validated model declaration matched to the completed physical training receipt; independent of this evaluation's actual controls"
        report["social_observation_capture"] = {"first_inputs": None, "diagnostic_rows": 0,
            "raw_12308_inputs_archived_every_control": False, "all_diagnostics_phase": "pre-action current owned COMMON state"}
        return policy, producer

    def record_social(path, packet, report, control):
        import numpy as np
        captured = report["social_observation_capture"]
        if captured["diagnostic_rows"] != control:
            raise ValueError("Social pre-action observation clock differs from original control clock")
        if control == 0:
            target = path / "social_initial_inputs.npz"
            with target.open("xb") as stream:
                np.savez_compressed(stream, raw_self=packet.raw_self.detach().cpu().numpy(),
                    raw_task=packet.raw_task.detach().cpu().numpy(), owners=packet.owners.detach().cpu().numpy())
            captured["first_inputs"] = str(target)
        diagnostic = packet.diagnostics
        def plain(value):
            if hasattr(value, "detach"):
                return value.detach().cpu().tolist()
            if isinstance(value, dict):
                return {key: plain(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [plain(item) for item in value]
            return value
        row = {"control_step": control, "producer": plain(diagnostic)}
        with (path / "social_observations.jsonl").open("x" if control == 0 else "a") as stream:
            json.dump(row, stream, allow_nan=False)
            stream.write("\n")
        captured["diagnostic_rows"] += 1

    def finish(path, report, world, app):
        owned["finish_called"] = True
        report["social_entry"] = preparation
        report["app_launch_attempted"] = owned["app_launch_attempted"]
        report["app_constructor_returned"] = owned["app"] is not None
        report["world_construction_attempted"] = owned["world_construction_attempted"]
        errors = []
        try:
            if _sources() != sources or _profile(HERE / "evaluation_profile.json")[1] != profile_identity:
                raise ValueError("Execution source or profile changed")
            for role, identity in selected.items():
                if _identity(identity["path"], readonly=role != "scene", expected_sha=identity["sha256"]) != identity:
                    raise ValueError("Selected artifact changed: " + role)
            report["social_sources_and_artifacts_unchanged"] = True
        except BaseException as error:
            report["social_sources_and_artifacts_unchanged"] = False
            errors.append("final identity: " + type(error).__name__ + ": " + str(error))
        sim = None
        try:
            sim = (world.simulator._sim if world is not None else owned["sim_type"].instance()
                   if owned["world_construction_attempted"] and owned["sim_type"] is not None else None)
        except BaseException as error:
            errors.append("context lookup: " + type(error).__name__ + ": " + str(error))
        if sim is not None:
            try:
                sim.stop()
                report["simulation_stop_returned"] = True
            except BaseException as error:
                errors.append("sim.stop: " + type(error).__name__ + ": " + str(error))
        if app is not None:
            try:
                app.close()
                report["app_close_returned"] = True
            except BaseException as error:
                errors.append("app.close: " + type(error).__name__ + ": " + str(error))
        report["cleanup_errors"] = errors
        if errors or not report.get("simulation_stop_returned") or not report.get("app_close_returned"):
            report["status"] = "failed"
        if report["status"] == "passed" and report["social_observation_capture"]["diagnostic_rows"] != args.control_steps:
            report["status"] = "failed"
            report["error"] = "Missing requested pre-action social diagnostic rows"
        report["social_entry"]["physical_execution_verified"] = report["status"] == "passed"
        nonfinite = []
        def finite_json(value, location="report"):
            if isinstance(value, float) and not math.isfinite(value):
                nonfinite.append(location)
                return str(value)
            if isinstance(value, dict):
                return {key: finite_json(item, location + "." + str(key)) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [finite_json(item, location + "." + str(index)) for index, item in enumerate(value)]
            return value
        prepared = finite_json(report)
        if nonfinite:
            prepared["nonfinite_report_fields_preserved_as_strings"] = nonfinite
            prepared["status"] = report["status"] = "failed"
            prepared["social_entry"]["physical_execution_verified"] = False
        with (path / "result.json").open("x") as stream:
            json.dump(prepared, stream, indent=2, default=lambda value: getattr(value, "value", str(value)), allow_nan=False)
            stream.write("\n")
        print(json.dumps({"status": report["status"], "actual_control_steps": report.get("actual_control_steps")}))

    namespace = {"__name__": "_social_four_actor_derived_driver", "__file__": str(DRIVER),
        "_social_profile_reader": lambda path: _profile(path)[0], "_social_admission": admission,
        "_social_record": record_social, "_social_partial": _partial, "_social_finish": finish,
        "_social_launch": launch, "_social_world_constructor": construct_world}
    exec(compile(source, str(DRIVER), "exec"), namespace)
    namespace["parse_args"] = lambda: driver_args
    try:
        return namespace["main"](evaluation_actor_factory=lambda *unused, **ignored: None)
    except BaseException as error:
        # App construction and early Torch imports precede the original try.
        # Dispose only objects this invocation actually owns, without retries.
        if not output.exists():
            output.mkdir(parents=True)
        failure = {"status": "failed", "error": type(error).__name__ + ": " + str(error),
                   "failure_phase": "entry or early original-driver failure", "original_finalizer_called": owned["finish_called"]}
        if not owned["finish_called"]:
            finish(output, failure, owned["world"], owned["app"])
        else:
            with (output / "entry_failure.json").open("x") as stream:
                json.dump(failure, stream, indent=2, allow_nan=False)
                stream.write("\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
