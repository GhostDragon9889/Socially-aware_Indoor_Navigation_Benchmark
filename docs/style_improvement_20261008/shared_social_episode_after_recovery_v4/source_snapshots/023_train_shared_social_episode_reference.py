#!/usr/bin/env python3
"""Explicit episode-bound procedural Social AMP entry, fresh Pop-Art training only.

Reuses the original control, collector, expert and cleanup. Only a caller-owned
probe namespace selects the separately derived World. No SDK class is patched.
GPU execution requires externally observed completion of the entire S3 owner.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import train_shared_social as canonical  # noqa: E402

HERE = ROOT / "docs/style_improvement_20261006/shared_social_episode_reference_runtime_preparation_v1"
WORLD = HERE / "shared_group_world.py"
CURRICULUM = ROOT / "CrowdSim/humanoid/training/episode_mixed_trajectories.py"
REFERENCE = ROOT / "CrowdSim/humanoid/training/episode_trajectories.py"
REFERENCE_SHA = "f4283656f29a31547905ee640316aac63e7ba2cdbcd9321f1763475247a17576"
PARENT_WORLD_SHA = "e76182d2be5a3fa4dad19a04d71275d91929ce3ea2d63a1c23f081aa1a59880b"
WORLD_SHA = "635bd6183404141502bca82f6e7e3143c223a75c64c5cd080a1b3eb89031b360"
CURRICULUM_SHA = "61ec4446472800c4c6ed922194cb42f8d6988cabe732853d865f5386d4793e1a"
CANONICAL_SHA = "a1247911b78cc9adce2ee67da42e2afefb43c785fcdbd8afe02e97b1b572cacc"
PROFILE_SHAS = {"smoke.json": "df9ba4aae81226a9daf9d89aa7b249e4841291b9fbd07212aeec89b77714d7d9",
    "smoke256.json": "5c619d1bc0ac9f0cba3391eb45f0c77a3224de8523a29a76d418cea9baf65738",
    "training.json": "0d00173b490d18f3c38f75934c6f73a16641b3cf6e7f64dac5bb543955cd4a7b"}
COMPLETION = ROOT / "docs/style_improvement_20261005/shared_social_budget12500_v1/external_owner_completion.json"
BRIDGE_SHA = "a9eccb65600be8b2f3f057ae8b13df297c36da405922398c84f55886e4bcd2a9"


def pinned(path, expected, *, readonly=False):
    row = canonical.record(path)
    if row["sha256"] != expected or readonly and Path(path).stat().st_mode & 0o222:
        raise ValueError("Exact episode-reference source/profile changed: " + str(path))
    return row


def world_sources():
    """Cold-read only the pinned dependency declaration, without importing Torch."""
    pinned(WORLD, WORLD_SHA, readonly=True)
    statements = [node for node in ast.parse(WORLD.read_bytes()).body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "PINS" for target in node.targets)]
    if len(statements) != 1:
        raise ValueError("Require the one explicit World dependency declaration")
    namespace = {"ROOT": ROOT, "STAGES": ROOT / "docs/style_improvement_20261004",
                 "LAB": ROOT.parent / "IsaacLab-Protomotions/source"}
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(WORLD), "exec"), namespace)
    result = {}
    for name, (path, size, sha, readonly) in namespace["PINS"].items():
        row = pinned(path, sha, readonly=readonly)
        if row["bytes"] != size:
            raise ValueError("World dependency size differs: " + name)
        result[name] = row
    return result


def replace_fresh_curriculum(corpus, original, runtime, device, probe, records, report, torch):
    """Prepare the exact new source type before any physical World is created."""
    if bool(original.amass.initialized.any()) or bool(original.proc_initialized.any()) or any(original.statistics.values()):
        raise ValueError("Replace only an unused fresh curriculum before World creation")
    source_rng, global_rng = corpus.generator.get_state().clone(), torch.random.get_rng_state().clone()
    probe._load(REFERENCE, "_shared_social_original_training.episode_trajectories", records["episode_reference_primitive"])
    implementation = probe._load(CURRICULUM, "_shared_social_original_training.episode_reference_curriculum",
                                 records["episode_reference_curriculum"])
    candidate = implementation.EpisodeAnchoredMixedCurriculum(corpus, runtime["actors"], episode_controls=runtime["episode_controls"], device=device,
        bounds_xy=None, tail_policy="linear_velocity", procedural_config=original.procedural.config, seed=runtime["seed"])
    if (not torch.equal(corpus.generator.get_state(), source_rng) or not torch.equal(torch.random.get_rng_state(), global_rng)
            or not torch.equal(candidate.private_corpus.generator.get_state(), original.private_corpus.generator.get_state())
            or not torch.equal(candidate.procedural.generator.get_state(), original.procedural.generator.get_state())):
        raise ValueError("Fresh curriculum replacement consumed or changed source RNG")
    report["actual_import_origins"]["episode_reference_curriculum"] = probe._origin(
        implementation.EpisodeAnchoredMixedCurriculum, "episode_reference_curriculum", records)
    report["episode_reference_experiment"]["actual_reference_configuration"] = candidate.describe()["config"]
    return candidate


def prepare_entry():
    pinned(Path(canonical.__file__), CANONICAL_SHA)
    world, curriculum = pinned(WORLD, WORLD_SHA, readonly=True), pinned(CURRICULUM, CURRICULUM_SHA)
    reference = pinned(REFERENCE, REFERENCE_SHA)
    binding = {"entry": canonical.record(Path(__file__)), "World": world, "curriculum": curriculum,
        "canonical_entry": canonical.record(canonical.__file__), "World_dependencies": world_sources(), "reference_primitive": reference,
        "change": {"procedural_reference_ownership": ["periodic_live_root", "episode_bound"], "AMASS_reset_future_frames": 31, "training_clip_filtering": False,
                   "expert_sampling_rule_changed": False, "physical_control_reset_cleanup_changed": False},
        "binding_stage": "CPU_preflight_before_training",
        "runtime_smoke_verified": False, "training_executed": False, "paper_effect_achieved": False}
    module, canonical_binding = canonical.prepare_entry("popart")
    binding["canonical_binding"] = canonical_binding
    parent_sources, parent_load = module._sources, module._load
    parent_setup, parent_run, parent_profile = module._runner_setup, module._run, module._profile

    def sources():
        if canonical.record(Path(__file__)) != binding["entry"]:
            raise ValueError("Episode-reference entry changed after source admission")
        if (pinned(WORLD, WORLD_SHA, readonly=True) != world or pinned(CURRICULUM, CURRICULUM_SHA) != curriculum
                or pinned(REFERENCE, REFERENCE_SHA) != reference):
            raise ValueError("Episode-reference World/curriculum changed")
        rows = parent_sources()
        rows["canonical_training_entry"] = rows["entry"]
        rows.update(entry=binding["entry"], episode_reference_World=world, episode_reference_curriculum=curriculum, episode_reference_primitive=reference)
        return rows

    def load(path, name, row):
        loaded = parent_load(path, name, row)
        if Path(path) == module.PINS["probe_setup"][0]:
            if loaded.PINS["World"][2] != PARENT_WORLD_SHA:
                raise ValueError("Caller probe does not select the original World basis")
            loaded.PINS = {**loaded.PINS,
                "World": (WORLD, world["bytes"], world["sha256"], True),
                "episode_reference_curriculum": (CURRICULUM, curriculum["bytes"], curriculum["sha256"], False),
                "episode_reference_primitive": (REFERENCE, reference["bytes"], reference["sha256"], False)}
        return loaded

    def setup(args, runtime, probe, records, report, torch):
        values = list(parent_setup(args, runtime, probe, records, report, torch))
        values[3] = replace_fresh_curriculum(values[2], values[3], runtime, args.device, probe, records, report, torch)
        return tuple(values)

    def run(args, profile, profile_record, report, rows):
        report["episode_reference_experiment"] = {"binding": binding,
            "caller_probe_namespace_World_selection_changed": True, "original_source_files_modified": False,
            "author_recipe_equivalence_claimed": False}
        return parent_run(args, profile, profile_record, report, rows)

    def profile(path):
        path = Path(path).resolve()
        if path.parent != HERE or path.name not in PROFILE_SHAS:
            raise ValueError("Choose an exact registered episode-reference smoke/smoke256/training profile")
        pinned(path, PROFILE_SHAS[path.name], readonly=True)
        return parent_profile(path)

    module._sources, module._load, module._runner_setup = sources, load, setup
    module._run, module._profile = run, profile
    return module, binding


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--check-sources", action="store_true")
    checks.add_argument("--check-profile", action="store_true")
    checks.add_argument("--check-ready", action="store_true", help="Cold-read whole S3 owner completion without starting SDK")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    args = parser.parse_args(argv)
    module, binding = prepare_entry()
    sources = module._sources()
    if args.config is not None:
        profile, profile_identity = module._profile(args.config)
    elif args.check_profile or not args.check_sources:
        parser.error("--config is required")
    if args.check_sources or args.check_profile:
        print(json.dumps({"status": "episode_reference_CPU_preflight_passed", "binding": binding,
            "entry_sources": sources, "profile": profile_identity if args.config else None,
            "Torch_imported": "torch" in sys.modules, "SDK_started": False, "runtime_ready": False}))
        return 0
    if args.output is None and not args.check_ready:
        parser.error("--output is required")
    bridge_path = ROOT / "CrowdSim/humanoid/application/native_execution_bridge.py"
    pinned(bridge_path, BRIDGE_SHA)
    from CrowdSim.humanoid.application import native_execution_bridge as bridge
    try:
        completion = bridge.admit_S3_owner_completion(COMPLETION)
    except FileNotFoundError as error:
        if Path(error.filename) != COMPLETION:
            raise
        print(json.dumps({"status": "waiting_for_external_S3_owner_completion", "certificate": str(COMPLETION),
            "Torch_imported": "torch" in sys.modules, "SDK_started": False, "runtime_ready": False}))
        return 1
    if args.check_ready:
        print(json.dumps({"status": "episode_reference_serial_owner_admitted", "completion": completion,
                          "Torch_imported": "torch" in sys.modules, "SDK_started": False}))
        return 0
    saved = sys.argv
    try:
        sys.argv = [str(Path(__file__)), "--config", str(args.config), "--output", str(args.output), "--device", args.device]
        return module.main()
    finally:
        sys.argv = saved


if __name__ == "__main__":
    raise SystemExit(main())
