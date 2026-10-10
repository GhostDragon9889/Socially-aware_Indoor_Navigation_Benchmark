"""Actual private-source ABI on fixed2848 CPU data; no physical World/NN."""
import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts import train_shared_social_episode_reference as entry  # noqa: E402
from scripts.diagnose_shared_social_training import identity  # noqa: E402

HERE = Path(__file__).resolve().parent


def check(condition, message):
    if not condition:
        raise ValueError(message)


def same_reference(left, right, torch):
    for index in (0, 1):
        for name, value in vars(left[index]).items():
            check(torch.equal(value, getattr(right[index], name)), "Reference field differs: " + name)
    for index in (2, 3, 4):
        check(torch.equal(left[index], right[index]), "Reference next/ID/frame differs")


check(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not (HERE / "actual_entry_probe.json").exists(), "Fresh CPU probe required")
module, binding = entry.prepare_entry()
check("torch" not in sys.modules, "Cold source preparation imported Torch")
rows = module._sources()
probe = module._load(module.PINS["probe_setup"][0], "_episode_reference_actual_CPU_probe", rows["probe_setup"])
records = probe._sources()
check(records["World"] == binding["World"] and records["episode_reference_curriculum"] == binding["curriculum"], "Caller-owned source selection differs")
manifest = ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json"
manifest_record = identity(manifest)
check(manifest_record["sha256"] == "68d41be28908f28a09c47f7658af39d55fad01b32212bd7c0ced2b1e6b1489ca", "Fixed2848 manifest changed")
registered = json.loads(manifest.read_text())["splits"]["train"]
bank = Path(registered["path"])
bank = bank if bank.is_absolute() else manifest.parent / bank
bank_record = identity(bank)
check(bank_record["sha256"] == registered["sha256"] == "46e0241108e200409c08d1115a270205fe0b6ce827e366a06e8a09f6a2cc82c0"
    and bank_record["bytes"] == registered["bytes"] and registered["clip_count"] == len(registered["clips"]) == 2848, "Train bank registry changed")
inputs = [manifest_record, bank_record]
sources = {row["path"]: row for row in [identity(__file__), *rows.values(), *records.values(), *binding["World_dependencies"].values()]}
guards = json.loads((ROOT / "docs/style_improvement_20261005/shared_social_budget12500_v1/terminal_owner_state.json").read_text())["bindings"]["guarded_sources"]
check(len(guards) == 23 and all(identity(row["path"]) == row for row in guards), "Active S3 source changed")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from CrowdSim.humanoid.amass_corpus import AMASSCorpus  # noqa: E402

torch.set_num_threads(4)
check(not torch.cuda.is_initialized(), "CPU data probe initialized CUDA")
global_rng = torch.random.get_rng_state().clone()
world = probe._load(entry.WORLD, "_episode_reference_actual_CPU_World", records["World"])
dependencies = world._dependencies()
check((world.DT, world.SUBSTEPS, world.HISTORY, world.REPLAN) == (1 / 120, 4, 10, 15)
    and callable(dependencies.action_to_pd), "Original control/history/PD changed")
tree = ast.parse(entry.WORLD.read_bytes())
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SharedGroupWorld")
method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
assignments = {node.targets[0].id: node for node in ast.walk(method) if isinstance(node, ast.Assign)
    and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("curriculum_module", "origin")}
conditions = [next(node for node in ast.walk(method) if isinstance(node, ast.If) and text in ast.unparse(node.test))
              for text in ("EpisodeAnchoredMixedCurriculum", "curriculum.amass.initialized")]
guard = ast.Module(body=[deepcopy(assignments[key]) for key in ("curriculum_module", "origin")]
                   + [deepcopy(node) for node in conditions], type_ignores=[])
cohorts, artifacts = {}, []
for profile_name in ("smoke.json", "training.json"):
    profile, profile_record = module._profile(HERE / profile_name)
    inputs.append(profile_record)
    runtime = profile["runtime"]
    count, cap = runtime["actors"], runtime["episode_controls"]
    package = ModuleType("_shared_social_original_training")
    package.__path__ = [str(ROOT / "CrowdSim/humanoid/training")]
    check(package.__name__ not in sys.modules, "Refuse an existing private training package")
    sys.modules[package.__name__] = package
    for key, filename in (("procedural", "trajectories"), ("amass", "amass_trajectories"), ("mixed", "mixed_trajectories")):
        row = binding["World_dependencies"][key]
        probe._load(Path(row["path"]), package.__name__ + "." + filename, row)
    corpus = AMASSCorpus(manifest, device="cpu", seed=7, dof_velocity_frame="child_joint")
    check(corpus.ids.tolist() == list(range(2848)), "AMASS IDs filtered or changed")
    for clip in registered["clips"]:
        mid = clip["motion_id"]
        check(int(corpus.data["motion_num_frames"][mid]) == clip["frames"]
            and corpus.data["motion_files"][mid] == clip["source_path"]
            and abs(float(corpus.data["motion_dt"][mid]) - 1 / 30) < 1e-7, "Actual source frames/rate differ")
    original = sys.modules[package.__name__ + ".mixed_trajectories"].MixedTrajectoryCurriculum(corpus, count,
        procedural_config=sys.modules[package.__name__ + ".trajectories"].PacerTrajectoryConfig(), seed=7)
    report = {"actual_import_origins": {}, "episode_reference_experiment": {}}
    candidate = entry.replace_fresh_curriculum(corpus, original, runtime, "cpu", probe, records, report, torch)
    scope = {**world.SharedGroupWorld.__init__.__globals__, "self": SimpleNamespace(num_envs=count, device=torch.device("cpu")),
             "curriculum": candidate, "episode_steps": cap}
    exec(compile(guard, str(entry.WORLD), "exec"), scope)
    ids, episodes, ages = torch.arange(count), torch.ones(count, dtype=torch.long), torch.zeros(count, dtype=torch.long)
    caller_rng = corpus.generator.get_state().clone()
    baseline = original.sample_reference(ids)
    caller_after = corpus.generator.get_state().clone()
    corpus.generator.set_state(caller_rng)
    actual = candidate.sample_reference(ids)
    same_reference(baseline, actual, torch)
    check(torch.equal(caller_after, corpus.generator.get_state())
        and torch.equal(original.private_corpus.generator.get_state(), candidate.private_corpus.generator.get_state()), "Reset RNG exposure differs")
    origins = torch.stack((ids.float() * .2, ids.float() * .3), -1)
    source_xy = actual[0].root_pos[:, :2].clone()
    for curriculum in (original, candidate):
        curriculum.bind_reset(ids, actual[3], actual[4], source_xy, origins, episodes)
    state = SimpleNamespace(root_pos=actual[0].root_pos.clone(), root_vel=actual[0].root_vel.clone(), root_rot=actual[0].root_rot.clone())
    state.root_pos[:, :2] = origins
    old = original.build_plan(state, ages, episode_ids=episodes)
    initial = candidate.build_plan(state, ages, episode_ids=episodes)
    check(torch.equal(old.native_xy, initial.native_xy)
        and torch.equal(old.xy[count // 2:], candidate.episode_reference.paths[:, :51]), "Initial reference targets differ")
    checkpoint_rng = candidate.generator.get_state().clone()
    controls = 1 if cap == 4 else 15
    flags = torch.zeros(count, dtype=torch.bool)
    for _ in range(controls):
        ages += 1
        original.completed_step(ages, flags, flags)
        candidate.completed_step(ages, flags, flags)
    plan_age = torch.full((count,), controls)
    view = initial
    if cap != 4:
        state.root_pos[count // 2:, :2] += 100  # Explicit CPU counterfactual, not a captured physical state.
        old = original.build_plan(state, ages, episode_ids=episodes)
        view = candidate.build_plan(state, ages, episode_ids=episodes)
        plan_age.zero_()
        check(torch.equal(old.native_xy[:count // 2], view.native_xy[:count // 2])
            and torch.equal(checkpoint_rng, candidate.generator.get_state()), "AMASS view changed or procedural query redrew")
    future = candidate.future_targets(ages, plan_age, view.native_xy, episodes)
    check(torch.equal(future[:count // 2], original.future_targets(ages, plan_age, old.native_xy, episodes)[:count // 2]), "AMASS future changed")
    target = candidate.interpolate_reference(plan_age[:, None].float() / 30, view.native_xy)
    check(torch.equal(target[count // 2:], candidate.episode_reference.query(ages[count // 2:], episodes[count // 2:], [0.])), "POST reference clock differs")
    candidate.validate_runner_state(ages, plan_age, view.native_xy, actual[3], episodes, replan_steps=15, origins_xy=origins)
    saved = candidate.state_dict()
    restored = type(candidate)(corpus, count, episode_controls=cap,
        procedural_config=sys.modules[package.__name__ + ".trajectories"].PacerTrajectoryConfig(), seed=runtime["seed"])
    untouched = corpus.generator.get_state().clone()
    restored.load_state_dict(saved)
    check(torch.equal(corpus.generator.get_state(), untouched)
        and torch.equal(restored.future_targets(ages, plan_age, view.native_xy, episodes), future), "Reference restore or caller-owned RNG differs")
    expert_rng = corpus.generator.get_state().clone()
    left = corpus.sample_reference(4096)
    expert_after = corpus.generator.get_state().clone()
    corpus.generator.set_state(expert_rng)
    right = candidate.corpus.sample_reference(4096)
    same_reference(left, right, torch)
    check(torch.equal(expert_after, corpus.generator.get_state()), "Expert source draw order differs")
    path = HERE / (profile_name.removesuffix(".json") + "_actual_CPU.npz")
    np.savez_compressed(path, reset_motion_ids=actual[3].numpy(), reset_frames=actual[4].numpy(), reset_source_xy=source_xy.numpy(),
        declared_reset_origins_xy=origins.numpy(), initial_native_xy=initial.native_xy.numpy(), later_native_xy=view.native_xy.numpy(),
        later_future10=future.numpy(), later_reward_target_xy=target.numpy(), episode_paths=candidate.episode_reference.paths.numpy(),
        expert_motion_ids=left[3].numpy(), expert_frames=left[4].numpy())
    artifacts.append(identity(path))
    cohorts[profile_name] = {"actors": count, "physical_episode_cap_controls": cap,
        "reference_buffer_cap_controls": candidate.episode_reference.episode_controls,
        "CPU_completed_controls": controls, "all_fixed_train_clips": len(corpus.ids),
        "source_history_next_reference_and_reset_RNG_exact": True, "original_AMASS_native_and_future_exact": True,
        "initial_all_native_targets_exact": True, "initial_procedural_first51_exact": True,
        "reference_state_roundtrip_exact": True, "caller_owned_corpus_RNG_unchanged_by_restore": True,
        "expert_source_windows_exact": 4096, "procedural_path_generation_count": candidate.procedural.plans_generated,
        "procedural_plan_view_count": candidate.statistics["procedural_plan_views"], "actual_entry_report": report}
    for key in list(sys.modules):
        if key == package.__name__ or key.startswith(package.__name__ + "."):
            del sys.modules[key]
check(torch.equal(torch.random.get_rng_state(), global_rng) and not torch.cuda.is_initialized(), "Global RNG/CUDA changed")
check(not any(name.split(".")[0] in {"isaacsim", "isaaclab", "omni", "pxr", "warp"} for name in sys.modules), "SDK imported")
check(all(identity(row["path"]) == row for row in inputs + list(sources.values()) + guards), "Input/source changed during CPU probe")
result = {"schema": "shared_social.episode_reference_private_entry_CPU.v1", "status": "actual_fixed2848_CPU_interfaces_passed",
    "inputs": inputs, "sources": list(sources.values()), "artifacts": artifacts, "cohorts": cohorts,
    "World_type_source_configuration_admission_executed": True, "original_PD_dependency_loaded": True,
    "source_guards_preserved": len(guards), "global_RNG_consumed": False, "CUDA_initialized": False,
    "NN_constructed": False, "SDK_started": False, "physical_World_created": False,
    "training_executed": False, "quality_improvement_proven": False,
    "scope": "Actual private-package type/source/CPU references, with declared reset origins and root shifts; not physical simulation or full SDK resume"}
(HERE / "actual_entry_probe.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print(json.dumps({"result": identity(HERE / "actual_entry_probe.json"), "sources": len(sources), "cohorts": {
    key: {name: value[name] for name in ("actors", "physical_episode_cap_controls", "reference_buffer_cap_controls", "expert_source_windows_exact")}
    for key, value in cohorts.items()}, "SDK_started": False}, indent=2))
