"""Real2848 CPU/CUDA reference admission and clocks; no SDK or policy training."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts import train_shared_social_episode_after_recovery as entry  # noqa: E402

HERE = Path(__file__).resolve().parent
module, binding, _ = entry.prepare_training(HERE / "serial_release.json", "smoke8")
rows = module._sources()
probe = module._load(module.PINS["probe_setup"][0], "_reference_clock_actual_probe", rows["probe_setup"])
import torch  # noqa: E402
from CrowdSim.humanoid.amass_corpus import AMASSCorpus  # noqa: E402

cohorts = []
for device in ("cpu", "cuda:0"):
    for stage in ("smoke8", "pilot2000"):
        selected = next(row for row in entry.release.stages() if row["stage"] == stage)
        runtime = selected["runtime"]
        actors, cap = runtime["actors"], runtime["episode_controls"]
        package = ModuleType("_shared_social_original_training")
        package.__path__ = [str(ROOT / "CrowdSim/humanoid/training")]
        assert package.__name__ not in sys.modules
        sys.modules[package.__name__] = package
        try:
            for key, name in (("procedural", "trajectories"), ("amass", "amass_trajectories"), ("mixed", "mixed_trajectories")):
                identity = entry.original.world_sources()[key]
                probe._load(Path(identity["path"]), package.__name__ + "." + name, identity)
            corpus = AMASSCorpus(ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json",
                                 device=device, seed=7, dof_velocity_frame="child_joint")
            assert len(corpus.ids) == 2848
            source = sys.modules[package.__name__ + ".trajectories"]
            mixed_module = sys.modules[package.__name__ + ".mixed_trajectories"]
            config = source.PacerTrajectoryConfig()
            original = mixed_module.MixedTrajectoryCurriculum(corpus, actors, device=device, procedural_config=config, seed=7)
            report = {"actual_import_origins": {}, "episode_reference_experiment": {}}
            candidate = entry.original.replace_fresh_curriculum(corpus, original, runtime, device, probe, rows, report, torch)
            ids = torch.arange(actors, device=device)
            rng = corpus.generator.get_state().clone()
            reference = candidate.sample_reference(ids)
            after = corpus.generator.get_state().clone()
            corpus.generator.set_state(rng)
            baseline = original.sample_reference(ids)
            assert torch.equal(after, corpus.generator.get_state())
            for left, right in zip(reference, baseline):
                if isinstance(left, torch.Tensor):
                    assert torch.equal(left, right)
                else:
                    assert all(torch.equal(value, getattr(right, name)) for name, value in vars(left).items())
            current = reference[0]
            episodes, ages = torch.ones(actors, dtype=torch.long, device=device), torch.zeros(actors, dtype=torch.long, device=device)
            origin = torch.stack((ids.float() * 3, torch.zeros(actors, device=device)), -1)
            candidate.bind_reset(ids, reference[3], reference[4], current.root_pos[:, :2], origin, episodes)
            root = current.root_pos.clone()
            root[:, :2] = origin
            state = SimpleNamespace(root_pos=root, root_rot=current.root_rot.clone(), root_vel=current.root_vel.clone())
            first = candidate.build_plan(state, ages, episode_ids=episodes)
            assert first.native_times_s[-1].item() == 5.
            assert first.native_xy.shape == (actors, 151, 2) and torch.isfinite(first.native_xy).all()
            saved_paths, saved_rng = candidate.episode_reference.paths.clone(), candidate.generator.get_state().clone()
            for _ in range(min(15, cap)):
                ages += 1
                flags = torch.zeros(actors, dtype=torch.bool, device=device)
                candidate.completed_step(ages, flags, flags)
            state.root_pos[actors // 2:, :2] += 100
            later = candidate.build_plan(state, ages, episode_ids=episodes)
            assert torch.equal(saved_paths, candidate.episode_reference.paths)
            assert torch.equal(saved_rng, candidate.generator.get_state())
            future = candidate.future_targets(ages, torch.zeros_like(ages), later.native_xy, episodes)
            assert future.shape == (actors, 10, 2) and torch.isfinite(future).all()
            restored = type(candidate)(corpus, actors, episode_controls=cap, device=device, procedural_config=config, seed=7)
            restored.load_state_dict(deepcopy(candidate.state_dict()))
            assert torch.equal(restored.episode_reference.paths, saved_paths)
            reference_cap = candidate.episode_reference.episode_controls
            end_ages = torch.full((actors // 2,), reference_cap, dtype=torch.long, device=device)
            endpoint = candidate.episode_reference.query(end_ages, episodes[actors // 2:], [5.])
            assert torch.isfinite(endpoint).all()
            try:
                candidate.episode_reference.query(end_ages, episodes[actors // 2:], [5.000001])
            except ValueError:
                pass
            else:
                raise AssertionError("Strict five-second guard was disabled")
            cohorts.append({"device": device, "actors": actors, "physical_episode_controls": cap,
                            "actual_native_offset_endpoint_s": later.native_times_s[-1].item(),
                            "unchanged_AMASS_reset_tensors_and_RNG": True, "real_AMASS_clips": len(corpus.ids),
                            "replans_redraw_path": False, "state_restore_checked": True,
                            "strict_greater_than_five_seconds_rejected": True})
        finally:
            for name in list(sys.modules):
                if name == package.__name__ or name.startswith(package.__name__ + "."):
                    del sys.modules[name]

result = {"status": "passed_actual_CPU_CUDA_reference_clock_probe", "binding": binding,
          "cohorts": cohorts, "SDK_started": False, "training_updates": 0,
          "CUDA_initialized": torch.cuda.is_initialized(), "physics_or_quality_proven": False}
with (HERE / "reference_clock_probe_result.json").open("x") as stream:
    json.dump(result, stream, indent=2)
print(json.dumps({key: value for key, value in result.items() if key != "binding"}))
