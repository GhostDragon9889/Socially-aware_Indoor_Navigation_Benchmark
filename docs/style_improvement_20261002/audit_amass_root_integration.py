"""Independent CPU admission/clock/source audit of opt-in AMASS commands.

Reads the training bank only. Source-clock harnesses below are not physics
rollouts; they exercise the actual command methods and extracted runner helpers.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from CrowdSim.humanoid.amass_corpus import AMASSCorpus, file_sha256
from CrowdSim.humanoid.application.configuration import ConfigurationError, validate_training_source
from CrowdSim.humanoid.training.amass_trajectories import AMASSRootTrajectoryCurriculum
from CrowdSim.humanoid.training.contracts import generalized_advantages

ROOT = Path(__file__).resolve().parents[2]
BEFORE = ROOT / "docs/style_improvement_20261002/amass_root_curriculum_before"
SOURCE_PATHS = (
    "CrowdSim/humanoid/training/amass_trajectories.py", "CrowdSim/humanoid/amass_corpus.py",
    "CrowdSim/humanoid/amp_world.py", "CrowdSim/humanoid/application/configuration.py",
    "scripts/train_paper_amp.py", "configs/diffusion_amp/amp_style_amass_root_curriculum.yaml",
    "CrowdSim/humanoid/training/contracts.py",
)


def extract(tree, name, namespace):
    fn = deepcopy(next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name))
    fn.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), "actual_source_helper", "exec"), namespace)
    return namespace[name]


def equal_reference(a, b):
    for index in (0, 1):
        if vars(a[index]).keys() != vars(b[index]).keys():
            return False
        if not all(torch.equal(value, vars(b[index])[key]) for key, value in vars(a[index]).items()):
            return False
    return all(torch.equal(a[i], b[i]) for i in (2, 3, 4))


def rejected(call, exception=ValueError):
    try:
        call()
    except exception:
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--floor-report", type=Path,
                        default=ROOT / "docs/style_improvement_20261002/amass_root_curriculum_independent_floor_bounds.json")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Fresh report required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Explicit CUDA_VISIBLE_DEVICES='' is required")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    source_before = {path: file_sha256(ROOT / path) for path in SOURCE_PATHS}
    checks = []

    def check(name, passed, evidence=None):
        checks.append({"name": name, "passed": bool(passed), "evidence": evidence})
        if not passed:
            raise AssertionError(name)

    manifest = ROOT / "data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json"
    corpus = AMASSCorpus(manifest, split="train", device="cpu", seed=7, dof_velocity_frame="child_joint")
    floor = json.loads(args.floor_report.read_text())
    check("independent_floor_proof_binds_actual_manifest_and_bank", floor["status"] == "passed"
          and floor["manifest"]["sha256"] == file_sha256(manifest)
          and floor["train_bank"]["sha256"] == corpus.entry["sha256"])
    check("all_2848_retained_without_sampling_filter", len(corpus.ids) == 2848)

    old_corpus_tree = ast.parse((BEFORE / "CrowdSim/humanoid/amass_corpus.py").read_text())
    current_corpus_tree = ast.parse((ROOT / "CrowdSim/humanoid/amass_corpus.py").read_text())
    def find(tree, name):
        return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    check("before_and_current_index_sampler_AST_identical",
          ast.dump(find(old_corpus_tree, "_indices")) == ast.dump(find(current_corpus_tree, "_indices")))
    old_sample = extract(old_corpus_tree, "sample_reference", {"torch": torch})
    rng = corpus.generator.get_state().clone()
    old = old_sample(corpus, 512, 10)
    old_rng = corpus.generator.get_state().clone()
    corpus.generator.set_state(rng)
    current = corpus.sample_reference(512, 10)
    check("default_reference_all_tensor_fields_bitwise_before", equal_reference(old, current))
    check("default_reference_shared_rng_bitwise_before", torch.equal(old_rng, corpus.generator.get_state()))

    runner_tree = ast.parse((ROOT / "scripts/train_paper_amp.py").read_text())
    old_runner_tree = ast.parse((BEFORE / "scripts/train_paper_amp.py").read_text())
    def outer(tree):
        return next(n for n in ast.walk(tree) if isinstance(n, ast.For)
                    and isinstance(n.target, ast.Name) and n.target.id == "update")
    check("complete_existing_control_update_For_AST_identical", ast.dump(outer(old_runner_tree)) == ast.dump(outer(runner_tree)))
    sample = corpus.sample_reference(4096, 10, future_frames=31)
    check("future31_sample_real_history_and_suffix_never_cross_clip",
          bool((sample[4] >= 9).all()) and bool((sample[4] + 31 < corpus.data["motion_num_frames"][sample[3]]).all()))

    # Four actual train sources, final legal reset phase; each has31 real steps.
    ids = corpus.ids[torch.linspace(0, len(corpus.ids) - 1, 4).long()]
    frames = corpus.data["motion_num_frames"][ids] - 32
    starts = corpus.data["length_starts"][ids]
    source = corpus.data["gts"][starts + frames, 0, :2]
    origins = torch.tensor([[48., 48.], [48., 108.], [108., 48.], [108., 108.]])
    episodes = torch.ones(4, dtype=torch.long)
    ages = torch.zeros(4, dtype=torch.long)
    curriculum = AMASSRootTrajectoryCurriculum(corpus, 4, bounds_xy=[0., 0., 192., 192.], tail_policy="linear_velocity")
    corpus_rng = corpus.generator.get_state().clone()
    curriculum.bind_reset(torch.arange(4), ids, frames, source, origins, episodes)
    plan = curriculum.build_plan(ages, episode_ids=episodes)
    check("native151_and_canonical51_have_same_actual_every_third_frame", torch.equal(plan.native_xy[:, ::3], plan.xy))
    plan_age = torch.zeros(4, dtype=torch.long)
    namespace = {"torch": torch, "plans": plan.native_xy, "plan_frequency_hz": 30,
                 "args": SimpleNamespace(trajectory_source="amass_root", num_envs=4),
                 "world": SimpleNamespace(device=torch.device("cpu"))}
    interpolate = extract(runner_tree, "interpolate", namespace)
    task_namespace = {"args": namespace["args"], "curriculum": curriculum,
                      "world": SimpleNamespace(age=ages, episode_ids=episodes, device=torch.device("cpu"))}
    task = extract(runner_tree, "task_trajectory", task_namespace)
    check("off_grid_reward_target_queries_rejected", rejected(lambda: interpolate(torch.full((4, 1), .017))))
    phase_errors, target_errors, horizon_errors = [], [], []
    final_source = corpus.data["gts"][starts + corpus.data["motion_num_frames"][ids] - 1, 0, :2]
    independent_velocity = (final_source - corpus.data["gts"][starts + corpus.data["motion_num_frames"][ids] - 11, 0, :2]) / (10 / 30)
    for completed in range(1, 32):
        ages += 1
        final = curriculum.completed_step(ages, torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool))
        plan_age += 1
        expected = corpus.data["gts"][starts + frames + ages, 0, :2] + curriculum.offset_xy
        target = interpolate((plan_age / 30)[:, None])[:, 0]
        target_errors.append(float((target - expected).abs().max()))
        phase_errors.append(bool(torch.equal(final, ages == 31)))
        future = task()
        raw = frames[:, None] + ages[:, None] + torch.arange(15, 151, 15)
        last = corpus.data["motion_num_frames"][ids, None] - 1
        expected_future = corpus.data["gts"][starts[:, None] + raw.minimum(last), 0, :2] + curriculum.offset_xy[:, None]
        expected_future += (raw - last).clamp_min(0)[..., None] / 30 * independent_velocity[:, None]
        horizon_errors.append(float((future - expected_future).abs().max()))
        if completed == 16:
            curriculum.validate_runner_state(ages, plan_age, namespace["plans"], ids, episodes,
                                            replan_steps=15, origins_xy=origins)
            buffer = io.BytesIO()
            torch.save(curriculum.state_dict(), buffer)
            buffer.seek(0)
            stored = torch.load(buffer, weights_only=True, map_location="cpu")
            restored_corpus = AMASSCorpus(manifest, split="train", device="cpu", seed=123, dof_velocity_frame="child_joint")
            restored = AMASSRootTrajectoryCurriculum(restored_corpus, 4, bounds_xy=[0., 0., 192., 192.], tail_policy="linear_velocity")
            restored.load_state_dict(stored)
            restored.validate_runner_state(ages, plan_age, namespace["plans"], ids, episodes,
                                          replan_steps=15, origins_xy=origins)
            check("weights_only_resume_all_actor_state_and_counters_equal",
                  all(torch.equal(getattr(restored, name), getattr(curriculum, name)) for name in curriculum.STATE_FIELDS)
                  and restored.statistics == curriculum.statistics)
            check("resume_current_full_horizon_commands_bitwise_equal",
                  torch.equal(restored.future_targets(ages, episode_ids=episodes), future))
            after1 = corpus.sample_reference(32, future_frames=31)
            after2 = restored_corpus.sample_reference(32, future_frames=31)
            check("resume_next_actual_reference_and_shared_rng_bitwise_equal", equal_reference(after1, after2)
                  and torch.equal(corpus.generator.get_state(), restored_corpus.generator.get_state()))
            # Return to the unchanged pre-harness sampling state for no-RNG claims.
            corpus.generator.set_state(corpus_rng)
        if completed % 15 == 0 and completed < 31:
            plan = curriculum.build_plan(ages, episode_ids=episodes)
            namespace["plans"] = plan.native_xy
            plan_age.zero_()
    check("all31_completed_reward_targets_bitwise_real_next_source_frames", max(target_errors) == 0,
          {"actual_source_motion_ids": ids.tolist(), "initial_frames": frames.tolist(), "max_abs_error_m": max(target_errors)})
    check("source_end_truncates_exact_final_real_frame_without_prior_early_truncation", all(phase_errors))
    check("current_phase_task_full5s_queries_match_independent_formula", max(horizon_errors) == 0,
          {"max_abs_error_m": max(horizon_errors), "queries": 31 * 4 * 10})
    check("post_source_end_physical_clock_advance_rejected",
          rejected(lambda: curriculum.completed_step(ages + 1, torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool))))
    check("bind_replan_and_task_queries_do_not_consume_rng", torch.equal(corpus_rng, corpus.generator.get_state()))
    before_state = curriculum.state_dict()
    check("resume_corrupt_endpoint_velocity_rejected", rejected(lambda: corrupt_velocity(curriculum, before_state)))
    check("resume_reference_ids_wrong_dtype_rejected", rejected(lambda: curriculum.validate_runner_state(
        ages, plan_age, namespace["plans"], ids.float(), episodes, replan_steps=15, origins_xy=origins)))

    fresh = AMASSRootTrajectoryCurriculum(corpus, 4, bounds_xy=[0., 0., 192., 192.], tail_policy="linear_velocity")
    fresh.bind_reset(torch.arange(4), ids, frames, source, origins, episodes)
    check("newly_reset_resume_without_expired_plan_marker_rejected", rejected(lambda: fresh.validate_runner_state(
        torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long), namespace["plans"], ids,
        episodes, replan_steps=15, origins_xy=origins)))
    fresh.validate_runner_state(torch.zeros(4, dtype=torch.long), torch.full((4,), 15, dtype=torch.long), namespace["plans"],
                                ids, episodes, replan_steps=15, origins_xy=origins)
    check("legitimate_newly_reset_expired_plan_resume_accepted", True)

    missing = SimpleNamespace(trajectory_source="amass_root", evaluate_steps=0)
    check("missing_explicit_amass_profile_rejected_before_Kit", rejected(lambda: validate_training_source(missing, {}), ConfigurationError))
    validate_training_source(SimpleNamespace(trajectory_source="diffusion", evaluate_steps=0), {})
    validate_training_source(SimpleNamespace(trajectory_source="pacer_curriculum", evaluate_steps=0), {})
    check("legacy_undeclared_diffusion_and_procedural_profiles_still_accepted", True)
    reward = torch.ones(1, 1)
    values, next_values = torch.full((1, 1), 2.), torch.full((1, 1), 3.)
    advantages, _ = generalized_advantages(reward, values, next_values, torch.zeros(1, 1, dtype=torch.bool),
                                          torch.ones(1, 1, dtype=torch.bool), gamma=.99, gae_lambda=.95)
    terminal, _ = generalized_advantages(reward, values, next_values, torch.ones(1, 1, dtype=torch.bool),
                                        torch.ones(1, 1, dtype=torch.bool), gamma=.99, gae_lambda=.95)
    check("source_end_time_limit_bootstrap_retained_fall_bootstrap_suppressed",
          torch.allclose(advantages, torch.full((1, 1), 1.97)) and torch.equal(terminal, torch.full((1, 1), -1.)))
    source_after = {path: file_sha256(ROOT / path) for path in SOURCE_PATHS}
    check("source_files_unchanged_during_independent_CPU_audit", source_before == source_after)
    report = {"schema": "navisaaclab.amass_root_curriculum_independent_integration.v1", "status": "passed",
              "CPU_only": True, "Kit_started": False, "GPU_used": False, "actual_physics_rollout": False,
              "heldout_captures_read": False, "training_data_modified": False, "source_files_modified": False,
              "checks": checks, "check_count": len(checks), "sources": source_after,
              "manifest": str(manifest), "manifest_sha256": file_sha256(manifest), "source_bank_sha256": corpus.entry["sha256"],
              "independent_floor_report": {"path": str(args.floor_report.resolve()), "sha256": file_sha256(args.floor_report)},
              "script_sha256": file_sha256(__file__),
              "limitations": ["Source-clock harnesses are CPU command tests, not simulated physical actors.",
                              "The generated linear tail is engineering extrapolation, not motion capture or author protocol.",
                              "Resume validates logical state and RNG; PhysX contact/solver caches are not serialized.",
                              "Floor proof certifies reference root commands, not bodycollider or physical tracking safety."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "passed", "checks": len(checks), "output": str(args.output)}))


def corrupt_velocity(curriculum, state):
    value = deepcopy(state)
    value["actors"]["end_velocity_xy"][0, 0] += .01
    curriculum.load_state_dict(value)


if __name__ == "__main__":
    main()
