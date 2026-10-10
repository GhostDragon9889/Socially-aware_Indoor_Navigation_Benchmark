"""Independent CPU plumbing checks over the real frozen AMASS training bank.

No SDK or physics is executed. The tests compare the mixed facade with actual
AMASS records and the existing independent procedural controller, rather than
implementing a second fake curriculum with the same recurrences.
"""
from __future__ import annotations

import ast
from copy import copy, deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from CrowdSim.humanoid.amass_corpus import AMASSCorpus
from CrowdSim.humanoid.training.mixed_trajectories import MixedTrajectoryCurriculum
from CrowdSim.humanoid.training.trajectories import PacerTrajectoryConfig, PacerTrajectoryCurriculum

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/"data/diffusion_amp/amass_style_2848/manifest_labeled_v7.json"
RECIPE = PacerTrajectoryConfig(speed_max_mps=2., scalar_accel_max_mps2=1.5,
                              vector_accel_max_mps2=1.5)


@pytest.fixture(scope="module")
def frozen_corpus():
    if not MANIFEST.exists():
        pytest.skip("Licensed local AMASS v7 payload is not present")
    torch.set_num_threads(4)
    return AMASSCorpus(MANIFEST, "train", device="cpu", seed=29, dof_velocity_frame="child_joint")


def independent_corpus(source, seed=29):
    value = copy(source)
    value.generator = torch.Generator(device="cpu").manual_seed(seed)
    return value


def make(source):
    return MixedTrajectoryCurriculum(independent_corpus(source), 4, seed=7,
        bounds_xy=[0., 0., 192., 192.], procedural_config=RECIPE)


def bind(mixed, *, near_end=False):
    c = mixed.corpus
    mid = c.ids[:2].repeat(2)
    frames = (c.data["motion_num_frames"][mid]-32) if near_end else torch.full((4,), 9, dtype=torch.long)
    origins = torch.tensor([[48., 48.], [48., 52.], [52., 48.], [52., 52.]])
    source = c.data["gts"][c.data["length_starts"][mid]+frames, 0, :2]
    mixed.bind_reset(torch.arange(4), mid, frames, source, origins, torch.ones(4, dtype=torch.long))
    robot = SimpleNamespace(root_pos=torch.cat((origins, torch.ones(4, 1)), -1),
        root_vel=torch.tensor([[.1, .2, 0.], [.2, .1, 0.], [.3, .1, 0.], [.1, .4, 0.]]),
        root_rot=torch.tensor([[0., 0., 0., 1.]]).repeat(4, 1))
    return robot, origins, mid, frames


def completed(mixed, ages, count):
    flag = None
    for _ in range(count):
        ages += 1
        flag = mixed.completed_step(ages, torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool))
    return flag


def actual_xy(corpus, motion, raw_frames, offset):
    last = corpus.data["motion_num_frames"][motion]-1
    start = corpus.data["length_starts"][motion]
    exact = corpus.data["gts"][start+raw_frames.minimum(last), 0, :2]+offset
    end_speed = (corpus.data["gts"][start+last, 0, :2]-corpus.data["gts"][start+last-10, 0, :2])/(10/30)
    return exact+(raw_frames-last).clamp_min(0).float()[:, None]/30*end_speed


@pytest.mark.needs_lfs
def test_actual_bank_arbitrary_global_actor_order_and_independent_reset_rng(frozen_corpus):
    mixed = make(frozen_corpus)
    original = independent_corpus(frozen_corpus)
    real = independent_corpus(frozen_corpus, seed=8)
    expected_proc = original.sample_reference(2, 10, future_frames=1)
    expected_real = real.sample_reference(2, 10, future_frames=31)
    got = mixed.sample_reference(torch.tensor([3, 0, 2, 1]))
    expected_ids = torch.stack((expected_proc[3][0], expected_real[3][0], expected_proc[3][1], expected_real[3][1]))
    expected_frames = torch.stack((expected_proc[4][0], expected_real[4][0], expected_proc[4][1], expected_real[4][1]))
    assert torch.equal(got[3], expected_ids) and torch.equal(got[4], expected_frames)
    absolute = mixed.corpus.data["length_starts"][expected_ids]+expected_frames
    direct = mixed.corpus.state_at(absolute)
    history = mixed.corpus.state_at((absolute[:, None]-torch.arange(10)).reshape(-1))
    for name in vars(direct):
        assert torch.equal(getattr(got[0], name), getattr(direct, name)), name
        assert torch.equal(getattr(got[1], name), getattr(history, name)), name
    assert torch.equal(got[2], mixed.corpus.data["dps"][absolute+1])
    assert torch.equal(mixed.corpus.generator.get_state(), original.generator.get_state())
    assert torch.equal(mixed.private_corpus.generator.get_state(), real.generator.get_state())
    assert mixed.private_corpus.data is mixed.corpus.data
    assert mixed.private_corpus.generator is not mixed.corpus.generator


@pytest.mark.needs_lfs
def test_actual_real151_and_procedural51_keep_distinct_timing_under_reordered_subset(frozen_corpus):
    mixed = make(frozen_corpus)
    robot, origins, mid, frames = bind(mixed)
    ages = torch.zeros(4, dtype=torch.long)
    result = mixed.build_plan(robot, ages, actor_ids=torch.tensor([3, 1, 2, 0]), episode_ids=torch.ones(4, dtype=torch.long))
    previous = PacerTrajectoryCurriculum(RECIPE, seed=7).build_plan(robot, actor_ids=torch.tensor([3, 2]))
    assert torch.equal(result.xy[[0, 2]], previous.xy)
    for actor, row in ((0, 3), (1, 1)):
        offset = origins[actor]-mixed.corpus.data["gts"][mixed.corpus.data["length_starts"][mid[actor]]+frames[actor], 0, :2]
        expected = actual_xy(mixed.corpus, mid[actor], frames[actor]+torch.arange(151), offset)
        torch.testing.assert_close(result.native_xy[row], expected, atol=2e-6, rtol=0)
        assert torch.equal(result.native_xy[row, ::3], result.xy[row])
    # Independent legacy interpolation at all151 control queries, including
    # non-knot times. No30Hz AMASS quantization is imposed on procedural51.
    times = torch.arange(151)/30
    lower = (times*10).floor().long().clamp_max(49)
    weight = (times*10-lower)[:, None]
    expected = torch.lerp(previous.xy[:, lower], previous.xy[:, lower+1], weight)
    assert torch.equal(result.native_xy[[0, 2]], expected)


@pytest.mark.needs_lfs
def test_live_root_perturbation_cannot_reanchor_real_commands_or_need_future3d(frozen_corpus):
    mixed = make(frozen_corpus)
    robot, _, _, _ = bind(mixed)
    ages = torch.zeros(4, dtype=torch.long)
    before = mixed.build_plan(robot, ages, actor_ids=torch.tensor([1, 0]))
    robot.root_pos += 1000
    robot.root_rot = torch.tensor([[0., 0., 1., 0.]]).repeat(4, 1)
    after = mixed.build_plan(robot, ages, actor_ids=torch.tensor([1, 0]))
    assert torch.equal(before.native_xy, after.native_xy)
    assert set(vars(robot)) == {"root_pos", "root_vel", "root_rot"}


@pytest.mark.needs_lfs
def test_full_future_horizon_uses_real_current_phase_and_old_procedural_clamp(frozen_corpus):
    mixed = make(frozen_corpus)
    robot, origins, mid, frames = bind(mixed)
    ages = torch.zeros(4, dtype=torch.long)
    plans = mixed.build_plan(robot, ages).native_xy
    completed(mixed, ages, 12)
    target = mixed.future_targets(ages, torch.full((4,), 12, dtype=torch.long), plans, torch.ones(4, dtype=torch.long))
    for actor in (0, 1):
        offset = origins[actor]-mixed.corpus.data["gts"][mixed.corpus.data["length_starts"][mid[actor]]+frames[actor], 0, :2]
        expected = actual_xy(mixed.corpus, mid[actor], frames[actor]+12+torch.arange(1, 11)*15, offset)
        torch.testing.assert_close(target[actor], expected, atol=2e-6, rtol=0)
    assert torch.equal(target[2:, -1], mixed.proc_canonical_last_plans[:, -1])
    assert mixed.statistics["amass_actor_transitions"] == mixed.statistics["procedural_actor_transitions"] == 24


@pytest.mark.needs_lfs
def test_source_end_applies_only_real_slots_with_actual_equal_transition_exposure(frozen_corpus):
    mixed = make(frozen_corpus)
    robot, _, _, _ = bind(mixed, near_end=True)
    ages = torch.zeros(4, dtype=torch.long)
    mixed.build_plan(robot, ages)
    assert not completed(mixed, ages, 30).any()
    assert completed(mixed, ages, 1).tolist() == [True, True, False, False]
    assert mixed.statistics["amass_actor_transitions"] == mixed.statistics["procedural_actor_transitions"] == 62
    assert mixed.amass.statistics["clip_end_truncations"] == 2


@pytest.mark.needs_lfs
def test_full_saved_plan_and_all_streams_resume_and_reject_corruption(frozen_corpus):
    mixed = make(frozen_corpus)
    robot, origins, mid, _ = bind(mixed)
    ages = torch.zeros(4, dtype=torch.long)
    plans = mixed.build_plan(robot, ages).native_xy
    completed(mixed, ages, 3)
    saved = mixed.state_dict()
    parent_rng = mixed.corpus.generator.get_state()
    restored = make(frozen_corpus)
    restored.load_state_dict(saved)
    restored.corpus.generator.set_state(parent_rng)
    restored.validate_runner_state(ages, torch.full((4,), 3, dtype=torch.long), plans, mid, torch.ones(4, dtype=torch.long),
        replan_steps=15, origins_xy=origins)
    for actors in (torch.tensor([1, 3, 0, 2]), torch.tensor([3, 2]), torch.tensor([0])):
        left, right = mixed.sample_reference(actors), restored.sample_reference(actors)
        for index in (2, 3, 4):
            assert torch.equal(left[index], right[index])
        assert torch.equal(left[0].root_pos, right[0].root_pos)
    for bad in ("assignment", "child_rng", "counter"):
        invalid = deepcopy(saved)
        if bad == "assignment":
            invalid["config"]["amass_actor_ids"] = [1, 0]
        elif bad == "child_rng":
            invalid["generator_state"] = torch.Generator().manual_seed(889).get_state()
        else:
            invalid["statistics"]["procedural_actor_transitions"] += 1
        with pytest.raises(ValueError):
            make(frozen_corpus).load_state_dict(invalid)
    shifted = plans.clone()
    shifted[3, 77, 0] += .1
    with pytest.raises(ValueError, match="canonical51"):
        restored.validate_runner_state(ages, torch.full((4,), 3, dtype=torch.long), shifted, mid, torch.ones(4, dtype=torch.long),
            replan_steps=15, origins_xy=origins)


def test_actual_driver_helper_transfers_all_three_rng_buffers_only_and_keeps_old_path():
    source = ast.parse((ROOT/"scripts/train_paper_amp.py").read_text())
    function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "curriculum_rng_to_cpu")
    namespace = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "actual_driver_cpu_rng_helper", "exec"), namespace)
    class Buffer:
        def __init__(self, name):
            self.name, self.calls = name, 0
        def cpu(self):
            self.calls += 1
            return ("CPU", self.name)
    a, b, c, physical = (Buffer(name) for name in ("top", "real", "proc", "physical"))
    original = {"generator_state": a, "amass_state": {"generator_state": b, "physical": physical},
                "procedural_state": {"generator_state": c}, "physical": physical}
    result = namespace["curriculum_rng_to_cpu"](original, mixed=True)
    assert [a.calls, b.calls, c.calls, physical.calls] == [1, 1, 1, 0]
    assert result["amass_state"]["generator_state"] == ("CPU", "real")
    assert result["procedural_state"]["generator_state"] == ("CPU", "proc")
    assert original["amass_state"]["generator_state"] is b
    assert result["physical"] is physical
    namespace["curriculum_rng_to_cpu"](original, mixed=False)
    assert [a.calls, b.calls, c.calls, physical.calls] == [2, 1, 1, 0]


def test_complete_control_loop_ast_equals_frozen_before_mixed():
    def control(path):
        loops = [node for node in ast.walk(ast.parse(path.read_text())) if isinstance(node, ast.For)
                 and isinstance(node.target, ast.Name) and node.target.id == "step"]
        assert len(loops) == 1
        from runner_branch_contract import training_branch
        return ast.dump(training_branch(loops[0]), include_attributes=False)
    assert control(ROOT/"scripts/train_paper_amp.py") == control(
        ROOT/"docs/style_improvement_20261002/mixed_curriculum_before/scripts/train_paper_amp.py")
