"""Actual CPU event simulations of staggered resets and persistent clocks."""

import copy

import numpy as np
import pytest
import torch

from CrowdSim.humanoid.planning_clock import due_agents


def _simulate(reset_schedule, *, synchronized, saved=None, stop=None):
    """Runner timing: reset before the new actor's first action; then age once."""
    steps, actors = reset_schedule.shape
    interval = 15
    state = ({"control_step": 0, "plan_age": torch.full((actors,), interval, dtype=torch.long),
              "last_plan": np.full(actors, -1, dtype=np.int64), "calls": [], "intervals": [],
              "reset_checks": 0} if saved is None else copy.deepcopy(saved))
    for tick in range(state["control_step"], steps if stop is None else stop):
        resets = np.flatnonzero(reset_schedule[tick])
        state["plan_age"][torch.as_tensor(resets)] = interval
        due = due_agents(state["plan_age"], state["control_step"], interval, synchronized)
        selected = due.numpy()
        assert np.isin(resets, selected).all(), "A spawned actor reached its first action without a fresh plan"
        state["reset_checks"] += len(resets)
        if synchronized and tick % interval == 0:
            assert np.array_equal(selected, np.arange(actors))
        previous = state["last_plan"][selected]
        gaps = tick - previous[previous >= 0]
        state["intervals"].extend(gaps.tolist())
        state["last_plan"][selected] = tick
        if len(selected):
            state["calls"].append((tick, tuple(selected.tolist())))
        state["plan_age"][due] = 0
        assert (state["plan_age"] < interval).all(), "Overdue trajectory used for a control action"
        state["plan_age"] += 1
        state["control_step"] += 1
        assert (state["plan_age"] <= interval).all()
    return state


def test_default_preserves_old_actor_age_rule_independent_of_counter():
    age = torch.tensor([0, 14, 15, 3, 19], dtype=torch.int32)
    original = age.clone()
    expected = (age >= 15).nonzero().flatten()
    for counter in (0, 1, 15, 32, 10**12):
        assert torch.equal(due_agents(age, counter, 15), expected)
    assert torch.equal(age, original)


def test_global_ticks_include_all_and_offgrid_spawn_is_immediate():
    age = torch.tensor([0, 2, 15, 14])
    assert torch.equal(due_agents(age, 30, 15, True), torch.arange(4))
    assert torch.equal(due_agents(age, 31, 15, True), torch.tensor([2]))
    assert torch.equal(due_agents(torch.empty(0, dtype=torch.long), 0, 15, True), torch.empty(0, dtype=torch.long))


@pytest.mark.parametrize("synchronized", [False, True])
def test_random_resets_preserve_max_interval_and_fresh_spawn(synchronized):
    resets = np.random.default_rng(7).random((1200, 64)) < .006
    result = _simulate(resets, synchronized=synchronized)
    assert result["reset_checks"] == int(resets.sum()) > 300
    assert max(result["intervals"]) <= 15
    assert result["control_step"] == 1200
    assert not torch.cuda.is_initialized()


def test_resume_absolute_counter_matches_uninterrupted_across_update_boundaries():
    resets = np.random.default_rng(91).random((731, 64)) < .006
    expected = _simulate(resets, synchronized=True)
    # Neither boundary is a rollout multiple nor a global planning tick.
    state = _simulate(resets, synchronized=True, stop=137)
    assert state["control_step"] % 15 != 0
    saved = {**state, "plan_age": state["plan_age"].clone()}
    # These are the runner's serializable logical clock values, not hidden state.
    state = _simulate(resets, synchronized=True, saved=saved, stop=489)
    actual = _simulate(resets, synchronized=True, saved=state)
    assert expected["calls"] == actual["calls"]
    assert expected["intervals"] == actual["intervals"]
    assert np.array_equal(expected["last_plan"], actual["last_plan"])
    assert torch.equal(expected["plan_age"], actual["plan_age"])
    assert actual["control_step"] == expected["control_step"] == 731
    wrong_counter = {**saved, "control_step": 0}
    assert due_agents(saved["plan_age"], saved["control_step"], 15, True).numel() != due_agents(wrong_counter["plan_age"], 0, 15, True).numel()


def test_explicit_stage_switch_from_staggered_existing_ages_keeps_interval_bound():
    resets = np.random.default_rng(11).random((731, 64)) < .006
    old = _simulate(resets, synchronized=False, stop=137)
    before_ages = old["plan_age"].clone()
    actual = _simulate(resets, synchronized=True, saved=old)
    assert torch.equal(old["plan_age"], before_ages)
    assert max(actual["intervals"]) <= 15
    new_calls = [(tick, ids) for tick, ids in actual["calls"] if tick >= 137]
    assert all(len(ids) == 64 for tick, ids in new_calls if tick % 15 == 0)
    assert (150, tuple(range(64))) in new_calls
    assert actual["reset_checks"] == int(resets.sum())


def test_staggered_fall_resets_reduce_small_batch_calls_after_global_alignment():
    resets = np.random.default_rng(20261001).random((1200, 64)) < .006
    asynchronous = _simulate(resets, synchronized=False)
    synchronized = _simulate(resets, synchronized=True)
    old_calls, new_calls = asynchronous["calls"], synchronized["calls"]
    assert len(old_calls) > 1000
    assert len(new_calls) < len(old_calls) * .5
    full_batches = [ids for tick, ids in new_calls if tick % 15 == 0]
    assert len(full_batches) == 80 and all(len(ids) == 64 for ids in full_batches)
    assert np.median([len(ids) for _, ids in old_calls]) < 8
    # Immediate reset plans remain mandatory; those calls are not hidden.
    offgrid = [tick for tick, _ in new_calls if tick % 15]
    assert len(offgrid) > 0


@pytest.mark.parametrize("age,counter,interval,sync,error", [
    (torch.zeros(2, 2, dtype=torch.long), 0, 15, True, TypeError),
    (torch.zeros(2), 0, 15, True, TypeError),
    (torch.zeros(2, dtype=torch.long), -1, 15, True, ValueError),
    (torch.zeros(2, dtype=torch.long), 0, 0, True, ValueError),
    (torch.zeros(2, dtype=torch.long), 0, 15, "yes", TypeError),
    (torch.zeros(2, dtype=torch.long), True, 15, True, ValueError),
])
def test_clock_rejects_ambiguous_contracts(age, counter, interval, sync, error):
    with pytest.raises(error):
        due_agents(age, counter, interval, sync)
