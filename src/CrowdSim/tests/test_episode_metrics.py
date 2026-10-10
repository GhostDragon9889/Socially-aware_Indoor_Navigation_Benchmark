import pytest
import torch

from CrowdSim.ppo.episode_metrics import terminal_counts, terminal_masks
from CrowdSim.ppo.train_ppo import RobotTrainingLogger


def test_terminal_event_overlap_counts_one_completed_episode():
    done = torch.tensor([True, False])
    info = {"reached": torch.tensor([True, True]), "timeout": torch.tensor([True, False])}
    counts = {key: float(value) for key, value in terminal_counts(done, info).items()}
    assert counts["episodes"] == counts["success"] == counts["reached"] == counts["timeout"] == 1
    assert counts["collision"] == counts["stuck"] == 0
    metrics = RobotTrainingLogger._outcome_stats(counts)
    assert metrics["outcome/total_episodes"] == 1
    assert metrics["outcome/success_rate"] == metrics["outcome/reached_rate"] == 1
    assert metrics["outcome/timeout_rate"] == 1


def test_collision_excludes_success_even_when_reaching_on_last_step():
    done = torch.tensor([True, True, False])
    info = {"reached": torch.tensor([True, True, True]), "collision": torch.tensor([True, False, False])}
    masks = terminal_masks(done, info)
    assert masks["success"].tolist() == [False, True, False]
    counts = {key: float(value) for key, value in terminal_counts(done, info).items()}
    assert counts["episodes"] == 2 and counts["success"] == 1
    assert RobotTrainingLogger._outcome_stats(counts)["outcome/success_rate"] == .5


def test_missing_events_are_zero_and_wrong_shapes_are_rejected():
    counts = terminal_counts(torch.tensor([True]), {})
    assert float(counts["episodes"]) == 1
    assert float(counts["success"]) == float(counts["collision"]) == 0
    with pytest.raises(ValueError, match="Terminal event"):
        terminal_counts(torch.tensor([True]), {"reached": torch.tensor([True, False])})
    with pytest.raises(ValueError, match="explicit episodes"):
        RobotTrainingLogger._outcome_stats({"reached": 1, "timeout": 1})
