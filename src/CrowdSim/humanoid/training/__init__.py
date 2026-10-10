"""Trainable paper method reconstruction; distinct from reference checkpoints."""

from .rewards import paper_task_reward, paper_total_reward
from .trainer import PaperAMPTrainer, TrainingConfig

__all__ = ["PaperAMPTrainer", "TrainingConfig", "paper_task_reward", "paper_total_reward"]
