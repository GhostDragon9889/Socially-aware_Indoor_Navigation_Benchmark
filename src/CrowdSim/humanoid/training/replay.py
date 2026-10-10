"""Bounded raw-state policy replay for AMP; full content survives resume.

AMP uses replay to prevent the discriminator overfitting the latest policy.
Raw features are stored so current normalization is shared by live and replay
examples. Sampling uses the trainer's checkpointed Torch RNG, not a hidden RNG.
"""
from __future__ import annotations

import torch


class AMPPolicyReplay:
    def __init__(self, capacity, history_shape, *, device="cpu"):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("Replay capacity must be a positive integer")
        if len(history_shape) != 2 or any(type(v) is not int or v < 1 for v in history_shape):
            raise ValueError("Replay history shape must contain two positive integers")
        self.capacity = capacity
        self.history_shape = tuple(history_shape)
        self.device = torch.device(device)
        self.history = torch.empty((capacity, *history_shape), device=device)
        self.actions = torch.empty((capacity, 69), device=device)
        self.size = 0
        self.cursor = 0
        self.accepted = 0

    @torch.no_grad()
    def add(self, history, actions, *, keep_probability=1.):
        if type(keep_probability) not in (int, float) or not 0 <= keep_probability <= 1:
            raise ValueError("Replay keep probability must be in [0,1]")
        if history.shape != (len(history), *self.history_shape) or actions.shape != (len(history), 69):
            raise ValueError("Replay samples do not match the history/action contract")
        if not torch.isfinite(history).all() or not torch.isfinite(actions).all() or (actions.abs() > 1 + 1e-6).any():
            raise ValueError("Replay samples must be finite with normalized actions")
        h, a = history.detach().to(self.device), actions.detach().to(self.device)
        # Fill the initial buffer; subsequently retain the declared fraction.
        if self.size == self.capacity and keep_probability < 1:
            keep = torch.rand(len(h), device=self.device) < keep_probability
            h, a = h[keep], a[keep]
        accepted = len(h)
        self.accepted += accepted
        if accepted >= self.capacity:
            self.history.copy_(h[-self.capacity:])
            self.actions.copy_(a[-self.capacity:])
            self.size, self.cursor = self.capacity, 0
            return accepted
        first = min(accepted, self.capacity - self.cursor)
        self.history[self.cursor:self.cursor + first].copy_(h[:first])
        self.actions[self.cursor:self.cursor + first].copy_(a[:first])
        remaining = accepted - first
        if remaining:
            self.history[:remaining].copy_(h[first:])
            self.actions[:remaining].copy_(a[first:])
        self.cursor = (self.cursor + accepted) % self.capacity
        self.size = min(self.capacity, self.size + accepted)
        return accepted

    def sample(self, count):
        if type(count) is not int or count < 1 or self.size == 0:
            raise ValueError("Replay sampling requires a nonempty buffer and positive count")
        index = torch.randint(self.size, (count,), device=self.device)
        return self.history[index], self.actions[index]

    def state_dict(self):
        return {"capacity": self.capacity, "history_shape": self.history_shape,
                "size": self.size, "cursor": self.cursor, "accepted": self.accepted,
                "history": self.history[:self.size].detach().clone(),
                "actions": self.actions[:self.size].detach().clone()}

    @torch.no_grad()
    def load_state_dict(self, state):
        expected = {"capacity", "history_shape", "size", "cursor", "accepted", "history", "actions"}
        if not isinstance(state, dict) or set(state) != expected or state["capacity"] != self.capacity or tuple(state["history_shape"]) != self.history_shape:
            raise ValueError("Replay resume schema/configuration changed")
        size, cursor, accepted = state["size"], state["cursor"], state["accepted"]
        if any(type(v) is not int for v in (size, cursor, accepted)) or not 0 <= size <= self.capacity or not 0 <= cursor < self.capacity or accepted < size:
            raise ValueError("Invalid replay counters")
        if size < self.capacity and (cursor != size or accepted != size):
            raise ValueError("Partially filled replay has inconsistent counters")
        h, a = state["history"], state["actions"]
        if not isinstance(h, torch.Tensor) or not isinstance(a, torch.Tensor) or h.shape != (size, *self.history_shape) or a.shape != (size, 69) or h.dtype != torch.float32 or a.dtype != torch.float32:
            raise ValueError("Invalid replay tensor shape/dtype")
        if not torch.isfinite(h).all() or not torch.isfinite(a).all() or (a.abs() > 1 + 1e-6).any():
            raise ValueError("Invalid replay tensor values")
        self.history[:size].copy_(h.to(self.device))
        self.actions[:size].copy_(a.to(self.device))
        self.size, self.cursor, self.accepted = size, cursor, accepted
