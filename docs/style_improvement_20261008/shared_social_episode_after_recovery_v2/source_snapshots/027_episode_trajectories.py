"""Opt-in episode-bound procedural references, independent of live-root replans.

The existing smooth generator is retained. A ten-second episode needs fifteen
seconds of reference to support an unclamped five-second future observation.
This is a local curriculum candidate; World/SDK integration is separate.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace

import torch

from .trajectories import PacerTrajectoryConfig, PacerTrajectoryCurriculum


class EpisodeProceduralReference:
    SCHEMA = "episode_bound_procedural_reference_v1"

    def __init__(self, num_envs, *, episode_controls=300, config=None, seed=7, device="cpu"):
        if type(num_envs) is not int or num_envs < 1 or type(episode_controls) is not int or episode_controls < 1:
            raise ValueError("Require positive integer actor and episode counts")
        config = PacerTrajectoryConfig() if config is None else config
        if not isinstance(config, PacerTrajectoryConfig):
            raise ValueError("Require explicit PacerTrajectoryConfig")
        config.validate()
        if config.horizon_s != 5 or config.knot_dt_s != .1 or episode_controls % 3:
            raise ValueError("Require five-second lookahead, 0.1s knots and a knot-aligned 30Hz episode cap")
        self.num_envs, self.episode_controls, self.device = num_envs, episode_controls, torch.device(device)
        self.source = PacerTrajectoryCurriculum(replace(config, horizon_s=episode_controls / 30 + 5), seed=seed)
        self.config = {"num_envs": num_envs, "episode_controls": episode_controls,
            "source_config": asdict(self.source.config), "lookahead_s": 5., "control_hz": 30}
        self.paths = torch.zeros(num_envs, round(self.source.config.horizon_s * 10) + 1, 2, device=self.device)
        self.episode_ids = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

    def _actors(self, actor_ids):
        ids = torch.arange(self.num_envs, device=self.device) if actor_ids is None else torch.as_tensor(actor_ids, device=self.device)
        if (ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64) or not len(ids)
                or len(ids.unique()) != len(ids) or bool(((ids < 0) | (ids >= self.num_envs)).any())):
            raise ValueError("Require unique in-range integer actor IDs")
        return ids.long()

    def _integer(self, value, name):
        if not isinstance(value, torch.Tensor) or value.shape != (self.num_envs,) or value.dtype not in (torch.int32, torch.int64):
            raise ValueError(name + " must be a full integer actor vector")
        return value.to(self.device)

    @torch.no_grad()
    def reset(self, robot_state, episode_ids, actor_ids=None):
        ids = self._actors(actor_ids)
        episodes = self._integer(episode_ids, "episode IDs")
        if (bool((episodes[ids] < 1).any())
                or bool((self.initialized[ids] & (episodes[ids] <= self.episode_ids[ids])).any())):
            raise ValueError("Reset must bind a fresh strictly increasing episode ID")
        for name, width in (("root_pos", 3), ("root_vel", 3), ("root_rot", 4)):
            field = getattr(robot_state, name, None)
            if not isinstance(field, torch.Tensor) or field.shape != (self.num_envs, width) or field.device != self.device:
                raise ValueError("Reset requires full root state on the configured device")
        prediction = self.source.build_plan(robot_state, actor_ids=ids)
        self.paths[ids] = prediction.xy
        self.episode_ids[ids] = episodes[ids]
        self.initialized[ids] = True

    def query(self, ages, episode_ids, offsets_s, actor_ids=None):
        """Absolute XY; no current root input, RNG draws, clamping or state update."""
        ids = self._actors(actor_ids)
        ages, episodes = self._integer(ages, "completed control ages"), self._integer(episode_ids, "episode IDs")
        offsets = torch.as_tensor(offsets_s, device=self.device)
        if (offsets.ndim != 1 or not len(offsets) or not offsets.is_floating_point()
                or not torch.isfinite(offsets).all() or bool(((offsets < 0) | (offsets > 5)).any())
                or bool(((ages[ids] < 0) | (ages[ids] > self.episode_controls)).any())
                or not bool(self.initialized[ids].all()) or not torch.equal(episodes[ids], self.episode_ids[ids])):
            raise ValueError("Require bound episode clocks and explicit finite offsets within five seconds")
        times = ages[ids, None].float() / 30 + offsets.float()[None]
        fractional = times * 10
        lower = fractional.floor().long().clamp_max(self.paths.shape[1] - 2)
        rows = ids[:, None]
        return torch.lerp(self.paths[rows, lower], self.paths[rows, lower + 1], (fractional - lower)[..., None])

    def state_dict(self):
        return {"schema": self.SCHEMA, "config": deepcopy(self.config), "source": self.source.state_dict(),
            "paths": self.paths.detach().cpu().clone(), "episode_ids": self.episode_ids.detach().cpu().clone(),
            "initialized": self.initialized.detach().cpu().clone()}

    def load_state_dict(self, state):
        if (not isinstance(state, dict) or set(state) != {"schema", "config", "source", "paths", "episode_ids", "initialized"}
                or state["schema"] != self.SCHEMA or state["config"] != self.config):
            raise ValueError("Require exact episode reference schema and configuration")
        for name in ("paths", "episode_ids", "initialized"):
            saved, live = state[name], getattr(self, name)
            if (not isinstance(saved, torch.Tensor) or saved.device.type != "cpu"
                    or saved.shape != live.shape or saved.dtype != live.dtype or not torch.isfinite(saved).all()):
                raise ValueError("Invalid episode reference state: " + name)
        if (bool((state["episode_ids"][state["initialized"]] < 1).any())
                or bool((state["episode_ids"][~state["initialized"]] != 0).any())
                or bool((state["paths"][~state["initialized"]] != 0).any())):
            raise ValueError("Saved path and episode initialization disagree")
        source = PacerTrajectoryCurriculum(self.source.config, seed=0)
        source.load_state_dict(state["source"])
        if source.plans_generated < int(state["initialized"].sum()):
            raise ValueError("Saved private generator has fewer plans than bound actors")
        self.source.load_state_dict(state["source"])
        for name in ("paths", "episode_ids", "initialized"):
            getattr(self, name).copy_(state[name].to(self.device))
