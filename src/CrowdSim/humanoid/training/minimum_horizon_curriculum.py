"""Opt-in future81 AMASS reset experiment; the original curriculum is untouched.

All selected clips must support history10 plus81 real future transitions.
Only the real-reference reset frame law changes. Expert sampling, procedural
reset RNG, targets and physical group reset semantics remain caller-owned.
This type needs explicit admission by a future training World/entry.
"""
from __future__ import annotations

from copy import deepcopy

import torch

from .amass_trajectories import AMASSRootTrajectoryCurriculum
from .mixed_trajectories import MixedTrajectoryCurriculum


class MinimumHorizonAMASSCurriculum(AMASSRootTrajectoryCurriculum):
    FUTURE_FRAMES = 81
    SCHEMA = "amass_root_trajectory_minimum_future81_v1"

    def __init__(self, corpus, num_envs, **kwargs):
        ids = corpus.ids.detach().cpu().long()
        if bool((corpus.data["motion_num_frames"][ids] < self.HISTORY_FRAMES + self.FUTURE_FRAMES).any()):
            raise ValueError("Every selected AMASS clip must retain history10 plus future81; no filtering fallback")
        super().__init__(corpus, num_envs, **kwargs)


class MinimumHorizonMixedCurriculum(MixedTrajectoryCurriculum):
    FUTURE_FRAMES = 81
    SCHEMA = "fixed_actor_mixed_minimum_future81_v1"

    def __init__(self, corpus, num_envs, *, device="cpu", bounds_xy=None,
                 tail_policy="linear_velocity", procedural_config=None, seed=7):
        # Reject the entire experiment before changing any source generator.
        ids = corpus.ids.detach().cpu().long()
        if bool((corpus.data["motion_num_frames"][ids] < self.HISTORY_FRAMES + self.FUTURE_FRAMES).any()):
            raise ValueError("Every selected AMASS clip must retain history10 plus future81; no filtering fallback")
        super().__init__(corpus, num_envs, device=device, bounds_xy=bounds_xy,
                         tail_policy=tail_policy, procedural_config=procedural_config, seed=seed)
        # Constructors allocate/reset holders but consume no random draws.
        # Replace only this instance's unused AMASS holder; do not patch classes.
        self.amass = MinimumHorizonAMASSCurriculum(self.private_corpus, self.half,
            device=self.device, bounds_xy=bounds_xy, tail_policy=tail_policy)
        self.config["amass_config"] = deepcopy(self.amass.config)
        self.config["real_reset_future_frames"] = self.FUTURE_FRAMES
        self.config["reset_horizon_experiment"] = {
            "baseline_real_future_frames": 31, "candidate_real_future_frames": self.FUTURE_FRAMES,
            "changed_sampling_unit": "minimum remaining source transitions for AMASS reference reset frame only",
            "training_clip_filtering": False, "expert_sampling_changed": False,
            "procedural_reset_sampling_changed": False, "author_recipe_equivalence_claimed": False}

    @torch.no_grad()
    def sample_reference(self, actor_ids, history_frames=10):
        actors = self._actors(actor_ids)
        if type(history_frames) is not int or history_frames != self.HISTORY_FRAMES:
            raise ValueError("Mixed reference history must contain ten newest-first frames")
        motion = torch.empty(len(actors), dtype=torch.long)
        frames = torch.empty_like(motion)
        for real, source, future in ((True, self.private_corpus, self.FUTURE_FRAMES), (False, self.corpus, 1)):
            selected = torch.nonzero((actors < self.half) == real, as_tuple=False).flatten().cpu()
            if not len(selected):
                continue
            result = source.sample_reference(len(selected), history_frames, future_frames=future)
            motion[selected], frames[selected] = result[3].cpu(), result[4].cpu()
            self.statistics["amass_reference_samples" if real else "procedural_reference_samples"] += len(selected)
        absolute = self.corpus.data["length_starts"][motion] + frames
        history = absolute[:, None] - torch.arange(history_frames)
        return (self.corpus.state_at(absolute), self.corpus.state_at(history.reshape(-1)),
                self.corpus.data["dps"][absolute + 1].to(self.device), motion, frames)

    @torch.no_grad()
    def bind_reset(self, actor_ids, motion_ids, frames, source_root_xy, reset_root_xy, episode_ids):
        actors = self._actors(actor_ids)
        mid = self._long(motion_ids, (len(actors),), "reset motion IDs").cpu()
        phase = self._long(frames, (len(actors),), "reset source frames").cpu()
        if not bool(torch.isin(mid, self.corpus.ids.cpu()).all()):
            raise ValueError("Reset source ID must belong to the declared corpus")
        count = self.corpus.data["motion_num_frames"][mid]
        required = torch.where(actors.cpu() < self.half, self.FUTURE_FRAMES, 1)
        if bool((phase < self.HISTORY_FRAMES - 1).any()) or bool((count - 1 - phase < required).any()):
            raise ValueError("Reset source must retain history10 and AMASS future81/procedural future1")
        return super().bind_reset(actor_ids, motion_ids, frames, source_root_xy, reset_root_xy, episode_ids)
