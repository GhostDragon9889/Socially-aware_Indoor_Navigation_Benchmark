"""Opt-in fixed-actor mixed commands, without changing physical robot states.

The first half follows translated AMASS root XY; the second half keeps the
existing procedural51-knot interpolation. Equal actor counts mean equal actual
control-transition exposure, not equal resets or plans. This is an engineering
curriculum, not the unpublished NavIsaacLab author training distribution.
"""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import torch

from .amass_trajectories import AMASSRootTrajectoryCurriculum
from .trajectories import PacerTrajectoryConfig, PacerTrajectoryCurriculum


class MixedTrajectoryCurriculum:
    SCHEMA = "fixed_actor_mixed_trajectory_curriculum_v1"
    HISTORY_FRAMES = 10
    FUTURE_FRAMES = 31
    PROC_FIELDS = ("reference_ids", "initial_frames", "source_xy", "reset_xy", "episode_ids",
                   "last_observed_age", "last_plan_age", "initialized", "plan_valid", "canonical_last_plans")
    STAT_FIELDS = ("control_steps", "amass_actor_transitions", "procedural_actor_transitions",
                   "amass_resets", "procedural_resets", "amass_plans", "procedural_plans",
                   "amass_reference_samples", "procedural_reference_samples",
                   "procedural_fall_terminations", "procedural_cap_truncations")

    def __init__(self, corpus, num_envs, *, device="cpu", bounds_xy=None,
                 tail_policy="linear_velocity", procedural_config=None, seed=7):
        if type(num_envs) is not int or num_envs < 2 or num_envs % 2:
            raise ValueError("Mixed curriculum needs an even integer num_envs >=2")
        if type(seed) is not int or not 0 <= seed < 2**63-1:
            raise ValueError("Mixed seed must allow an independent seed+1 CPU generator")
        self.corpus, self.num_envs, self.half = corpus, num_envs, num_envs//2
        self.device = torch.device(device)
        if corpus.device != self.device:
            raise ValueError("Mixed corpus and curriculum destination devices must match")
        if not isinstance(corpus.generator, torch.Generator) or corpus.generator.device.type != "cpu":
            raise ValueError("Original corpus requires a CPU generator")
        if not isinstance(procedural_config, PacerTrajectoryConfig):
            raise ValueError("procedural_config must be a PacerTrajectoryConfig")
        procedural_config.validate()
        if procedural_config.horizon_s != 5 or procedural_config.knot_dt_s != .1:
            raise ValueError("Mixed commands require original procedural51 knots over five seconds")
        self.private_corpus = copy(corpus)
        self.private_corpus.generator = torch.Generator(device="cpu").manual_seed(seed+1)
        self.amass = AMASSRootTrajectoryCurriculum(self.private_corpus, self.half,
            device=self.device, bounds_xy=bounds_xy, tail_policy=tail_policy)
        self.procedural = PacerTrajectoryCurriculum(procedural_config, seed=seed)
        self.generator = self.procedural.generator
        self.config = {"num_envs": num_envs, "seed": seed,
            "amass_actor_ids": list(range(self.half)), "procedural_actor_ids": list(range(self.half, num_envs)),
            "amass_config": deepcopy(self.amass.config), "procedural_config": asdict(procedural_config),
            "amass_reset_seed": seed+1, "procedural_plan_seed": seed,
            "procedural_reset_rng": "original corpus; also used by unchanged expert/reset sampling",
            "history_frames": 10, "real_reset_future_frames": 31, "procedural_reset_future_frames": 1,
            "assignment": "fixed global first-half AMASS; second-half procedural; no branch ID in actor observation"}
        for name in self.PROC_FIELDS:
            shape = (self.half, 51, 2) if name == "canonical_last_plans" else (self.half, 2) if name in {"source_xy", "reset_xy"} else (self.half,)
            dtype = torch.float32 if name in {"source_xy", "reset_xy", "canonical_last_plans"} else torch.bool if name in {"initialized", "plan_valid"} else torch.long
            setattr(self, "proc_"+name, torch.zeros(shape, dtype=dtype, device=self.device))
        self.statistics = dict.fromkeys(self.STAT_FIELDS, 0)

    @property
    def plans_generated(self):
        return self.statistics["amass_plans"]+self.statistics["procedural_plans"]

    def _actors(self, ids):
        ids = torch.as_tensor(ids, device=self.device)
        if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64) or not len(ids) or len(ids.unique()) != len(ids) or bool(((ids < 0) | (ids >= self.num_envs)).any()):
            raise ValueError("Global actor IDs must be unique in-range integers")
        return ids.long()

    def _long(self, value, shape, name):
        AMASSRootTrajectoryCurriculum._integer(value, shape, name)
        return value.to(self.device)

    def _ages(self, ages, episode_ids=None):
        ages = self._long(ages, (self.num_envs,), "global actor ages")
        if not bool(self.amass.initialized.all()) or not bool(self.proc_initialized.all()):
            raise ValueError("Both mixed actor groups must be bound to actual resets")
        if bool((ages < 0).any()) or not torch.equal(ages[self.half:], self.proc_last_observed_age):
            raise ValueError("Procedural age differs from its completed-step clock")
        self.amass._phase(ages[:self.half], torch.arange(self.half, device=self.device),
            None if episode_ids is None else episode_ids[:self.half])
        if episode_ids is not None:
            episode_ids = self._long(episode_ids, (self.num_envs,), "global episode IDs")
            if not torch.equal(episode_ids[self.half:], self.proc_episode_ids):
                raise ValueError("Procedural episode IDs differ from bound resets")
        return ages

    @torch.no_grad()
    def sample_reference(self, actor_ids, history_frames=10):
        actors = self._actors(actor_ids)
        if type(history_frames) is not int or history_frames != self.HISTORY_FRAMES:
            raise ValueError("Mixed reference history must contain ten newest-first frames")
        motion = torch.empty(len(actors), dtype=torch.long)
        frames = torch.empty_like(motion)
        for real, source, future in ((True, self.private_corpus, 31), (False, self.corpus, 1)):
            selected = torch.nonzero((actors < self.half) == real, as_tuple=False).flatten().cpu()
            if not len(selected):
                continue
            # Keep each source's exact existing draw order, including its frame draw.
            result = source.sample_reference(len(selected), history_frames, future_frames=future)
            motion[selected], frames[selected] = result[3].cpu(), result[4].cpu()
            self.statistics["amass_reference_samples" if real else "procedural_reference_samples"] += len(selected)
        absolute = self.corpus.data["length_starts"][motion]+frames
        history = absolute[:, None]-torch.arange(history_frames)
        return (self.corpus.state_at(absolute), self.corpus.state_at(history.reshape(-1)),
                self.corpus.data["dps"][absolute+1].to(self.device), motion, frames)

    @torch.no_grad()
    def bind_reset(self, actor_ids, motion_ids, frames, source_root_xy, reset_root_xy, episode_ids):
        actors = self._actors(actor_ids)
        batch = len(actors)
        mid = self._long(motion_ids, (batch,), "reset motion IDs")
        frames = self._long(frames, (batch,), "reset source frames")
        episodes = self._long(episode_ids, (batch,), "reset episode IDs")
        for value in (source_root_xy, reset_root_xy):
            if not isinstance(value, torch.Tensor) or value.shape != (batch, 2) or not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError("Reset source/absolute roots require finite floating XY")
        if bool((episodes < 1).any()) or not bool(torch.isin(mid.cpu(), self.corpus.ids.cpu()).all()):
            raise ValueError("Reset motion/episode IDs must belong to the declared corpus")
        count = self.corpus.data["motion_num_frames"][mid.cpu()]
        required = torch.where(actors.cpu() < self.half, 31, 1)
        if bool((frames.cpu() < 9).any()) or bool((count-1-frames.cpu() < required).any()):
            raise ValueError("Reset source window crosses history/future clip boundaries")
        absolute = self.corpus.data["length_starts"][mid.cpu()]+frames.cpu()
        if not torch.equal(source_root_xy.detach().cpu(), self.corpus.data["gts"][absolute, 0, :2]):
            raise ValueError("Reset source XY differs from its actual source frame")
        real = actors < self.half
        if bool(real.any()):
            self.amass.bind_reset(actors[real], mid[real], frames[real], source_root_xy[real], reset_root_xy[real], episodes[real])
            self.statistics["amass_resets"] += int(real.sum())
        if bool((~real).any()):
            local = actors[~real]-self.half
            self.proc_reference_ids[local] = mid[~real]
            self.proc_initial_frames[local] = frames[~real]
            self.proc_source_xy[local] = source_root_xy[~real].to(self.device)
            self.proc_reset_xy[local] = reset_root_xy[~real].to(self.device)
            self.proc_episode_ids[local] = episodes[~real]
            self.proc_last_observed_age[local] = 0
            self.proc_last_plan_age[local] = 0
            self.proc_initialized[local] = True
            self.proc_plan_valid[local] = False
            self.statistics["procedural_resets"] += len(local)

    @torch.no_grad()
    def completed_step(self, ages, terminated, cap):
        ages = self._long(ages, (self.num_envs,), "completed global ages")
        for value in (terminated, cap):
            if not isinstance(value, torch.Tensor) or value.shape != (self.num_envs,) or value.dtype != torch.bool:
                raise ValueError("Termination/cap flags require global boolean tensors")
        if not bool(self.proc_initialized.all()) or not torch.equal(ages[self.half:], self.proc_last_observed_age+1):
            raise ValueError("Procedural clock must advance exactly once per physical control transition")
        flags = self.amass.completed_step(ages[:self.half], terminated[:self.half], cap[:self.half])
        self.proc_last_observed_age.copy_(ages[self.half:])
        self.statistics["control_steps"] += 1
        for name in ("amass_actor_transitions", "procedural_actor_transitions"):
            self.statistics[name] += self.half
        self.statistics["procedural_fall_terminations"] += int(terminated[self.half:].sum())
        self.statistics["procedural_cap_truncations"] += int((cap[self.half:] & ~terminated[self.half:]).sum())
        result = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        result[:self.half] = flags.to(self.device)
        return result

    @staticmethod
    def _lerp51(seconds, canonical):
        fractional = (seconds*10).clamp(0, 50)
        lower = fractional.floor().long().clamp_max(49)
        upper = lower+1
        rows = torch.arange(len(canonical), device=canonical.device)[:, None]
        return torch.lerp(canonical[rows, lower], canonical[rows, upper], (fractional-lower)[..., None])

    def _upsample(self, canonical):
        seconds = torch.arange(151, device=canonical.device, dtype=canonical.dtype)[None].expand(len(canonical), -1)/30
        return self._lerp51(seconds, canonical)

    @torch.no_grad()
    def build_plan(self, robot_state, ages, actor_ids=None, episode_ids=None):
        ages = self._ages(ages, episode_ids)
        actors = self._actors(torch.arange(self.num_envs, device=self.device) if actor_ids is None else actor_ids)
        for field, width in (("root_pos", 3), ("root_vel", 3), ("root_rot", 4)):
            value = getattr(robot_state, field, None)
            if not isinstance(value, torch.Tensor) or value.shape != (self.num_envs, width) or not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError("Mixed build_plan requires the full finite global robot state")
        xy = torch.empty(len(actors), 51, 2, device=self.device)
        native = torch.empty(len(actors), 151, 2, device=self.device)
        real = actors < self.half
        if bool(real.any()):
            prediction = self.amass.build_plan(ages[:self.half], actor_ids=actors[real],
                episode_ids=None if episode_ids is None else episode_ids[:self.half])
            xy[real], native[real] = prediction.xy, prediction.native_xy
            self.statistics["amass_plans"] += int(real.sum())
        if bool((~real).any()):
            prediction = self.procedural.build_plan(robot_state, actor_ids=actors[~real])
            local = actors[~real]-self.half
            xy[~real], native[~real] = prediction.xy, self._upsample(prediction.xy)
            self.proc_canonical_last_plans[local] = prediction.xy
            self.proc_last_plan_age[local] = ages[actors[~real]]
            self.proc_plan_valid[local] = True
            self.statistics["procedural_plans"] += len(local)
        return SimpleNamespace(xy=xy, native_xy=native, times_s=torch.arange(51, device=self.device)*.1,
                               native_times_s=torch.arange(151, device=self.device)/30, metadata=self.describe())

    def _plans(self, plans):
        if not isinstance(plans, torch.Tensor) or plans.shape != (self.num_envs, 151, 2) or not plans.is_floating_point() or not torch.isfinite(plans).all():
            raise ValueError("Mixed native plans require finite floating [global actors,151,2]")

    def interpolate_reference(self, seconds, plans):
        self._plans(plans)
        if not isinstance(seconds, torch.Tensor) or seconds.ndim != 2 or seconds.shape[0] != self.num_envs or not seconds.is_floating_point() or not torch.isfinite(seconds).all():
            raise ValueError("Mixed reference times require finite floating [global actors,queries]")
        if not bool(self.proc_plan_valid.all()):
            raise ValueError("Procedural actors must have plans for their current episodes")
        scaled = seconds[:self.half]*30
        if not torch.allclose(scaled, scaled.round(), atol=2e-5, rtol=0):
            raise ValueError("AMASS reference queries must lie on the actual30Hz source grid")
        index = scaled.round().long().clamp(0, 150)
        rows = torch.arange(self.half, device=plans.device)[:, None]
        real = plans[:self.half][rows, index.to(plans.device)]
        proc = self._lerp51(seconds[self.half:].to(self.device), self.proc_canonical_last_plans)
        return torch.cat((real, proc), 0)

    def future_targets(self, ages, plan_age, plans, episode_ids=None):
        self._ages(ages, episode_ids)
        plan_age = self._long(plan_age, (self.num_envs,), "global local-plan ages")
        if bool((plan_age < 0).any()):
            raise ValueError("Local plan ages cannot be negative")
        self._plans(plans)
        if not bool(self.proc_plan_valid.all()):
            raise ValueError("Procedural future commands require current-episode plans")
        real = self.amass.future_targets(ages[:self.half], episode_ids=None if episode_ids is None else episode_ids[:self.half])
        seconds = plan_age[self.half:, None]/30+torch.arange(1, 11, device=self.device)[None]*.5
        proc = self._lerp51(seconds, self.proc_canonical_last_plans)
        return torch.cat((real, proc), 0)

    def describe(self):
        return deepcopy({"source": "engineering_fixed_actor_amass_procedural_mixture", "config": self.config,
            "statistics": dict(self.statistics), "amass": self.amass.describe(), "procedural": self.procedural.describe(),
            "mixture_unit": "50% actual actor/control transitions; resets/plans are not balanced",
            "physical_state_modification": False, "actor_branch_id_observation": False, "reference_coordinates": "world XY only",
            "rng_scope": "real resets private seed+1; procedural resets/expert original corpus; plans private seed; not original whole-batch reset RNG sequence",
            "deployment": "training-only curriculum; no collision-free or full paper reproduction claim"})

    def state_dict(self):
        return {"schema": self.SCHEMA, "config": deepcopy(self.config), "generator_state": self.generator.get_state().clone(),
            "amass_state": deepcopy(self.amass.state_dict()), "procedural_state": self.procedural.state_dict(),
            "procedural_actors": {n: getattr(self, "proc_"+n).detach().cpu().clone() for n in self.PROC_FIELDS},
            "statistics": dict(self.statistics)}

    def load_state_dict(self, state):
        expected = {"schema", "config", "generator_state", "amass_state", "procedural_state", "procedural_actors", "statistics"}
        if not isinstance(state, dict) or set(state) != expected or state["schema"] != self.SCHEMA or state["config"] != self.config:
            raise ValueError("Mixed resume requires exact source/config/fixed assignment/schema/keys")
        actors, stats = state["procedural_actors"], state["statistics"]
        if not isinstance(actors, dict) or set(actors) != set(self.PROC_FIELDS):
            raise ValueError("Mixed resume requires all procedural actor fields")
        for name, value in actors.items():
            template = getattr(self, "proc_"+name)
            if not isinstance(value, torch.Tensor) or value.shape != template.shape or value.dtype != template.dtype or not torch.isfinite(value).all():
                raise ValueError(f"Invalid saved procedural actor field {name}")
        if not isinstance(stats, dict) or set(stats) != set(self.STAT_FIELDS) or any(type(n) is not int or n < 0 for n in stats.values()):
            raise ValueError("Mixed statistics require exact nonnegative integer counters")
        if any(stats[n] != stats["control_steps"]*self.half for n in ("amass_actor_transitions", "procedural_actor_transitions")):
            raise ValueError("Mixed actual group transitions differ from fixed assignment/control-step exposure")
        if not bool(actors["initialized"].all()) or bool((actors["episode_ids"] < 1).any()):
            raise ValueError("Mixed resume requires initialized procedural episodes")
        mid, frames = actors["reference_ids"].cpu(), actors["initial_frames"].cpu()
        if not bool(torch.isin(mid, self.corpus.ids.cpu()).all()) or bool((frames < 9).any()) or bool((self.corpus.data["motion_num_frames"][mid]-1-frames < 1).any()):
            raise ValueError("Saved procedural reset source IDs/windows are invalid")
        absolute = self.corpus.data["length_starts"][mid]+frames
        if not torch.equal(actors["source_xy"].cpu(), self.corpus.data["gts"][absolute, 0, :2]):
            raise ValueError("Saved procedural source XY differs from actual source")
        if bool((actors["last_observed_age"] < 0).any()) or bool((actors["last_plan_age"] < 0).any()) or bool((actors["last_plan_age"] > actors["last_observed_age"]).any()):
            raise ValueError("Invalid procedural saved source/control clock")
        # Validate children on isolated state holders before changing any live RNG.
        amass = copy(self.amass)
        amass.corpus = copy(self.private_corpus)
        amass.corpus.generator = torch.Generator(device="cpu")
        for name in amass.STATE_FIELDS:
            setattr(amass, name, getattr(self.amass, name).clone())
        amass.load_state_dict(state["amass_state"])
        procedural = PacerTrajectoryCurriculum(self.procedural.config, seed=0)
        procedural.load_state_dict(state["procedural_state"])
        rng = state["generator_state"]
        if not isinstance(rng, torch.Tensor) or rng.device.type != "cpu" or rng.dtype != torch.uint8 or not torch.equal(rng, procedural.generator.get_state()):
            raise ValueError("Mixed top-level and procedural RNG states must agree exactly")
        if stats["amass_actor_transitions"] != amass.statistics["control_actor_steps"] or stats["amass_resets"] != amass.statistics["resets"] or stats["amass_plans"] != amass.plans_generated or stats["procedural_plans"] != procedural.plans_generated:
            raise ValueError("Mixed parent/child statistics disagree")
        self.amass.load_state_dict(state["amass_state"])
        self.procedural.load_state_dict(state["procedural_state"])
        for name, value in actors.items():
            getattr(self, "proc_"+name).copy_(value.to(self.device))
        self.statistics = dict(stats)

    def validate_runner_state(self, ages, plan_age, plans, reference_ids, episode_ids, *, replan_steps, origins_xy):
        ages = self._ages(ages, episode_ids)
        plan_age = self._long(plan_age, (self.num_envs,), "global local-plan ages")
        reference_ids = self._long(reference_ids, (self.num_envs,), "global reference IDs")
        self._plans(plans)
        if type(replan_steps) is not int or not 1 <= replan_steps <= 30 or bool(((plan_age < 0) | (plan_age > replan_steps)).any()):
            raise ValueError("Mixed local plan ages/replan interval are invalid")
        if not isinstance(origins_xy, torch.Tensor) or origins_xy.shape != (self.num_envs, 2) or not origins_xy.is_floating_point() or not torch.isfinite(origins_xy).all():
            raise ValueError("Mixed reset origins require finite global XY")
        self.amass.validate_runner_state(ages[:self.half], plan_age[:self.half], plans[:self.half], reference_ids[:self.half], episode_ids[:self.half], replan_steps=replan_steps, origins_xy=origins_xy[:self.half])
        if not torch.equal(reference_ids[self.half:], self.proc_reference_ids) or not torch.allclose(origins_xy[self.half:].to(self.device), self.proc_reset_xy, atol=2e-5, rtol=0):
            raise ValueError("Saved procedural reset IDs/origins differ from bound source")
        active = ages[self.half:] > 0
        if bool((plan_age[self.half:][~active] != replan_steps).any()):
            raise ValueError("Newly reset procedural actors must replan before their next step")
        if not bool(self.proc_plan_valid[active].all()) or not torch.equal((ages[self.half:]-plan_age[self.half:])[active], self.proc_last_plan_age[active]):
            raise ValueError("Saved procedural plan clock/current episode is inconsistent")
        actual = self._upsample(self.proc_canonical_last_plans[active])
        if not torch.allclose(plans[self.half:][active].to(self.device), actual, atol=2e-5, rtol=0):
            raise ValueError("Saved procedural151 plan differs from its original canonical51 knots")


__all__ = ["MixedTrajectoryCurriculum"]
