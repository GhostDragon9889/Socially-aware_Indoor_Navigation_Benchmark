"""Opt-in real AMASS root-XY commands, independent of future 3D poses.

This is a training curriculum experiment, not the unpublished author protocol.
An episode has one fixed source phase and world translation. Replanning only
refreshes a window; it never reanchors the reference at the live physical root.
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import torch

from CrowdSim.humanoid.amass_corpus import file_sha256


class AMASSRootTrajectoryCurriculum:
    FUTURE_FRAMES = 31  # Real transitions after reset; at least one second.
    HISTORY_FRAMES = 10
    SCHEMA = "amass_root_trajectory_curriculum_v1"
    STATE_FIELDS = ("motion_ids", "initial_frames", "offset_xy", "end_velocity_xy", "remaining_steps",
                    "episode_ids", "last_observed_age", "last_plan_age", "initialized")
    STAT_FIELDS = ("resets", "control_actor_steps", "plans_generated", "future_knots",
                   "held_future_knots", "beyond_source_future_knots", "extrapolated_future_knots",
                   "clip_end_truncations", "cap_only_truncations",
                   "clip_end_and_cap", "fall_terminations", "fall_at_clip_end")

    def __init__(self, corpus, num_envs, *, device="cpu", bounds_xy=None, root_margin_m=1., tail_policy="hold"):
        if type(num_envs) is not int or num_envs < 1 or corpus.split != "train":
            raise ValueError("AMASS root curriculum requires a nonempty training-only actor batch")
        self.corpus, self.device, self.num_envs = corpus, torch.device(device), num_envs
        if not isinstance(tail_policy, str) or tail_policy not in {"hold", "linear_velocity"}:
            raise ValueError("AMASS tail_policy must explicitly be hold or linear_velocity")
        self.tail_policy = tail_policy
        self.ids = corpus.ids.detach().cpu().long().clone()
        if not len(self.ids) or len(self.ids.unique()) != len(self.ids):
            raise ValueError("AMASS curriculum requires unique actual manifest motion IDs")
        self.counts = corpus.data["motion_num_frames"].detach().cpu().long()
        if bool((self.counts[self.ids] <= 9+self.FUTURE_FRAMES).any()):
            raise ValueError("Every selected clip must retain real history10 and future31; no filtering fallback")
        dt = corpus.data["motion_dt"][self.ids].detach().cpu()
        if not bool(torch.isfinite(dt).all()) or not torch.allclose(dt, torch.full_like(dt, 1/30), atol=1e-7, rtol=0):
            raise ValueError("AMASS root commands require an explicitly resampled30Hz bank")
        if not torch.isfinite(torch.tensor(root_margin_m)) or root_margin_m < 0:
            raise ValueError("Root floor margin must be finite and nonnegative")
        self.bounds = None if bounds_xy is None else torch.as_tensor(bounds_xy, dtype=torch.float32).cpu()
        if self.bounds is not None and (self.bounds.shape != (4,) or not torch.isfinite(self.bounds).all()
                or not bool((self.bounds[2:]-self.bounds[:2] > 2*root_margin_m).all())):
            raise ValueError("Declared curriculum floor bounds cannot contain the root margin")
        self.records = {int(r["motion_id"]): r for r in corpus.entry["clips"]}
        self.config = {"manifest_sha256": file_sha256(corpus.manifest_path),
            "source_bank_sha256": corpus.entry["sha256"], "split": "train", "motion_ids": self.ids.tolist(),
            "future_frames": self.FUTURE_FRAMES, "history_frames": self.HISTORY_FRAMES,
            "control_dt_s": 1/30, "knot_dt_s": .1, "horizon_s": 5.,
            "bounds_xy": self.bounds.tolist() if self.bounds is not None else None, "root_margin_m": float(root_margin_m),
            "dof_velocity_frame": corpus.output_dof_velocity_frame,
            "reset_sampling": "uniform clip then uniform legal frame; no silent filtering or redistribution",
            "tail_policy": tail_policy,
            "end_velocity_formula": "(root_xy[-1]-root_xy[-11])/(10/30); eleven positions, ten actual30Hz intervals",
            "tail": "command-only assumption beyond the source; truncate at last real source frame"}
        for name in self.STATE_FIELDS:
            shape = (num_envs, 2) if name in {"offset_xy", "end_velocity_xy"} else (num_envs,)
            dtype = torch.float32 if name in {"offset_xy", "end_velocity_xy"} else torch.bool if name == "initialized" else torch.long
            setattr(self, name, torch.zeros(shape, dtype=dtype, device=self.device))
        self.statistics = dict.fromkeys(self.STAT_FIELDS, 0)

    @property
    def plans_generated(self):
        return self.statistics["plans_generated"]

    def _actors(self, ids):
        ids = torch.as_tensor(ids, device=self.device)
        if ids.ndim != 1 or ids.dtype not in (torch.int32, torch.int64) or not len(ids) or len(ids.unique()) != len(ids) or bool(((ids < 0) | (ids >= self.num_envs)).any()):
            raise ValueError("AMASS actor IDs must be unique in-range integers")
        return ids.long()

    @staticmethod
    def _integer(value, shape, name):
        if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != torch.long:
            raise ValueError(f"{name} requires an int64 tensor of shape{shape}")

    def _validate_paths(self, mid, phase, offset):
        starts, roots = self.corpus.data["length_starts"], self.corpus.data["gts"]
        for row, (motion, frame) in enumerate(zip(mid.tolist(), phase.tolist(), strict=True)):
            start = int(starts[motion])+frame
            path = roots[start:int(starts[motion])+int(self.counts[motion]), 0, :2]+offset[row]
            if not torch.isfinite(path).all():
                raise ValueError("Nonfinite actual source root trajectory")
            if self.bounds is not None:
                margin = self.config["root_margin_m"]
                if bool(((path < self.bounds[:2]+margin) | (path > self.bounds[2:]-margin)).any()):
                    raise ValueError("Actual translated clip path exceeds declared floor margin; no reanchor/resample fallback")
                if self.tail_policy == "linear_velocity":
                    end = path[-1]+5*self._end_velocity(torch.tensor([motion]))[0]
                    if bool(((end < self.bounds[:2]+margin) | (end > self.bounds[2:]-margin)).any()):
                        raise ValueError("Extrapolated five-second command exceeds declared floor margin; no fallback or resampling")

    def _end_velocity(self, mid):
        last = self.corpus.data["length_starts"][mid]+self.counts[mid]-1
        roots = self.corpus.data["gts"]
        velocity = (roots[last, 0, :2]-roots[last-10, 0, :2])/(10/30)
        if not torch.isfinite(velocity).all():
            raise ValueError("Nonfinite actual endpoint velocity estimate")
        return velocity

    def _sample_xy(self, mid, raw_frames, offset, velocity):
        last = self.counts[mid, None]-1
        beyond = raw_frames > last
        frames = raw_frames.minimum(last)
        absolute = self.corpus.data["length_starts"][mid, None]+frames
        xy = self.corpus.data["gts"][absolute, 0, :2].to(self.device)+offset[:, None]
        if self.tail_policy == "linear_velocity":
            seconds = (raw_frames-last).clamp_min(0).to(device=self.device, dtype=xy.dtype)/30
            xy = torch.where(beyond.to(self.device)[..., None], xy+seconds[..., None]*velocity[:, None], xy)
        if not torch.isfinite(xy).all():
            raise FloatingPointError("Nonfinite actual or explicitly extended AMASS root command")
        return xy, frames, beyond

    def bind_reset(self, actor_ids, motion_ids, frames, source_root_xy, reset_root_xy, episode_ids):
        """Bind the actual sampled reset once. No physical state is written here."""
        actors = self._actors(actor_ids)
        for name, value in (("motion_ids", motion_ids), ("frames", frames), ("episode_ids", episode_ids)):
            self._integer(value, (len(actors),), name)
        mid, phase = motion_ids.cpu(), frames.cpu()
        if not torch.isin(mid, self.ids).all():
            raise ValueError("Reset motion ID is outside the training manifest")
        remaining = self.counts[mid]-1-phase
        if bool((phase < 9).any()) or bool((remaining < self.FUTURE_FRAMES).any()) or bool((episode_ids < 1).any()):
            raise ValueError("Reset phase must have actual history10 and future31 within its source clip")
        for value in (source_root_xy, reset_root_xy):
            if not isinstance(value, torch.Tensor) or value.shape != (len(actors), 2) or not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError("Reset root XY must be finite floating-point[B,2]")
        offset = reset_root_xy.detach().cpu()-source_root_xy.detach().cpu()
        starts = self.corpus.data["length_starts"]
        roots = self.corpus.data["gts"]
        actual_source_root = roots[starts[mid]+phase, 0, :2]
        if not torch.equal(source_root_xy.detach().cpu(), actual_source_root):
            raise ValueError("Reset source root must match its actual motion/frame")
        self._validate_paths(mid, phase, offset)
        self.motion_ids[actors] = mid.to(self.device)
        self.initial_frames[actors] = phase.to(self.device)
        self.offset_xy[actors] = offset.to(self.device)
        self.end_velocity_xy[actors] = self._end_velocity(mid).to(self.device)
        self.remaining_steps[actors] = remaining.to(self.device)
        self.episode_ids[actors] = episode_ids.to(self.device)
        self.last_observed_age[actors] = 0
        self.last_plan_age[actors] = 0
        self.initialized[actors] = True
        self.statistics["resets"] += len(actors)

    def _phase(self, ages, actors, episode_ids=None):
        self._integer(ages, (self.num_envs,), "source ages")
        current = ages.to(self.device)[actors]
        if not bool(self.initialized[actors].all()) or bool(((current < 0) | (current > self.remaining_steps[actors])).any()):
            raise ValueError("Source phase is unbound or outside its actual clip")
        if not torch.equal(current, self.last_observed_age[actors]):
            raise ValueError("Source phase differs from its committed30Hz physical clock")
        if episode_ids is not None:
            self._integer(episode_ids, (self.num_envs,), "episode IDs")
            if not torch.equal(episode_ids.to(self.device)[actors], self.episode_ids[actors]):
                raise ValueError("Source phase belongs to a different physical episode")
        return current

    @torch.no_grad()
    def build_plan(self, ages, actor_ids=None, *, episode_ids=None):
        actors = self._actors(torch.arange(self.num_envs, device=self.device) if actor_ids is None else actor_ids)
        age = self._phase(ages, actors, episode_ids)
        mid = self.motion_ids[actors].cpu()
        frame = (self.initial_frames[actors]+age).cpu()
        raw = frame[:, None]+torch.arange(51)*3
        xy, source_frames, beyond = self._sample_xy(mid, raw, self.offset_xy[actors], self.end_velocity_xy[actors])
        native_xy, native_frames, native_beyond = self._sample_xy(mid, frame[:, None]+torch.arange(151),
            self.offset_xy[actors], self.end_velocity_xy[actors])
        self.last_plan_age[actors] = age
        self.statistics["plans_generated"] += len(actors)
        self.statistics["future_knots"] += len(actors)*50
        tail_count = int(beyond[:, 1:].sum())
        self.statistics["beyond_source_future_knots"] += tail_count
        self.statistics["held_future_knots" if self.tail_policy == "hold" else "extrapolated_future_knots"] += tail_count
        return SimpleNamespace(xy=xy, time=torch.arange(51, device=self.device)*.1,
            native_xy=native_xy, native_time=torch.arange(151, device=self.device)/30,
            motion_ids=mid, source_frames=source_frames, native_source_frames=native_frames,
            beyond_source=beyond, native_beyond_source=native_beyond,
            tail_hold=beyond if self.tail_policy == "hold" else torch.zeros_like(beyond), selected_sample=None,
            source_phase=frame, source_offset_xy=self.offset_xy[actors].detach().clone())

    @torch.no_grad()
    def future_targets(self, ages, *, episode_ids=None):
        """Ten targets from the current physical phase, no replan-window clamp."""
        actors = torch.arange(self.num_envs, device=self.device)
        current = self._phase(ages, actors, episode_ids)
        phase = (self.initial_frames+current).cpu()
        xy, _, _ = self._sample_xy(self.motion_ids.cpu(), phase[:, None]+torch.arange(15, 151, 15),
            self.offset_xy, self.end_velocity_xy)
        return xy

    def completed_step(self, ages, terminated, cap_truncated):
        """Observe the same completed physical step; never advance physics."""
        self._integer(ages, (self.num_envs,), "source ages")
        ages = ages.to(self.device)
        for value in (terminated, cap_truncated):
            if not isinstance(value, torch.Tensor) or value.shape != (self.num_envs,) or value.dtype != torch.bool:
                raise ValueError("Physical termination masks require bool[actors]")
        if not bool(self.initialized.all()) or not torch.equal(ages, self.last_observed_age+1) or bool((ages > self.remaining_steps).any()):
            raise ValueError("Exactly one completed30Hz step is required; reset a completed clip before stepping")
        clip_end = ages >= self.remaining_steps
        terminated, cap_truncated = terminated.to(self.device), cap_truncated.to(self.device)
        alive = ~terminated
        self.statistics["control_actor_steps"] += self.num_envs
        self.statistics["clip_end_truncations"] += int((clip_end & alive).sum())
        self.statistics["cap_only_truncations"] += int((cap_truncated & ~clip_end & alive).sum())
        self.statistics["clip_end_and_cap"] += int((clip_end & cap_truncated & alive).sum())
        self.statistics["fall_terminations"] += int(terminated.sum())
        self.statistics["fall_at_clip_end"] += int((terminated & clip_end).sum())
        self.last_observed_age.copy_(ages)
        return clip_end

    def describe(self):
        stats = dict(self.statistics)
        stats["sampled_plan_tail_hold_fraction"] = stats["held_future_knots"]/stats["future_knots"] if stats["future_knots"] else None
        stats["sampled_plan_beyond_source_fraction"] = stats["beyond_source_future_knots"]/stats["future_knots"] if stats["future_knots"] else None
        stats["sampled_plan_extrapolated_fraction"] = stats["extrapolated_future_knots"]/stats["future_knots"] if stats["future_knots"] else None
        return {"variant": "engineering_real_amass_root_curriculum", "author_training_protocol": False,
            "config": deepcopy(self.config), "statistics": stats,
            "commands": "real prefix source root XY + explicitly engineered clip-outside tail; no future3D poses/actions",
            "phase": "initial sampled source frame + committed physical episode age",
            "translation": "once at reset: intended world resetXY minus source resetXY; never live-root reanchoring",
            "RNG": "existing corpus generator at reset only; replans consume no random numbers",
            "canonical_plan": "51@10Hz, actual real-prefix source knots plus declared tail; counters refer to50 future knots",
            "consumed_plan": "151@30Hz, exact real-prefix source frames plus declared tail; no10Hz interpolation approximation",
            "body_origin_floor_margin_is_not_fullbody_safety_guarantee": True}

    def state_dict(self):
        return {"schema": self.SCHEMA, "config": deepcopy(self.config), "statistics": dict(self.statistics),
                "generator_state": self.corpus.generator.get_state().clone(),
                "actors": {name: getattr(self, name).detach().cpu().clone() for name in self.STATE_FIELDS}}

    def load_state_dict(self, state):
        if not isinstance(state, dict) or set(state) != {"schema", "config", "statistics", "generator_state", "actors"} or state["schema"] != self.SCHEMA or state["config"] != self.config:
            raise ValueError("AMASS curriculum resume requires its exact source/config/schema")
        stats, actors = state["statistics"], state["actors"]
        if not isinstance(stats, dict) or set(stats) != set(self.STAT_FIELDS) or any(type(v) is not int or v < 0 for v in stats.values()):
            raise ValueError("AMASS curriculum requires complete nonnegative counters")
        if (stats["beyond_source_future_knots"] > stats["future_knots"] or stats["future_knots"] != stats["plans_generated"]*50
                or stats["held_future_knots"]+stats["extrapolated_future_knots"] != stats["beyond_source_future_knots"]
                or stats["extrapolated_future_knots" if self.tail_policy == "hold" else "held_future_knots"] != 0):
            raise ValueError("AMASS curriculum knot counters disagree")
        if not isinstance(actors, dict) or set(actors) != set(self.STATE_FIELDS):
            raise ValueError("AMASS curriculum requires all actor phase/offset/end state")
        for name in self.STATE_FIELDS:
            expected, value = getattr(self, name), actors[name]
            if not isinstance(value, torch.Tensor) or value.shape != expected.shape or value.dtype != expected.dtype or not torch.isfinite(value).all():
                raise ValueError(f"Invalid saved AMASS actor field{name}")
        cpu = {k:v.detach().cpu() for k,v in actors.items()}
        mid = cpu["motion_ids"]
        if (not bool(cpu["initialized"].all()) or not bool(torch.isin(mid, self.ids).all())
                or bool((cpu["episode_ids"] < 1).any()) or bool((cpu["initial_frames"] < 9).any())):
            raise ValueError("Saved AMASS actors are not bound to legal training clips")
        remaining = self.counts[mid]-1-cpu["initial_frames"]
        if not torch.equal(remaining, cpu["remaining_steps"]) or bool((remaining < self.FUTURE_FRAMES).any()) or bool(((cpu["last_observed_age"] < 0) | (cpu["last_observed_age"] > remaining)).any()) or bool(((cpu["last_plan_age"] < 0) | (cpu["last_plan_age"] > cpu["last_observed_age"])).any()):
            raise ValueError("Saved AMASS source phase/end/plan clocks disagree")
        self._validate_paths(mid, cpu["initial_frames"], cpu["offset_xy"])
        if not torch.allclose(cpu["end_velocity_xy"], self._end_velocity(mid), atol=2e-5, rtol=0):
            raise ValueError("Saved endpoint velocity differs from the actual last eleven source positions")
        rng = state["generator_state"]
        if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or rng.device.type != "cpu" or rng.shape != self.corpus.generator.get_state().shape:
            raise ValueError("AMASS curriculum requires complete CPU corpus RNG state")
        validator = torch.Generator(device="cpu")
        validator.set_state(rng)
        for name, value in actors.items():
            getattr(self, name).copy_(value.to(self.device))
        self.statistics = dict(stats)
        self.corpus.generator.set_state(rng)

    def validate_runner_state(self, ages, plan_age, plans, reference_ids, episode_ids, *, replan_steps, origins_xy):
        actors = torch.arange(self.num_envs, device=self.device)
        current = self._phase(ages, actors, episode_ids)
        self._integer(plan_age, (self.num_envs,), "plan ages")
        self._integer(reference_ids, (self.num_envs,), "world reference IDs")
        if not torch.equal(reference_ids.to(self.device), self.motion_ids):
            raise ValueError("Saved world reference IDs differ from AMASS commands")
        if type(replan_steps) is not int or not 1 <= replan_steps <= 30:
            raise ValueError("Saved AMASS replan interval must match the supported runner contract")
        if bool((plan_age < 0).any()) or bool((plan_age > replan_steps).any()):
            raise ValueError("Invalid saved AMASS local plan ages")
        if not isinstance(origins_xy, torch.Tensor) or origins_xy.shape != (self.num_envs, 2) or not torch.isfinite(origins_xy).all():
            raise ValueError("Saved AMASS source translation requires declared world reset origins")
        reset_absolute = self.corpus.data["length_starts"][self.motion_ids.cpu()]+self.initial_frames.cpu()
        expected_offset = origins_xy.detach().cpu()-self.corpus.data["gts"][reset_absolute, 0, :2]
        if not torch.allclose(self.offset_xy.cpu(), expected_offset, atol=2e-5, rtol=0):
            raise ValueError("Saved AMASS fixed offset differs from its source reset and world origin")
        # Newly reset actors have no plan yet, and the next loop replans them.
        active = current > 0
        if bool((plan_age.to(self.device)[~active] != replan_steps).any()):
            raise ValueError("Newly reset AMASS actors must replan before their next control step")
        if not torch.equal((current-plan_age.to(self.device))[active], self.last_plan_age[active]):
            raise ValueError("Saved AMASS local plan age differs from the source clock")
        if not isinstance(plans, torch.Tensor) or plans.shape != (self.num_envs, 151, 2) or not torch.isfinite(plans).all():
            raise ValueError("Saved AMASS root plan must be finite[actors,151,2]")
        mid = self.motion_ids[active].cpu()
        phase = (self.initial_frames[active]+self.last_plan_age[active]).cpu()
        actual_plan, _, _ = self._sample_xy(mid, phase[:, None]+torch.arange(151), self.offset_xy[active], self.end_velocity_xy[active])
        if not torch.allclose(plans.to(self.device)[active], actual_plan, atol=2e-5, rtol=0):
            raise ValueError("Saved AMASS plan differs from its declared source frame and fixed offset")


__all__ = ["AMASSRootTrajectoryCurriculum"]
