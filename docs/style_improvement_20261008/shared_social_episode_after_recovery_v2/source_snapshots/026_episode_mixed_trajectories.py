"""Independent mixed curriculum: original AMASS31 plus episode-bound procedural.

Only reference ownership changes. The caller retains physics, rewards, expert
sampling and group resets. Original repeated plan views now query a fixed path.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import torch

from .episode_trajectories import EpisodeProceduralReference
from .mixed_trajectories import MixedTrajectoryCurriculum


class EpisodeAnchoredMixedCurriculum(MixedTrajectoryCurriculum):
    SCHEMA = "fixed_actor_mixed_episode_procedural_v1"
    STAT_FIELDS = MixedTrajectoryCurriculum.STAT_FIELDS + ("procedural_plan_views",)

    def __init__(self, corpus, num_envs, *, episode_controls=300, device="cpu", bounds_xy=None,
                 tail_policy="linear_velocity", procedural_config=None, seed=7):
        if type(episode_controls) is not int or episode_controls < 1:
            raise ValueError("Require a positive physical episode control cap")
        super().__init__(corpus, num_envs, device=device, bounds_xy=bounds_xy,
                         tail_policy=tail_policy, procedural_config=procedural_config, seed=seed)
        self.episode_controls = episode_controls
        # Four-control SDK smoke keeps its physical cap; only the knot buffer rounds up.
        self.episode_reference = EpisodeProceduralReference(self.half,
            episode_controls=(episode_controls + 2) // 3 * 3, config=procedural_config, seed=seed, device=device)
        self.procedural = self.episode_reference.source
        self.generator = self.procedural.generator
        self.config["procedural_config"] = asdict(self.procedural.config)
        self.config["episode_reference"] = deepcopy(self.episode_reference.config)
        self.config["physical_episode_controls"] = episode_controls
        self.config["reference_ownership"] = "procedural path bound once at episode reset; unchanged AMASS31 binding"

    def _cap(self, ages):
        if bool((ages[self.half:] > self.episode_controls).any()):
            raise ValueError("Procedural clock exceeds the caller's physical episode cap")

    @torch.no_grad()
    def build_plan(self, robot_state, ages, actor_ids=None, episode_ids=None):
        ages = self._ages(ages, episode_ids)
        self._cap(ages)
        actors = self._actors(torch.arange(self.num_envs, device=self.device) if actor_ids is None else actor_ids)
        for name, width in (("root_pos", 3), ("root_vel", 3), ("root_rot", 4)):
            value = getattr(robot_state, name, None)
            if (not isinstance(value, torch.Tensor) or value.shape != (self.num_envs, width)
                    or value.device != self.device or not value.is_floating_point() or not torch.isfinite(value).all()):
                raise ValueError("Episode mixed planning requires full finite on-device root state")
        real = actors < self.half
        xy = torch.empty(len(actors), 51, 2, device=self.device)
        native = torch.empty(len(actors), 151, 2, device=self.device)
        if bool(real.any()):
            prediction = super().build_plan(robot_state, ages, actor_ids=actors[real], episode_ids=episode_ids)
            xy[real], native[real] = prediction.xy, prediction.native_xy
        if bool((~real).any()):
            local = actors[~real] - self.half
            fresh = local[~self.proc_plan_valid[local]]
            if len(fresh):
                if bool((ages[self.half:][fresh] != 0).any()):
                    raise ValueError("A new procedural episode must bind its path at reset age0")
                state = SimpleNamespace(**{name: getattr(robot_state, name)[self.half:]
                    for name in ("root_pos", "root_vel", "root_rot")})
                self.episode_reference.reset(state, self.proc_episode_ids, fresh)
                self.statistics["procedural_plans"] += len(fresh)
            offsets = torch.arange(51, device=self.device).float() * .1
            xy[~real] = self.episode_reference.query(ages[self.half:], self.proc_episode_ids, offsets, local)
            native[~real] = self.episode_reference.query(ages[self.half:], self.proc_episode_ids,
                                                        torch.arange(151, device=self.device).float() / 30, local)
            self.proc_canonical_last_plans[local] = xy[~real]
            self.proc_last_plan_age[local] = ages[actors[~real]]
            self.proc_plan_valid[local] = True
            self.statistics["procedural_plan_views"] += len(local)
        return SimpleNamespace(xy=xy, native_xy=native, times_s=torch.arange(51, device=self.device) * .1,
            native_times_s=torch.arange(151, device=self.device) / 30, metadata=self.describe())

    def interpolate_reference(self, seconds, plans):
        # Keep the original AMASS grid lookup and all original native/time validation.
        result = super().interpolate_reference(seconds, plans)
        controls = seconds[self.half:] * 30
        if controls.shape[1] < 1 or bool((controls < 0).any()) or not torch.allclose(controls, controls.round(), atol=2e-5, rtol=0):
            raise ValueError("Episode reward references must use actual30Hz local control queries")
        columns = []
        for column in controls.unbind(1):
            ages = self.proc_last_plan_age + column.round().long()
            if bool((ages > self.episode_controls).any()):
                raise ValueError("Reward reference exceeds the physical episode cap")
            columns.append(self.episode_reference.query(ages, self.proc_episode_ids, [0.])[:, 0])
        result[self.half:] = torch.stack(columns, 1)
        return result

    def future_targets(self, ages, plan_age, plans, episode_ids=None):
        ages = self._ages(ages, episode_ids)
        self._cap(ages)
        plan_age = self._long(plan_age, (self.num_envs,), "global local-plan ages")
        self._plans(plans)
        if bool((plan_age < 0).any()) or not bool(self.proc_plan_valid.all()):
            raise ValueError("Require nonnegative local plan ages and bound procedural paths")
        real = self.amass.future_targets(ages[:self.half], episode_ids=None if episode_ids is None else episode_ids[:self.half])
        proc = self.episode_reference.query(ages[self.half:], self.proc_episode_ids,
                                           torch.arange(1, 11, device=self.device).float() * .5)
        return torch.cat((real, proc))

    def describe(self):
        result = super().describe()
        result["source"] = "engineering_fixed_actor_amass_episode_procedural_mixture"
        result["procedural_plan_counts"] = "paths generated at episode reset; separate procedural_plan_views count"
        result["author_recipe_equivalence_claimed"] = False
        return result

    def state_dict(self):
        # Original corpus RNG belongs to the caller/expert sampler, as in the baseline.
        return {**super().state_dict(), "episode_reference": self.episode_reference.state_dict()}

    def load_state_dict(self, state):
        if not isinstance(state, dict) or "episode_reference" not in state:
            raise ValueError("Require saved episode-bound reference state")
        temporary = EpisodeProceduralReference(self.half,
            episode_controls=self.episode_reference.episode_controls,
            config=type(self.procedural.config)(**{**asdict(self.procedural.config), "horizon_s": 5.}), device=self.device)
        temporary.load_state_dict(state["episode_reference"])
        original = {key: value for key, value in state.items() if key != "episode_reference"}
        if "procedural_state" not in original or "procedural_actors" not in original:
            raise ValueError("Require original mixed procedural ownership state")
        left, right = original["procedural_state"], state["episode_reference"]["source"]
        if (not isinstance(left, dict) or set(left) != set(right)
                or any(left[key] != right[key] for key in ("schema", "config", "plans_generated"))
                or not isinstance(left["generator_state"], torch.Tensor)
                or not torch.equal(left["generator_state"], right["generator_state"])):
            raise ValueError("Mixed and episode reference private generator states disagree")
        fields = original["procedural_actors"]
        if not isinstance(fields, dict) or set(fields) != set(self.PROC_FIELDS):
            raise ValueError("Require every original procedural actor field")
        for name, value in fields.items():
            template = getattr(self, "proc_" + name)
            if (not isinstance(value, torch.Tensor) or value.shape != template.shape
                    or value.dtype != template.dtype or not torch.isfinite(value).all()):
                raise ValueError("Invalid saved procedural actor field: " + name)
        active = original["procedural_actors"]["plan_valid"].to(self.device)
        episodes = original["procedural_actors"]["episode_ids"].to(self.device)
        phase = original["procedural_actors"]["last_plan_age"].to(self.device)
        if not bool(temporary.initialized[active].all()) or not torch.equal(temporary.episode_ids[active], episodes[active]):
            raise ValueError("Saved active path belongs to another episode")
        if bool(active.any()):
            expected = temporary.query(phase, episodes, torch.arange(51, device=self.device).float() * .1,
                                       active.nonzero().flatten())
            if not torch.equal(expected, original["procedural_actors"]["canonical_last_plans"][active.cpu()].to(self.device)):
                raise ValueError("Saved procedural view differs from its persistent path")
        super().load_state_dict(original)  # Original atomic child/clock/counter validation remains.
        for name in ("paths", "episode_ids", "initialized"):
            getattr(self.episode_reference, name).copy_(getattr(temporary, name))

    def validate_runner_state(self, ages, plan_age, plans, reference_ids, episode_ids, *, replan_steps, origins_xy):
        ages = self._ages(ages, episode_ids)
        self._cap(ages)
        plan_age = self._long(plan_age, (self.num_envs,), "global local-plan ages")
        reference_ids = self._long(reference_ids, (self.num_envs,), "global reference IDs")
        self._plans(plans)
        if type(replan_steps) is not int or not 1 <= replan_steps <= 30 or bool(((plan_age < 0) | (plan_age > replan_steps)).any()):
            raise ValueError("Mixed local plan ages/replan interval are invalid")
        if not isinstance(origins_xy, torch.Tensor) or origins_xy.shape != (self.num_envs, 2) or not torch.isfinite(origins_xy).all():
            raise ValueError("Mixed reset origins require finite global XY")
        self.amass.validate_runner_state(ages[:self.half], plan_age[:self.half], plans[:self.half], reference_ids[:self.half],
            episode_ids[:self.half], replan_steps=replan_steps, origins_xy=origins_xy[:self.half])
        if not torch.equal(reference_ids[self.half:], self.proc_reference_ids) or not torch.allclose(origins_xy[self.half:].to(self.device), self.proc_reset_xy, atol=2e-5, rtol=0):
            raise ValueError("Saved procedural reset IDs/origins differ")
        active = ages[self.half:] > 0
        if bool((plan_age[self.half:][~active] != replan_steps).any()) or not bool(self.proc_plan_valid[active].all()):
            raise ValueError("Reset actors must replan; active actors require bound paths")
        if not torch.equal((ages[self.half:] - plan_age[self.half:])[active], self.proc_last_plan_age[active]):
            raise ValueError("Saved procedural view clock differs")
        if bool(active.any()):
            expected = self.episode_reference.query(self.proc_last_plan_age, self.proc_episode_ids,
                torch.arange(151, device=self.device).float() / 30, active.nonzero().flatten())
            if not torch.equal(plans[self.half:][active].to(self.device), expected):
                raise ValueError("Saved native view differs from its episode-bound reference")
