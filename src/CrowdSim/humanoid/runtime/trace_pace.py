"""TRACE→AMP clock and coordinate boundary, with actual measured state feedback.

Planning is at a locally chosen 2 Hz and PD control at 30 Hz (PACER's 10 future XY targets at
0..4.5 seconds). The final horizon is explicitly held between replans; no root
or joint pose is overwritten. Trace and Pace Sec. 3.3 replans every TWO seconds;
the shorter interval is an explicit reliability adaptation, not that protocol.
"""

from dataclasses import dataclass
import math

import torch


def trace_inputs_from_world(world, *, stationary_route_targets=None, pedestrian_extent_m=None):
    """Nav actual velocity → ORCA-trained TRACE speed and velocity heading.

    Author orca_dataset.py defines heading=atan2(vy,vx). Use the measured
    pelvis yaw only below .1m/s, where velocity heading becomes unstable.
    The geometric raster MUST use the same new heading, independently of the
    actual body orientation used by AMP. Never rotate an existing crop: that
    invents unknown pixels and changes the walkable domain.
    """
    packet = world.planner_inputs()

    def convert(value):
        speed = value[..., 2:4].norm(dim=-1, keepdim=True)
        heading = torch.where(
            speed >= 0.1,
            torch.atan2(value[..., 3:4], value[..., 2:3]),
            value[..., 4:5],
        )
        return torch.cat(
            (
                value[..., :2],
                speed,
                heading,
            ),
            -1,
        )

    converted = {
        name: convert(packet[name]) for name in ("states", "history", "neighbors")
    }
    if stationary_route_targets is not None:
        targets = torch.as_tensor(
            stationary_route_targets, device=converted["states"].device
        )
        if (
            targets.ndim != 3
            or targets.shape[0] != len(converted["states"])
            or targets.shape[-1] != 2
            or not torch.isfinite(targets).all()
        ):
            raise ValueError("Finite stationary intent targets[B,K,2] required")
        delta = targets - converted["states"][:, None, :2]
        usable = delta.norm(dim=-1) >= 0.25
        first = usable.long().argmax(1)
        tangent = delta[torch.arange(len(delta), device=delta.device), first]
        intent = torch.atan2(tangent[:, 1], tangent[:, 0])
        use_intent = (converted["states"][:, 2] < 0.1) & usable.any(1)
        converted["states"][:, 3] = torch.where(
            use_intent, intent, converted["states"][:, 3]
        )
        converted["history"][:, -1] = converted["states"]
        # Only an undefined near-stationary velocity heading is re-expressed.
        # Past moving headings, measured XY/speed and all body poses stay actual.
    scene = getattr(world, "validation_scene", None)
    raster = packet["raster"]
    if scene is not None:
        nav = packet["states"].clone()
        nav[..., 4] = converted["states"][..., 3]
        raster = scene.raster(nav)
    elif raster is not None:
        # This adapter supports the explicitly unobstructed flat fixture or a
        # geometry provider. An obstacle-bearing old crop cannot be reused.
        if not bool((raster[:, 0] == 1).all() and (raster[:, 1] == 0).all()):
            raise ValueError(
                "Heading-aligned obstacle raster requires a geometry provider"
            )
    if pedestrian_extent_m is not None:
        extent = validate_pedestrian_extent(pedestrian_extent_m)
        batch = len(converted["states"])
        packet["extents"] = converted["states"].new_tensor(extent).expand(batch, 3)
        packet["neighbor_extents"] = packet["extents"][:, None].expand(
            batch, converted["neighbors"].shape[1], 3
        )
    return {
        **packet,
        **converted,
        "raster": raster,
    }


def validate_pedestrian_extent(value):
    """Explicit body envelope used consistently by TRACE, guidance and guards."""
    if not isinstance(value, (list, tuple)) or len(value) != 3 or any(
        type(v) not in (int, float) or not math.isfinite(v) or not 0 < v <= 3
        for v in value
    ):
        raise ValueError("Pedestrian extent must be three positive finite dimensions <=3m")
    return tuple(float(v) for v in value)


def timed_amp_waypoints(plan, elapsed_s, *, first_sample_s=0.0):
    """Public PACER samples t+0,...,t+4.5; legacy+.5 is explicit opt-in."""
    if isinstance(first_sample_s, bool) or first_sample_s not in (0.0, 0.5):
        raise ValueError("Public PACER start0 or explicit legacy start.5 required")
    if not math.isfinite(elapsed_s) or elapsed_s < 0:
        raise ValueError("Plan age must be finite and nonnegative")
    time = plan.time
    if (
        time.ndim != 1
        or len(time) < 2
        or not torch.isfinite(time).all()
        or not bool((time[1:] > time[:-1]).all())
        or float(time[0]) != 0
    ):
        raise ValueError(
            "Strictly increasing finite plan times starting at zero are required"
        )
    query = torch.arange(10, device=time.device, dtype=time.dtype) * 0.5 + elapsed_s + first_sample_s
    clipped = query.clamp(max=time[-1])
    right = torch.searchsorted(time, clipped).clamp(1, len(time) - 1)
    left = right - 1
    fraction = (clipped - time[left]) / (time[right] - time[left])
    targets = torch.lerp(plan.xy[:, left], plan.xy[:, right], fraction[None, :, None])
    return targets, int((query > time[-1]).sum())


@dataclass
class TracePaceClosedLoop:
    """One explicit AMP controller, actual DDPM planner; no fallback families."""

    planner: object
    controller: object
    guidance_config: object
    replan_controls: int = 15
    num_samples: int = 4
    seed: int = 7
    plan: object = None
    plan_control: int = -1
    last_diagnostics: object = None
    last_control: int = -1
    allow_safety_braking: bool = False
    safety_braking_plans: int = 0
    last_rejection_diagnostics: object = None
    braking_agent_buffer_m: float = 0.05
    stationary_route_heading: bool = False
    mixed_safety_brakes: bool = False
    guard_horizon_s: float = 1.0
    guard_tracking_margin_m: float = 0.0
    map_guard_clearance_m: float | None = None
    persistent_braking_target: bool = False
    actual_body_map_guard: bool = False
    body_guard_horizon_s: float = .3
    body_guard_gap_m: float = .025
    body_guard_geometry: object = None
    pedestrian_extent_m: tuple | None = None
    amp_sample_start_s: float = 0.0
    braking_anchor_xy: object = None
    braking_anchor_mask: object = None
    settle_braking_target: bool = False
    settled_braking_helper: object = None

    def __post_init__(self):
        if type(self.settle_braking_target) is not bool or (self.settle_braking_target and not(self.persistent_braking_target and self.actual_body_map_guard)):
            raise ValueError("Measured settled holding intent requires persistent braking and actual body guard")
        if type(self.actual_body_map_guard) is not bool or (self.actual_body_map_guard and not self.allow_safety_braking):
            raise ValueError("Body map guard requires explicit safety braking")
        for name,lower,upper in (('body_guard_horizon_s',.3,1.),('body_guard_gap_m',.025,.2)):
            value=getattr(self,name)
            if type(value) not in (int,float) or not math.isfinite(value) or not lower<=value<=upper:raise ValueError('Finite body guard margin must retain baseline minimum: '+name)
        if self.pedestrian_extent_m is not None:
            self.pedestrian_extent_m = validate_pedestrian_extent(self.pedestrian_extent_m)
        if type(self.replan_controls) is not int or not 1 <= self.replan_controls <= 15:
            raise ValueError("Replan at least every .5s at 30Hz")
        if type(self.num_samples) is not int or self.num_samples < 1:
            raise ValueError("Positive sample budget required")
        if type(self.allow_safety_braking) is not bool:
            raise ValueError("Safety braking must be explicitly enabled")
        if type(self.stationary_route_heading) is not bool:
            raise ValueError("Stationary route heading must be explicitly enabled")
        if isinstance(self.amp_sample_start_s, bool) or self.amp_sample_start_s not in (0.0, 0.5):
            raise ValueError("Public PACER start0 or explicit legacy start.5 required")
        if type(self.persistent_braking_target) is not bool or (
            self.persistent_braking_target and not self.allow_safety_braking
        ):
            raise ValueError("Persistent braking target requires explicit safety braking")
        if type(self.mixed_safety_brakes) is not bool or (
            self.mixed_safety_brakes and not self.allow_safety_braking
        ):
            raise ValueError("Mixed safety candidates require explicit safety braking")
        if (
            isinstance(self.guard_horizon_s, bool)
            or not math.isfinite(self.guard_horizon_s)
            or not .5 <= self.guard_horizon_s <= 3.
            or abs(self.guard_horizon_s * 10 - round(self.guard_horizon_s * 10)) > 1e-6
            or isinstance(self.guard_tracking_margin_m, bool)
            or not math.isfinite(self.guard_tracking_margin_m)
            or not 0 <= self.guard_tracking_margin_m <= 1.
        ):
            raise ValueError("Guard requires a .1s-grid horizon in[.5,3] and tracking margin in[0,1]m")
        if self.map_guard_clearance_m is not None and (type(self.map_guard_clearance_m) not in (float,int) or not math.isfinite(self.map_guard_clearance_m) or not 0<=self.map_guard_clearance_m<=1.):
            raise ValueError('Explicit physical map guard gap must be finite in[0,1]m')
        if (
            not math.isfinite(self.braking_agent_buffer_m)
            or not 0
            <= self.braking_agent_buffer_m
            <= self.guidance_config.agent_buffer_m
        ):
            raise ValueError(
                "Brake clearance must retain physical radii and fit the nominal comfort buffer"
            )

    def braking_plan(self, world, packet, scene_ids, *, admit=True):
        """Authored safety command, explicitly distinct from a DDPM sample.

        Measured world velocity decelerates to zero at the configured rate.
        Pose/yaw is not overwritten; AMP still generates every physical action.
        The whole shared braking combination must pass the same geometry gate.
        """
        from CrowdSim.humanoid.diffusion.guidance import PedestrianGuidance
        from CrowdSim.humanoid.diffusion.planner import DiffusionPlan
        from dataclasses import replace

        state = packet["states"]
        speed = state[:, 2]
        velocity = world.planar()[:, 2:4]
        time = torch.arange(1, 53, device=state.device, dtype=state.dtype) * 0.1
        rate = self.guidance_config.max_acceleration_m_s2
        capped = torch.minimum(time[None], speed[:, None] / rate)
        distance = speed[:, None] * capped - 0.5 * rate * capped.square()
        direction = velocity / speed[:, None].clamp_min(1e-8)
        world_xy = state[:, None, :2] + direction[:, None] * distance[..., None]
        delta = world_xy - state[:, None, :2]
        c, s = state[:, 3, None].cos(), state[:, 3, None].sin()
        local_xy = torch.stack(
            (
                delta[..., 0] * c + delta[..., 1] * s,
                -delta[..., 0] * s + delta[..., 1] * c,
            ),
            -1,
        )
        future_speed = (speed[:, None] - rate * time).clamp_min(0)
        previous = torch.cat((speed[:, None], future_speed[:, :-1]), 1)
        zeros = torch.zeros_like(future_speed)
        trajectory = torch.cat(
            (
                local_xy,
                future_speed[..., None],
                zeros[..., None],
                ((future_speed - previous) / 0.1)[..., None],
                zeros[..., None],
            ),
            -1,
        )[:, None]
        context = self.planner.prepare_context(**packet)
        brake_guidance = replace(
            self.guidance_config,
            agent_buffer_m=self.braking_agent_buffer_m,
            comfort_recovery_speed_m_s=0.0,
        )
        guide = PedestrianGuidance(state, context, brake_guidance, scene_ids=scene_ids)
        diagnostics = None
        if admit:
            _, diagnostics = guide.select(trajectory)
        return DiffusionPlan(
            xy=torch.cat((state[:, None, :2], world_xy[:, :50]), 1),
            yaw=state[:, 3, None].expand(-1, 51),
            vel=torch.cat((speed[:, None], future_speed[:, :50]), 1),
            time=torch.arange(51, device=state.device, dtype=state.dtype) * 0.1,
            sample_states=trajectory,
            selected_sample=torch.full(
                (len(state),), -1, device=state.device, dtype=torch.long
            ),
            reference="explicit analytic safety braking; NOT a diffusion sample",
            guidance_diagnostics=diagnostics,
            sampling_dynamics_limits={
                "kind": "analytic safety braking",
                "deceleration_m_s2": rate,
                "measured_initial_velocity": True,
                "agent_buffer_m": self.braking_agent_buffer_m,
            },
        )

    def retain_braking_targets(self):
        """Hold the first inertial stopping destination until diffusion resumes.

        Admission still uses freshly measured velocity and a causal braking
        trajectory. The AMP desired stopping target is held fixed instead of
        repeatedly advancing v²/(2a) with the moving physical root. This goal
        is a control command, not a claim of instantaneous physical stopping.
        """
        braking = self.plan.selected_sample < 0
        if self.braking_anchor_xy is None:
            self.braking_anchor_xy = self.plan.xy.new_zeros(len(braking), 2)
            self.braking_anchor_mask = torch.zeros_like(braking)
        new = braking & ~self.braking_anchor_mask
        self.braking_anchor_xy[new] = self.plan.xy[new, -1]
        self.braking_anchor_mask = braking.clone()

    def apply_braking_targets(self, targets):
        if not self.persistent_braking_target:
            return targets
        return torch.where(
            self.braking_anchor_mask[:, None, None],
            self.braking_anchor_xy[:, None], targets,
        )

    def actual_velocity_hazards(self, world, packet, scene_ids):
        """Short reactive forecast from measured motion, independent of DDPM.

        AMP can lag a route change. Check every30Hz tick instead of waiting for
        the next2Hz replan. Stationary actors inside a comfort buffer can recover;
        only closing pairs trigger a brake, avoiding a stationary deadlock.
        """
        from CrowdSim.humanoid.diffusion.guidance import PedestrianGuidance

        state = packet["states"]
        velocity = world.planar()[:, 2:4]
        context = self.planner.prepare_context(**packet)
        guide = PedestrianGuidance(
            state, context, self.guidance_config, scene_ids=scene_ids
        )
        intervals = round(self.guard_horizon_s * 10)
        time = torch.arange(1, intervals + 1, device=state.device, dtype=state.dtype) * 0.1
        delta = velocity[:, None] * time[None, :, None]
        c, s = state[:, 3, None].cos(), state[:, 3, None].sin()
        local = torch.stack(
            (
                delta[..., 0] * c + delta[..., 1] * s,
                -delta[..., 0] * s + delta[..., 1] * c,
            ),
            -1,
        )
        projected = state.new_zeros(len(state), 1, intervals, 6)
        projected[:, 0, :, :2] = local
        projected[:, 0, :, 2] = state[:, 2, None]
        now = state.new_zeros(len(state), 1, 1, 6)
        now[..., 2] = state[:, None, None, 2]
        future_parts = guide.components(projected)
        now_parts = guide.components(now)
        future_clear = future_parts["minimum_map_clearance_m"][:, 0]
        now_clear = now_parts["minimum_map_clearance_m"][:, 0]
        from CrowdSim.humanoid.runtime.map_velocity_guard import map_velocity_hazard
        map_hazard=map_velocity_hazard(future_clear,now_clear,self.guidance_config.clearance_m,self.guard_tracking_margin_m,physical_threshold_m=self.map_guard_clearance_m)
        xy = state[:, None, :2] + delta
        pair_distance = (xy[:, None] - xy[None, :]).norm(dim=-1).amin(-1)
        separation = state[:, None, :2] - state[None, :, :2]
        relative_velocity = velocity[:, None] - velocity[None, :]
        closing = (separation * relative_velocity).sum(-1) / separation.norm(
            dim=-1
        ).clamp_min(1e-6)
        radius = context["extent"][:, :2].norm(dim=-1) / 2
        pair_hazard = (
            (pair_distance < radius[:, None] + radius[None, :]
             + self.guidance_config.agent_buffer_m + self.guard_tracking_margin_m)
            & (closing < -0.05)
            & guide.pair_mask
        ).any(1)
        body_hazard = torch.zeros_like(map_hazard)
        body_diagnostics = None
        if self.actual_body_map_guard:
            from CrowdSim.humanoid.runtime.body_map_guard import collision_spheres,body_map_hazard
            scene = world.validation_scene
            if not callable(getattr(scene,'signed_distance',None)):
                raise ValueError('Body guard requires the source-bound native map SDF')
            if self.body_guard_geometry is None:
                self.body_guard_geometry=collision_spheres(world.mapping.body_names,contact_offset_m=world.simulator.config.sim.physx.contact_offset)
            body_hazard,body_diagnostics=body_map_hazard(scene,world.state(),self.body_guard_geometry,horizon_s=self.body_guard_horizon_s,gap_m=self.body_guard_gap_m)
        hazard = map_hazard | pair_hazard | body_hazard
        return hazard, {
            "early_actual_velocity_guard": True,
            "actual_body_map_guard": body_diagnostics,
            "horizon_s": self.guard_horizon_s,
            "physical_tracking_margin_m": self.guard_tracking_margin_m,
            "map_guard_clearance_m": self.guidance_config.clearance_m+self.guard_tracking_margin_m if self.map_guard_clearance_m is None else self.map_guard_clearance_m,
            "map_guard_mode": "comfort" if self.map_guard_clearance_m is None else "explicit_physical_gap",
            "at_risk_actors": hazard.tolist(),
            "map_at_risk": map_hazard.tolist(),
            "closing_pair_at_risk": pair_hazard.tolist(),
        }

    def action(self, world, control, route_targets, *, waypoint_times=None, route_target_provider=None):
        if type(control) is not int or control != self.last_control + 1:
            raise ValueError(
                "Contiguous integer physical controls starting at zero required"
            )
        if abs(world.dt - 1 / 30) > 1e-8:
            raise ValueError("This adapter requires measured 30Hz control")
        if route_target_provider is not None and not callable(route_target_provider):
            raise ValueError("Fresh route target provider must be callable")
        packet = ids = guard_diagnostics = None
        early_hazard = False
        if self.allow_safety_braking:
            packet = trace_inputs_from_world(
                world,
                pedestrian_extent_m=self.pedestrian_extent_m,
                stationary_route_targets=route_targets
                if self.stationary_route_heading
                else None,
            )
            ids = (
                torch.zeros(world.num_envs, device=world.device, dtype=torch.long)
                if world.shared_world
                else torch.arange(world.num_envs, device=world.device)
            )
            hazard, guard_diagnostics = self.actual_velocity_hazards(world, packet, ids)
            early_hazard = bool(hazard.any())
        already_braking = (
            self.plan.selected_sample < 0
            if self.plan is not None
            else torch.zeros(world.num_envs, device=world.device, dtype=torch.bool)
        )
        if (
            self.plan is None
            or control - self.plan_control >= self.replan_controls
            or (early_hazard and bool((hazard & ~already_braking).any()))
        ):
            if route_target_provider is not None:
                # Reactive replans must not reuse an old A* query. Refresh from
                # this same measured physical frame, then re-express the raster.
                route_targets = route_target_provider()
                packet = None
            packet = (
                trace_inputs_from_world(
                    world,
                    pedestrian_extent_m=self.pedestrian_extent_m,
                stationary_route_targets=route_targets
                    if self.stationary_route_heading
                    else None,
                )
                if packet is None
                else packet
            )
            ids = (
                torch.zeros(world.num_envs, device=world.device, dtype=torch.long)
                if world.shared_world
                else torch.arange(world.num_envs, device=world.device)
            )
            from CrowdSim.humanoid.diffusion.guidance import TrajectoryAdmissionError

            self.last_rejection_diagnostics = None
            if early_hazard and not self.mixed_safety_brakes:
                self.last_rejection_diagnostics = guard_diagnostics
                self.plan = self.braking_plan(world, packet, ids)
                self.safety_braking_plans += 1
            else:
                try:
                    brakes = (
                        self.braking_plan(world, packet, ids, admit=False).sample_states
                        if self.mixed_safety_brakes
                        else None
                    )
                    self.plan = self.planner.plan(
                        **packet,
                        waypoints=route_targets,
                        waypoint_times=waypoint_times,
                        pedestrian_guidance=self.guidance_config,
                        scene_ids=ids,
                        num_samples=self.num_samples,
                        seed=self.seed + control,
                        analytic_brake_candidates=brakes,
                        force_brake_mask=hazard
                        if self.mixed_safety_brakes and early_hazard
                        else None,
                    )
                    if bool((self.plan.selected_sample < 0).any()):
                        self.safety_braking_plans += 1
                    if early_hazard:
                        self.last_rejection_diagnostics = guard_diagnostics
                except TrajectoryAdmissionError as error:
                    self.last_rejection_diagnostics = error.diagnostics
                    if not self.allow_safety_braking:
                        raise
                    self.plan = self.braking_plan(world, packet, ids)
                    self.safety_braking_plans += 1
            self.plan_control = control
            self.last_diagnostics = self.plan.guidance_diagnostics
            if self.persistent_braking_target:
                self.retain_braking_targets()
        targets, held = timed_amp_waypoints(
            self.plan, (control - self.plan_control) * world.dt,
            first_sample_s=self.amp_sample_start_s,
        )
        settling_events = []
        if self.settle_braking_target:
            from CrowdSim.humanoid.runtime.settled_braking_target import SettledBrakingTarget
            if self.settled_braking_helper is None:
                self.settled_braking_helper = SettledBrakingTarget(world.num_envs)
            body_gap = guard_diagnostics["actual_body_map_guard"]["current_minimum_sphere_map_clearance_m"]
            for event in self.settled_braking_helper.update(world.planar()[:, :2], self.plan.selected_sample < 0, body_gap):
                self.braking_anchor_xy[event["actor"]] = event["actual_root_xy"]
                settling_events.append({"physical_control": control, "actor": event["actor"],
                    "actual_root_xy": event["actual_root_xy"].tolist(),
                    "measured1s_bbox_diagonal_m": event["measured1s_bbox_diagonal_m"],
                    "scope": "desired holding point only; body, velocity, guards and source unchanged"})
        targets = self.apply_braking_targets(targets)
        # Real current pose, velocities, geometry and terrain at EVERY PD tick.
        action = self.controller.act(world.state(), targets, world, return_raw=True)
        if not torch.isfinite(action).all():
            raise FloatingPointError("Nonfinite AMP action; no substitute animation")
        self.last_control = control
        return action, {
            "braking_target_settling_events": settling_events,
            "replanned": control == self.plan_control,
            "terminal_held_waypoints": held,
            "safety_braking": bool((self.plan.selected_sample < 0).any()),
        }

    def invalidate_after_reset(self):
        self.plan = None
        self.plan_control = -1
        self.braking_anchor_xy = None
        self.braking_anchor_mask = None
        self.settled_braking_helper = None
