"""Clean-trajectory constraints for TRACE planning and shared sample selection.

Reference: nv-tlabs/trace, tbsim/utils/guidance_loss.py at
594b2af0824c5853e1a1884e167cd5ed097dd81c (CVPR 2023 Trace and Pace).
Timed goal and disk separation follow its physical trajectory costs. Map SDF,
outside-map rejection and locomotion limits are explicit local adaptations;
these costs are planning heuristics, never proof of physical collision safety.
"""

from dataclasses import asdict, dataclass
import math

import numpy as np
import torch
import torch.nn.functional as F


class TrajectoryAdmissionError(RuntimeError):
    def __init__(self, parts):
        super().__init__(
            "No sampled trajectory passes the map/peer/locomotion admission gate"
        )
        self.diagnostics = {
            name: [
                [float(x) if math.isfinite(float(x)) else None for x in row]
                for row in value.detach().cpu().tolist()
            ]
            for name, value in parts.items()
        }


@dataclass(frozen=True)
class PedestrianGuidanceConfig:
    goal_weight: float = 1.0
    map_weight: float = 4.0
    agent_weight: float = 4.0
    dynamics_weight: float = 1.0
    target_speed_weight: float = 0.0
    target_speed_m_s: float = 0.6
    turn_speed_decay_s: float = 0.0
    speed_ramp_time_s: float = 0.0
    yaw_smoothness_weight: float = 0.0
    speed_smoothness_weight: float = 0.0
    bounded_guidance_gradient: float = 0.0
    clean_guidance_step_size: float = 1.0
    final_refinement_steps: int = 0
    final_refinement_learning_rate: float = .15
    final_refinement_anchor_weight: float = .02
    max_speed_m_s: float = 1.4
    max_acceleration_m_s2: float = 1.5
    max_yaw_rate_rad_s: float = 0.6
    clearance_m: float = 0.15
    agent_buffer_m: float = 0.2
    map_worst_point_cost: bool = False
    guide_clean: bool = True
    reject_unsafe_samples: bool = True
    selection_beam_width: int = 64
    comfort_recovery_speed_m_s: float = 0.05

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "final_refinement_steps":
                if type(value) is not int or not 0 <= value <= 128:
                    raise ValueError("final_refinement_steps must be an integer in0..128")
            elif name == "selection_beam_width":
                if type(value) is not int or not 1 <= value <= 256:
                    raise ValueError("selection_beam_width must be an integer in1..256")
            elif name in ("guide_clean", "reject_unsafe_samples", "map_worst_point_cost"):
                if type(value) is not bool:
                    raise ValueError(name + " must be boolean")
            elif (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(name + " must be finite and nonnegative")
        if (
            min(self.max_speed_m_s, self.max_acceleration_m_s2, self.max_yaw_rate_rad_s)
            <= 0
        ):
            raise ValueError("Locomotion limits must be positive")
        if not 0 < self.final_refinement_learning_rate <= .25:
            raise ValueError("Final refinement learning rate must be in(0,.25]")
        if self.clean_guidance_step_size > 1.:
            raise ValueError("Clean guidance step size must be in[0,1]")
        if self.bounded_guidance_gradient > 1.:
            raise ValueError("Bounded guidance gradient must be in[0,1]")
        if self.target_speed_m_s > self.max_speed_m_s:
            raise ValueError("Target speed exceeds the locomotion limit")


def motion_admission(parts, cfg):
    """Per-actor map and locomotion constraints; final peer mixing still required."""
    return (
        (parts["minimum_map_clearance_m"] >= 0)
        & (parts["minimum_speed_m_s"] >= -0.05)
        & (parts["maximum_speed_limit_violation_m_s"] <= 0.1)
        & (parts["maximum_acceleration_m_s2"] <= cfg.max_acceleration_m_s2 + 0.15)
        & (parts["maximum_yaw_rate_rad_s"] <= cfg.max_yaw_rate_rad_s + 0.05)
    )


class PedestrianGuidance:
    """Physical local trajectories[B,S,52,6], one explicit scene ID per actor.

    A scene ID describes actors sampled jointly in one physical world, not
    independent environments. Cross-sample combinations are checked jointly;
    independently picking per-actor minima can destroy planned clearance.
    Unknown semantic pixels (.5) are treated as blocked for this safety cost.
    """

    def __init__(
        self,
        states,
        context,
        config,
        *,
        scene_ids,
        local_targets=None,
        target_indices=None,
    ):
        from scipy.ndimage import distance_transform_edt

        self.config = config
        self.states = states.detach()
        self.extent = context["extent"].detach()
        self.scene_ids = torch.as_tensor(scene_ids, device=states.device)
        if self.scene_ids.shape != (len(states),) or self.scene_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("Explicit integer scene_ids[B] are required")
        self.targets, self.indices = local_targets, target_indices
        free = (
            ((context["image"][:, 0] >= 0.75) & (context["image"][:, 1] <= 0.25))
            .cpu()
            .numpy()
        )
        fields = []
        for mask in free:
            if not mask.any():
                raise ValueError("No known walkable cells in the planning raster")
            # Pad blocked border, including an otherwise fully empty scene.
            padded = np.pad(mask, 1, constant_values=False)
            field = distance_transform_edt(padded) - distance_transform_edt(~padded)
            fields.append(field[1:-1, 1:-1] / 12.0)
        self.sdf = torch.as_tensor(
            np.stack(fields), device=states.device, dtype=states.dtype
        )[:, None]
        self.pair_mask = (
            self.scene_ids[:, None] == self.scene_ids[None, :]
        ) & ~torch.eye(len(states), device=states.device, dtype=torch.bool)

    def components(self, trajectory):
        if (
            trajectory.ndim != 4
            or trajectory.shape[0] != len(self.states)
            or trajectory.shape[-1] != 6
        ):
            raise ValueError("Physical trajectories[B,S,T,6] required")
        b, s, t, _ = trajectory.shape
        xy = trajectory[..., :2]
        zeros = trajectory[..., 0].sum(-1) * 0
        goal = (
            zeros
            if self.targets is None
            else torch.linalg.vector_norm(
                xy[:, :, self.indices] - self.targets[:, None], dim=-1
            ).mean(-1)
        )
        # Nine circle samples around a conservative circumscribed footprint.
        angles = torch.arange(8, device=xy.device, dtype=xy.dtype) * math.pi / 4
        offsets = torch.cat(
            (xy.new_zeros(1, 2), torch.stack((angles.cos(), angles.sin()), -1)), 0
        )
        radius = self.extent[:, :2].norm(dim=-1) / 2
        points = xy[..., None, :] + radius[:, None, None, None, None] * offsets
        px = points[..., 0] * 12 + 56
        py = points[..., 1] * 12 + 112
        # Outside penalties preserve a gradient beyond grid_sample's border.
        outside = (
            torch.sqrt(
                (F.relu(-px) + F.relu(px - 223)).square()
                + (F.relu(-py) + F.relu(py - 223)).square()
                + 1e-12
            )
            / 12
        )
        grid = torch.stack(
            (px.clamp(0, 223) * 2 / 223 - 1, py.clamp(0, 223) * 2 / 223 - 1), -1
        )
        clearance = (
            F.grid_sample(
                self.sdf,
                grid.reshape(b, s * t * 9, 1, 2),
                align_corners=True,
                padding_mode="border",
            ).reshape(b, s, t, 9)
            - outside
        )
        deficit = F.relu(self.config.clearance_m - clearance).square()
        # A single hand-side footprint sample must not be diluted by eight
        # clear samples. Retain time averaging and the same admission gate.
        map_loss = (deficit.amax(-1).mean(-1) if self.config.map_worst_point_cost
                    else deficit.mean((-1, -2)))
        c, sn = (
            self.states[:, 3].cos()[:, None, None],
            self.states[:, 3].sin()[:, None, None],
        )
        world = (
            torch.stack(
                (xy[..., 0] * c - xy[..., 1] * sn, xy[..., 0] * sn + xy[..., 1] * c), -1
            )
            + self.states[:, None, None, :2]
        )
        dist = (
            (world[:, None] - world[None, :]).square().sum(-1).clamp_min(1e-12).sqrt()
        )
        threshold = radius[:, None] + radius[None, :] + self.config.agent_buffer_m
        violation = (
            F.relu(threshold[:, :, None, None] - dist)
            * self.pair_mask[:, :, None, None]
        )
        agent = (
            violation.square().sum(1).mean(-1)
            / self.pair_mask.sum(1).clamp_min(1)[:, None]
        )
        min_peer = dist.masked_fill(
            ~self.pair_mask[:, :, None, None], float("inf")
        ).amin((1, 3))
        max_speed = trajectory[..., 2].abs().amax(-1)
        future_time = torch.arange(1, t + 1, device=xy.device, dtype=xy.dtype) * 0.1
        speed_envelope = (
            self.states[:, 2, None] - self.config.max_acceleration_m_s2 * future_time
        ).clamp_min(self.config.max_speed_m_s)
        speed_violation = F.relu(trajectory[..., 2] - speed_envelope[:, None]).amax(-1)
        # TRACE returns unconstrained network actions alongside integrated
        # states. Unicycle clips actions internally; those raw channels are
        # NOT the realized acceleration/yaw rate of the route given to AMP.
        previous_speed = torch.cat(
            (self.states[:, 2, None, None].expand(b, s, 1), trajectory[:, :, :-1, 2]),
            -1,
        )
        previous_yaw = torch.cat((xy.new_zeros(b, s, 1), trajectory[:, :, :-1, 3]), -1)
        acceleration = (trajectory[..., 2] - previous_speed) / 0.1
        yaw_rate = (
            torch.remainder(trajectory[..., 3] - previous_yaw + math.pi, 2 * math.pi)
            - math.pi
        )
        yaw_rate = yaw_rate / 0.1
        dynamics = (
            F.relu(trajectory[..., 2] - speed_envelope[:, None]).square()
            + F.relu(-trajectory[..., 2]).square()
            + F.relu(acceleration.abs() - self.config.max_acceleration_m_s2).square()
            + F.relu(yaw_rate.abs() - self.config.max_yaw_rate_rad_s).square()
        ).mean(-1)
        desired_speed = torch.full_like(trajectory[..., 2], self.config.target_speed_m_s)
        if self.config.turn_speed_decay_s > 0:
            # Smooth differentiable curvature cost; a tight turn may slow down.
            # This is a soft clean-trajectory objective, never an imposed pose/speed.
            desired_speed = desired_speed / torch.sqrt(1 + (self.config.turn_speed_decay_s*yaw_rate).square())
        if self.config.speed_ramp_time_s > 0:
            blend = (future_time/self.config.speed_ramp_time_s).clamp(max=1.)
            desired_speed = torch.lerp(self.states[:,2,None,None], desired_speed, blend)
        speed = (trajectory[..., 2] - desired_speed).square().mean(-1)
        # Penalize changes of realized route yaw rate / acceleration inside
        # the future. Raw network control channels remain non-authoritative.
        yaw_smoothness = (torch.diff(yaw_rate,dim=-1)/.1).square().mean(-1) if t>1 else zeros
        speed_smoothness = (torch.diff(acceleration,dim=-1)/.1).square().mean(-1) if t>1 else zeros
        return {
            "yaw_smoothness": yaw_smoothness,
            "speed_smoothness": speed_smoothness,
            "goal": goal,
            "map": map_loss,
            "agent": agent,
            "dynamics": dynamics,
            "target_speed": speed,
            "minimum_map_clearance_m": clearance.amin((-1, -2)),
            "minimum_peer_distance_m": min_peer,
            "maximum_agent_overlap_m": violation.amax((1, 3)),
            "maximum_physical_overlap_m": (
                F.relu(radius[:, None, None, None] + radius[None, :, None, None] - dist)
                * self.pair_mask[:, :, None, None]
            ).amax((1, 3)),
            "minimum_speed_m_s": trajectory[..., 2].amin(-1),
            "maximum_acceleration_m_s2": acceleration.abs().amax(-1),
            "maximum_yaw_rate_rad_s": yaw_rate.abs().amax(-1),
            "raw_network_maximum_acceleration_m_s2": trajectory[..., 4].abs().amax(-1),
            "raw_network_maximum_yaw_rate_rad_s": trajectory[..., 5].abs().amax(-1),
            "maximum_speed_m_s": max_speed,
            "maximum_speed_limit_violation_m_s": speed_violation,
        }

    def score(self, trajectory):
        parts = self.components(trajectory)
        config = self.config
        score = (
            config.goal_weight * parts["goal"]
            + config.map_weight * parts["map"]
            + config.agent_weight * parts["agent"]
            + config.dynamics_weight * parts["dynamics"]
            + config.target_speed_weight * parts["target_speed"]
            + config.yaw_smoothness_weight * parts["yaw_smoothness"]
            + config.speed_smoothness_weight * parts["speed_smoothness"]
        )
        return score, parts

    def compute_guidance_loss(self, trajectory, batch):
        score, parts = self.score(trajectory)
        # Per-trajectory gradient magnitude must not shrink when more actors
        # or samples join a batch. Weights are calibrated for this sum, whereas
        # the author's default mean changes strength with population size.
        return score.sum(), {key: value.detach() for key, value in parts.items()}

    def select(self, trajectory, *, fixed_samples=None):
        """Bounded joint search, checking geometry AFTER mixing sample columns.

        Same-column selection can reject a scene even though a safe cross-column
        combination exists. Picking separate minima can introduce collisions.
        A beam enumerates only compatible combinations. Exhaustion is a failure,
        never permission to return an unchecked best-effort route.
        """
        _, parts = self.score(trajectory)
        cfg = self.config
        costs = (
            cfg.goal_weight * parts["goal"]
            + cfg.map_weight * parts["map"]
            + cfg.dynamics_weight * parts["dynamics"]
            + cfg.target_speed_weight * parts["target_speed"]
            + cfg.yaw_smoothness_weight * parts["yaw_smoothness"]
            + cfg.speed_smoothness_weight * parts["speed_smoothness"]
        )
        safe = motion_admission(parts, cfg)
        if not cfg.reject_unsafe_samples:
            # Explicit offline diagnostic mode; still retain pair checks.
            safe = torch.ones_like(safe)
        if fixed_samples is not None:
            fixed = torch.as_tensor(fixed_samples, device=trajectory.device)
            if (
                fixed.shape != (len(trajectory),)
                or fixed.dtype != torch.long
                or bool(((fixed < -1) | (fixed >= trajectory.shape[1])).any())
            ):
                raise ValueError(
                    "Fixed candidate indices require int64[B], -1 for unrestricted"
                )
            candidate = torch.arange(trajectory.shape[1], device=trajectory.device)[
                None
            ]
            safe &= (fixed[:, None] < 0) | (candidate == fixed[:, None])
        xy = trajectory[..., :2]
        c, sn = self.states[:, 3, None, None].cos(), self.states[:, 3, None, None].sin()
        world = (
            torch.stack(
                (xy[..., 0] * c - xy[..., 1] * sn, xy[..., 0] * sn + xy[..., 1] * c), -1
            )
            + self.states[:, None, None, :2]
        )
        radius = self.extent[:, :2].norm(dim=-1) / 2
        compatibility = {}
        selected = torch.empty(len(costs), device=costs.device, dtype=torch.long)
        for scene in self.scene_ids.unique():
            actors = (self.scene_ids == scene).nonzero().flatten().tolist()
            options = {i: safe[i].nonzero().flatten().tolist() for i in actors}
            if any(not value for value in options.values()):
                raise TrajectoryAdmissionError(parts)
            order = sorted(actors, key=lambda i: (len(options[i]), i))
            for i in order:
                for j in order:
                    if i >= j:
                        continue
                    distance = (world[i, :, None] - world[j, None, :]).norm(dim=-1)
                    current_distance = (self.states[i, :2] - self.states[j, :2]).norm()
                    # Actual tracking may enter the social comfort buffer.
                    # Recover it gradually; physical radii remain mandatory.
                    future_time = (
                        torch.arange(
                            1,
                            trajectory.shape[2] + 1,
                            device=world.device,
                            dtype=world.dtype,
                        )
                        * 0.1
                    )
                    desired = torch.minimum(
                        radius[i] + radius[j] + cfg.agent_buffer_m,
                        current_distance + cfg.comfort_recovery_speed_m_s * future_time,
                    )
                    desired = desired.clamp_min(radius[i] + radius[j])
                    compatibility[i, j] = (
                        (
                            (distance >= desired[None, None] - 0.02)
                            & (distance >= radius[i] + radius[j] - 1e-5)
                        )
                        .all(-1)
                        .cpu()
                        .numpy()
                    )
            # Fully compatible pairs impose no joint constraint. Solve their
            # independent components separately so unrelated choices cannot
            # consume the beam and prune a feasible constrained pair's prefix.
            # Every cross-component candidate combination has already passed
            # the full52-step pair test above; final joint diagnostics remain.
            edges = {i:set() for i in actors}
            for (i,j), matrix in compatibility.items():
                if i in edges and j in edges and not matrix[np.ix_(options[i],options[j])].all():
                    edges[i].add(j);edges[j].add(i)
            components=[];remaining=set(actors)
            while remaining:
                pending=[min(remaining)];component=set()
                while pending:
                    actor=pending.pop()
                    if actor in component:continue
                    component.add(actor);pending.extend(edges[actor]-component)
                remaining-=component;components.append(component)
            scalar = costs.detach().cpu().numpy()
            for component in components:
                component_order=[actor for actor in order if actor in component]
                beam = [(0.0, {})]
                for actor in component_order:
                    expanded = []
                    for total, choice in beam:
                        for candidate in options[actor]:
                            valid = all(
                                compatibility[min(actor, previous), max(actor, previous)][
                                    candidate if actor < previous else prior,
                                    prior if actor < previous else candidate,
                                ]
                                for previous, prior in choice.items()
                            )
                            if valid:
                                expanded.append(
                                    (
                                        total + float(scalar[actor, candidate]),
                                        {**choice, actor: candidate},
                                    )
                                )
                    if not expanded:
                        raise TrajectoryAdmissionError(parts)
                    beam = sorted(
                        expanded, key=lambda item: (item[0], tuple(sorted(item[1].items())))
                    )[: cfg.selection_beam_width]
                for actor, candidate in beam[0][1].items():
                    selected[actor] = candidate
        chosen = trajectory[
            torch.arange(len(selected), device=trajectory.device), selected
        ][:, None]
        # Recompute and disclose the chosen combination's joint clearance.
        joint_parts = self.components(chosen)
        diagnostics = {key: value.detach() for key, value in parts.items()}
        diagnostics.update(
            {
                "selected_joint_" + key: value.detach()
                for key, value in joint_parts.items()
            }
        )
        return selected, diagnostics
