"""Trainable five-state trajectory DDPM for the NavIsaacLab method contract.

This is an independent implementation: state=(x,y,u,v,theta), without TRACE's
unicycle restriction. Public author weights and architectural hyperparameters
were not supplied. Selected network details are explicit reproducibility
assumptions, not a claim of identical author training or benchmark scores.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from CrowdSim.humanoid.waypoint_weighting import near_horizon_waypoint_weights, validate_near_horizon_mass
from CrowdSim.humanoid.diffusion.base_models import MLP
from CrowdSim.humanoid.diffusion.temporal import TemporalMapUnet
from CrowdSim.humanoid.diffusion.trace_helpers import MapEncoder, cosine_beta_schedule, query_feature_grid


PAPER_SCHEMA = "crowdsim.paper_five_state_diffusion.v1"


@dataclass(frozen=True)
class PaperDiffusionConfig:
    history_frames: int = 31
    horizon: int = 52
    dt: float = 0.1
    diffusion_steps: int = 100
    base_dim: int = 32
    history_feature_dim: int = 128
    conditioning_feature_dim: int = 256
    map_grid_feature_dim: int = 32
    state_mean: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    state_std: tuple[float, ...] = (5.0, 5.0, 1.5, 1.5, math.pi)
    map_conditioning_dropout: float = 0.1
    neighbor_conditioning_dropout: float = 0.1
    unknown_map_value: float = 0.5
    architecture_provenance: str = "Independent paper-state adaptation of TRACE temporal U-Net; author hyperparameters unavailable"

    def __post_init__(self):
        if self.history_frames != 31 or self.horizon != 52 or self.dt != 0.1:
            raise ValueError("Five-state adapter requires 31 history frames, 52 future frames, 0.1s dt")
        if self.diffusion_steps < 2 or len(self.state_mean) != 5 or len(self.state_std) != 5:
            raise ValueError("At least two diffusion steps and five state normalization coefficients required")
        if not all(math.isfinite(x) for x in (*self.state_mean, *self.state_std)) or min(self.state_std) <= 0:
            raise ValueError("State normalization must be finite with positive standard deviations")
        if not all(0 <= p <= 1 for p in (self.map_conditioning_dropout, self.neighbor_conditioning_dropout, self.unknown_map_value)):
            raise ValueError("Conditioning dropout and unknown map values must be within [0,1]")


def _finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains nonfinite available values")


def world_to_agent_state(values, current):
    """Rotate both XY and world velocity; yaw is a relative wrapped angle."""
    shape = (current.shape[0],) + (1,) * (values.ndim - 2)
    cosine = current[:, 4].cos().reshape(shape)
    sine = current[:, 4].sin().reshape(shape)
    origin = current[:, :2].reshape(*shape, 2)
    xy = values[..., :2] - origin
    velocity = values[..., 2:4]
    local_xy = torch.stack((xy[..., 0] * cosine + xy[..., 1] * sine,
                            -xy[..., 0] * sine + xy[..., 1] * cosine), dim=-1)
    local_velocity = torch.stack((velocity[..., 0] * cosine + velocity[..., 1] * sine,
                                  -velocity[..., 0] * sine + velocity[..., 1] * cosine), dim=-1)
    yaw = torch.remainder(values[..., 4] - current[:, 4].reshape(shape) + math.pi, 2 * math.pi) - math.pi
    return torch.cat((local_xy, local_velocity, yaw[..., None]), dim=-1)


@torch.no_grad()
def waypoint_guidance_gradient(mean, target_normalized, indices, *, num_samples=1):
    """Independent per-trajectory soft goal gradient, invariant to batch size.

    Reduce over waypoint and XY coordinates, then sum trajectories. Averaging
    over batch/sample dimensions would weaken the same agent's guidance when
    the caller changes planning chunks.
    """
    batch = target_normalized.shape[0]
    if mean.ndim != 3 or mean.shape[-1] != 5 or target_normalized.ndim != 3 or target_normalized.shape[-1] != 2:
        raise ValueError("Guidance requires mean[B*S,T,5] and targets[B,K,2]")
    count = target_normalized.shape[1]
    if count < 1 or indices.ndim != 1 or len(indices) != count:
        raise ValueError("Guidance requires nonempty matching waypoint indices")
    if not isinstance(num_samples, int) or num_samples < 1 or mean.shape[0] != batch * num_samples:
        raise ValueError("Guidance batch/sample shape mismatch")
    # d(sum_B,S mean_K,XY residual^2)/d(mean_XY) = residual/K.
    # Accumulate repeated frame indices exactly as indexing's autograd backward
    # does; assignment would incorrectly discard repeated waypoint gradients.
    residual = mean[:, indices, :2].reshape(batch, num_samples, count, 2) - target_normalized[:, None]
    gradient = torch.zeros_like(mean)
    scatter_indices = indices[None, :, None].expand(batch * num_samples, count, 2)
    gradient[..., :2].scatter_add_(1, scatter_indices, residual.reshape(batch * num_samples, count, 2) / count)
    return gradient


def _world_prediction_xy(clean_physical, current, num_samples):
    local = clean_physical.reshape(len(current), num_samples, -1, 5)[..., :2]
    cosine, sine = current[:, 4].cos()[:, None, None], current[:, 4].sin()[:, None, None]
    return torch.stack((local[..., 0] * cosine - local[..., 1] * sine,
                        local[..., 0] * sine + local[..., 1] * cosine), -1) + current[:, None, None, :2]


def scene_agent_collision_cost(world_xy, scene_ids, radius=.3):
    """Unordered same-scene disk overlaps, summed over future timesteps.

    All actors' sample s denotes the same joint scene future. The scalar disk
    penalty 1-distance/(2*radius) follows TRACE; summing unordered pairs avoids
    its duplicate i->j,j->i contribution. No recorded future or constant-
    velocity proxy is used. Radius is an explicit reconstruction assumption.
    Return [B,B,S] costs (upper triangle only) for gradients and scene filtering.
    """
    if (world_xy.ndim != 4 or world_xy.shape[-1] != 2 or scene_ids.shape != (len(world_xy),)
            or not math.isfinite(radius) or radius <= 0):
        raise ValueError("Scene collision requires worldXY[B,S,T,2], scene_ids[B] and positive finite radius")
    distance = torch.linalg.vector_norm(world_xy[:, None] - world_xy[None, :], dim=-1)
    paired = torch.triu(scene_ids[:, None] == scene_ids[None, :], diagonal=1)
    penalty = torch.relu(1. - distance / (2 * radius)).sum(-1)
    return penalty * paired[..., None]


def signed_aabb_distance(points, centers, sizes):
    """Exact 2D box signed distance; negative inside, Euclidean outside.

    At the symmetric box centre, abs has no unique derivative. Choose the
    positive-face subgradient explicitly rather than a zero derivative that
    cannot recover a fully embedded prediction. This changes no field values.
    """
    delta = points - centers
    absolute = torch.where(delta >= 0, delta, -delta)
    q = absolute - sizes / 2
    return torch.linalg.vector_norm(q.clamp_min(0), dim=-1) + q.clamp_max(0).amax(-1)


def scene_geometry_collision_cost(world_xy, bounds, centers, sizes, radius=.5):
    """Analytic hinge penetration of actual scene boxes and bounded domain.

    This is a declared extension beyond TRACE's partially overlapping box
    samples: it also gives gradients when a future root is fully inside an
    obstacle or outside the domain. Sum timesteps/obstacles; return [B,S].
    Geometry is supplied explicitly, never inferred from unknown raster cells.
    """
    obstacle_distance = signed_aabb_distance(world_xy[..., None, :], centers, sizes)
    obstacle = torch.relu(radius - obstacle_distance).sum(-1)
    domain_distance = signed_aabb_distance(world_xy, (bounds[:2] + bounds[2:]) / 2, bounds[2:] - bounds[:2])
    domain = torch.relu(radius + domain_distance)
    return (obstacle + domain).sum(-1)


def _validate_scene_collision_callback(cost_fn, weight, geometry):
    """An explicit world-grid callback cannot silently replace analytic boxes."""
    if cost_fn is None:
        return
    if not callable(cost_fn):
        raise ValueError("scene_collision_cost_fn must be callable")
    if type(weight) not in (float, int) or not math.isfinite(weight) or weight <= 0:
        raise ValueError("scene_collision_cost_fn requires a finite positive scene_collision_weight")
    if any(value is not None for value in geometry):
        raise ValueError("scene_collision_cost_fn and AABB geometry parameters are mutually exclusive")


def _scene_callback_cost(world_xy, cost_fn, radius, *, require_gradient=False):
    """Validate one provider cost shared by clean guidance and candidate scoring.

    The planner has already transformed predictions to world XY exactly once.
    The provider owns grid coordinates, unknown/exterior handling and reduction
    units. A differentiable constant-zero free-space cost is valid; a detached
    or world-disconnected cost is not. autograd.grad never populates .grad.
    """
    if (not isinstance(world_xy, torch.Tensor) or world_xy.ndim != 4 or world_xy.shape[-1] != 2
            or any(size < 1 for size in world_xy.shape[:3]) or world_xy.dtype not in (torch.float32, torch.float64)
            or type(radius) not in (float, int) or not math.isfinite(radius) or radius <= 0):
        raise ValueError("Scene callback requires finite FP32/64 worldXY[B,S,T,2] and positive radius")
    _finite(world_xy, "scene callback world predictions")
    cost = cost_fn(world_xy, radius=radius)
    if (not isinstance(cost, torch.Tensor) or cost.shape != world_xy.shape[:2]
            or cost.dtype != world_xy.dtype or cost.device != world_xy.device):
        raise ValueError("Scene callback cost must be floating[B,S] with the exact world prediction dtype/device")
    _finite(cost, "scene callback collision cost")
    if bool((cost < 0).any()):
        raise ValueError("Scene callback collision cost must be nonnegative")
    if require_gradient:
        if not world_xy.requires_grad or not cost.requires_grad:
            raise ValueError("Scene callback collision cost must retain its worldXY autograd connection")
        gradient, = torch.autograd.grad(cost.sum(), world_xy, allow_unused=True, retain_graph=True)
        if gradient is None:
            raise ValueError("Scene callback collision cost is disconnected from worldXY")
        _finite(gradient, "scene callback worldXY gradient")
    return cost


def trajectory_dynamics_cost(world_xy, current, dt=.1, max_speed=3., max_acceleration=2.):
    """Squared speed/acceleration excess of the actually consumed XY path.

    Include the actual current position and world velocity at the first
    transition. These are soft guidance costs, not clipping or feasibility
    guarantees. Limits are declared PACER procedural-training assumptions.
    Return separate [B,S] costs, summed over the52 future frames.
    """
    start = current[:, None, None, :2].expand(-1, world_xy.shape[1], -1, -1)
    velocity = torch.diff(torch.cat((start, world_xy), dim=-2), dim=-2) / dt
    initial_velocity = current[:, None, None, 2:4].expand_as(start)
    acceleration = torch.diff(torch.cat((initial_velocity, velocity), dim=-2), dim=-2) / dt
    speed_cost = torch.relu(torch.linalg.vector_norm(velocity, dim=-1) - max_speed).square().sum(-1)
    acceleration_cost = torch.relu(torch.linalg.vector_norm(acceleration, dim=-1) - max_acceleration).square().sum(-1)
    return speed_cost, acceleration_cost


def reduce_waypoint_distances(distances, waypoint_weights=None):
    """Retain the original mean exactly, or apply explicit normalized priorities."""
    if waypoint_weights is None:
        return distances.mean(dim=-1)
    if (not isinstance(waypoint_weights, torch.Tensor) or waypoint_weights.ndim != 1
            or waypoint_weights.shape[0] != distances.shape[-1]
            or waypoint_weights.device != distances.device or not waypoint_weights.is_floating_point()
            or not bool(torch.isfinite(waypoint_weights).all()) or bool((waypoint_weights < 0).any())
            or not bool(torch.isclose(waypoint_weights.sum(), waypoint_weights.new_tensor(1.), atol=1e-6, rtol=0))):
        raise ValueError("Waypoint weights require finite nonnegative normalized floating[K] on the trajectory device")
    return (distances * waypoint_weights).sum(dim=-1)


def clean_waypoint_guidance_gradient(noisy, clean_physical, target_physical, indices, *, num_samples=1,
                                    waypoint_weight=1., waypoint_weights=None, current=None, scene_ids=None,
                                    agent_collision_weight=0., agent_collision_radius=.3,
                                    scene_collision_weight=0., scene_collision_radius=.5,
                                    scene_bounds_xy=None, obstacle_centers_xy=None, obstacle_sizes_xy=None, scene_collision_cost_fn=None,
                                    trajectory_speed_weight=0., max_trajectory_speed_mps=3.,
                                    trajectory_accel_weight=0., max_trajectory_accel_mps2=2.):
    """TRACE clean-output guidance, with a full denoiser input Jacobian.

    The objective is Euclidean waypoint distance in metres, as in the pinned
    TRACE TargetPosAtTimeLoss. Average waypoints within each trajectory, then
    sum trajectories so the same actor receives the same guidance irrespective
    of planning chunk/sample count. TRACE averages its scene batches instead.
    Explicit waypoint_weights use the same normalized weighted sum as filtering.
    Gradients are requested only for the noisy input; parameter .grad fields
    and the trained checkpoint are untouched.
    """
    _validate_scene_collision_callback(scene_collision_cost_fn, scene_collision_weight,
        (scene_bounds_xy, obstacle_centers_xy, obstacle_sizes_xy))
    if noisy.ndim != 3 or noisy.shape[-1] != 5 or clean_physical.shape != noisy.shape:
        raise ValueError("Clean guidance requires noisy/clean[B*S,T,5] and targets[B,K,2]")
    if scene_collision_cost_fn is not None:
        if (not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples < 1
                or not isinstance(current, torch.Tensor) or current.ndim != 2 or current.shape[1] != 5
                or noisy.shape[0] != len(current) * num_samples or current.dtype != noisy.dtype
                or current.device != noisy.device):
            raise ValueError("Scene callback guidance requires matching current[B,5], noisy[B*S,T,5] dtype/device and positive samples")
        _finite(current, "scene callback current states")
    loss = noisy.sum() * 0.
    if target_physical is not None:
        if target_physical.ndim != 3 or target_physical.shape[-1] != 2:
            raise ValueError("Clean guidance requires targets[B,K,2]")
        batch, count, _ = target_physical.shape
        if (count < 1 or indices.ndim != 1 or len(indices) != count
                or not isinstance(num_samples, int) or num_samples < 1
                or noisy.shape[0] != batch * num_samples):
            raise ValueError("Clean guidance waypoint/batch/sample shape mismatch")
        residual = clean_physical[:, indices, :2].reshape(batch, num_samples, count, 2) - target_physical[:, None]
        loss = loss + waypoint_weight * reduce_waypoint_distances(
            torch.linalg.vector_norm(residual, dim=-1), waypoint_weights).sum()
    elif waypoint_weights is not None:
        raise ValueError("Explicit waypoint weights require target waypoints")
    if max(agent_collision_weight, scene_collision_weight, trajectory_speed_weight, trajectory_accel_weight) > 0:
        world_xy = _world_prediction_xy(clean_physical, current, num_samples)
    if agent_collision_weight > 0:
        loss = loss + agent_collision_weight * scene_agent_collision_cost(world_xy, scene_ids, agent_collision_radius).sum()
    if scene_collision_weight > 0:
        if scene_collision_cost_fn is None:
            cost = scene_geometry_collision_cost(
                world_xy, scene_bounds_xy, obstacle_centers_xy, obstacle_sizes_xy, scene_collision_radius)
        else:
            cost = _scene_callback_cost(world_xy, scene_collision_cost_fn, scene_collision_radius, require_gradient=True)
        loss = loss + scene_collision_weight * cost.sum()
    if trajectory_speed_weight > 0 or trajectory_accel_weight > 0:
        speed, acceleration = trajectory_dynamics_cost(world_xy, current, max_speed=max_trajectory_speed_mps,
                                                        max_acceleration=max_trajectory_accel_mps2)
        loss = loss + trajectory_speed_weight * speed.sum() + trajectory_accel_weight * acceleration.sum()
    gradient, = torch.autograd.grad(loss, noisy, create_graph=False)
    _finite(gradient, "clean-output waypoint gradient")
    return gradient.detach()


class _HistoryEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        dimensions = config.history_frames * 6  # 5 state coordinates + availability
        self.mlp = MLP(dimensions, config.history_feature_dim,
                       (dimensions, dimensions, config.history_feature_dim), normalization=True)

    def forward(self, normalized_state, available):
        state = torch.where(available[..., None], normalized_state, torch.zeros_like(normalized_state))
        features = torch.cat((state, available[..., None].to(state.dtype)), dim=-1)
        return self.mlp(features.flatten(start_dim=1))


class PaperDiffusionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_buffer("state_mean", torch.tensor(config.state_mean, dtype=torch.float32))
        self.register_buffer("state_std", torch.tensor(config.state_std, dtype=torch.float32))
        self.ego_encoder = _HistoryEncoder(config)
        self.neighbor_encoder = _HistoryEncoder(config)
        self.conditioning = MLP(2 * config.history_feature_dim, config.conditioning_feature_dim,
                                (config.conditioning_feature_dim, config.conditioning_feature_dim), normalization=True)
        self.map_encoder = MapEncoder("resnet18", (2, 224, 224), grid_feature_dim=config.map_grid_feature_dim)
        self.temporal = TemporalMapUnet(config.horizon, 5 + config.map_grid_feature_dim,
                                       config.conditioning_feature_dim, 5, config.base_dim, (2, 4, 8))
        betas = cosine_beta_schedule(config.diffusion_steps)
        alphas = 1 - betas
        cumulative = alphas.cumprod(dim=0)
        previous = torch.cat((torch.ones(1), cumulative[:-1]))
        self.register_buffer("betas", betas)
        self.register_buffer("cumulative_alpha", cumulative)
        self.register_buffer("posterior_variance", betas * (1 - previous) / (1 - cumulative))
        self.register_buffer("posterior_mean_clean", betas * previous.sqrt() / (1 - cumulative))
        self.register_buffer("posterior_mean_noisy", (1 - previous) * alphas.sqrt() / (1 - cumulative))

    def normalize(self, physical):
        return (physical - self.state_mean) / self.state_std

    def denormalize(self, normalized):
        return normalized * self.state_std + self.state_mean

    def encode_context(self, context, *, dropout=False):
        history = self.normalize(context["history"])
        available = context["history_mask"]
        neighbors = self.normalize(context["neighbors"])
        neighbor_available = context["neighbor_mask"]
        raster = context["raster"]
        batch, count, frames, _ = neighbors.shape
        if dropout:
            map_drop = torch.rand(batch, device=history.device) < self.config.map_conditioning_dropout
            neighbor_drop = torch.rand(batch, device=history.device) < self.config.neighbor_conditioning_dropout
            raster = torch.where(map_drop[:, None, None, None], torch.full_like(raster, self.config.unknown_map_value), raster)
            neighbor_available = neighbor_available & ~neighbor_drop[:, None, None]
        ego = self.ego_encoder(history, available)
        encoded = self.neighbor_encoder(neighbors.reshape(batch * count, frames, 5), neighbor_available.reshape(batch * count, frames)).reshape(batch, count, -1)
        valid = neighbor_available.any(dim=-1)
        encoded = encoded.masked_fill(~valid[..., None], -torch.inf).amax(dim=1)
        encoded = torch.where(valid.any(dim=1, keepdim=True), encoded, torch.zeros_like(encoded))
        _, grid = self.map_encoder(raster)
        if "map_condition_mask" in context:
            grid = grid.masked_fill(~context["map_condition_mask"][:, None, None, None], 0.)
        return {"conditioning": self.conditioning(torch.cat((ego, encoded), dim=-1)), "map_grid": grid}

    def forward(self, noisy, timesteps, encoded):
        xy = self.denormalize(noisy.detach())[..., :2]
        _, _, height, width = encoded["map_grid"].shape
        raster_xy = (xy * 12 + xy.new_tensor([56, 112])) * xy.new_tensor([width / 224, height / 224])
        # Leave finite coordinates unchanged. In particular, guidance must not
        # silently repair a divergent trajectory by clipping its world XY.
        # torch._assert_async is traceable by the optional compiled forward;
        # on CPU it fails before NaN -> int64-min feature-grid indexing.
        torch._assert_async(torch.isfinite(raster_xy).all(), "Nonfinite diffusion map coordinates before feature-grid lookup")
        features = query_feature_grid(raster_xy, encoded["map_grid"])
        return self.temporal(torch.cat((noisy, features), dim=-1), encoded["conditioning"], timesteps)


@dataclass(frozen=True)
class PaperDiffusionTrainBatch:
    context: dict[str, torch.Tensor]
    clean_trajectory: torch.Tensor  # LOCAL [B,52,5] physical x,y,u,v,theta

    @classmethod
    def from_world(cls, planner, states, history, neighbors, raster, future, **kwargs):
        current = planner.tensor(states)
        future = planner.tensor(future)
        if future.shape != (current.shape[0], 52, 5):
            raise ValueError("future must be world[x,y,vx,vy,yaw] [B,52,5]")
        _finite(future, "future")
        return cls(planner.prepare_context(current, history, neighbors, raster, **kwargs),
                   world_to_agent_state(future, current))


@dataclass(frozen=True)
class PaperDiffusionPlan:
    xy: torch.Tensor
    yaw: torch.Tensor
    vel: torch.Tensor  # speed magnitude, for interfaces requiring a scalar
    velocity: torch.Tensor  # unconstrained WORLD vx/vy, including lateral motion
    time: torch.Tensor
    sample_states: torch.Tensor  # full LOCAL physical [B,S,52,5]
    selected_sample: torch.Tensor
    reference: str = "Independently trained five-state method implementation; not released author checkpoint"


class PaperDiffusionPlanner:
    def __init__(self, config=None, *, checkpoint_path=None, device="cpu", compile_forward=False):
        self.device = torch.device(device)
        artifact = None
        if checkpoint_path is not None:
            artifact = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if artifact.get("schema") != PAPER_SCHEMA:
                raise ValueError("Expected independent five-state diffusion artifact; TRACE checkpoints are incompatible")
            saved_config = PaperDiffusionConfig(**artifact["config"])
            if config is not None and asdict(config) != asdict(saved_config):
                raise ValueError("Checkpoint and supplied five-state configuration differ")
            config = saved_config
        self.config = config or PaperDiffusionConfig()
        self.model = PaperDiffusionModel(self.config).to(self.device)
        self.trained_steps = 0
        if artifact is not None:
            for key, value in artifact["model"].items():
                _finite(value, f"checkpoint.{key}")
            self.model.load_state_dict(artifact["model"], strict=True)
            self.trained_steps = int(artifact["step"])
        self.model.eval()
        self._compiled_forward = None
        self.set_compiled_forward(compile_forward)

    @property
    def compiled_forward_enabled(self):
        return self._compiled_forward is not None

    def set_compiled_forward(self, enabled=True, *, backend="inductor"):
        """Optional inference-only compile, preserving the original model/state.

        The bound forward function is compiled with dynamic=True/mode=default;
        no CUDA graphs/reduced-overhead mode, precision, sampler, schedule,
        config, or RNG draws are changed. Compilation is lazy and errors remain
        explicit. Training, state_dict and checkpoint restoration use the
        unwrapped model. backend=eager permits a lightweight actual Dynamo
        graph-capture CPU regression; production defaults to Inductor.
        """
        if not isinstance(enabled, bool):
            raise ValueError("compile_forward must be a boolean")
        self._compiled_forward = (torch.compile(self.model.forward, dynamic=True, mode="default", backend=backend)
                                  if enabled else None)

    def tensor(self, value, *, boolean=False):
        return torch.as_tensor(value, device=self.device, dtype=torch.bool if boolean else torch.float32)

    def prepare_context(self, states, history, neighbors, raster, *, history_mask=None,
                        neighbor_mask=None, map_condition_mask=None):
        current, history = self.tensor(states), self.tensor(history)
        if current.ndim != 2 or current.shape[1] != 5 or history.shape != (current.shape[0], 31, 5):
            raise ValueError("states[B,5] and history[B,31,5] required")
        batch = current.shape[0]
        _finite(current, "states")
        mask = torch.ones((batch, 31), device=self.device, dtype=torch.bool) if history_mask is None else self.tensor(history_mask, boolean=True)
        if mask.shape != (batch, 31) or not bool(mask[:, -1].all()):
            raise ValueError("history_mask[B,31] must include the current ego frame")
        _finite(history[mask], "available history")
        if not torch.allclose(history[:, -1], current, atol=1e-4, rtol=1e-5):
            raise ValueError("Current history frame must match states")
        history = torch.where(mask[..., None], history, torch.zeros_like(history))
        other = current.new_empty((batch, 0, 31, 5)) if neighbors is None else self.tensor(neighbors)
        if other.ndim != 4 or other.shape[0] != batch or other.shape[2:] != (31, 5):
            raise ValueError("neighbors[B,N,31,5] required")
        count = other.shape[1]
        neighbor_available = torch.ones((batch, count, 31), device=self.device, dtype=torch.bool) if neighbor_mask is None else self.tensor(neighbor_mask, boolean=True)
        if neighbor_available.shape != (batch, count, 31):
            raise ValueError("neighbor_mask[B,N,31] required")
        _finite(other[neighbor_available], "available neighbors")
        other = torch.where(neighbor_available[..., None], other, torch.zeros_like(other))
        if count == 0:
            other = current.new_zeros((batch, 1, 31, 5))
            neighbor_available = torch.zeros((batch, 1, 31), device=self.device, dtype=torch.bool)
        image = self.tensor(raster)
        if image.shape != (batch, 2, 224, 224):
            raise ValueError("raster[B,2,224,224] required")
        _finite(image, "raster")
        if bool(((image < 0) | (image > 1)).any()):
            raise ValueError("Raster values must be within [0,1]")
        map_available = torch.ones(batch, device=self.device, dtype=torch.bool) if map_condition_mask is None else self.tensor(map_condition_mask, boolean=True)
        if map_available.shape != (batch,):
            raise ValueError("map_condition_mask[B] required")
        return {"history": world_to_agent_state(history, current), "history_mask": mask,
                "neighbors": world_to_agent_state(other, current), "neighbor_mask": neighbor_available,
                "raster": image, "map_condition_mask": map_available}

    def training_loss(self, clean_trajectory, context, timesteps=None, *, noise=None,
                      drop_conditioning=True, training=True):
        clean = self.tensor(clean_trajectory)
        if clean.ndim != 3 or clean.shape[1:] != (52, 5):
            raise ValueError("clean_trajectory[B,52,5] LOCAL physical x,y,u,v,theta required")
        _finite(clean, "clean_trajectory")
        batch = clean.shape[0]
        t = torch.randint(self.config.diffusion_steps, (batch,), device=self.device) if timesteps is None else torch.as_tensor(timesteps, device=self.device, dtype=torch.long)
        if t.shape != (batch,) or bool(((t < 0) | (t >= self.config.diffusion_steps)).any()):
            raise ValueError("timesteps[B] outside diffusion schedule")
        random_noise = torch.randn_like(clean) if noise is None else self.tensor(noise)
        if random_noise.shape != clean.shape:
            raise ValueError("noise must have shape[B,52,5]")
        _finite(random_noise, "noise")
        self.model.train(training)
        encoded = self.model.encode_context(context, dropout=drop_conditioning and training)
        normalized = self.model.normalize(clean)
        alpha = self.model.cumulative_alpha[t, None, None]
        noisy = alpha.sqrt() * normalized + (1 - alpha).sqrt() * random_noise
        prediction = self.model(noisy, t, encoded)
        residual = prediction - normalized
        physical_yaw_error = residual[..., 4] * self.model.state_std[4]
        wrapped_yaw_error = torch.remainder(physical_yaw_error + math.pi, 2 * math.pi) - math.pi
        residual = torch.cat((residual[..., :4], (wrapped_yaw_error / self.model.state_std[4])[..., None]), dim=-1)
        loss = residual.square().mean()
        return loss, {"state_mse": residual[..., :4].square().mean().detach(),
                      "yaw_mse": residual[..., 4].square().mean().detach()}

    def train_step(self, batch, optimizer, *, max_grad_norm=1.0, **loss_kwargs):
        if not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be finite and positive")
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = self.training_loss(batch.clean_trajectory, batch.context, **loss_kwargs)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite diffusion training loss; optimizer not stepped")
        loss.backward()
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        if not all(bool(torch.isfinite(g).all()) for g in gradients):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Nonfinite diffusion gradient; optimizer not stepped")
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        self.trained_steps += 1
        return {"loss": float(loss.detach()), "grad_norm": float(gradient_norm),
                **{key: float(value) for key, value in metrics.items()}}

    @torch.no_grad()
    def plan(self, states, history, neighbors, raster, waypoints=None, *, waypoint_times=None,
             num_samples=1, seed=None, guidance_weight=1.0, guidance_mode="legacy_posterior",
             scene_ids=None, agent_collision_weight=0., agent_collision_radius=.3,
             scene_collision_weight=0., scene_collision_radius=.5,
             scene_bounds_xy=None, obstacle_centers_xy=None, obstacle_sizes_xy=None, scene_collision_cost_fn=None,
             trajectory_speed_weight=0., max_trajectory_speed_mps=3.,
             trajectory_accel_weight=0., max_trajectory_accel_mps2=2.,
             near_horizon_mass=0.0, **context_kwargs):
        """Sample a trajectory with explicit, versioned goal-guidance semantics.

        legacy_posterior preserves the published local experiments exactly.
        trace_clean adapts TRACE's public-code clean-output, input-Jacobian guidance to our
        independently trained five-state model. It has different loss units
        and weight scaling; it is not an original TRACE checkpoint/model.
        trace_clean_variance retains the covariance factor printed in TRACE
        Eq.6, which the public sampler explicitly omits for empirical reasons.
        Neither mode guides/noises the final t=0 transition, following TRACE.
        Optional near_horizon_mass redistributes goal weight toward times<=1s;
        this is an evaluation engineering variant, not an author setting.
        Zero retains the original mean gradient and sample-filter operations.
        Optional scene_collision_cost_fn consumes already-world XY and radius,
        returns finite nonnegative[B,S], and is shared by clean guidance and
        candidate filtering. It is mutually exclusive with analytic AABBs.
        Unknown/exterior clearance and cost units belong to that provider.
        """
        if self.trained_steps < 1:
            raise RuntimeError("Five-state diffusion has no trained weights; train or load a training artifact")
        if not isinstance(num_samples, int) or num_samples < 1 or not math.isfinite(guidance_weight) or guidance_weight < 0:
            raise ValueError("Positive num_samples and finite nonnegative guidance_weight required")
        if guidance_mode not in ("legacy_posterior", "trace_clean", "trace_clean_variance"):
            raise ValueError("guidance_mode must be legacy_posterior, trace_clean or trace_clean_variance")
        near_horizon_mass = validate_near_horizon_mass(near_horizon_mass)
        if near_horizon_mass > 0 and (guidance_mode == "legacy_posterior" or waypoints is None):
            raise ValueError("Near-horizon waypoint priority requires a clean guidance mode and explicit waypoints")
        if (not math.isfinite(agent_collision_weight) or agent_collision_weight < 0
                or not math.isfinite(agent_collision_radius) or agent_collision_radius <= 0):
            raise ValueError("Agent collision weight must be nonnegative and radius positive, both finite")
        if agent_collision_weight > 0 and guidance_mode == "legacy_posterior":
            raise ValueError("Scene agent collision guidance requires a clean-output guidance mode")
        if (not math.isfinite(scene_collision_weight) or scene_collision_weight < 0
                or not math.isfinite(scene_collision_radius) or scene_collision_radius <= 0):
            raise ValueError("Scene collision weight must be nonnegative and radius positive, both finite")
        if scene_collision_weight > 0 and guidance_mode == "legacy_posterior":
            raise ValueError("Scene geometry collision guidance requires a clean-output guidance mode")
        if (any(not math.isfinite(w) or w < 0 for w in (trajectory_speed_weight, trajectory_accel_weight))
                or any(not math.isfinite(limit) or limit <= 0 for limit in (max_trajectory_speed_mps, max_trajectory_accel_mps2))):
            raise ValueError("Dynamics guidance weights must be nonnegative and limits positive, all finite")
        dynamics_guided = trajectory_speed_weight > 0 or trajectory_accel_weight > 0
        if dynamics_guided and guidance_mode == "legacy_posterior":
            raise ValueError("Trajectory dynamics guidance requires a clean-output guidance mode")
        _validate_scene_collision_callback(scene_collision_cost_fn, scene_collision_weight,
            (scene_bounds_xy, obstacle_centers_xy, obstacle_sizes_xy))
        if scene_collision_cost_fn is not None and guidance_mode == "legacy_posterior":
            raise ValueError("Scene collision callbacks require a clean-output guidance mode")
        if scene_collision_weight > 0 and scene_collision_cost_fn is None:
            if scene_bounds_xy is None or obstacle_centers_xy is None or obstacle_sizes_xy is None:
                raise ValueError("Explicit bounds and obstacle centers/sizes required for scene collision guidance")
            scene_bounds_xy, obstacle_centers_xy, obstacle_sizes_xy = (
                self.tensor(value) for value in (scene_bounds_xy, obstacle_centers_xy, obstacle_sizes_xy))
            if (scene_bounds_xy.shape != (4,) or not bool((scene_bounds_xy[2:] > scene_bounds_xy[:2]).all())
                    or obstacle_centers_xy.ndim != 2 or obstacle_centers_xy.shape[-1] != 2
                    or obstacle_sizes_xy.shape != obstacle_centers_xy.shape or not bool((obstacle_sizes_xy > 0).all())):
                raise ValueError("Expected common bounds[4], obstacle_centers[M,2] and positive obstacle_sizes[M,2]")
            for value, name in ((scene_bounds_xy, "scene bounds"), (obstacle_centers_xy, "obstacle centers"), (obstacle_sizes_xy, "obstacle sizes")):
                _finite(value, name)
        current = self.tensor(states)
        context = self.prepare_context(current, history, neighbors, raster, **context_kwargs)
        self.model.eval()
        encoded = self.model.encode_context(context)
        encoded = {key: value.repeat_interleave(num_samples, dim=0) for key, value in encoded.items()}
        batch = current.shape[0]
        if agent_collision_weight > 0:
            scene_ids = torch.as_tensor(scene_ids, device=self.device) if scene_ids is not None else None
            if (scene_ids is None or scene_ids.shape != (batch,) or scene_ids.dtype == torch.bool
                    or scene_ids.is_complex() or not bool(torch.isfinite(scene_ids).all())
                    or (scene_ids.is_floating_point() and not bool((scene_ids == scene_ids.round()).all()))):
                raise ValueError("Explicit finite integer scene_ids[B] required for joint agent guidance")
            scene_ids = scene_ids.long()
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        x = torch.randn((batch * num_samples, 52, 5), device=self.device, generator=generator)
        target = indices = waypoint_weights = None
        if waypoints is not None:
            target = self.tensor(waypoints)
            if target.ndim == 2:
                target = target[:, None]
            if target.ndim != 3 or target.shape[0] != batch or target.shape[-1] != 2 or target.shape[1] < 1:
                raise ValueError("waypoints[B,K,2] required")
            _finite(target, "waypoints")
            times = torch.arange(1, target.shape[1] + 1, device=self.device).mul(50 / target.shape[1]).round().mul(0.1) if waypoint_times is None else self.tensor(waypoint_times)
            if times.shape != (target.shape[1],) or not bool(((times > 0) & (times <= 5.2)).all()) or not torch.allclose(times / 0.1, (times / 0.1).round(), atol=1e-4):
                raise ValueError("waypoint times must be exact 0.1s-grid times in 0..5.2s")
            indices = (times / 0.1).round().long() - 1
            if near_horizon_mass > 0:
                # Use the actual indexed frame time, including tolerated grid
                # rounding, so a1s waypoint stays in the inclusive near window.
                frame_times = [(index+1)*self.config.dt for index in indices.detach().cpu().tolist()]
                waypoint_weights = current.new_tensor(near_horizon_waypoint_weights(frame_times, near_horizon_mass))
            dummy = torch.cat((target, target.new_zeros((*target.shape[:-1], 3))), dim=-1)
            target = world_to_agent_state(dummy, current)[..., :2]
            target_normalized = (target - self.model.state_mean[:2]) / self.model.state_std[:2]
        forward = self._compiled_forward if self.compiled_forward_enabled else self.model
        for step in range(self.config.diffusion_steps - 1, -1, -1):
            t = torch.full((batch * num_samples,), step, device=self.device, dtype=torch.long)
            _finite(x, f"diffusion noisy state at step {step}; reduce guidance weight if divergent")
            guided = ((target is not None and guidance_weight > 0) or agent_collision_weight > 0 or scene_collision_weight > 0 or dynamics_guided) and step > 0
            if guided and guidance_mode in ("trace_clean", "trace_clean_variance"):
                with torch.enable_grad():
                    noisy = x.detach().requires_grad_(True)
                    predicted_clean = forward(noisy, t, encoded)
                    _finite(predicted_clean, f"denoiser prediction at step {step}")
                    gradient = clean_waypoint_guidance_gradient(
                        noisy, self.model.denormalize(predicted_clean), target, indices,
                        num_samples=num_samples, waypoint_weight=guidance_weight, waypoint_weights=waypoint_weights,
                        current=current, scene_ids=scene_ids,
                        agent_collision_weight=agent_collision_weight, agent_collision_radius=agent_collision_radius,
                        scene_collision_weight=scene_collision_weight, scene_collision_radius=scene_collision_radius,
                        scene_bounds_xy=scene_bounds_xy, obstacle_centers_xy=obstacle_centers_xy, obstacle_sizes_xy=obstacle_sizes_xy,
                        scene_collision_cost_fn=scene_collision_cost_fn,
                        trajectory_speed_weight=trajectory_speed_weight, max_trajectory_speed_mps=max_trajectory_speed_mps,
                        trajectory_accel_weight=trajectory_accel_weight, max_trajectory_accel_mps2=max_trajectory_accel_mps2)
                # Official TRACE changes the clean prediction, recomputes its
                # DDPM posterior, and explicitly omits variance scaling.
                scale = self.model.posterior_variance[step] if guidance_mode == "trace_clean_variance" else 1.
                predicted_clean = predicted_clean.detach() - scale * gradient
            else:
                predicted_clean = forward(x, t, encoded)
            _finite(predicted_clean, f"diffusion clean prediction at step {step}; reduce guidance weight if divergent")
            mean = self.model.posterior_mean_clean[step] * predicted_clean + self.model.posterior_mean_noisy[step] * x
            if guided and guidance_mode == "legacy_posterior":
                gradient = waypoint_guidance_gradient(mean, target_normalized, indices, num_samples=num_samples)
                mean = mean - guidance_weight * self.model.posterior_variance[step] * gradient
            if step > 0:
                mean = mean + self.model.posterior_variance[step].sqrt() * torch.randn(x.shape, device=self.device, generator=generator)
            x = mean
        physical = self.model.denormalize(x.reshape(batch, num_samples, 52, 5))
        _finite(physical, "sampled trajectory")
        selected = torch.zeros(batch, device=self.device, dtype=torch.long) if target is None else reduce_waypoint_distances(
            torch.linalg.vector_norm(physical[:, :, indices, :2] - target[:, None], dim=-1), waypoint_weights).argmin(dim=1)
        if (agent_collision_weight > 0 or scene_collision_weight > 0 or dynamics_guided) and num_samples > 1:
            # A joint scene sample cannot be assembled by picking each actor's
            # independently best sample: that can introduce new collisions.
            world_xy = _world_prediction_xy(physical, current, num_samples)
            goal_cost = physical.new_zeros(batch, num_samples) if target is None else reduce_waypoint_distances(
                torch.linalg.vector_norm(physical[:, :, indices, :2] - target[:, None], dim=-1), waypoint_weights)
            independent_cost = guidance_weight * goal_cost
            if scene_collision_weight > 0:
                if scene_collision_cost_fn is None:
                    cost = scene_geometry_collision_cost(
                        world_xy, scene_bounds_xy, obstacle_centers_xy, obstacle_sizes_xy, scene_collision_radius)
                else:
                    cost = _scene_callback_cost(world_xy, scene_collision_cost_fn, scene_collision_radius)
                independent_cost = independent_cost + scene_collision_weight * cost
            if dynamics_guided:
                speed, acceleration = trajectory_dynamics_cost(world_xy, current, max_speed=max_trajectory_speed_mps,
                                                                max_acceleration=max_trajectory_accel_mps2)
                independent_cost = independent_cost + trajectory_speed_weight * speed + trajectory_accel_weight * acceleration
            if agent_collision_weight > 0:
                collision = scene_agent_collision_cost(world_xy, scene_ids, agent_collision_radius)
                for scene_id in scene_ids.unique():
                    members = scene_ids == scene_id
                    cost = independent_cost[members].sum(0) + agent_collision_weight * collision[members][:, members].sum((0, 1))
                    selected[members] = cost.argmin()
            else:
                selected = independent_cost.argmin(-1)
        picked = physical[torch.arange(batch, device=self.device), selected, :50]
        cosine, sine = current[:, 4].cos()[:, None], current[:, 4].sin()[:, None]
        xy = torch.stack((picked[..., 0] * cosine - picked[..., 1] * sine,
                          picked[..., 0] * sine + picked[..., 1] * cosine), dim=-1) + current[:, None, :2]
        velocity = torch.stack((picked[..., 2] * cosine - picked[..., 3] * sine,
                                picked[..., 2] * sine + picked[..., 3] * cosine), dim=-1)
        velocity = torch.cat((current[:, None, 2:4], velocity), dim=1)
        yaw = torch.remainder(picked[..., 4] + current[:, None, 4] + math.pi, 2 * math.pi) - math.pi
        return PaperDiffusionPlan(torch.cat((current[:, None, :2], xy), dim=1),
                                  torch.cat((current[:, 4:5], yaw), dim=1),
                                  velocity.norm(dim=-1), velocity,
                                  torch.arange(51, device=self.device) * 0.1, physical, selected)

    def save_training_checkpoint(self, path, *, optimizer, step=None, dataset_metadata,
                                 scheduler=None):
        from protomotions.utils.checkpoint import atomic_torch_save

        completed = self.trained_steps if step is None else int(step)
        if completed < 1:
            raise ValueError("Cannot publish an untrained artifact as trained five-state diffusion")
        numpy_rng = np.random.get_state()
        artifact = {"schema": PAPER_SCHEMA, "config": asdict(self.config), "model": self.model.state_dict(),
                    "optimizer": optimizer.state_dict(), "step": completed, "dataset_metadata": dataset_metadata,
                    "reference": "Independent five-state adaptation trained from scratch; not author paper checkpoint",
                    "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
                            "python": random.getstate(), "numpy": [numpy_rng[0], numpy_rng[1].tolist(), numpy_rng[2], numpy_rng[3], numpy_rng[4]]}}
        if scheduler is not None:
            artifact["scheduler"] = scheduler.state_dict()
        atomic_torch_save(artifact, Path(path))

    def restore_training_checkpoint(self, path, *, optimizer=None, scheduler=None,
                                    restore_rng=True):
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if artifact.get("schema") != PAPER_SCHEMA or artifact.get("config") != asdict(self.config):
            raise ValueError("Five-state checkpoint schema, normalization or architecture mismatch")
        for key, value in artifact["model"].items():
            _finite(value, f"checkpoint.{key}")
        self.model.load_state_dict(artifact["model"], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(artifact["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(artifact["scheduler"])
        if restore_rng:
            rng = artifact["rng"]
            torch.set_rng_state(rng["torch"])
            random.setstate(rng["python"])
            numpy_rng = rng["numpy"]
            np.random.set_state((numpy_rng[0], np.asarray(numpy_rng[1], dtype=np.uint32), numpy_rng[2], numpy_rng[3], numpy_rng[4]))
            if rng["cuda"] and self.device.type == "cuda":
                if not torch.cuda.is_available() or len(rng["cuda"]) != torch.cuda.device_count():
                    raise ValueError("Saved CUDA RNG topology differs from current hardware")
                torch.cuda.set_rng_state_all(rng["cuda"])
        self.trained_steps = int(artifact["step"])
        return artifact
