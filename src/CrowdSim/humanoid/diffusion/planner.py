"""Actual TRACE trajectory inference/training in world-coordinate CrowdSim inputs.

The public checkpoint is a TRACE baseline. Neither this adapter nor goal guidance
establishes identity with the unreleased NavIsaacLab authors' adapted checkpoint.
"""

from __future__ import annotations

from protomotions.repository import repository_root

from dataclasses import dataclass
from contextlib import nullcontext
from pathlib import Path
import json
import math
import random
from typing import Any

import numpy as np
import torch

from .model import DiffuserModel
from .trace_helpers import convert_state_to_state_and_action


_ROOT = repository_root(__file__)
DEFAULT_CHECKPOINT = _ROOT / "data/pretrained_models/paper_references/trace/ckpt/trace/orca_mixed/iter40000.ckpt"
DEFAULT_CONFIG = DEFAULT_CHECKPOINT.parent / "config.json"
SCHEMA = "crowdsim.trace_diffusion.v1"


def _tensor(value, device, *, boolean=False):
    return torch.as_tensor(value, device=device, dtype=torch.bool if boolean else torch.float32)


def _require_finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains nonfinite available values")


def _local(states, current):
    delta = states[..., :2] - current[:, None, :2]
    cosine, sine = current[:, 3].cos(), current[:, 3].sin()
    xy = torch.stack((delta[..., 0] * cosine[:, None] + delta[..., 1] * sine[:, None],
                      -delta[..., 0] * sine[:, None] + delta[..., 1] * cosine[:, None]), dim=-1)
    yaw = torch.remainder(states[..., 3] - current[:, None, 3] + math.pi, 2 * math.pi) - math.pi
    return torch.cat((xy, states[..., 2:3], yaw[..., None]), dim=-1)


@dataclass(frozen=True)
class DiffusionPlan:
    """Selected world trajectory: time[51], xy[B,51,2], yaw/vel[B,51]."""

    xy: torch.Tensor
    yaw: torch.Tensor
    vel: torch.Tensor
    time: torch.Tensor
    sample_states: torch.Tensor  # [B,S,52,6], full physical LOCAL network output
    selected_sample: torch.Tensor
    reference: str = "TRACE baseline or explicitly trained adapter; not author paper checkpoint"
    guidance_diagnostics: dict | None = None
    sampling_dynamics_limits: dict | None = None


@dataclass(frozen=True)
class DiffusionTrainBatch:
    """Context dictionary and physical agent-local clean[x,y,v,yaw,a,yawrate]."""

    context: dict[str, torch.Tensor]
    clean_trajectory: torch.Tensor

    @classmethod
    def from_world(cls, planner, states, history, neighbors, raster, future, **context_kwargs):
        context = planner.prepare_context(states, history, neighbors, raster, **context_kwargs)
        state = _tensor(states, planner.device)
        future = _tensor(future, planner.device)
        if future.shape != (state.shape[0], 52, 3):
            raise ValueError("future must be world[x,y,yaw] [B,52,3] at t=0.1..5.2s")
        _require_finite(future, "future")
        padded = torch.cat((future[..., :2], torch.zeros_like(future[..., :1]), future[..., 2:3]), dim=-1)
        local = _local(padded, state)[..., [0, 1, 3]]
        clean = convert_state_to_state_and_action(local, context["curr_speed"], 0.1)
        return cls(context, clean)


class _WaypointGuidance:
    """TRACE TargetPosAtTimeLoss generalized to explicitly timed waypoints."""

    def __init__(self, local_xy, indices, weight):
        self.local_xy, self.indices, self.weight = local_xy, indices, weight

    def compute_guidance_loss(self, trajectory, batch):
        positions = trajectory[:, :, self.indices, :2]
        per_sample = torch.linalg.vector_norm(positions - self.local_xy[:, None], dim=-1).mean(dim=-1)
        return self.weight * per_sample.mean(), {"waypoint": per_sample.detach()}


class DiffusionPlanner:
    """Strictly loaded TRACE temporal U-Net + actual 100-step DDPM sampling.

    States/history are world [x,y,speed,yaw]. History is oldest→current at
    0.1s intervals, exactly 31 frames. Masks may contain missing past samples;
    current ego must be available. Raster is agent-aligned semantic layers,
    layer0 walkable, layer1 obstruction (ORCA occupancy adaptation), positive
    local x/y map to increasing column/row. Masked values can be NaN and are
    replaced with zero before encoders. Empty neighbors are genuinely masked.
    """

    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT, config_path=DEFAULT_CONFIG,
                 *, device="cpu", use_ema=True):
        self.device = torch.device(device)
        self.config = json.loads(Path(config_path).read_text())
        algorithm = self.config["algo"]
        if algorithm["horizon"] != 52 or algorithm["history_num_frames"] != 30 or algorithm["step_time"] != 0.1:
            raise ValueError("Adapter requires TRACE 31-frame history, 52-frame future and 0.1s dt")
        if algorithm.get("predict_epsilon", False) or algorithm["diffuser_input_mode"] != "state_and_action":
            raise ValueError("Adapter requires TRACE clean action (x0) prediction")
        self.model = DiffuserModel(
            map_encoder_model_arch=algorithm["map_encoder_model_arch"], input_image_shape=(2, 224, 224),
            map_feature_dim=algorithm["map_feature_dim"], map_grid_feature_dim=algorithm["map_grid_feature_dim"],
            diffuser_model_arch=algorithm["diffuser_model_arch"], horizon=52, observation_dim=4,
            action_dim=2, output_dim=2, cond_feature_dim=algorithm["cond_feat_dim"],
            rasterized_map=algorithm["rasterized_map"], use_map_feat_global=algorithm["use_map_feat_global"],
            use_map_feat_grid=algorithm["use_map_feat_grid"], hist_num_frames=31,
            hist_feature_dim=algorithm["history_feature_dim"], n_timesteps=algorithm["n_diffusion_steps"],
            loss_type=algorithm["loss_type"], action_weight=algorithm["action_weight"],
            loss_discount=algorithm["loss_discount"], dim_mults=algorithm["dim_mults"],
            dynamics_type=algorithm["dynamics"]["type"], dynamics_kwargs=algorithm["dynamics"],
            base_dim=algorithm["base_dim"], diffuser_norm_info=algorithm["diffuser_norm_info"],
            agent_hist_norm_info=algorithm["agent_hist_norm_info"],
            neighbor_hist_norm_info=algorithm["neighbor_hist_norm_info"],
            cond_fill_value=algorithm["conditioning_drop_fill"], dt=0.1,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.source_checkpoint = str(Path(checkpoint_path).resolve())
        if checkpoint.get("schema") == SCHEMA:
            if checkpoint["config"] != self.config:
                raise ValueError("Training artifact and supplied configuration differ")
            state = checkpoint["model"]
            self.weight_source = "trained_adapter"
        else:
            prefix = "ema_policy." if use_ema else "nets.policy."
            state = {key.removeprefix(prefix): value for key, value in checkpoint["state_dict"].items() if key.startswith(prefix)}
            if not state:
                raise ValueError(f"Checkpoint has no {prefix} parameters")
            self.weight_source = prefix
        for name, value in state.items():
            _require_finite(value, f"checkpoint.{name}")
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.loaded_state_entries = len(state)

    def prepare_context(self, states, history, neighbors, raster, *, history_mask=None,
                        neighbor_mask=None, extents=None, neighbor_extents=None):
        state, history = _tensor(states, self.device), _tensor(history, self.device)
        if state.ndim != 2 or state.shape[1] != 4 or history.shape != (state.shape[0], 31, 4):
            raise ValueError("states[B,4] and history[B,31,4] required")
        batch = state.shape[0]
        _require_finite(state, "states")
        available = torch.ones((batch, 31), device=self.device, dtype=torch.bool) if history_mask is None else _tensor(history_mask, self.device, boolean=True)
        if available.shape != (batch, 31) or not bool(available[:, -1].all()):
            raise ValueError("history mask[B,31] must include current ego")
        _require_finite(history[available], "available history")
        if not torch.allclose(history[:, -1], state, atol=1e-4, rtol=1e-5):
            raise ValueError("history final frame must match current states")
        history = torch.where(available[..., None], history, torch.zeros_like(history))
        local = _local(history, state)
        extent = _tensor([0.5, 0.5, 1.7], self.device).expand(batch, 3) if extents is None else _tensor(extents, self.device)
        if extent.shape != (batch, 3) or bool((extent <= 0).any()):
            raise ValueError("extents[B,3] must be positive")
        _require_finite(extent, "extent")
        if neighbors is None:
            other = state.new_zeros((batch, 0, 31, 4))
        else:
            other = _tensor(neighbors, self.device)
        if other.ndim != 4 or other.shape[0] != batch or other.shape[2:] != (31, 4):
            raise ValueError("neighbors[B,N,31,4] required")
        count = other.shape[1]
        other_available = torch.ones((batch, count, 31), device=self.device, dtype=torch.bool) if neighbor_mask is None else _tensor(neighbor_mask, self.device, boolean=True)
        if other_available.shape != (batch, count, 31):
            raise ValueError("neighbor mask[B,N,31] required")
        _require_finite(other[other_available], "available neighbors")
        other = torch.where(other_available[..., None], other, torch.zeros_like(other))
        other_extent = extent[:, None].expand(batch, count, 3) if neighbor_extents is None else _tensor(neighbor_extents, self.device)
        if other_extent.shape != (batch, count, 3):
            raise ValueError("neighbor extents[B,N,3] required")
        _require_finite(other_extent, "neighbor extents")
        if bool((other_extent[other_available.any(dim=-1)] <= 0).any()):
            raise ValueError("available neighbor extents must be positive")
        if count == 0:
            other, other_available = state.new_zeros((batch, 1, 31, 4)), torch.zeros((batch, 1, 31), device=self.device, dtype=torch.bool)
            other_extent = state.new_zeros((batch, 1, 3))
            count = 1
        other_local = _local(other.reshape(batch, count * 31, 4), state).reshape(batch, count, 31, 4)
        image = _tensor(raster, self.device)
        if image.shape != (batch, 2, 224, 224):
            raise ValueError("raster[B,2,224,224] required; silently resizing channels is prohibited")
        _require_finite(image, "raster")
        if bool(((image < 0) | (image > 1)).any()):
            raise ValueError("raster semantic values must lie in [0,1]")
        transform = state.new_tensor([[12, 0, 56], [0, 12, 112], [0, 0, 1]]).expand(batch, 3, 3)
        return {"history_positions": local[..., :2], "history_yaws": local[..., 3:4],
                "history_speeds": local[..., 2], "history_availabilities": available,
                "extent": extent, "curr_speed": state[:, 2], "image": image,
                "raster_from_agent": transform,
                "all_other_agents_history_positions": other_local[..., :2],
                "all_other_agents_history_yaws": other_local[..., 3:4],
                "all_other_agents_history_speeds": other_local[..., 2],
                "all_other_agents_history_availabilities": other_available,
                "all_other_agents_extents": other_extent}

    def plan(self, states, history, neighbors, raster, waypoints=None, *, waypoint_times=None,
             guidance_weight=1.0, num_samples=1, seed=None, pedestrian_guidance=None,
             scene_ids=None, analytic_brake_candidates=None, force_brake_mask=None, **context_kwargs):
        if not isinstance(num_samples, int) or num_samples < 1:
            raise ValueError("num_samples must be a positive integer")
        if not math.isfinite(guidance_weight) or guidance_weight < 0:
            raise ValueError("guidance_weight must be finite and nonnegative")
        state = _tensor(states, self.device)
        context = self.prepare_context(state, history, neighbors, raster, **context_kwargs)
        self.model.eval()
        target = indices = None
        if waypoints is not None:
            target = _tensor(waypoints, self.device)
            if target.ndim == 2:
                target = target[:, None]
            if target.ndim != 3 or target.shape[0] != state.shape[0] or target.shape[-1] != 2 or target.shape[1] < 1:
                raise ValueError("waypoints[B,K,2] required")
            _require_finite(target, "waypoints")
            times = torch.arange(1, target.shape[1] + 1, device=self.device).mul(50 / target.shape[1]).round().mul(0.1) if waypoint_times is None else _tensor(waypoint_times, self.device)
            if times.shape != (target.shape[1],) or not bool(((times > 0) & (times <= 5.2)).all()) or not torch.allclose(times / 0.1, (times / 0.1).round(), atol=1e-4):
                raise ValueError("waypoint_times[K] must be 0.1s-grid future times within 5.2s")
            indices = (times / 0.1).round().long() - 1
            dummy = torch.cat((target, torch.zeros_like(target)), dim=-1)
            target = _local(dummy, state)[..., :2]
        pedestrian = None
        if pedestrian_guidance is not None:
            from .guidance import PedestrianGuidance, PedestrianGuidanceConfig
            if not isinstance(pedestrian_guidance, PedestrianGuidanceConfig) or scene_ids is None:
                raise ValueError("Pedestrian guidance requires typed config and explicit scene IDs")
            pedestrian = PedestrianGuidance(state, context, pedestrian_guidance,
                scene_ids=scene_ids, local_targets=target, target_indices=indices)
        self.model.current_guidance = pedestrian or (_WaypointGuidance(target, indices, guidance_weight)
            if target is not None and guidance_weight > 0 else None)
        original_dynamics = sampling_limits = None
        if pedestrian is not None:
            # Explicit inference adaptation to the selected AMP capability.
            # Apply bounds INSIDE every DDPM dynamics reconstruction, not by
            # clipping world positions after planning (which breaks continuity).
            original_dynamics = (self.model.dyn.vbound, self.model.dyn.acce_bound)
            self.model._pedestrian_guidance_step_size = pedestrian_guidance.clean_guidance_step_size
            self.model._bounded_guidance_gradient = pedestrian_guidance.bounded_guidance_gradient
            self.model._pedestrian_speed_ceiling = pedestrian_guidance.max_speed_m_s
            self.model._pedestrian_yaw_rate_ceiling = pedestrian_guidance.max_yaw_rate_rad_s
            self.model.dyn.vbound = [0., pedestrian_guidance.max_speed_m_s]
            self.model.dyn.acce_bound = [-pedestrian_guidance.max_acceleration_m_s2,
                                         pedestrian_guidance.max_acceleration_m_s2]
            sampling_limits = {"speed_m_s": self.model.dyn.vbound.copy(),
                               "acceleration_m_s2": self.model.dyn.acce_bound.copy(),
                               "yaw_rate_rad_s": pedestrian_guidance.max_yaw_rate_rad_s,
                               "kind": "explicit AMP capability bounds inside DDPM dynamics"}
        try:
            devices = [] if self.device.type == "cpu" else [self.device.index if self.device.index is not None else torch.cuda.current_device()]
            rng_context = torch.random.fork_rng(devices=devices) if seed is not None else nullcontext()
            with rng_context, torch.no_grad():
                if seed is not None:
                    torch.random.default_generator.manual_seed(seed)
                    if self.device.type == "cuda":
                        torch.cuda.default_generators[devices[0]].manual_seed(seed)
                auxiliary = self.model.get_aux_info(context)
                generated = self.model.conditional_sample(context, aux_info=auxiliary, num_samp=num_samples,
                    guide_clean=pedestrian_guidance.guide_clean if pedestrian_guidance is not None else False)["pred_traj"]
                physical = self.model.descale_traj(generated)
            _require_finite(physical, "sampled trajectory")
            refinement = None
            if pedestrian is not None and pedestrian_guidance.final_refinement_steps:
                from .refinement import refine_samples
                physical, refinement = refine_samples(physical, state, pedestrian, self.model.dyn)
                sampling_limits['final_clean_refinement']=refinement
                _require_finite(physical, 'refined sampled trajectory')
            diagnostics = None
            fixed_samples = None
            if analytic_brake_candidates is not None:
                if pedestrian is None:
                    raise ValueError("Analytic safety candidates require joint pedestrian admission")
                brake = _tensor(analytic_brake_candidates, self.device)
                if brake.shape != (len(state), 1, 52, 6):
                    raise ValueError("Exactly one explicit analytic candidate[B,1,52,6] required")
                _require_finite(brake, "analytic safety candidate")
                physical = torch.cat((physical, brake), 1)
                if force_brake_mask is not None:
                    forced = torch.as_tensor(force_brake_mask, device=self.device)
                    if forced.shape != (len(state),) or forced.dtype != torch.bool:
                        raise ValueError("Forced brake mask must be genuine bool[B]")
                    fixed_samples = torch.where(forced, num_samples, -1).long()
            elif force_brake_mask is not None:
                raise ValueError("Forced braking needs explicit analytic candidates")
            if pedestrian is not None:
                selected, diagnostics = pedestrian.select(physical, fixed_samples=fixed_samples)
                if refinement is not None:
                    diagnostics = {key.replace('raw_network_', 'candidate_intent_'): value
                                   for key, value in diagnostics.items()}
            elif target is None:
                selected = torch.zeros(state.shape[0], device=self.device, dtype=torch.long)
            else:
                cost = torch.linalg.vector_norm(physical[:, :, indices, :2] - target[:, None], dim=-1).mean(dim=-1)
                selected = cost.argmin(dim=1)
            picked = physical[torch.arange(state.shape[0], device=self.device), selected, :50]
            cosine, sine = state[:, 3].cos()[:, None], state[:, 3].sin()[:, None]
            world_xy = torch.stack((picked[..., 0] * cosine - picked[..., 1] * sine,
                                    picked[..., 0] * sine + picked[..., 1] * cosine), dim=-1) + state[:, None, :2]
            world_yaw = torch.remainder(picked[..., 3] + state[:, None, 3] + math.pi, 2 * math.pi) - math.pi
            reference = "TRACE baseline or explicitly trained adapter; not author paper checkpoint"
            if refinement is not None:
                reference += '; finite bounded final clean-trajectory refinement'
            if analytic_brake_candidates is not None:
                selected = torch.where(selected == num_samples, -1, selected)
                if bool((selected < 0).any()):
                    reference = "explicit mixed TRACE diffusion + analytic safety braking" + ('; finite bounded final clean-trajectory refinement' if refinement is not None else '')
            return DiffusionPlan(torch.cat((state[:, None, :2], world_xy), dim=1),
                                 torch.cat((state[:, 3:4], world_yaw), dim=1),
                                 torch.cat((state[:, 2:3], picked[..., 2]), dim=1),
                                 torch.arange(51, device=self.device) * 0.1, physical, selected, reference=reference,
                                 guidance_diagnostics=diagnostics, sampling_dynamics_limits=sampling_limits)
        finally:
            self.model.current_guidance = None
            if original_dynamics is not None:
                self.model.dyn.vbound, self.model.dyn.acce_bound = original_dynamics
                del self.model._pedestrian_guidance_step_size
                del self.model._bounded_guidance_gradient
                del self.model._pedestrian_speed_ceiling
                del self.model._pedestrian_yaw_rate_ceiling

    def training_loss(self, clean_trajectory, context, timesteps=None, *, noise=None,
                      drop_conditioning=True, training=True):
        """Exact TRACE x0-action prediction, dynamics reconstruction, full6 MSE.

        clean_trajectory uses physical local units, never already normalized.
        t[B] is uniform on [0,99] when omitted. Supplied noise is [B,52,6].
        """
        clean = _tensor(clean_trajectory, self.device)
        if clean.ndim != 3 or clean.shape[1:] != (52, 6):
            raise ValueError("clean_trajectory[B,52,6] in physical local units required")
        batch = clean.shape[0]
        _require_finite(clean, "clean_trajectory")
        self.model.train(training)
        prepared = dict(context)
        if drop_conditioning and training:
            algorithm = self.config["algo"]
            map_drop = torch.rand(batch, device=self.device) < algorithm["conditioning_drop_map_p"]
            neighbor_drop = torch.rand(batch, device=self.device) < algorithm["conditioning_drop_neighbor_p"]
            prepared["image"] = torch.where(map_drop[:, None, None, None], torch.full_like(context["image"], algorithm["conditioning_drop_fill"]), context["image"])
            prepared["all_other_agents_history_availabilities"] = context["all_other_agents_history_availabilities"] & ~neighbor_drop[:, None, None]
        auxiliary = self.model.get_aux_info(prepared)
        t = torch.randint(self.model.n_timesteps, (batch,), device=self.device) if timesteps is None else torch.as_tensor(timesteps, device=self.device, dtype=torch.long)
        if t.shape != (batch,) or bool(((t < 0) | (t >= self.model.n_timesteps)).any()):
            raise ValueError("timesteps[B] outside diffusion schedule")
        normalized = self.model.scale_traj(clean)
        if noise is None:
            return self.model.p_losses(normalized, t, auxiliary)
        noise = _tensor(noise, self.device)
        if noise.shape != clean.shape:
            raise ValueError("noise must have shape[B,52,6]")
        _require_finite(noise, "noise")
        x_noisy = self.model.q_sample(normalized, t, noise)
        actions = x_noisy[..., [4, 5]]
        reconstructed = self.model.convert_action_to_state_and_action(actions, auxiliary)
        features = self.model.query_map_feats(reconstructed.detach(), auxiliary["map_grid_feat"], auxiliary["raster_from_agent"])
        clean_actions = self.model.model(torch.cat((reconstructed, features), dim=-1), auxiliary["cond_feat"], t)
        full_clean = self.model.convert_action_to_state_and_action(clean_actions, auxiliary)
        return self.model.loss_fn(full_clean, normalized)

    def save_training_checkpoint(self, path, *, optimizer, step, dataset_metadata,
                                 scheduler=None):
        """Self-contained atomic artifact; no opaque Hydra/Python config objects."""
        from protomotions.utils.checkpoint import atomic_torch_save

        rng = np.random.get_state()
        artifact: dict[str, Any] = {
            "schema": SCHEMA, "config": self.config, "model": self.model.state_dict(),
            "optimizer": optimizer.state_dict(), "step": int(step), "dataset_metadata": dataset_metadata,
            "source_checkpoint": self.source_checkpoint, "reference": "Explicit adapter training; not author paper checkpoint",
            "rng": {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    "python": random.getstate(), "numpy": [rng[0], rng[1].tolist(), rng[2], rng[3], rng[4]]},
        }
        if scheduler is not None:
            artifact["scheduler"] = scheduler.state_dict()
        atomic_torch_save(artifact, path)

    def restore_training_checkpoint(self, path, *, optimizer=None, scheduler=None,
                                    restore_rng=True):
        """Restore the single matching training artifact, including optimizer/RNG."""
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if artifact.get("schema") != SCHEMA or artifact.get("config") != self.config:
            raise ValueError("Training checkpoint schema or normalization/configuration mismatch")
        for name, value in artifact["model"].items():
            _require_finite(value, f"checkpoint.{name}")
        self.model.load_state_dict(artifact["model"], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(artifact["optimizer"])
        if scheduler is not None:
            if "scheduler" not in artifact:
                raise ValueError("Training checkpoint lacks requested scheduler state")
            scheduler.load_state_dict(artifact["scheduler"])
        if restore_rng:
            rng = artifact["rng"]
            torch.set_rng_state(rng["torch"])
            if rng["cuda"]:
                if not torch.cuda.is_available() or len(rng["cuda"]) != torch.cuda.device_count():
                    raise ValueError("Checkpoint CUDA RNG topology differs from current hardware")
                torch.cuda.set_rng_state_all(rng["cuda"])
            random.setstate(rng["python"])
            state = rng["numpy"]
            np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), state[2], state[3], state[4]))
        self.weight_source = "trained_adapter"
        return artifact
