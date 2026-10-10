"""Unmerged CPU prototype: bounded batch-marginal positive-expert weighting.

No samples, RNG, models, losses outside positive BCE, or runtime are created.
The discriminator remains UNCONDITIONAL: this is not CALM/CMP conditional D.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class WeightingSpec:
    velocity_scale_forward: float = .5
    velocity_scale_lateral: float = .35
    path_turn_scale_rad_s: float = 1.0
    direction_min_speed_m_s: float = .12
    lower: float = .5
    upper: float = 2.0
    control_dt_s: float = 1/30

    def validate(self):
        for name in ('velocity_scale_forward', 'velocity_scale_lateral',
                     'path_turn_scale_rad_s', 'direction_min_speed_m_s', 'control_dt_s'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite positive numeric')
        if self.lower != .5 or self.upper != 2. or type(self.lower) is bool or type(self.upper) is bool:
            raise ValueError('This candidate has explicit fixed bounded weights [.5,2]')
        if not math.isclose(self.control_dt_s, 1/30, rel_tol=0, abs_tol=1e-12):
            raise ValueError('Candidate windows must use the actual30Hz control clock')


def _float_tensor(value, name, shape_tail):
    if not isinstance(value, torch.Tensor) or value.device.type != 'cpu':
        raise ValueError(f'{name} must be a CPU Tensor for this unmerged candidate')
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f'{name} must use float32 or float64')
    if value.ndim != len(shape_tail)+1 or tuple(value.shape[1:]) != shape_tail or not len(value):
        raise ValueError(f'{name} shape must be [nonempty batch,{shape_tail}]')
    if not torch.isfinite(value).all():
        raise ValueError(f'{name} must be finite')


def _integers(value, name, shape):
    if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
            or value.dtype != torch.int64 or tuple(value.shape) != shape):
        raise ValueError(f'{name} must be an int64 CPU Tensor of shape {shape}')


def _wrap_angle(value):
    return torch.atan2(value.sin(), value.cos())


def _path_turn(first_velocity, second_velocity, center_separation_s, min_speed):
    valid = ((first_velocity.norm(dim=-1) >= min_speed)
             & (second_velocity.norm(dim=-1) >= min_speed))
    first = torch.atan2(first_velocity[:, 1], first_velocity[:, 0])
    second = torch.atan2(second_velocity[:, 1], second_velocity[:, 0])
    return torch.where(valid, _wrap_angle(second-first)/center_separation_s, 0.)


@torch.no_grad()
def command_context(raw_task, *, capture_phase, observation_space, spec=WeightingSpec()):
    """Actual raw pre-action task1044 -> intended [vx,vy,path_turn_rate].

    task[:20] already contains root-centered, body-heading-local futureXY at
    +.5,+1,...,+5 seconds. No physical root velocity or normalized RMS is used.
    The caller's metadata is a contract, not authentication of capture timing.
    """
    spec.validate()
    if capture_phase != 'pre_action' or observation_space != 'raw_task_1044':
        raise ValueError('Need actual raw PRE-ACTION task, not post-step or normalized observations')
    _float_tensor(raw_task, 'raw_task', (1044,))
    xy = raw_task[:, :20].reshape(-1, 10, 2)
    first = xy[:, 0]/.5
    second = (xy[:, 1]-xy[:, 0])/.5
    turn = _path_turn(first, second, .5, spec.direction_min_speed_m_s)
    result = torch.cat((first, turn[:, None]), -1)
    if not torch.isfinite(result).all():
        raise ValueError('Command descriptor calculation overflowed')
    return result


@torch.no_grad()
def expert_context(root_xy, root_xyzw, motion_ids, local_frames, clip_counts,
                   *, history_episode_ids=None, spec=WeightingSpec()):
    """Same-clip chronological t-9,...,t,t+1 actual geometry -> descriptor.

    Mean translational velocity spans10 intervals (.333s), expressed in the
    current t root heading. Path turn uses two5-interval velocities; it is NOT
    root body-yaw rate. Actual t->t+1 body-yaw rate is returned as diagnostics.
    """
    spec.validate()
    _float_tensor(root_xy, 'root_xy', (11, 2))
    _float_tensor(root_xyzw, 'root_xyzw', (11, 4))
    count = len(root_xy)
    if root_xyzw.dtype != root_xy.dtype or len(root_xyzw) != count:
        raise ValueError('Root pose tensors must share dtype/batch')
    _integers(motion_ids, 'motion_ids', (count, 11))
    _integers(local_frames, 'local_frames', (count, 11))
    _integers(clip_counts, 'clip_counts', (count,))
    if ((motion_ids < 0).any() or (motion_ids != motion_ids[:, :1]).any()
            or (local_frames < 0).any() or (local_frames >= clip_counts[:, None]).any()
            or not torch.equal(local_frames[:, 1:]-local_frames[:, :-1], torch.ones(count, 10, dtype=torch.int64))):
        raise ValueError('Expert window crosses a motion boundary or is not contiguous history10+next')
    if history_episode_ids is not None:
        _integers(history_episode_ids, 'history_episode_ids', (count, 11))
        if (history_episode_ids < 0).any() or (history_episode_ids != history_episode_ids[:, :1]).any():
            raise ValueError('Expert window crosses an episode/reset boundary')
    norms = root_xyzw.norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), rtol=0, atol=1e-3):
        raise ValueError('Need actual unit XYZW quaternions')
    q = root_xyzw
    projected_x = 1-2*(q[..., 1].square()+q[..., 2].square())
    projected_y = 2*(q[..., 3]*q[..., 2]+q[..., 0]*q[..., 1])
    if (torch.stack((projected_x, projected_y), -1).norm(dim=-1) <= 1e-4).any():
        raise ValueError('Body-heading projection is degenerate')
    yaw = torch.atan2(projected_y, projected_x)
    velocity = (root_xy[:, 10]-root_xy[:, 0])/(10*spec.control_dt_s)
    current_yaw = yaw[:, 9]
    forward = velocity[:, 0]*current_yaw.cos()+velocity[:, 1]*current_yaw.sin()
    lateral = -velocity[:, 0]*current_yaw.sin()+velocity[:, 1]*current_yaw.cos()
    first = (root_xy[:, 5]-root_xy[:, 0])/(5*spec.control_dt_s)
    second = (root_xy[:, 10]-root_xy[:, 5])/(5*spec.control_dt_s)
    turn = _path_turn(first, second, 5*spec.control_dt_s, spec.direction_min_speed_m_s)
    body_yaw_rate = _wrap_angle(yaw[:, 10]-yaw[:, 9])/spec.control_dt_s
    result = torch.stack((forward, lateral, turn), -1)
    if not torch.isfinite(result).all() or not torch.isfinite(body_yaw_rate).all():
        raise ValueError('Expert descriptor calculation overflowed')
    return result, body_yaw_rate


@torch.no_grad()
def renormalize_bounded(raw_weights, spec=WeightingSpec()):
    """clip(lambda*raw,.5,2) solves mean(weight)=1 without leaving bounds."""
    spec.validate()
    if (not isinstance(raw_weights, torch.Tensor) or raw_weights.ndim != 1
            or raw_weights.device.type != 'cpu' or not len(raw_weights)
            or raw_weights.dtype not in (torch.float32, torch.float64)
            or not torch.isfinite(raw_weights).all()
            or (raw_weights < spec.lower).any() or (raw_weights > spec.upper).any()):
        raise ValueError('Raw weights must be nonempty finite CPU floating vector bounded [.5,2]')
    # Declared [.5,2] raw scores imply finite lambda<=4; no arbitrary subnormal input.
    values = raw_weights.double()
    low = torch.zeros((), dtype=torch.float64)
    high = spec.upper/values.min()
    for _ in range(64):
        middle = (low+high)/2
        mean = (middle*values).clamp(spec.lower, spec.upper).mean()
        if mean < 1:
            low = middle
        else:
            high = middle
    result = (((low+high)/2)*values).clamp(spec.lower, spec.upper).to(raw_weights.dtype)
    if (not torch.isfinite(result).all() or (result < spec.lower).any()
            or (result > spec.upper).any()
            or not torch.allclose(result.double().mean(), torch.ones((), dtype=torch.float64), rtol=0, atol=5e-7)):
        raise ValueError('Bounded mean-one normalization failed')
    return result


@torch.no_grad()
def expert_weights(commands, experts, *, spec=WeightingSpec(), chunk_size=64):
    """Batch-marginal affinity; no sampling, hidden RNG, or context-aware D.

    Affinity_i = mean_c exp(-.5 ||(expert_i-command_c)/scale||²).
    Return bounded mean-one weights and transparent relative-batch ESS.
    """
    spec.validate()
    _float_tensor(commands, 'commands', (3,))
    _float_tensor(experts, 'experts', (3,))
    if commands.dtype != experts.dtype:
        raise ValueError('Command/expert descriptors must share dtype')
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('chunk_size must be positive integer')
    scale = torch.tensor((spec.velocity_scale_forward, spec.velocity_scale_lateral,
                          spec.path_turn_scale_rad_s), dtype=torch.float64)
    affinity = torch.zeros(len(experts), dtype=torch.float64)
    maximum_affinity_per_command = []
    for start in range(0, len(commands), chunk_size):
        difference = (commands[start:start+chunk_size, None].double()-experts[None].double())/scale
        pair_affinity = (-.5*difference.square().sum(-1)).exp()
        affinity += pair_affinity.sum(0)
        maximum_affinity_per_command.append(pair_affinity.amax(1))
    affinity /= len(commands)
    raw = spec.lower+(spec.upper-spec.lower)*affinity
    weights = renormalize_bounded(raw, spec).to(experts.dtype)
    effective_probability = weights.double()/weights.double().sum()
    description = {'mode': 'unmerged_batch_marginal_positive_BCE_only',
                   'command_count': len(commands), 'expert_count': len(experts),
                   'lower': spec.lower, 'upper': spec.upper, 'mean': weights.double().mean().item(),
                   'batch_relative_ESS': (1/effective_probability.square().sum()).item(),
                   'batch_relative_probability': effective_probability.tolist(),
                   'expert_average_affinity_min': affinity.min().item(),
                   'expert_average_affinity_max': affinity.max().item(),
                   'command_support_affinity_threshold_diagnostic_only': .1,
                   'command_fraction_with_any_sampled_expert_affinity_ge_0_1': (torch.cat(maximum_affinity_per_command) >= .1).double().mean().item(),
                   'no_support_in_sampled_batch': bool((torch.cat(maximum_affinity_per_command) < .1).all()),
                   'sampling_changed': False, 'context_conditioned_D': False,
                   'prior_scope': 'Actual already-sampled uniform-clip/legal-frame expert batch; these are loss weights, not full-bank draw probabilities'}
    return weights, description


def classification_loss(positive_logits, negative_logits, *, weights=None):
    """Legacy exact formula if None; only positive per-example BCE weighted.

    Gradient penalty remains UNWEIGHTED in the caller. No reg/replay/AMP reward
    change is hidden here. A selected subset needs explicit bounded mean-one
    renormalization before being passed, never silent p/q compensation.
    """
    if (not isinstance(positive_logits, torch.Tensor) or not isinstance(negative_logits, torch.Tensor)
            or positive_logits.ndim != 1 or negative_logits.ndim != 1
            or not len(positive_logits) or not len(negative_logits)
            or positive_logits.dtype not in (torch.float32, torch.float64)
            or negative_logits.dtype != positive_logits.dtype
            or positive_logits.device.type != 'cpu' or negative_logits.device.type != 'cpu'
            or not torch.isfinite(positive_logits).all() or not torch.isfinite(negative_logits).all()):
        raise ValueError('Need finite nonempty CPU floating logit vectors')
    if weights is None:
        return .5*(F.binary_cross_entropy_with_logits(positive_logits, torch.ones_like(positive_logits))
                   + F.binary_cross_entropy_with_logits(negative_logits, torch.zeros_like(negative_logits)))
    if (not isinstance(weights, torch.Tensor) or weights.shape != positive_logits.shape
            or weights.dtype != positive_logits.dtype or weights.device != positive_logits.device
            or weights.requires_grad or not torch.isfinite(weights).all()
            or (weights < .5).any() or (weights > 2).any()
            or not torch.allclose(weights.double().mean(), torch.ones((), dtype=torch.float64), rtol=0, atol=5e-7)):
        raise ValueError('Explicit detached weights must remain [.5,2], mean1 and match selected expert logits')
    positive = (weights*F.binary_cross_entropy_with_logits(
        positive_logits, torch.ones_like(positive_logits), reduction='none')).mean()
    negative = F.binary_cross_entropy_with_logits(negative_logits, torch.zeros_like(negative_logits))
    return .5*(positive+negative)
