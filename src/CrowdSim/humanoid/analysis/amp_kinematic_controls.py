"""Frozen pose controls for the published PACER 195D observation layout.

Stationary and rigid translation have consistent position/velocity kinematics.
No contact, force balance or dynamic feasibility is asserted. These controls
are never added to the real AMASS expert dataset or used as quality scores.
"""
from __future__ import annotations

from types import SimpleNamespace

CONDITIONS = ("identity", "repeat_current", "stationary_pose", "rigid_translation_pose")
BODY_FIELDS = ("rigid_body_pos", "rigid_body_rot", "rigid_body_vel", "rigid_body_ang_vel", "dof_pos", "dof_vel")


def controls(history):
    import torch

    if (history.ndim != 3 or tuple(history.shape[1:]) != (10, 195) or len(history) == 0
            or history.dtype != torch.float32 or history.device.type != "cpu"
            or not bool(torch.isfinite(history).all())):
        raise ValueError("Require nonempty finite CPU float32 history10/195")
    # Encoder order: root6D, root linear3, angular3, joint6D114,
    # selected joint velocities57, root-heading key-body positions12.
    stationary = history[:, :1].clone()
    stationary[..., 6:12] = 0
    stationary[..., 126:183] = 0
    translating = history[:, :1].clone()
    translating[..., 9:12] = 0
    translating[..., 126:183] = 0
    return {"identity": history, "repeat_current": history[:, :1].expand(-1, 10, -1),
            "stationary_pose": stationary.expand(-1, 10, -1),
            "rigid_translation_pose": translating.expand(-1, 10, -1)}


def encode_pose_sequence(state, *, translating, dt=1 / 30):
    """Independent oracle: translate full body states, then call original encoder."""
    import math
    import torch
    from CrowdSim.humanoid.amp.observations import build_amp_observation

    if type(translating) is not bool or type(dt) not in (float, int) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Require an explicit boolean motion and positive finite control dt")
    batch = len(state.dof_pos)
    current = build_amp_observation(state)
    velocity = state.rigid_body_vel[:, 0] if translating else torch.zeros_like(state.rigid_body_vel[:, 0])
    ages = -torch.arange(10, dtype=torch.float32) * dt  # newest first
    fields = {name: getattr(state, name)[:, None].expand(-1, 10, *getattr(state, name).shape[1:])
              for name in BODY_FIELDS}
    fields["rigid_body_pos"] = fields["rigid_body_pos"] + ages[None, :, None, None] * velocity[:, None, None]
    fields["rigid_body_vel"] = velocity[:, None, None].expand(-1, 10, 24, -1)
    fields["rigid_body_ang_vel"] = torch.zeros_like(fields["rigid_body_ang_vel"])
    fields["dof_vel"] = torch.zeros_like(fields["dof_vel"])
    sequence = SimpleNamespace(**{name: value.reshape(batch * 10, *value.shape[2:]) for name, value in fields.items()})
    return current, build_amp_observation(sequence).reshape(batch, 10, 195)


def score(proxy, history, actions, *, batch_size=256):
    """Use only the exact original AMP input, full discriminator and reward methods."""
    import torch

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Require a positive integer CPU batch size")
    population = controls(history)
    if (tuple(actions.shape) != (len(history), 69) or actions.dtype != torch.float32
            or actions.device.type != "cpu"):
        raise ValueError("Require exactly one CPU float32 action69 per history")
    moments = {name: value.clone() for name, value in proxy.amp_normalizer.state_dict().items()}
    versions = tuple(value._version for value in proxy.model.discriminator.parameters())
    scores, summaries = {}, {}
    with torch.no_grad():
        for name, changed in population.items():
            logits, rewards = [], []
            for start in range(0, len(history), batch_size):
                h, a = changed[start:start + batch_size], actions[start:start + batch_size]
                logits.append(proxy.model.disc_logits(*proxy._amp_inputs(h, a)))
                rewards.append(proxy.amp_reward(h, a))
            logit, reward = torch.cat(logits), torch.cat(rewards)
            if not bool(torch.isfinite(logit).all() and torch.isfinite(reward).all()):
                raise ValueError("Nonfinite original discriminator/reward output")
            delta = logit - scores["identity"]["logits"] if scores else torch.zeros_like(logit)
            scores[name] = {"logits": logit, "rewards": reward}
            summaries[name] = {"rows": len(history), "mean_logit": float(logit.double().mean()),
                "mean_AMP_reward": float(reward.double().mean()), "mean_paired_logit_delta": float(delta.double().mean()),
                "fraction_reward_increased": float((reward > scores["identity"]["rewards"]).double().mean())}
    if (any(not torch.equal(value, moments[name]) for name, value in proxy.amp_normalizer.state_dict().items())
            or versions != tuple(value._version for value in proxy.model.discriminator.parameters())
            or any(value.grad is not None for value in proxy.model.discriminator.parameters())):
        raise ValueError("Frozen inference changed model/RMS or accumulated gradients")
    return summaries, scores
