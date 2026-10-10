"""Frozen CPU discriminator interventions on actual archived training inputs.

All ten frames still enter the original D2019. Per-frame RMS shares moments;
it does not turn the flattened discriminator into a frame-independent score.
These counterfactual inputs diagnose sensitivity, not physical motion quality.
"""
from __future__ import annotations

from types import MethodType, SimpleNamespace

ORDERS = {
    "identity": tuple(range(10)),
    "reverse_past_keep_current": (0, 9, 8, 7, 6, 5, 4, 3, 2, 1),
    "permute_past_keep_current": (0, 3, 1, 9, 4, 7, 2, 8, 6, 5),
}
CONDITIONS = (*ORDERS, "repeat_current", "cyclic_action_mismatch", "zero_actions")


def interventions(history, actions):
    import torch

    if (history.ndim != 3 or tuple(history.shape[1:]) != (10, 195)
            or tuple(actions.shape) != (len(history), 69) or len(history) < 2
            or history.dtype != torch.float32 or actions.dtype != torch.float32
            or history.device.type != "cpu" or actions.device.type != "cpu"):
        raise ValueError("Require complete materialized CPU float32 history10/195 and action69 populations")
    result = {name: (history[:, order], actions) for name, order in ORDERS.items()}
    result["repeat_current"] = (history[:, :1].expand(-1, 10, -1), actions)
    result["cyclic_action_mismatch"] = (history, actions.roll(1, dims=0))
    result["zero_actions"] = (history, torch.zeros_like(actions))
    return result


def load_discriminator(state):
    """Construct only the exact saved D and RMS, using original numerical code."""
    import torch
    from CrowdSim.humanoid.training import trainer as original

    config = SimpleNamespace(**state["config"])
    if ((config.history_frames, config.amp_frame_dim, config.discriminator_input,
         config.amp_normalization, tuple(config.discriminator_layers)) != (10, 195, "state_action", "per_frame", (1024, 512))
            or config.normalize_observations is not True or config.normalization_clip != 5.):
        raise ValueError("Require the actual D2019/shared195 RMS checkpoint family")
    discriminator = original._mlp(2019, config.discriminator_layers, 1).cpu()
    weights = {key.removeprefix("discriminator."): value for key, value in state["model"].items()
               if key.startswith("discriminator.")}
    if any(value.device.type != "cpu" or value.dtype != torch.float32
           or not bool(torch.isfinite(value).all()) for value in weights.values()):
        raise ValueError("Saved discriminator weights must be finite CPU float32")
    discriminator.load_state_dict(weights, strict=True)
    discriminator.eval().requires_grad_(False)
    normalizer = original.RunningNormalizer(195, config.normalization_clip).cpu()
    normalizer.load_state_dict(state["normalizers"]["amp_normalizer"], strict=True)
    if (not all(bool(torch.isfinite(value).all()) for value in normalizer.state_dict().values())
            or not bool((normalizer.variance >= 0).all())):
        raise ValueError("Saved AMP RMS must be finite with nonnegative variance")
    model = SimpleNamespace(discriminator_input=config.discriminator_input, discriminator=discriminator)
    model.disc_logits = MethodType(original.PaperAMPNetwork.disc_logits, model)
    proxy = SimpleNamespace(config=config, provenance=state["provenance"], device=torch.device("cpu"),
                            action_mask=torch.tensor(config.action_mask, dtype=torch.float32),
                            amp_normalizer=normalizer, model=model)
    proxy._amp_inputs = MethodType(original.PaperAMPTrainer._amp_inputs, proxy)
    proxy.amp_reward = MethodType(original.PaperAMPTrainer.amp_reward, proxy)
    return proxy


def measure(proxy, history, actions, *, batch_size=256):
    import torch

    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Require a positive integer CPU batch size")
    population = interventions(history, actions)
    moments = {key: value.clone() for key, value in proxy.amp_normalizer.state_dict().items()}
    parameter_versions = tuple(value._version for value in proxy.model.discriminator.parameters())
    scores, summaries = {}, {}
    with torch.no_grad():
        normalized, _ = proxy._amp_inputs(history, actions)
        for name, order in ORDERS.items():
            transformed, _ = proxy._amp_inputs(*population[name])
            if not torch.equal(transformed, normalized[:, order]):
                raise ValueError("Shared per-frame RMS must commute exactly with history permutation")
        for name, (changed_history, changed_actions) in population.items():
            logits, rewards = [], []
            for start in range(0, len(history), batch_size):
                h, a = changed_history[start:start + batch_size], changed_actions[start:start + batch_size]
                inputs = proxy._amp_inputs(h, a)
                logits.append(proxy.model.disc_logits(*inputs))
                # Original reward implementation, not a copied transform.
                rewards.append(proxy.amp_reward(h, a))
            logit, reward = torch.cat(logits), torch.cat(rewards)
            if not bool(torch.isfinite(logit).all() and torch.isfinite(reward).all()):
                raise ValueError("Nonfinite frozen discriminator output")
            scores[name] = {"logits": logit, "rewards": reward}
            delta = logit - scores["identity"]["logits"]
            summaries[name] = {"rows": len(history), "mean_logit": float(logit.double().mean()),
                "mean_expert_probability": float(logit.sigmoid().double().mean()),
                "mean_AMP_reward": float(reward.double().mean()),
                "mean_paired_logit_delta": float(delta.double().mean()),
                "mean_absolute_paired_logit_delta": float(delta.double().abs().mean()),
                "fraction_logit_above_zero": float((logit > 0).double().mean())}
    if (any(not torch.equal(value, moments[key]) for key, value in proxy.amp_normalizer.state_dict().items())
            or parameter_versions != tuple(value._version for value in proxy.model.discriminator.parameters())
            or any(value.grad is not None for value in proxy.model.discriminator.parameters())):
        raise ValueError("Diagnostic modified frozen model/RMS or accumulated gradients")
    return summaries, scores
