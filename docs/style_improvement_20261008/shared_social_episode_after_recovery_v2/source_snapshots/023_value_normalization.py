"""Opt-in scalar Pop-Art output preservation for a separate training experiment.

Proposition 1, van Hasselt et al. (2016), https://arxiv.org/abs/1602.07714.
Not connected to the sealed baseline trainer. Call after updating value RMS and
before critic optimization, using a detached copy of the previous RMS buffers.
Only the scalar critic output layer changes; Adam state is left untouched.
The arithmetic preserves the unnormalized function, not optimizer trajectories.
"""
from __future__ import annotations


def preserve_unnormalized_critic(head, previous_rms, current_rms):
    """Apply scalar Pop-Art with the existing RMS epsilon1e-5; no forward call.

Validate all input/output tensors before either parameter write. Population
moment estimation, rewards, GAE, actor/encoder and optimizer are caller-owned.
Input RMS buffers are never modified. Float32 cast error must be independently
checked on actual critic features before integrating a new training recipe.
"""
    import torch

    if (type(head) is not torch.nn.Linear or head.out_features != 1 or head.bias is None
            or head.weight.dtype != torch.float32 or head.bias.dtype != torch.float32
            or head.weight.device != head.bias.device):
        raise TypeError("Require the exact scalar float32 critic Linear output with bias")
    stats = []
    for state in (previous_rms, current_rms):
        if type(state) is not dict or set(state) != {"mean", "variance", "count"}:
            raise ValueError("Require complete scalar population RMS buffers")
        for name, shape in (("mean", (1,)), ("variance", (1,)), ("count", ())):
            value = state[name]
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.float64
                    or tuple(value.shape) != shape or value.device != head.weight.device
                    or not bool(torch.isfinite(value).all())):
                raise ValueError("Invalid scalar RMS tensor: " + name)
        if bool((state["variance"] < 0).any()) or float(state["count"]) < 0:
            raise ValueError("RMS variance/count must be nonnegative")
        stats.append((state["mean"].detach(), (state["variance"].detach() + 1e-5).sqrt()))
    if float(current_rms["count"]) <= float(previous_rms["count"]):
        raise ValueError("Require a real positive new batch exposure before Pop-Art")
    old_mean, old_scale = stats[0]
    new_mean, new_scale = stats[1]
    with torch.no_grad():
        ratio = old_scale / new_scale
        weight = (head.weight.double() * ratio[:, None]).to(head.weight.dtype)
        bias = ((old_scale * head.bias.double() + old_mean - new_mean) / new_scale).to(head.bias.dtype)
        if not bool(torch.isfinite(weight).all() & torch.isfinite(bias).all()):
            raise FloatingPointError("Pop-Art transform would produce nonfinite parameters")
        head.weight.copy_(weight)
        head.bias.copy_(bias)
    return {"strategy": "scalar_popart_output_preservation_v1", "epsilon": 1e-5,
            "old_count": float(previous_rms["count"]), "new_count": float(current_rms["count"]),
            "old_mean": float(old_mean[0]), "new_mean": float(new_mean[0]),
            "old_scale": float(old_scale[0]), "new_scale": float(new_scale[0]),
            "weight_ratio": float(ratio[0]), "optimizer_state_modified": False,
            "actor_encoder_modified": False, "critic_forward_performed": False,
            "physical_quality_verified": False}
