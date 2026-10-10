"""Pure scheduling for bounded asynchronous or synchronized replanning.

The caller owns the monotonic absolute control counter and serializes it with
plan_age. Increment it exactly once after a physical control step, regardless
of rollout/update boundaries. A newly reset actor must have age=replan_steps
before its first subsequent action, so it is planned immediately off-grid too.
No clock state, RNG, simulation calls, or model weights are held in this module.
"""

from numbers import Integral

import torch


def due_agents(plan_age, control_step, replan_steps, synchronized=False):
    """Return sorted int64 actor IDs on the age Tensor's original device.

    Default: preserve the original per-actor ``age >= interval`` schedule.
    Synchronized: additionally plan every actor on each global interval tick.
    Between ticks, overdue/reset actors still plan immediately. This keeps the
    maximum age at the existing interval; the next global tick aligns actors
    that reset at arbitrary times. Inputs are never mutated.

    Ages are nonnegative integer control-step counts owned by the runner. Only
    metadata is validated here: no device-to-host value checks are introduced
    into the scheduling hot path. Global tick0 plans all initial actors.
    """
    if not isinstance(plan_age, torch.Tensor) or plan_age.ndim != 1 or plan_age.dtype not in (
        torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
    ):
        raise TypeError("plan_age must be a one-dimensional integer Tensor")
    if isinstance(control_step, bool) or not isinstance(control_step, Integral) or control_step < 0:
        raise ValueError("control_step must be a nonnegative absolute integer counter")
    if isinstance(replan_steps, bool) or not isinstance(replan_steps, Integral) or replan_steps < 1:
        raise ValueError("replan_steps must be a positive integer")
    if not isinstance(synchronized, bool):
        raise TypeError("synchronized must be a boolean")
    if synchronized and control_step % replan_steps == 0:
        return torch.arange(plan_age.numel(), device=plan_age.device, dtype=torch.long)
    return (plan_age >= replan_steps).nonzero(as_tuple=False).flatten()
