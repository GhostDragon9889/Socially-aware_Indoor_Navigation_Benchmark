"""Named PACER contact predicate with an explicit local normal-force adaptation.

Only a current postphysics all-partner body net NORMAL force sample is valid.
The recorded peer-body-pair control maxima cannot supply this input. This
primitive does not implement full PACER reset, fetch SDK data or terminate a
runner. Gym friction-channel equivalence has not been established.
"""
from __future__ import annotations

from protomotions.repository import repository_root

import ast
from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch

REPO = repository_root(__file__)
BODY_SOURCE = REPO / "CrowdSim/humanoid/amp/observations.py"
BODY_SHA = "a64dc2eba5c132cf68a248f2324ea20446dd264a2ea8f9f139faae4738f9aeba"
EXCLUDED_BODIES = frozenset(("R_Ankle", "L_Ankle", "R_Toe", "L_Toe"))
THRESHOLD_N = 50.


def _canonical_bodies():
    # Only decode the existing tuple literal, avoiding that module's imports.
    data = BODY_SOURCE.read_bytes()
    if len(data) != 10438 or hashlib.sha256(data).hexdigest() != BODY_SHA:
        raise ValueError("Named SMPL observation source identity changed")
    nodes = [node for node in ast.parse(data).body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == "PACER_BODY_NAMES"
                     for target in node.targets)]
    if len(nodes) != 1:
        raise ValueError("Expected one canonical body-name tuple")
    value = ast.literal_eval(nodes[0].value)
    if type(value) is not tuple or len(value) != 24 or len(set(value)) != 24:
        raise ValueError("Invalid canonical body-name tuple")
    return value


@dataclass(frozen=True)
class NonfootNormalContact:
    resultant_force_w: torch.Tensor
    magnitude_n: torch.Tensor
    above_threshold: torch.Tensor
    progress_eligible: torch.Tensor
    contact_flag: torch.Tensor


def nonfoot_net_normal_contact(net_normal_force_w, progress, *, body_names):
    """Exclude named feet, sum body vectors, norm>50N and progress>1.

    ``net_normal_force_w`` is finite float32[N,24,3] at one actual physics
    timestamp, world frame, all contact partners including the floor. It is
    not a filtered pair-force matrix, impulse, four-substep maximum or sum of
    force magnitudes. The caller must establish actual SDK source/timestamp.
    ``progress`` is nonnegative int64[N] on the same CPU/indexed CUDA device.
    No inputs, clocks, flags, history, RNG or physics are mutated.
    """
    names = tuple(body_names)
    canonical = _canonical_bodies()
    if (len(names) != 24 or any(type(name) is not str for name in names)
            or len(set(names)) != 24 or set(names) != set(canonical)):
        raise ValueError("Contact body names must be an exact named SMPL bijection")
    force = net_normal_force_w
    if (not isinstance(force, torch.Tensor) or force.ndim != 3
            or force.shape[0] < 1 or tuple(force.shape[1:]) != (24, 3)
            or force.dtype != torch.float32 or force.device.type not in ("cpu", "cuda")
            or (force.device.type == "cuda" and force.device.index is None)
            or not bool(torch.isfinite(force).all())):
        raise ValueError("A finite float32[N,24,3] current net normal force tensor is required")
    if (not isinstance(progress, torch.Tensor) or progress.shape != (force.shape[0],)
            or progress.dtype != torch.int64 or progress.device != force.device
            or bool((progress < 0).any())):
        raise ValueError("Progress requires nonnegative int64[N] on the force device")
    indices = torch.tensor([index for index, name in enumerate(names)
                            if name not in EXCLUDED_BODIES], dtype=torch.int64, device=force.device)
    resultant = force.index_select(1, indices).sum(1)
    magnitude = torch.linalg.vector_norm(resultant, dim=-1)
    if not bool(torch.isfinite(resultant).all()) or not bool(torch.isfinite(magnitude).all()):
        raise ValueError("Net normal force aggregation overflowed")
    above = magnitude > THRESHOLD_N
    eligible = progress > 1
    return NonfootNormalContact(resultant, magnitude, above, eligible, above & eligible)
