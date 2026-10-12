"""Current all-partner net NORMAL forces; no view creation, stepping or reset.

Prototype for the eight per-actor views from the prepared two-group smoke.
Grouped four-slot views require a separate adaptation. Import is stdlib-only.
"""
from __future__ import annotations

import ast
import hashlib
import math
from pathlib import Path
import stat

from protomotions.repository import repository_root
ROOT = repository_root(__file__)
BRIDGE_PATH = ROOT / "resources/runtime_sources/style_improvement_20261004/group_articulation_tensor_bridge_staging_v1/tensor_bridge.py"
BRIDGE_SHA = "2f5dc721f739d26ace02cc0831864523e605587f7e960f0fd32b82326ec8c9ec"
NAMES_PATH = ROOT / "CrowdSim/humanoid/amp/observations.py"
NAMES_SHA = "a64dc2eba5c132cf68a248f2324ea20446dd264a2ea8f9f139faae4738f9aeba"


def _source(path, size, sha, *, readonly=False):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or readonly and info.st_mode & 0o222:
        raise ValueError("Invalid source file: " + str(path))
    data = path.read_bytes()
    if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
        raise ValueError("Source identity changed: " + str(path))
    return data


def _canonical_names():
    tree = ast.parse(_source(NAMES_PATH, 10438, NAMES_SHA))
    nodes = [node for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == "PACER_BODY_NAMES" for target in node.targets)]
    if len(nodes) != 1:
        raise ValueError("Expected one named SMPL tuple")
    names = ast.literal_eval(nodes[0].value)
    if type(names) is not tuple or len(names) != 24 or len(set(names)) != 24:
        raise ValueError("Invalid COMMON SMPL names")
    return names


def sample_current_net_normal(views, layouts, body_names_by_view, group_actor_rows, *,
                              bridge, device, physics_dt, physics_step, control_step):
    """Return new COMMON actor forces[N,24,3] and declared source metadata.

    Each existing view has 24 ordered sensors for one actor. ``layout`` uses
    the smoke's ``actor`` and ``sources`` keys. Names follow that sensor order.
    All layouts are checked before any getter. Getter failures can leave
    earlier reads completed; this is not a transaction and performs no writes.
    Pass actual physics dt (currently1/120), NOT control dt1/30. Caller stamps
    are merely retained; this function does not read or verify a physics clock.
    """
    import torch

    target = torch.device(device)
    if target.type == "cpu":
        target = torch.device("cpu")
    if target.type not in ("cpu", "cuda") or target.type == "cuda" and target.index is None:
        raise ValueError("Explicit CPU/indexed CUDA device required")
    if type(physics_dt) not in (int, float) or not math.isfinite(physics_dt) or physics_dt <= 0:
        raise ValueError("Explicit finite positive physics dt required")
    if any(type(value) is not int or value < 0 for value in (physics_step, control_step)):
        raise ValueError("Caller physics/control stamps must be nonnegative integers")
    rows = group_actor_rows
    if (not isinstance(rows, torch.Tensor) or rows.ndim != 2 or rows.shape[1] != 4 or not len(rows)
            or rows.dtype != torch.int64 or rows.device != target):
        raise ValueError("Complete group_actor_rows requires int64[G,4] on the force device")
    count = rows.numel()
    if bool(((rows < 0) | (rows >= count)).any()) or torch.unique(rows).numel() != count:
        raise ValueError("Groups must partition canonical actor rows0..N-1")
    if any(not isinstance(value, (list, tuple)) or len(value) != count for value in (views, layouts, body_names_by_view)):
        raise ValueError("Exactly one existing view/layout/body-name list per actor is required")
    if (Path(getattr(bridge, "__file__", "")).resolve() != BRIDGE_PATH
            or not callable(getattr(bridge, "assets_to_canonical", None))):
        raise ValueError("The sealed actor-row bridge is required")
    _source(BRIDGE_PATH, 6035, BRIDGE_SHA, readonly=True)
    canonical = _canonical_names()
    declared, all_paths, actors = [], [], []
    for view, layout, names in zip(views, layouts, body_names_by_view, strict=True):
        if type(layout) is not dict or type(layout.get("actor")) is not int or not 0 <= layout["actor"] < count:
            raise ValueError("Each layout must declare one canonical actor row")
        names = tuple(names)
        paths = list(layout.get("sources", ()))
        if (len(names) != 24 or any(type(name) is not str for name in names)
                or len(set(names)) != 24 or set(names) != set(canonical)):
            raise ValueError("Sensor names must be an exact COMMON24 bijection")
        if (len(paths) != 24 or len(set(paths)) != 24
                or any(type(path) is not str or not path.startswith("/") or any(char in path for char in "*?[]") for path in paths)
                or view.sensor_count != 24 or list(view.sensor_paths) != paths
                or not callable(getattr(view, "get_net_contact_forces", None))):
            raise ValueError("Actual ordered sensor paths/count must equal the exact declared24")
        declared.append((layout["actor"], paths, names, [names.index(name) for name in canonical]))
        actors.append(layout["actor"])
        all_paths.extend(paths)
    if sorted(actors) != list(range(count)) or len(set(all_paths)) != 24 * count:
        raise ValueError("Every actor/body sensor must be distinct and covered once")
    by_actor = [None] * count
    for view, (actor, _, _, order) in zip(views, declared, strict=True):
        payload = view.get_net_contact_forces(dt=physics_dt)
        if not isinstance(payload, torch.Tensor):
            import warp as wp
            if not isinstance(payload, wp.array):
                raise ValueError("Only the configured native Warp or a pure-Torch mock payload is supported")
            payload = wp.to_torch(payload)
        force = payload.detach().clone()
        if force.shape != (24, 3) or force.dtype != torch.float32 or force.device != target or not bool(torch.isfinite(force).all()):
            raise ValueError("Current native net force must be finite float32[24,3] on device")
        by_actor[actor] = force.index_select(0, torch.tensor(order, dtype=torch.int64, device=target))
    row_lists = rows.tolist()
    assets = tuple(torch.stack([by_actor[group[slot]] for group in row_lists]) for slot in range(4))
    result = bridge.assets_to_canonical(assets, rows, device=target)
    metadata = {"force_quantity": "net NORMAL vector", "frame": "world", "units": "N",
                "partner_scope": "all contact partners including floor; independent of view filters",
                "getter": "get_net_contact_forces", "physics_dt_s": physics_dt,
                "sample_phase": "caller-declared current control end; no temporal max",
                "caller_physics_step": physics_step, "caller_control_step": control_step,
                "clock_independently_verified": False, "actors": count,
                "body_names_in_output_order": list(canonical),
                "sensor_layouts": [{"actor": actor, "sources": paths, "body_names": list(names)} for actor, paths, names, _ in declared],
                "bridge_sha256": BRIDGE_SHA, "canonical_names_source_sha256": NAMES_SHA,
                "Gym_friction_equivalence_established": False}
    return result, metadata
