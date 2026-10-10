"""Optional exact feet-to-floor normal forces, separate from obstacle contacts.

The caller creates the physics world and resolves the live USD rigid-body and
enabled floor-collider paths. This module never starts Kit or changes that world.
Native views are injectable; CPU mocks prove contracts, not physical accuracy.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch

FOOT_NAMES = ("L_Ankle", "L_Toe", "R_Ankle", "R_Toe")


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _paths(values, name):
    paths = tuple(values)
    if not paths or any(not isinstance(p, str) or not p.startswith("/") or p == "/"
                        or any(char in p for char in "*?[]") or "//" in p or p.endswith("/")
                        or any(part in (".", "..") for part in p.split("/")) for p in paths):
        raise ValueError(f"{name} requires nonempty exact absolute prim paths")
    if len(set(paths)) != len(paths):
        raise ValueError(f"{name} contains duplicate paths")
    return paths


def resolve_floor_collision_paths(stage, floor_roots):
    """Resolve enabled CollisionAPI descendants of explicitly declared floors.

    The caller is responsible for identifying floor roots, rather than passing
    all static objects. This lazy USD-only helper does not import Kit.
    """
    from pxr import Usd, UsdPhysics

    result = []
    for path in _paths(floor_roots, "floor_roots"):
        root = stage.GetPrimAtPath(path)
        if not root:
            raise ValueError(f"Floor root does not exist: {path}")
        colliders = [str(prim.GetPath()) for prim in Usd.PrimRange(root)
                     if prim.HasAPI(UsdPhysics.CollisionAPI)
                     and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False]
        if not colliders:
            raise ValueError(f"Floor root has no enabled CollisionAPI descendant: {path}")
        result.extend(colliders)
    return _paths(result, "floor colliders")


def body_name_permutation(source_names, target_names):
    """Require a named bijection before moving native body channels to common."""
    source, target = tuple(source_names), tuple(target_names)
    if not source or any(not isinstance(name, str) or not name for name in source + target):
        raise ValueError("Body names must be nonempty strings")
    if len(set(source)) != len(source) or len(set(target)) != len(target) or set(source) != set(target):
        raise ValueError("Body channels require unique names and a complete bijection")
    return tuple(source.index(name) for name in target)


def _float_tensor(value, name, last_dimension=None):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating Torch tensor")
    if not value.numel() or (last_dimension is not None and (value.ndim == 0 or value.shape[-1] != last_dimension)):
        raise ValueError(f"{name} has an invalid shape")
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Nonfinite {name}")
    return value


def body_link_world_com(link_pos_w, link_quat_xyzw, local_com_pos):
    """Compose link origins and link-local COM offsets; all use the same order.

    ``get_coms()[..., :3]`` is local, not world. Principal-axis COM rotation is
    unnecessary for COM translation. Never label body origins as true COM.
    """
    pos = _float_tensor(link_pos_w, "link positions", 3)
    quat = _float_tensor(link_quat_xyzw, "link XYZW rotations", 4)
    offset = _float_tensor(local_com_pos, "local COM positions", 3)
    if pos.ndim != 3 or quat.shape != (*pos.shape[:-1], 4) or offset.shape != pos.shape:
        raise ValueError("COM composition requires [actors,bodies,3/4] matched channels")
    if any(value.device != pos.device or value.dtype != pos.dtype for value in (quat, offset)):
        raise ValueError("COM channels must have the same device and floating dtype")
    if not torch.allclose(torch.linalg.vector_norm(quat, dim=-1), torch.ones_like(quat[..., 0]), atol=1e-3, rtol=0):
        raise ValueError("COM composition requires unit XYZW rotations")
    vector = quat[..., :3]
    twice_cross = 2 * torch.linalg.cross(vector, offset, dim=-1)
    result = pos + offset + quat[..., 3:] * twice_cross + torch.linalg.cross(vector, twice_cross, dim=-1)
    return _float_tensor(result, "composed world COM", 3)


def mass_weighted_world_com(body_com_w, masses_kg):
    com = _float_tensor(body_com_w, "world body COM", 3)
    masses = _float_tensor(masses_kg, "body masses")
    if com.ndim != 3 or masses.shape != com.shape[:-1]:
        raise ValueError("Mass-weighted COM requires [actors,bodies,3] and [actors,bodies]")
    if masses.device != com.device or masses.dtype != com.dtype or not (masses > 0).all():
        raise ValueError("Masses must be strictly positive and match COM device/dtype/order")
    weighted = (com * masses.unsqueeze(-1)).sum(-2) / masses.sum(-1, keepdim=True)
    return _float_tensor(weighted, "mass-weighted world COM", 3)


def body_origin_tangential_speed(vel_w, ground_normal_w):
    """Body-origin speed parallel to a declared plane, not contact-point slip."""
    velocity = _float_tensor(vel_w, "body-origin world velocities", 3)
    normal = _float_tensor(ground_normal_w, "ground plane normal", 3)
    if normal.shape != (3,) or normal.device != velocity.device or normal.dtype != velocity.dtype:
        raise ValueError("Ground normal must be [3] with the velocity device/dtype")
    if not torch.allclose(torch.linalg.vector_norm(normal), normal.new_tensor(1.), atol=1e-6, rtol=0):
        raise ValueError("Ground normal must be unit length")
    tangential = velocity - (velocity * normal).sum(-1, keepdim=True) * normal
    return _float_tensor(torch.linalg.vector_norm(tangential, dim=-1), "body-origin tangential speed")


def _native_torch(value):
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "torch"):
        return value.torch
    import warp as wp

    return wp.to_torch(value)


def _filter_rows(raw, sensors, filters):
    """Accept exact shared columns, explicit per-sensor rows, or flattened rows."""
    items = list(raw)
    if len(items) == sensors and all(not isinstance(item, str) for item in items):
        rows = [tuple(row) for row in items]
    elif all(isinstance(item, str) for item in items) and len(items) == filters:
        rows = [tuple(items)] * sensors
    elif all(isinstance(item, str) for item in items) and len(items) == sensors * filters:
        rows = [tuple(items[i * filters:(i + 1) * filters]) for i in range(sensors)]
    else:
        raise RuntimeError("Native floor filter_paths layout is not an exact resolvable matrix")
    if any(len(row) != filters for row in rows):
        raise RuntimeError("Native floor filter_paths column count changed")
    return rows


class FootSupportMonitor:
    """Accumulate exactly four completed native substeps, without reset mixing.

    Force matrices contain sums of *normal force vectors* per sensor/filter.
    Load here sums their vector magnitudes over foot bodies. It is not the
    scalar sum of individual contact-patch forces on nonparallel surfaces.
    """

    @classmethod
    def from_factory(cls, factory, actor_foot_paths, floor_collider_paths, *,
                     contact_reporting_enabled, max_contact_data_count=512, **kwargs):
        sources, floors = cls._layout(actor_foot_paths, floor_collider_paths)
        _integer(max_contact_data_count, "max_contact_data_count", minimum=1)
        if contact_reporting_enabled is not True:
            raise ValueError("Explicitly enabled contact reporting is required")
        creator = getattr(factory, "create_rigid_contact_view", factory)
        if not callable(creator):
            raise ValueError("Native rigid contact view factory is unavailable")
        view = creator(list(sources), filter_patterns=[list(floors) for _ in sources],
                       max_contact_data_count=max_contact_data_count)
        return cls(view, actor_foot_paths, floors, contact_reporting_enabled=True, **kwargs)

    @staticmethod
    def _layout(actor_foot_paths, floor_collider_paths):
        actors = list(actor_foot_paths)
        if not actors or any(not isinstance(actor, Mapping) or not set(FOOT_NAMES).issubset(actor) for actor in actors):
            raise ValueError(f"Every actor requires semantic rigid paths for {FOOT_NAMES}")
        sources = _paths([actor[name] for actor in actors for name in FOOT_NAMES], "feet")
        if any(actor[name].rsplit("/", 1)[-1] != name for actor in actors for name in FOOT_NAMES):
            raise ValueError("Foot rigid-prim leaf names must match the declared SMPL semantics")
        floors = _paths(floor_collider_paths, "enabled floor colliders")
        if set(sources) & set(floors):
            raise ValueError("Floor filters and foot sensors must be disjoint")
        return sources, floors

    def __init__(self, view, actor_foot_paths, floor_collider_paths, *, contact_reporting_enabled,
                 physics_dt_s=1 / 120, substeps_per_control=4, threshold_n=1., to_torch=None):
        if contact_reporting_enabled is not True:
            raise ValueError("Explicitly enabled contact reporting is required")
        self.sources, self.floors = self._layout(actor_foot_paths, floor_collider_paths)
        self.num_actors = len(self.sources) // 4
        self.physics_dt_s = _positive(physics_dt_s, "physics_dt_s")
        self.substeps = _integer(substeps_per_control, "substeps_per_control", minimum=1)
        self.threshold_n = _positive(threshold_n, "threshold_n")
        if to_torch is not None and not callable(to_torch):
            raise ValueError("to_torch must be callable")
        self.to_torch = to_torch or _native_torch
        self.view = view
        if not callable(getattr(view, "get_contact_force_matrix", None)):
            raise RuntimeError("Native floor force-matrix API is unavailable")
        if getattr(view, "sensor_count", None) != len(self.sources) or getattr(view, "filter_count", None) != len(self.floors):
            raise RuntimeError("Native floor sensor/filter counts do not match exact paths")
        sensor_paths = tuple(getattr(view, "sensor_paths", ()))
        if len(sensor_paths) != len(self.sources) or set(sensor_paths) != set(self.sources):
            raise RuntimeError("Native floor sensor paths are not a complete exact bijection")
        if len(set(sensor_paths)) != len(sensor_paths):
            raise RuntimeError("Native floor sensor paths contain duplicates")
        rows = _filter_rows(getattr(view, "filter_paths", ()), len(self.sources), len(self.floors))
        if any(len(set(row)) != len(row) or set(row) != set(self.floors) for row in rows):
            raise RuntimeError("Native filters include missing floor paths or non-floor partners")
        self._sensor_indices = tuple(sensor_paths.index(path) for path in self.sources)
        self._filter_indices = tuple(tuple(rows[row].index(path) for path in self.floors)
                                     for row in self._sensor_indices)
        self.manifest = {
            "source": "PhysX RigidContactView.get_contact_force_matrix(dt=physics_dt_s)",
            "units": "N", "foot_names": FOOT_NAMES, "sensor_paths": self.sources,
            "floor_collider_paths": self.floors, "native_sensor_paths": sensor_paths,
            "native_filter_paths": rows, "physics_dt_s": self.physics_dt_s,
            "substeps_per_control": self.substeps, "threshold_n": self.threshold_n,
            "force_semantics": "filtered normal-force vector sums; friction excluded",
            "side_load_semantics": "sum foot-vector magnitudes for ankle and toe; not scalar patch-force sum",
            "contact_semantics": "strict > threshold per foot; side contact is any ankle/toe contact",
            "slip_semantics": "optional body-origin tangential-speed proxy on declared plane; not contact-point slip",
            "excluded": ["obstacle and peer filters", "self articulation", "friction"],
            "floor_geometry_role": "explicit caller-selected enabled floor colliders; no automatic all-static filter",
            "author_metric_equivalence": False, "native_physical_probe_passed": False,
        }
        self._last_control = None
        self._active = False
        self._forces = []
        self._velocities = []
        self._positions = []
        self._poisoned = False
        self.reset_count = 0

    def reset(self):
        """Explicitly discard an interrupted/reset control; never publish zeros."""
        self._active = False
        self._forces.clear()
        self._velocities.clear()
        self._positions.clear()
        self._poisoned = False
        self._last_control = None
        self.reset_count += 1

    def begin_control_step(self, *, episode_ids, control_step, physics_step_before):
        step = _integer(control_step, "control_step")
        physics = _integer(physics_step_before, "physics_step_before")
        episodes = tuple(episode_ids)
        if len(episodes) != self.num_actors or any(type(value) is not int or value < 0 for value in episodes):
            raise ValueError("episode_ids requires one nonnegative integer per actor")
        if self._poisoned or (self._active and len(self._forces) != self.substeps):
            raise RuntimeError("Interrupted floor measurement requires explicit reset()")
        if self._last_control is not None and step != self._last_control + 1:
            raise RuntimeError("Floor measurements require consecutive control steps or explicit reset()")
        if self._active and (physics != self._last_physics or episodes != self._episode_ids):
            raise RuntimeError("Reset or missing physics steps between controls requires explicit reset()")
        self._episode_ids, self._control_step = episodes, step
        self._last_control = step
        self._last_physics = physics
        self._forces, self._velocities, self._positions = [], [], []
        self._active = True

    def sample_completed_substep(self, dt, *, episode_ids, physics_step,
                                 foot_body_origin_vel_w=None, foot_body_origin_pos_w=None):
        """Call after physics+scene.update; clone buffers before their next reuse."""
        if not self._active or self._poisoned or len(self._forces) >= self.substeps:
            raise RuntimeError("No active incomplete floor measurement")
        try:
            duration = _positive(dt, "physics measurement dt")
            if not math.isclose(duration, self.physics_dt_s, rel_tol=0, abs_tol=1e-12):
                raise ValueError("Force impulse conversion must use the declared physics dt, not control dt")
            if tuple(episode_ids) != self._episode_ids:
                raise RuntimeError("An actor reset inside a control; explicit reset() is required")
            if _integer(physics_step, "physics_step") != self._last_physics + 1:
                raise RuntimeError("Missing/repeated physics substep in floor measurement")
            force = _float_tensor(self.to_torch(self.view.get_contact_force_matrix(dt=duration)), "floor normal forces", 3)
            expected = (len(self.sources), len(self.floors), 3)
            if tuple(force.shape) != expected:
                raise RuntimeError(f"Floor force shape changed: {tuple(force.shape)}, expected {expected}")
            sensor = torch.tensor(self._sensor_indices, device=force.device)
            columns = torch.tensor(self._filter_indices, device=force.device)
            ordered = force.index_select(0, sensor).gather(1, columns.unsqueeze(-1).expand(-1, -1, 3))
            ordered = ordered.reshape(self.num_actors, 4, len(self.floors), 3).detach().clone()
            snapshots = []
            for value, name, previous in ((foot_body_origin_vel_w, "foot body-origin velocities", self._velocities),
                                          (foot_body_origin_pos_w, "foot body-origin positions", self._positions)):
                if value is None:
                    snapshot = None
                else:
                    channel = _float_tensor(value, name, 3)
                    if channel.shape != (self.num_actors, 4, 3) or channel.device != force.device or channel.dtype != force.dtype:
                        raise ValueError(f"{name} must match force device/dtype and [actors,4,3] semantic foot order")
                    snapshot = channel.detach().clone()
                if previous and (snapshot is None) != (previous[0] is None):
                    raise ValueError(f"{name} availability changed inside a control")
                snapshots.append(snapshot)
            if self._forces and (ordered.device != self._forces[0].device or ordered.dtype != self._forces[0].dtype):
                raise ValueError("Floor force device/dtype changed inside a control")
            self._forces.append(ordered)
            self._velocities.append(snapshots[0])
            self._positions.append(snapshots[1])
            self._last_physics = physics_step
        except Exception:
            self._poisoned = True
            raise

    def measurements(self, *, ground_normal_w=None):
        if not self._active or self._poisoned or len(self._forces) != self.substeps:
            raise RuntimeError("Floor support requires all declared completed substeps; partial data cannot be published")
        force = torch.stack(self._forces)  # [substep,actor,foot,floor,xyz]
        foot_vector = force.sum(-2)
        foot_load = torch.linalg.vector_norm(foot_vector, dim=-1)
        side_load = foot_load.reshape(self.substeps, self.num_actors, 2, 2).sum(-1)
        for values, name in ((foot_vector, "summed foot force vectors"), (foot_load, "foot loads"),
                             (side_load, "side loads")):
            _float_tensor(values, name)
        foot_contact = foot_load > self.threshold_n
        side_contact = foot_contact.reshape(self.substeps, self.num_actors, 2, 2).any(-1)
        self._last_control = self._control_step
        result = {
            "control_step": self._control_step, "episode_ids": self._episode_ids,
            "physics_step_end": self._last_physics, "substep_valid_count": self.substeps,
            "monitor_reset_count": self.reset_count,
            "actor_substep_valid_count": torch.full((self.num_actors,), self.substeps, dtype=torch.int64, device=force.device),
            "foot_floor_normal_force_w": force.clone(),
            "foot_normal_force_vector_mean_w": foot_vector.mean(0),
            "foot_normal_load_mean_n": foot_load.mean(0), "foot_normal_load_peak_n": foot_load.amax(0),
            "side_normal_load_substeps_n": side_load, "side_normal_load_mean_n": side_load.mean(0),
            "side_normal_load_peak_n": side_load.amax(0),
            "foot_contact_substeps": foot_contact, "side_contact_substeps": side_contact,
            "side_contact_any": side_contact.any(0), "side_contact_all": side_contact.all(0),
            "double_support_fraction": side_contact.all(-1).float().mean(0),
            "no_support_fraction": (~side_contact.any(-1)).float().mean(0),
            "foot_body_origin_pos_w": None if self._positions[0] is None else torch.stack(self._positions),
            "foot_body_origin_vel_w": None if self._velocities[0] is None else torch.stack(self._velocities),
            "body_origin_tangential_speed_proxy_mps": None,
            "supported_body_origin_tangential_speed_proxy_mean_mps": None,
            "supported_speed_proxy_valid": None,
        }
        if self._velocities[0] is not None:
            if ground_normal_w is None:
                raise ValueError("A declared world ground-plane normal is required for the tangential-speed proxy")
            speed = body_origin_tangential_speed(result["foot_body_origin_vel_w"], ground_normal_w)
            count = foot_contact.sum(0)
            result["body_origin_tangential_speed_proxy_mps"] = speed
            result["supported_body_origin_tangential_speed_proxy_mean_mps"] = (speed * foot_contact).sum(0) / count.clamp_min(1)
            result["supported_speed_proxy_valid"] = count > 0
        return result
