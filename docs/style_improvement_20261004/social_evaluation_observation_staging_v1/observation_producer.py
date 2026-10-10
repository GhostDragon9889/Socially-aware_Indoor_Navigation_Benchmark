"""Raw social evaluation observations from a caller-owned original AMPWorld.

Stdlib at import. Only a future from_world on an existing SDK world executes
Torch geometry. No policy/normalizer, seed, simulation step, reset or App here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
import stat
import sys
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SCHEMA = "local.neutral_smpl.raw_social_evaluation_observations.v1"
GEOMETRY = {"grid_center": "root", "grid_heading": "root", "height_reference": "static_center_ground_z"}
PINS = {
    "world": ("CrowdSim/humanoid/amp_world.py", 24834, "1b63a4c8ac7d4bf2f9ed523edc97b41dca89888cdd6c7dcb2234826d240d7f83", False),
    "observations": ("CrowdSim/humanoid/amp/observations.py", 10438, "a64dc2eba5c132cf68a248f2324ea20446dd264a2ea8f9f139faae4738f9aeba", False),
    "state": ("protomotions/simulator/base_simulator/simulator_state.py", 34638, "31cf14fab319151602f43fcf6ad2751b71088cda3a3f4feefadfa61262102731", False),
    "base_simulator": ("protomotions/simulator/base_simulator/simulator.py", 68053, "6b080dd1b63885e34fcad523b8468ab6b63e838a3eaac765e7a230a2ca3ac782", False),
    "rotations": ("protomotions/utils/rotations.py", 28136, "ef1475d33d6f941ccdea4a612477483db4fad8236ba7786fbe8bf91a65c1cef8", False),
    "task": ("docs/style_improvement_20261004/social_task_observation_staging_v1/social_task_observation.py", 8512, "6975d95f369e2c00f1fa2ef34474e14164c36c82f70e4fb83a925335ee276ee1", True),
    "raster": ("docs/style_improvement_20261004/social_raster_torch_staging_v1/social_raster_torch.py", 9125, "234e6366a00301998349dbc837fae5878cac4c506e29ad605ab8b2023a6c5dbe", True),
}


def _record(path, *, readonly=False):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or readonly and info.st_mode & 0o222:
        raise ValueError("Expected regular single-link source: " + str(path))
    data = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


_EXECUTION_RECORD = _record(Path(__file__))


def _sources():
    rows = {}
    for key, (relative, size, sha, readonly) in PINS.items():
        row = _record(ROOT / relative, readonly=readonly)
        if row["bytes"] != size or row["sha256"] != sha:
            raise ValueError("Raw observation source changed: " + key)
        rows[key] = row
    if _record(Path(__file__)) != _EXECUTION_RECORD:
        raise ValueError("Observation producer changed after import")
    rows["execution_adapter"] = dict(_EXECUTION_RECORD)
    return rows


def _module(value, key):
    module = sys.modules.get(value.__module__)
    if module is None or Path(getattr(module, "__file__", "")).resolve() != ROOT / PINS[key][0]:
        raise TypeError("Actual imported source origin differs: " + key)
    return module


class RawSocialEvaluationObservations:
    """One original N1 or shared N4 world; explicit episode notifications.

    Fields returned by observe own the one COMMON snapshot. A failed observe
    faults this producer; it never recovers or changes the physical world.
    """

    @classmethod
    def from_world(cls, world, actor_ids, scene_ids, *, terrain_sampler, terrain_sampler_semantics,
                   policy_description=None, grid_center="root", grid_heading="root",
                   height_reference="static_ground", include_ego=False):
        instance = cls.__new__(cls)
        instance._capture(world, actor_ids, scene_ids, terrain_sampler, terrain_sampler_semantics,
                          grid_center, grid_heading, height_reference, include_ego)
        if policy_description is not None:
            instance.require_policy_contract(policy_description)
        return instance

    def _capture(self, world, actor_ids, scene_ids, sampler, semantics, center, heading, reference, include_ego):
        self.sources = _sources()
        module = _module(type(world), "world")
        if type(world) is not module.AMPWorld:
            raise TypeError("Only the original AMPWorld is supported, not grouped-training World/reset semantics")
        if (type(world.num_envs) is not int or world.num_envs not in (1, 4)
                or type(world.shared_world) is not bool or world.shared_world != (world.num_envs == 4)
                or world.dt != 1 / 30):
            raise ValueError("Require N1 isolated tracking or N4 shared evaluation at30Hz")
        if (center != "root" or heading != "root" or reference not in {"static_ground", "static_center_ground_z"}
                or include_ego is not False or not callable(sampler) or type(semantics) is not str or not semantics.strip()):
            raise ValueError("Explicit root/root/static-ground, include_ego=False and static sampler semantics required")
        if (getattr(world.state, "__func__", None) is not module.AMPWorld.state
                or getattr(world.ground_height_at, "__func__", None) is not module.AMPWorld.ground_height_at):
            raise TypeError("Do not override the original COMMON state/ground getter")
        observations = _module(module.build_self_observation, "observations")
        if module.build_self_observation is not observations.build_self_observation or type(world.mapping) is not observations.PacerSMPLMapping:
            raise TypeError("Original self357 function and named physical mapping required")
        self.state_module = _module(module.RobotState, "state")
        _module(observations.calc_heading_quat_inv, "rotations")
        base = _module(world.simulator.get_robot_state, "base_simulator")
        if getattr(world.simulator.get_robot_state, "__func__", None) is not base.Simulator.get_robot_state:
            raise TypeError("Original simulator COMMON RobotState getter required")
        self.torch, self.world, self.mapping = module.torch, world, world.mapping
        if sys.modules.get("torch") is not self.torch:
            raise TypeError("Use the caller world's already imported Torch module")
        self.device, self.count = world.device, world.num_envs
        self.sim, self.sampler, self.sampler_semantics = world.simulator._sim, sampler, semantics
        self.actor_ids = self._ids(actor_ids, (self.count,), "stable actor IDs").clone()
        self.scene_ids = self._ids(scene_ids, (self.count,), "scene IDs").clone()
        if self.torch.unique(self.actor_ids).numel() != self.count or self.torch.unique(self.scene_ids).numel() != 1:
            raise ValueError("Unique actor IDs and one explicitly shared scene required")
        source = (ROOT / PINS["task"][0]).read_bytes()
        if hashlib.sha256(source).hexdigest() != PINS["task"][2]:
            raise ValueError("Task source changed before execution")
        self.task = ModuleType("_raw_social_eval_frozen_task")
        self.task.__file__ = str(ROOT / PINS["task"][0])
        exec(compile(source, self.task.__file__, "exec"), self.task.__dict__)
        self.observations = observations
        self._active = self.torch.zeros(self.count, dtype=self.torch.bool, device=self.device)
        self._episodes = self.torch.full((self.count,), -1, dtype=self.torch.int64, device=self.device)
        self._ages = self.torch.zeros(self.count, dtype=self.torch.int64, device=self.device)
        self._headings = self.torch.zeros(self.count, dtype=self.torch.float32, device=self.device)
        self._defined_prior = self._active.clone()
        self._last_counter = self._counter()
        self._policy_contract, self._faulted = None, False

    def require_policy_contract(self, description):
        """Bind the caller's frozen actor descriptor; this does not load/prove weights."""
        if (type(description) is not dict or description.get("schema") not in {
                "local.neutral_smpl.frozen_social_actor_snapshot.v1", "local.neutral_smpl.frozen_social_checkpoint_inference.v1"}
                or description.get("grid_geometry_contract") != GEOMETRY or description.get("include_ego") is not False
                or type(description.get("observations")) is not dict
                or description["observations"].get("raw_self") != 357 or description["observations"].get("raw_task") != 12308):
            raise ValueError("Frozen social actor self357/task12308/geometry/include_ego descriptor mismatch")
        bound = {key: deepcopy(description[key]) for key in ("schema", "grid_geometry_contract", "include_ego", "observations")}
        if self._policy_contract is not None and bound != self._policy_contract:
            raise ValueError("A producer cannot change its bound policy observation contract")
        self._policy_contract = bound

    def _ids(self, value, shape, name):
        t = self.torch
        if (not isinstance(value, t.Tensor) or tuple(value.shape) != shape or value.dtype != t.int64
                or value.device != self.device or bool((value < 0).any())):
            raise ValueError(name + " requires nonnegative exact int64 on the caller world device")
        return value

    def _float(self, value, shape, name):
        t = self.torch
        if (not isinstance(value, t.Tensor) or tuple(value.shape) != shape or value.dtype != t.float32
                or value.device != self.device or not bool(t.isfinite(value).all())):
            raise ValueError(name + " requires finite float32 on the caller world device")
        return value

    def _counter(self):
        if self.world.simulator._sim is not self.sim:
            raise RuntimeError("Caller replaced the observation source SimulationContext")
        value = self.sim.get_physics_step_count()
        if type(value) is not int or value < 0:
            raise ValueError("Actual nonnegative integer physics counter required")
        return value

    def _clock(self):
        if self._faulted:
            raise RuntimeError("Observation producer faulted; no recovery/resume is implemented")
        counter = self._counter()
        if counter < self._last_counter:
            raise RuntimeError("Actual physics counter moved backwards")
        return (counter, self._ids(self.world.age, (self.count,), "age").clone(),
                self._ids(self.world.episode_ids, (self.count,), "episode IDs").clone())

    def _unchanged(self, before):
        after = self._clock()
        if after[0] != before[0] or not self.torch.equal(after[1], before[1]) or not self.torch.equal(after[2], before[2]):
            raise RuntimeError("Observation query/callback advanced physics or changed age/episode")

    def begin_episode(self, actor_rows=None):
        """Notify after actual world.reset(rows); clears only those prior headings."""
        before = self._clock()
        rows = self.torch.arange(self.count, device=self.device) if actor_rows is None else actor_rows
        self._ids(rows, (len(rows),), "reset actor rows")
        if not len(rows) or bool((rows >= self.count).any()) or self.torch.unique(rows).numel() != len(rows):
            raise ValueError("Reset notification requires nonempty unique valid actor rows")
        selected = self._active.clone().fill_(False)
        selected[rows] = True
        if (bool((before[1][rows] != 0).any()) or bool((before[2][rows] < 1).any())
                or bool((self._active[rows] & (before[2][rows] <= self._episodes[rows])).any())
                or bool((self._active & ~selected & (before[2] != self._episodes)).any())):
            raise ValueError("Require a genuine new reset episode at age0; notify all changed rows")
        self._unchanged(before)
        self._episodes[rows], self._ages[rows] = before[2][rows], before[1][rows]
        self._defined_prior[rows], self._headings[rows], self._active[rows] = False, 0., True
        self._last_counter = before[0]

    def observe(self, trajectory_world_xy):
        """One current owned COMMON state and same-phase raw inputs; no action taken."""
        try:
            if self._policy_contract is None:
                raise ValueError("Bind frozen actor.describe() before observing")
            before = self._clock()
            if (not bool(self._active.all()) or not self.torch.equal(before[2], self._episodes)
                    or bool((before[1] < self._ages).any())):
                raise ValueError("Missing reset notification or episode age moved backwards")
            trajectory = self._float(trajectory_world_xy, (self.count, 10, 2), "world futureXY+0.5..5s").detach().clone()
            state = self.world.state()
            if type(state) is not self.state_module.RobotState or state.state_conversion is not self.state_module.StateConversion.COMMON:
                raise TypeError("The one original COMMON RobotState snapshot is required")
            state = state.clone()  # Exact source clone owns every materialized tensor field.
            canonical = self.mapping.canonical_state(state)
            for key, width in (("rigid_body_pos", 3), ("rigid_body_rot", 4), ("rigid_body_vel", 3), ("rigid_body_ang_vel", 3)):
                self._float(getattr(canonical, key), (self.count, 24, width), "COMMON " + key)
            q = canonical.root_rot.clone()
            self._float(q, (self.count, 4), "actual COMMON Pelvis XYZW")
            norm = q.norm(dim=-1)
            if not self.torch.allclose(norm, self.torch.ones_like(norm), atol=1e-3, rtol=0):
                raise ValueError("All actual Pelvis quaternions must be unit XYZW; no implicit renormalization")
            roots = canonical.root_pos[:, :2]
            ground_query = roots.clone()
            ground = self.world.ground_height_at(ground_query)
            if not self.torch.equal(ground_query, roots):
                raise ValueError("Original ground callback changed its query")
            ground = self._float(ground, (self.count,), "current root ground Z").clone()
            up_z = 1 - 2 * q[:, :2].square().sum(-1)
            height = canonical.root_pos[:, 2] - ground
            physical_fall = (height < .45) | (up_z < .1)
            x, y, z, w = q.unbind(-1)
            defined = self.torch.hypot(2*w.square()-1+2*x.square(), 2*w*z+2*x*y) > 1e-12
            fallback = ~defined
            if bool((fallback & (~physical_fall | ~self._defined_prior)).any()):
                raise ValueError("Singular task heading requires actual fall and same episode prior defined heading")
            heading = self._headings.clone()
            rows = defined.nonzero().flatten()
            if len(rows):
                heading[rows] = self.task.heading_from_pelvis_xyzw(q[rows], device=self.device)
            raw_self = self.observations.build_self_observation(state, mapping=self.mapping)
            offsets = self.torch.arange(1, 11, dtype=self.torch.float32, device=self.device) / 2
            raw_task, owners = self.task.assemble_raw_task(roots, heading, canonical.root_vel[:, :2],
                self.actor_ids, self.scene_ids, self.torch.arange(self.count, device=self.device),
                trajectory, heading, offsets, roots.clone(), heading.clone(), ground.clone(), self.sampler,
                include_ego=False, device=self.device)
            self._float(raw_self, (self.count, 357), "raw self357")
            self._float(raw_task, (self.count, 12308), "raw social task12308")
            self._unchanged(before)
            if _sources() != self.sources:
                raise ValueError("Observation execution dependencies changed during query")
            self._headings.copy_(heading)
            self._defined_prior |= defined
            self._ages.copy_(before[1])
            self._last_counter = before[0]
            diagnostics = {"physics_step": before[0], "age": before[1].clone(), "episode_ids": before[2].clone(),
                "actor_ids": self.actor_ids.clone(), "scene_ids": self.scene_ids.clone(),
                "actual_pelvis_quat_xyzw": q, "actual_ground_height_world_z": ground,
                "actual_pelvis_height_above_ground_m": height, "actual_up_z": up_z,
                "physical_fall": physical_fall, "used_task_heading": heading.clone(),
                "task_heading_defined": defined, "singular_heading_fallback": fallback,
                "metadata": self.describe()}
            return SimpleNamespace(state=state, raw_self=raw_self.detach().clone(), raw_task=raw_task.detach().clone(),
                                   owners=owners.detach().clone(), diagnostics=diagnostics)
        except BaseException:
            self._faulted = True
            raise

    def describe(self):
        return {"schema": SCHEMA, "sources": deepcopy(self.sources), "actors": self.count,
            "shared_original_world": self.world.shared_world,
            "actor_ids": self.actor_ids.tolist(), "scene_ids": self.scene_ids.tolist(),
            "grid_geometry_contract": dict(GEOMETRY), "include_ego": False,
            "terrain_sampler_semantics": self.sampler_semantics,
            "terrain_semantics_is_caller_declaration_not_surface_validation": True,
            "bound_policy_observation_contract": deepcopy(self._policy_contract),
            "observation_phase": "current_common_snapshot_before_caller_action",
            "reset_contract": "notify begin_episode after each actual original reset; no automatic reset",
            "physical_fall_definition": "COMMON PelvisZ-ground<0.45 or actual unit-XYZW upZ<0.1; not cap/contact/truncation",
            "singular_task_heading": "actual fallen only, same-episode prior projected-X; original self357 unchanged",
            "policy_normalizer_step_reset_seed_or_checkpoint_called": False,
            "physics_or_policy_quality_claimed": False}
