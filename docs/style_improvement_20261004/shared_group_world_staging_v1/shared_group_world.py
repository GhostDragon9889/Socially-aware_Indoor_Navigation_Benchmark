"""Prepared four-Articulation physics/clock driver; no executable training entry.

Import is Torch tensor code only. SDK construction is explicit and belongs after
one caller-owned App launch. No policy, weights, seed, logger or test is created.
"""
from __future__ import annotations

import ast
import __future__
import hashlib
import math
from pathlib import Path
import stat
import sys
from types import MappingProxyType, ModuleType, SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[3]
STAGES = ROOT / "docs/style_improvement_20261004"
LAB = Path("/project/EXP/SangforAndSaut/IsaacLab-Protomotions/source")
DT, SUBSTEPS, HISTORY, REPLAN = 1 / 120, 4, 10, 15
PINS = {
    "factory": (STAGES / "group_articulation_scene_staging_v1/group_scene.py", 7213, "88e5a8d289a3a3d8329107faaca857f03cf38a5c640985af84b7b3658f6916b6", True),
    "bridge": (STAGES / "group_articulation_tensor_bridge_staging_v1/tensor_bridge.py", 6035, "2f5dc721f739d26ace02cc0831864523e605587f7e960f0fd32b82326ec8c9ec", True),
    "adapter": (STAGES / "group_articulation_state_adapter_staging_v1/state_adapter.py", 12750, "468a26311fcde8a92595fb03bf5d6294bb9a927910c805f24368f7068edc1e50", True),
    "task": (STAGES / "social_task_observation_staging_v1/social_task_observation.py", 8512, "6975d95f369e2c00f1fa2ef34474e14164c36c82f70e4fb83a925335ee276ee1", True),
    "group_reset": (STAGES / "social_group_reset_staging_v1/group_reset.py", 3244, "2340a7908454408b84366393a83762fc224d214b6673546a03ed1bade9fd3aa4", True),
    "contact": (STAGES / "nonfoot_net_normal_contact_staging_v1/nonfoot_contact.py", 4095, "9fa82a7771a2c875e1cc41610b0203e8c3d235a9c50f7668f78665ae373a863a", True),
    "sampler": (STAGES / "social_net_normal_sampler_staging_v1/net_normal_sampler.py", 7398, "04d21596a8f1fd54d9d9a43977c8a76137840a574fb4f04a73ed4ce94beb5d09", True),
    "raster": (STAGES / "social_raster_torch_staging_v1/social_raster_torch.py", 9125, "234e6366a00301998349dbc837fae5878cac4c506e29ad605ab8b2023a6c5dbe", True),
    "observations": (ROOT / "CrowdSim/humanoid/amp/observations.py", 10438, "a64dc2eba5c132cf68a248f2324ea20446dd264a2ea8f9f139faae4738f9aeba", False),
    "rotations": (ROOT / "protomotions/utils/rotations.py", 28136, "ef1475d33d6f941ccdea4a612477483db4fad8236ba7786fbe8bf91a65c1cef8", False),
    "controller": (ROOT / "CrowdSim/humanoid/amp/controller.py", 10030, "ca27410b6c50ab7b9cbaff0f360ec09206ecfd602b7a2c3968c3e5c6509b1d68", False),
    "mixed": (ROOT / "CrowdSim/humanoid/training/mixed_trajectories.py", 22594, "c79732a675ac737d4ce0aff0d2b76cfa30cebed2d9c115e1a35da39299694d67", False),
    "amass": (ROOT / "CrowdSim/humanoid/training/amass_trajectories.py", 21752, "ddca3403a59422045f051b5d9f653d6f411db1523840ed8bc021fc96ef82566e", False),
    "procedural": (ROOT / "CrowdSim/humanoid/training/trajectories.py", 14063, "81701988535cf847b8b86979a836ba37089e93553aa4e3fe5ecfc9529d62d2c9", False),
    "SDK_scene": (LAB / "isaaclab/isaaclab/scene/interactive_scene.py", 39939, "5d509530070ffb8f4616f769a6608299a386800d7e2582833181ba28dfee32d1", False),
    "SDK_sim": (LAB / "isaaclab/isaaclab/sim/simulation_context.py", 42425, "8c5d73af45f82b0d5c36b05a1c293094319c5d17a75f75a2fe679c382334ea89", False),
    "SDK_physx_manager": (LAB / "isaaclab_physx/isaaclab_physx/physics/physx_manager.py", 34974, "d83ef1b3a375dbbdbd538f7e5f552064ee57528ae7b35b9ba58719fd8a4cfda5", False),
}
PD_AST_SHA = "920e6f702a422e73c91ae6bc7f939650b65e8eb423ef46b0e14a216e5aaaed69"
_DEPS = None


def _dependencies():
    global _DEPS
    if _DEPS is None:
        sources, modules = {}, {}
        for key, (path, size, sha, readonly) in PINS.items():
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or readonly and info.st_mode & 0o222:
                raise ValueError("Invalid reviewed source: " + key)
            data = path.read_bytes()
            if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
                raise ValueError("Reviewed source identity changed: " + key)
            sources[key] = data
        # Do not import amp.__init__, controller or training.__init__: those import NN.
        for key in ("factory", "bridge", "adapter", "task", "contact", "sampler", "observations"):
            module = ModuleType("_shared_group_" + key)
            module.__file__ = str(PINS[key][0])
            sys.modules[module.__name__] = module
            exec(compile(sources[key], module.__file__, "exec"), module.__dict__)
            modules[key] = module
        nodes = [node for node in ast.parse(sources["controller"]).body
                 if isinstance(node, ast.FunctionDef) and node.name in ("_pd_parameters", "action_to_pd")]
        tree = ast.Module(body=nodes, type_ignores=[])
        if len(nodes) != 2 or hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest() != PD_AST_SHA:
            raise ValueError("Original two-function PD AST changed")
        pd = {"torch": torch, "math": math, "PacerSMPLMapping": modules["observations"].PacerSMPLMapping,
              "PACER_DOF_NAMES": modules["observations"].PACER_DOF_NAMES}
        exec(compile(tree, str(PINS["controller"][0]), "exec", flags=__future__.annotations.compiler_flag), pd)
        modules["action_to_pd"] = pd["action_to_pd"]
        modules["adapter"]._dependencies()  # SDK write/data byte pins and named conversion, no SDK import.
        _DEPS = SimpleNamespace(**modules)
    return _DEPS


def _float(value, shape, device, label):
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
            or value.dtype != torch.float32 or value.device != device or not bool(torch.isfinite(value).all())):
        raise ValueError(label + " requires finite float32 on-device " + str(tuple(shape)))
    return value


def _native_clone(value):
    if not isinstance(value, torch.Tensor):
        if hasattr(type(value), "torch"):
            value = value.torch  # Fresh getter first; owned clone before another refresh.
        else:
            import warp as wp
            value = wp.to_torch(value)
    return value.clone()


def _expanded_reset(rows, terminated, truncated):
    """Torch semantic adaptation of sealed NumPy expand_group_reset; no host array."""
    triggered = (terminated | truncated)[rows].any(1)
    reset = torch.zeros_like(terminated)
    reset[rows[triggered].flatten()] = True
    collateral = reset & ~(terminated | truncated)
    return terminated.clone(), truncated | collateral, reset, collateral


def _geometry_contract(value):
    allowed = {"grid_center": {"root", "head"}, "grid_heading": {"root", "head"},
               "height_reference": {"root_z", "static_center_ground_z"}}
    if type(value) is not dict or set(value) != set(allowed):
        raise ValueError("Explicit grid geometry contract requires the exact three frame/reference fields")
    if any(type(value[key]) is not str or value[key] not in choices for key, choices in allowed.items()):
        raise ValueError("Invalid declared grid center, heading or world-Z reference")
    return MappingProxyType(dict(value))  # Owned declaration; not independent callback validation.


class SharedGroupWorld:
    @classmethod
    def create_after_app(cls, app, simulation_cfg, single_smpl_template, base_scene_cfg, *,
                         num_actors, actor_offsets_xy, env_spacing, robot_config, curriculum,
                         ground_height_at, terrain_sampler, grid_geometry, grid_geometry_contract, include_ego,
                         normal_force_bundle_factory, episode_steps=300):
        """Construct ONE new G-env scene; caller launches/owns App and eventual close."""
        deps = _dependencies()  # Verify source bytes before any SDK import/construction.
        _geometry_contract(grid_geometry_contract)  # Reject incomplete declarations before SDK side effects.
        if app is None or not callable(getattr(app, "is_running", None)) or not app.is_running():
            raise ValueError("A live caller-owned single App is required")
        from isaaclab.scene import InteractiveScene
        from isaaclab.sim import SimulationContext
        from isaaclab_physx.physics import PhysxCfg
        for key, value in (("SDK_scene", InteractiveScene), ("SDK_sim", SimulationContext)):
            if Path(sys.modules[value.__module__].__file__).resolve() != PINS[key][0]:
                raise ValueError("Imported SDK source origin mismatch: " + key)
        if SimulationContext.instance() is not None:
            raise ValueError("Refuse existing simulation context, including old single-robot World")
        if not isinstance(simulation_cfg.physics, PhysxCfg) or not math.isclose(simulation_cfg.dt, DT, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Explicit PhysX120Hz configuration required")
        if simulation_cfg.render_interval != SUBSTEPS or single_smpl_template.spawn.activate_contact_sensors is not True:
            raise ValueError("Control render_interval4 and pre-spawn ContactReportAPI authoring required")
        cfg = deps.factory.create_group_scene_cfg(single_smpl_template, base_scene_cfg,
            num_actors=num_actors, actor_offsets_xy=actor_offsets_xy, env_spacing=env_spacing)
        sim = SimulationContext(simulation_cfg)
        scene = InteractiveScene(cfg)
        sim.reset()  # Native initialization; recorded baseline starts AFTER this call.
        return cls(sim, scene, robot_config, curriculum, ground_height_at=ground_height_at,
                   terrain_sampler=terrain_sampler, grid_geometry=grid_geometry,
                   grid_geometry_contract=grid_geometry_contract, include_ego=include_ego,
                   normal_force_bundle_factory=normal_force_bundle_factory, episode_steps=episode_steps)

    def __init__(self, sim, scene, robot_config, curriculum, *, ground_height_at, terrain_sampler,
                 grid_geometry, grid_geometry_contract, include_ego, normal_force_bundle_factory, episode_steps=300):
        self.deps = deps = _dependencies()
        self.device = deps.bridge._device(sim.device)
        if type(scene.cfg.num_envs) is not int or scene.cfg.num_envs < 1:
            raise ValueError("Positive integer group environment count required")
        self.num_envs = scene.cfg.num_envs * 4  # Flat policy actors, NOT the scene env count.
        self.rows = deps.bridge.mixed_group_rows(self.num_envs, device=self.device)
        if (scene.cfg.filter_collisions is not True or scene.cfg.replicate_physics is not True
                or "physx" not in scene.physics_backend or set(scene.articulations) != set(deps.factory.ACTOR_NAMES)
                or not math.isclose(sim.get_physics_dt(), DT, rel_tol=0, abs_tol=1e-12)):
            raise ValueError("Four-slot grouped PhysX scene/120Hz required")
        if type(episode_steps) is not int or episode_steps < 1 or type(include_ego) is not bool:
            raise ValueError("Positive episode cap and explicit self-raster choice required")
        if any(not callable(value) for value in (ground_height_at, terrain_sampler, grid_geometry, normal_force_bundle_factory)):
            raise ValueError("Explicit current static geometry/ground/view callbacks required")
        curriculum_module = sys.modules.get(type(curriculum).__module__)
        origin = Path(getattr(curriculum_module, "__file__", "")).resolve()
        if (type(curriculum) is not getattr(curriculum_module, "MixedTrajectoryCurriculum", None) or origin != PINS["mixed"][0]
                or curriculum.num_envs != self.num_envs or curriculum.device != self.device):
            raise ValueError("Caller-prepared original fixed-half mixed curriculum required")
        if bool(curriculum.amass.initialized.any()) or bool(curriculum.proc_initialized.any()):
            raise ValueError("This fresh-world stage cannot resume an already bound curriculum")
        self.curriculum, self.episode_steps = curriculum, episode_steps
        self.ground_height_at, self.terrain_sampler = ground_height_at, terrain_sampler
        self.grid_geometry, self.include_ego = grid_geometry, include_ego
        self.grid_geometry_contract = _geometry_contract(grid_geometry_contract)
        self.assets = tuple(scene[name] for name in deps.factory.ACTOR_NAMES)
        self.sim, self.scene = sim, scene
        self._faulted = False
        self.adapter = deps.adapter.FourArticulationAdapter(self.assets, self.rows, robot_config.kinematic_info, device=self.device)
        self.mapping = deps.observations.PacerSMPLMapping(tuple(robot_config.kinematic_info.body_names), tuple(robot_config.kinematic_info.dof_names))
        self.kinematic_info = robot_config.kinematic_info
        self.actor_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)
        self.scene_ids = torch.empty_like(self.actor_ids)
        self.scene_ids[self.rows] = torch.arange(len(self.rows), dtype=torch.int64, device=self.device)[:, None]
        origins = _float(scene.env_origins.clone(), (len(self.rows), 3), self.device, "group world origins")
        # SDK default_root_pose is env-local. Apply scene origins exactly once here.
        spawn = tuple(_float(_native_clone(asset.data.default_root_pose), (len(self.rows), 7), self.device, "default root pose")[:, :3] + origins for asset in self.assets)
        self.spawn_world_xyz = deps.bridge.assets_to_canonical(spawn, self.rows, device=self.device)
        limits = tuple(_float(_native_clone(asset.data.joint_pos_limits), (len(self.rows), 69, 2), self.device, "native actual limits").index_select(1, mapping.dof_convert_to_common)
                       for asset, mapping in zip(self.assets, self.adapter.mappings, strict=True))
        common_limits = deps.bridge.assets_to_canonical(limits, self.rows, device=self.device)
        if not torch.equal(common_limits, common_limits[:1].expand_as(common_limits)):
            raise ValueError("One policy PD decoder requires identical actual named joint limits")
        self.lower, self.upper = common_limits[0, :, 0].clone(), common_limits[0, :, 1].clone()
        if bool((self.lower > self.upper).any()):
            raise ValueError("Actual named DOF limits must be ordered")
        before_bundle = self._counter()
        self.bundle = normal_force_bundle_factory(sim, scene, self.assets, self.rows.clone())
        if self._counter() != before_bundle:
            raise RuntimeError("View construction must not step physics")
        if not isinstance(self.bundle, (tuple, list)) or len(self.bundle) != 3:
            raise ValueError("Normal-force bundle must be views, layouts, body_names_by_view")
        self.age, self.episode_ids = torch.zeros_like(self.actor_ids), torch.zeros_like(self.actor_ids)
        self.reference_ids = torch.zeros_like(self.actor_ids)
        self.plan_age = torch.zeros_like(self.actor_ids)
        self.plan_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.plans = torch.zeros((self.num_envs, 151, 2), dtype=torch.float32, device=self.device)
        self.amp_history = torch.zeros((self.num_envs, HISTORY, 195), dtype=torch.float32, device=self.device)
        self.amp_history_episode_ids = torch.zeros((self.num_envs, HISTORY), dtype=torch.int64, device=self.device)
        self.planar_history = torch.zeros((self.num_envs, 91, 5), dtype=torch.float32, device=self.device)
        self.history_available = torch.zeros((self.num_envs, 91), dtype=torch.bool, device=self.device)
        self.heading_history_defined = torch.zeros_like(self.history_available)
        self.last_defined_heading = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.heading_context_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.used_task_heading = self.last_defined_heading.clone()
        self.task_heading_defined = self.heading_context_valid.clone()
        self.singular_heading_fallback = self.heading_context_valid.clone()
        self.control_step, self._physics_step = 0, self._counter()
        self.current = self.adapter.snapshot()
        self._prepared = self._pending = None
        self.last_completion = None  # Constant-size declaration, not independent execution evidence.
        self.statistics = dict.fromkeys(("actor_controls", "physical_falls", "nonfoot_normal_contact_terminations",
            "original_terminal_actor_controls", "collateral_truncated_actor_controls", "source_end_actor_controls",
            "cap_actor_controls", "reset_actor_rows", "reset_groups", "initial_reset_actor_rows"), 0)
        self._reset(self.actor_ids, initialization=True)

    def _counter(self):
        value = self.sim.get_physics_step_count()
        if type(value) is not int or value < 0:
            raise ValueError("Actual nonnegative integer physics counter required")
        return value

    def _stationary_clock(self):
        if self._faulted:
            raise RuntimeError("Driver faulted after control/archive/reset side effects; recovery is not implemented")
        if self._counter() != self._physics_step:
            raise RuntimeError("Caller or callback advanced physics outside this driver")

    def _planar(self, state):
        return torch.cat((state.root_pos[:, :2], state.root_vel[:, :2], self.used_task_heading[:, None]), -1)

    def _select_task_heading(self, physical_fall):
        """Continue task coordinates only for a truly fallen, singular root pose.

        Keep actual quaternion. A prior real defined heading in this episode is
        mandatory; no zero guess. Original self357 heading algorithm stays separate.
        """
        q = self.current.robot_state.root_rot
        x, y, z, w = q.unbind(-1)
        direction_x = 2*w.square()-1+2*x.square()
        direction_y = 2*w*z+2*x*y
        defined = torch.hypot(direction_x, direction_y) > 1e-12  # Exact frozen helper boundary/math.
        fallback = ~defined
        if bool((fallback & (~physical_fall | ~self.heading_context_valid)).any()):
            raise ValueError("Undefined task heading requires an actual fall and this episode's prior defined heading")
        rows = defined.nonzero().flatten()
        if len(rows):
            heading = self.deps.task.heading_from_pelvis_xyzw(q.index_select(0, rows), device=self.device)
            self.last_defined_heading[rows] = heading
            self.heading_context_valid[rows] = True
        self.used_task_heading = self.last_defined_heading.clone()
        self.task_heading_defined, self.singular_heading_fallback = defined, fallback

    def _ground(self, xy):
        query = xy.clone()
        before = query.clone()
        result = self.ground_height_at(query)
        if not torch.equal(query, before):
            raise ValueError("Static ground callback changed its query")
        return _float(result, xy.shape[:-1], self.device, "absolute static ground Z")

    def _replan(self, ids):
        if len(ids):
            prediction = self.curriculum.build_plan(self.current.robot_state, self.age,
                actor_ids=ids, episode_ids=self.episode_ids)
            self.plans[ids] = _float(prediction.native_xy, (len(ids), 151, 2), self.device, "30Hz reference plan")
            self.plan_age[ids] = 0
            self.plan_valid[ids] = True

    def _observations(self):
        state = self.current.robot_state
        trajectory = self.curriculum.future_targets(self.age, self.plan_age, self.plans, episode_ids=self.episode_ids)
        canonical = self.mapping.canonical_state(state)
        yaw = self.used_task_heading
        owned = self.deps.adapter.CommonSnapshot(self.current.robot_state.clone(), self.current.root_state.clone())
        geometry = self.grid_geometry(owned, self.used_task_heading.clone(), self.task_heading_defined.clone())
        if not isinstance(geometry, (tuple, list)) or len(geometry) != 3:
            raise ValueError("grid_geometry must explicitly return centersXY, headings, referenceWorldZ")
        task, owners = self.deps.task.assemble_raw_task(canonical.root_pos[:, :2], yaw, canonical.root_vel[:, :2],
            self.actor_ids, self.scene_ids, self.actor_ids, trajectory, yaw,
            torch.arange(1, 11, device=self.device, dtype=torch.float32) * .5,
            *geometry, self.terrain_sampler, include_ego=self.include_ego, device=self.device)
        self_obs = self.deps.observations.build_self_observation(state, mapping=self.mapping)
        return SimpleNamespace(self_obs=self_obs, task_obs=task, owners=owners,
            amp_history=self.amp_history.clone(), history_episode_ids=self.amp_history_episode_ids.clone(),
            episode_ids=self.episode_ids.clone(), age=self.age.clone(), plan_age=self.plan_age.clone(),
            control_step=self.control_step, physics_step=self._physics_step)

    def prepare_control(self):
        """Raw policy input; pre-step newest-first state_action AMP history retained."""
        self._stationary_clock()
        if self._pending is not None or self._prepared is not None:
            raise RuntimeError("Complete the pending transition before preparing another control")
        due = self.actor_ids if self.control_step % REPLAN == 0 else ((~self.plan_valid) | (self.plan_age >= REPLAN)).nonzero().flatten()
        try:
            self._replan(due)  # Same synchronized global15-control clock plus reset-forced planning.
            self._prepared = self._observations()
            self._stationary_clock()
        except BaseException:
            # Planning may already have consumed RNG or changed curriculum/plan
            # state; retrying this failed preparation is not a fresh attempt.
            self._faulted = True
            raise
        return self._prepared

    def step(self, applied_actions):
        """Four real future SDK substeps. No NN, implicit reset or rendering loop."""
        self._stationary_clock()
        if self._prepared is None or self._pending is not None:
            raise RuntimeError("Exactly one prepared policy input is required")
        actions = _float(applied_actions, (self.num_envs, 69), self.device, "applied canonical action")
        frozen = torch.tensor([name.rsplit("_", 1)[0] in {"L_Hand", "R_Hand", "L_Toe", "R_Toe"}
                               for name in self.deps.observations.PACER_DOF_NAMES], device=self.device)
        if bool((actions.abs() > 1).any()) or bool((actions[:, frozen] != 0).any()):
            raise ValueError("Require already clamped/masked applied69, not raw policy samples")
        targets = self.deps.action_to_pd(actions, self.lower, self.upper, dof_names=self.mapping.dof_names, clip=False)
        try:
            return self._advance_control(actions, targets)
        except BaseException:
            self._faulted = True  # No automatic retry after possibly partial physical side effects.
            raise

    def _advance_control(self, actions, targets):
        before = self._physics_step
        power = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        last_effort = None
        for index in range(SUBSTEPS):
            self.adapter.set_common_position_targets(targets)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(DT)
            if self._counter() != before + index + 1:
                raise RuntimeError("Substep counter must advance exactly once")
            efforts, velocities = [], []
            for asset, mapping in zip(self.assets, self.adapter.mappings, strict=True):
                effort = _float(_native_clone(asset.root_view.get_dof_projected_joint_forces()), (len(self.rows), 69), self.device, "measured generalized effort")
                velocity = _float(_native_clone(asset.data.joint_vel), (len(self.rows), 69), self.device, "post-substep native joint velocity")
                efforts.append(effort.index_select(1, mapping.dof_convert_to_common))
                velocities.append(velocity.index_select(1, mapping.dof_convert_to_common))
            last_effort = self.deps.bridge.assets_to_canonical(tuple(efforts), self.rows, device=self.device)
            common_velocity = self.deps.bridge.assets_to_canonical(tuple(velocities), self.rows, device=self.device)
            power += (last_effort * common_velocity).sum(-1)
        self._physics_step = self._counter()
        self.current = self.adapter.snapshot()
        self.current.robot_state.dof_forces = last_effort.clone()
        self.age += 1
        self.plan_age += 1
        self.control_step += 1
        state = self.current.robot_state
        ground = self._ground(state.root_pos[:, :2])
        physical_fall = ((state.root_pos[:, 2] - ground) < .45) | ((1 - 2 * state.root_rot[:, :2].square().sum(-1)) < .1)
        self._select_task_heading(physical_fall)
        self.amp_history = torch.cat((self.deps.observations.build_amp_observation(self.current.robot_state, mapping=self.mapping)[:, None], self.amp_history[:, :-1]), 1)
        self.amp_history_episode_ids = torch.cat((self.episode_ids[:, None], self.amp_history_episode_ids[:, :-1]), 1)
        self.planar_history = torch.cat((self.planar_history[:, 1:], self._planar(self.current.robot_state)[:, None]), 1)
        self.history_available = torch.cat((self.history_available[:, 1:], torch.ones((self.num_envs, 1), dtype=torch.bool, device=self.device)), 1)
        self.heading_history_defined = torch.cat((self.heading_history_defined[:, 1:], self.task_heading_defined[:, None]), 1)
        forces, force_metadata = self.deps.sampler.sample_current_net_normal(*self.bundle, self.rows,
            bridge=self.deps.bridge, device=self.device, physics_dt=DT,
            physics_step=self._physics_step, control_step=self.control_step)
        self._stationary_clock()  # Sampler stores caller stamps; driver actually verifies this clock.
        contact = self.deps.contact.nonfoot_net_normal_contact(forces, self.age,
            body_names=self.deps.observations.PACER_BODY_NAMES)
        raw_terminal = physical_fall | contact.contact_flag
        cap = self.age >= self.episode_steps
        source_end = self.curriculum.completed_step(self.age, raw_terminal, cap)
        terminal, truncation, reset, collateral = _expanded_reset(self.rows, raw_terminal, cap | source_end)
        for key, mask in (("physical_falls", physical_fall), ("nonfoot_normal_contact_terminations", contact.contact_flag),
                          ("original_terminal_actor_controls", raw_terminal), ("collateral_truncated_actor_controls", collateral),
                          ("source_end_actor_controls", source_end), ("cap_actor_controls", cap)):
            self.statistics[key] += int(mask.sum())
        self.statistics["actor_controls"] += self.num_envs
        post = self._observations()  # All original episodes, including terminal/collateral rows.
        self._stationary_clock()
        target = self.curriculum.interpolate_reference(self.plan_age[:, None] / 30, self.plans)[:, 0]
        archived_snapshot = self.deps.adapter.CommonSnapshot(self.current.robot_state.clone(), self.current.root_state.clone())
        self._pending = SimpleNamespace(pre=self._prepared, post=post, snapshot=archived_snapshot,
            applied_actions=actions.clone(), target_xy=target.clone(), signed_power_w=_float(power / SUBSTEPS, (self.num_envs,), self.device, "signed generalized joint power"),
            terminated=terminal, truncated=truncation, reset_mask=reset, collateral_truncated=collateral,
            physical_fall=physical_fall, nonfoot_normal_contact=contact, source_end_truncated=source_end.clone(), cap_truncated=cap,
            control_step_before=self.control_step - 1, control_step_after=self.control_step,
            physics_step_before=before, physics_step_after=self._physics_step,
            singular_heading_fallback=self.singular_heading_fallback.clone(), task_heading_defined=self.task_heading_defined.clone(),
            actual_root_quat_xyzw=state.root_rot.clone(), used_task_heading=self.used_task_heading.clone(),
            self_heading_source="original observations.build_self_observation / rotations.calc_heading_quat_inv; no driver continuation",
            normal_force_metadata=force_metadata)
        self._reset_rows = reset.nonzero().flatten().clone()
        self._pending_episode = self.episode_ids.clone()
        self._prepared = None
        return self._pending

    def complete_transition(self, next_values, archive_callback):
        """Caller supplies terminal nextV; callback completes BEFORE any whole-group reset.

        Callback return is an in-process assertion, not independent archival proof.
        Exceptions propagate; SDK writes or callback side effects are not transactional.
        """
        self._stationary_clock()
        if self._pending is None or not callable(archive_callback):
            raise RuntimeError("One unique pending transition and explicit archive callback required")
        if (self._pending.control_step_after != self.control_step
                or not torch.equal(self.episode_ids, self._pending_episode)):
            raise RuntimeError("Pending transition must remain on the same control and episodes")
        values = _float(next_values, (self.num_envs,), self.device, "terminal next values").clone()
        ids = self._reset_rows.clone()
        try:
            archive_callback(self._pending, values)
            self._stationary_clock()
            if not torch.equal(self.episode_ids, self._pending_episode):
                raise RuntimeError("Archive callback changed actor episodes")
            if len(ids):
                self._reset(ids)
        except BaseException:
            self._faulted = True  # Partial callback/SDK side effects cannot be rolled back.
            raise
        self.last_completion = {"control_step": self.control_step, "physics_step": self._physics_step,
                                "archive_callback_returned": True, "archive_independently_verified": False,
                                "reset_actor_rows": ids.clone()}
        self._pending = None
        return ids

    def _reset(self, ids, *, initialization=False):
        self._stationary_clock()
        group_ids = self.deps.bridge.reset_group_env_ids(ids, self.rows, device=self.device)[0]
        if not len(ids):
            return
        # Original draw order/private and shared generators are retained; never reseed.
        reference, history, _, motion_ids, frames = self.curriculum.sample_reference(ids, HISTORY)
        positions = reference.root_pos.clone()
        positions[:, :2] = self.spawn_world_xyz[ids, :2]
        positions[:, 2] += self._ground(positions[:, :2])
        _, states, _ = self.deps.adapter._dependencies()
        fields = {key: getattr(self.current.root_state, key).clone() for key in ("root_pos", "root_rot", "root_vel", "root_ang_vel")}
        fields.update({key: getattr(self.current.robot_state, key).clone() for key in ("dof_pos", "dof_vel")})
        for key in fields:
            fields[key][ids] = positions if key == "root_pos" else getattr(reference, key)
        reset = states.ResetState(state_conversion=states.StateConversion.COMMON, **fields)
        self.adapter.write_common_reset(reset, ids)
        self.scene.reset(env_ids=group_ids)  # Actuator/sensor buffers, not physical default-state writes.
        self.sim.forward()  # Optional fabric/FK sync without a physics step; not the sole refresh.
        self._stationary_clock()
        # Writers keep skip_forward=False: timestamps/FK are invalidated. Fresh
        # body properties call _ensure_fk_fresh even when fabric is unavailable.
        self.current = self.adapter.snapshot()
        # Fresh episode: selected rows must have an actually defined heading.
        reset_heading = self.deps.task.heading_from_pelvis_xyzw(self.current.robot_state.root_rot.index_select(0, ids), device=self.device)
        self.last_defined_heading[ids] = reset_heading
        self.heading_context_valid[ids] = True
        self.used_task_heading[ids] = reset_heading
        self.task_heading_defined[ids] = True
        self.singular_heading_fallback[ids] = False
        self.age[ids], self.plan_age[ids] = 0, 0
        self.episode_ids[ids] += 1
        self.reference_ids[ids] = motion_ids.to(self.device)
        self.amp_history[ids] = self.deps.observations.build_amp_observation(history, mapping=self.mapping).reshape(len(ids), HISTORY, 195)
        self.amp_history[ids, 0] = self.deps.observations.build_amp_observation(self.current.robot_state, mapping=self.mapping)[ids]
        self.amp_history_episode_ids[ids] = self.episode_ids[ids, None]
        self.planar_history[ids] = self._planar(self.current.robot_state)[ids, None]
        self.history_available[ids] = False
        self.history_available[ids, -1] = True
        self.heading_history_defined[ids] = False
        self.heading_history_defined[ids, -1] = True
        self.curriculum.bind_reset(ids, motion_ids, frames, reference.root_pos[:, :2], positions[:, :2], self.episode_ids[ids])
        self.plans[ids] = 0
        self.plan_valid[ids] = False
        # Like the old runner's reset-forced due flag, plan once at the next
        # policy preparation. No extra procedural draw or future query at reset.
        if initialization:
            self.statistics["initial_reset_actor_rows"] += len(ids)
        else:
            self.statistics["reset_actor_rows"] += len(ids)
            self.statistics["reset_groups"] += len(group_ids)
        self._stationary_clock()

    def describe(self):
        """Declaration/state only; does not claim successful SDK execution or recovery."""
        original = self.curriculum.describe()
        legacy = {}
        for path, stats in (("mixed.statistics", original["statistics"]), ("mixed.amass.statistics", original["amass"]["statistics"])):
            for name in tuple(stats):
                if "fall" in name:
                    legacy.setdefault(path, {})[name] = stats.pop(name)
        return {"schema": "official_orca.shared_group_world_staging.v1", "num_actors": self.num_envs,
                "num_scene_envs": len(self.rows), "group_actor_rows": self.rows.clone(),
                "grid_geometry_contract": dict(self.grid_geometry_contract),
                "grid_callback_semantics_independently_verified": False,
                "physics_dt_s": DT, "control_dt_s": DT * SUBSTEPS, "control_step": self.control_step,
                "power_quantity": "signed generalized projected joint effort times matching post-substep DOF velocity, four-sample mean W",
                "curriculum": original, "legacy_aggregate_terminal_counts": legacy,
                "driver_reason_counts": dict(self.statistics),
                "legacy_count_note": "Original opaque state_dict retains fall-named keys; this driver passes all actual terminal causes including contact.",
                "physics_execution_verified": False, "full_training_runner_implemented": False,
                "SDK_checkpoint_recovery_implemented": False}
