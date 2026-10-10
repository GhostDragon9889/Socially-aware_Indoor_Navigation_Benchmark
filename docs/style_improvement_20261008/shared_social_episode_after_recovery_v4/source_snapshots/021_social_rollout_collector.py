"""Prepared exclusive social rollout collector; not an executable training CLI.

Receives an already constructed reviewed World and trainer. No App, network,
seed, weight, expert batch or physical checkpoint is created here. This source
has not been imported or executed; source checks are not rollout evidence.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import stat
import sys
from types import ModuleType

import torch

ROOT = Path(__file__).resolve().parents[3]
STAGES = ROOT / "docs/style_improvement_20261004"
PINS = {
    "world": (STAGES / "shared_group_world_owned_lifetime_revision_v2/shared_group_world.py", 39943,
              "e76182d2be5a3fa4dad19a04d71275d91929ce3ea2d63a1c23f081aa1a59880b", True),
    "trainer": (STAGES / "social_trainer_staging_v1/social_trainer.py", 24078,
                "81a6319f2520b3be547f30454c88b088638d559d492bba52fe850d1c1d5ba96e", True),
    "rewards": (ROOT / "CrowdSim/humanoid/training/rewards.py", 5491,
                "fc595dd9c1b2012aa76f37dcecd4ec5d7dad50fbd923840736a77c24622a6430", False),
    "contracts": (ROOT / "CrowdSim/humanoid/training/contracts.py", 4545,
                  "26c7379ba60842c67f062fa5c1a3459ddfb6e0843e1b4952e1d078aa6f78cdad", False),
    "original_world": (ROOT / "CrowdSim/humanoid/amp_world.py", 24834,
                       "1b63a4c8ac7d4bf2f9ed523edc97b41dca89888cdd6c7dcb2234826d240d7f83", False),
}
WORLD_METHOD_AST_PINS = {
    "prepare_control": "0506a5e39ce2ea249af26ab8d98ffaeff652721e04253889197cfc3296c22c16",
    "step": "57fb3d8c04b43d3b54a67f923ea38bb8364127f4d7c5e0fa0c9bf9bb950f8c8b",
    "complete_transition": "699b3216e1d7b9d5397c8427138c2ad59c885e0378f42372a0bd7291267dfd73",
}
PLANAR_AST_SHA = "9dd2278c79b88c733cf0a3f4d63ab58298a6c2abd288b702753ecc8bb7aecf2d"
_DEPS = None


def _dependencies():
    global _DEPS
    if _DEPS is None:
        sources = {}
        for key, (path, size, sha, readonly) in PINS.items():
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or readonly and info.st_mode & 0o222:
                raise ValueError("Source must be an ordinary reviewed single-link file: " + key)
            data = path.read_bytes()
            if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
                raise ValueError("Reviewed source identity changed: " + key)
            sources[key] = data
        cls = next(node for node in ast.parse(sources["world"]).body
                   if isinstance(node, ast.ClassDef) and node.name == "SharedGroupWorld")
        methods = {node.name: node for node in cls.body if isinstance(node, ast.FunctionDef)}
        for name, expected in WORLD_METHOD_AST_PINS.items():
            if hashlib.sha256(ast.dump(methods[name], include_attributes=False).encode()).hexdigest() != expected:
                raise ValueError("Reviewed World method contract changed: " + name)
        modules = {}
        # Direct whole-source loads bypass training.__init__ and its NN imports.
        for key in ("rewards", "contracts"):
            module = ModuleType("_social_rollout_" + key)
            module.__file__ = str(PINS[key][0])
            exec(compile(sources[key], module.__file__, "exec"), module.__dict__)
            modules[key] = module
        original = next(node for node in ast.parse(sources["original_world"]).body
                        if isinstance(node, ast.ClassDef) and node.name == "AMPWorld")
        planar = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == "planar")
        if hashlib.sha256(ast.dump(planar, include_attributes=False).encode()).hexdigest() != PLANAR_AST_SHA:
            raise ValueError("Original physical-quaternion planar formula changed")
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=[planar], type_ignores=[]), str(PINS["original_world"][0]), "exec"), namespace)
        _DEPS = (modules["rewards"], modules["contracts"], namespace["planar"])
    return _DEPS


def _class_origin(value, name, key):
    cls = type(value)
    module = sys.modules.get(cls.__module__)
    if (cls.__name__ != name or getattr(module, name, None) is not cls
            or Path(getattr(module, "__file__", "")).resolve() != PINS[key][0]):
        raise TypeError("An already constructed exact reviewed " + name + " is required")


def _tensor(value, shape, dtype, device, label):
    if (not isinstance(value, torch.Tensor) or tuple(value.shape) != tuple(shape)
            or value.dtype != dtype or value.device != device
            or value.is_floating_point() and not bool(torch.isfinite(value).all())):
        raise ValueError(label + " must have the declared shape/dtype/device and finite values")
    return value


def _owned(values):
    return {name: value.detach().clone() for name, value in values.items()}


class SocialRolloutCollector:
    """Single-thread exclusive collection; call trainer.update only after return.

    Freeze every RMS/counter at each key boundary. Compare a full owned model
    baseline at completion and around external callbacks; no Tensor._version
    heuristic. Normal act/value/amp_reward use the exact no-update source above.
    Exact guards synchronize devices and cost memory; no speed claim is made.
    """

    def __init__(self, world, trainer):
        self.reward_api, self.contracts, self.original_planar = _dependencies()
        _class_origin(world, "SharedGroupWorld", "world")
        _class_origin(trainer, "SocialAMPTrainer", "trainer")
        trainer.config.validate()
        if (world.device != trainer.device or world.num_envs != trainer.config.actor_count
                or world.include_ego != trainer.config.include_ego
                or trainer.config.discriminator_input != "state_action"):
            raise ValueError("World/trainer device, flat actor count, include_ego and state_action must agree")
        declared_geometry = {name: getattr(trainer.config, name)
                             for name in ("grid_center", "grid_heading", "height_reference")}
        if dict(world.grid_geometry_contract) != declared_geometry:
            raise ValueError("World grid geometry declaration differs from the saved training configuration")
        if world._faulted or world._pending is not None or world._prepared is not None:
            raise RuntimeError("Collector requires an idle nonfaulted World")
        self.world, self.trainer = world, trainer
        self.device, self.actors = world.device, world.num_envs
        self._faulted = self._collecting = False
        self.failed_steps = None  # Partial local diagnostic data, never a successful rollout.

    def _freeze(self):
        trainer = self.trainer
        normalizers = {}
        dimensions = {"self_normalizer": 357, "task_normalizer": 12308, "value_normalizer": 1,
                      "amp_normalizer": 1950 if trainer.config.amp_normalization == "per_history_coordinate" else 195}
        for name, dimension in dimensions.items():
            normalizer = getattr(trainer, name)
            buffers = dict(normalizer.named_buffers())
            if set(buffers) != {"mean", "variance", "count"}:
                raise ValueError("Expected complete original RMS buffers: " + name)
            for field, value in buffers.items():
                _tensor(value, () if field == "count" else (dimension,), torch.float64, self.device, name + "." + field)
            normalizers[name] = (normalizer, _owned(buffers), normalizer.clip)
        counters = trainer.update_count, trainer.physical_transition_count
        if any(type(value) is not int or value < 0 for value in counters):
            raise ValueError("Training counters must be nonnegative integers")
        if any(float(normalizers[name][1]["count"]) != counters[1]
               for name in ("self_normalizer", "task_normalizer", "value_normalizer")):
            raise ValueError("Fresh social self/task/value RMS counts must equal actual training transitions")
        mask = _tensor(trainer.action_mask, (69,), torch.float32, self.device, "action mask")
        if not bool(((mask == 0) | (mask == 1)).all()) or int(mask.sum()) != 57:
            raise ValueError("The exact neutral57 action projection is required")
        model = {}
        for kind, values in (("parameters", dict(trainer.model.named_parameters())),
                             ("buffers", dict(trainer.model.named_buffers()))):
            model[kind] = {name: (value, value.detach().clone()) for name, value in values.items()}
        modes = {name: (module, module.training) for name, module in trainer.model.named_modules()}
        return {"normalizers": normalizers, "counters": counters, "mask": mask.clone(), "model": model,
                "modes": modes, "configuration": trainer.config.to_document(),
                "grid_geometry_contract": dict(self.world.grid_geometry_contract)}

    def _guard_rms(self, frozen):
        trainer = self.trainer
        if ((trainer.update_count, trainer.physical_transition_count) != frozen["counters"]
                or trainer.config.to_document() != frozen["configuration"]
                or not torch.equal(trainer.action_mask, frozen["mask"])
                or dict(self.world.grid_geometry_contract) != frozen["grid_geometry_contract"]
                or self.world.include_ego != trainer.config.include_ego):
            raise RuntimeError("Trainer counters/configuration/action mask changed during collection")
        for name, (normalizer, expected, clip) in frozen["normalizers"].items():
            current = getattr(trainer, name)
            actual = dict(current.named_buffers())
            if (current is not normalizer or current.clip != clip or set(actual) != set(expected)
                    or any(value.dtype != actual[key].dtype or value.device != actual[key].device
                           or value.shape != actual[key].shape or not torch.equal(value, actual[key])
                           for key, value in expected.items())):
                raise RuntimeError("Complete RMS buffers changed during rollout: " + name)

    def _guard_model(self, frozen):
        self._guard_rms(frozen)
        for kind, actual in (("parameters", dict(self.trainer.model.named_parameters())),
                             ("buffers", dict(self.trainer.model.named_buffers()))):
            expected = frozen["model"][kind]
            if (set(actual) != set(expected)
                    or any(actual[key] is not original or actual[key].dtype != value.dtype
                           or actual[key].device != value.device or actual[key].shape != value.shape
                           or not torch.equal(actual[key], value)
                           for key, (original, value) in expected.items())):
                raise RuntimeError("Model " + kind + " changed during exclusive collection")
        modes = dict(self.trainer.model.named_modules())
        if (set(modes) != set(frozen["modes"])
                or any(modes[key] is not module or modes[key].training != training
                       for key, (module, training) in frozen["modes"].items())):
            raise RuntimeError("Model module identity/mode changed during collection")

    def _observation(self, item):
        shapes = {"self_obs": (357,), "task_obs": (12308,), "amp_history": (10, 195),
                  "history_episode_ids": (10,), "episode_ids": (), "age": (), "plan_age": ()}
        result = {}
        for key, suffix in shapes.items():
            dtype = torch.float32 if key in {"self_obs", "task_obs", "amp_history"} else torch.int64
            result[key] = _tensor(getattr(item, key), (self.actors, *suffix), dtype, self.device, key)
        if (bool((result["history_episode_ids"] != result["episode_ids"][:, None]).any())
                or bool((result["age"] < 0).any()) or bool((result["plan_age"] < 0).any())):
            raise ValueError("AMP history cannot cross resets and episode ages must be nonnegative")
        if any(type(getattr(item, key)) is not int or getattr(item, key) < 0 for key in ("control_step", "physics_step")):
            raise ValueError("Actual monotonic World control/physics stamps are required")
        return _owned(result)

    def _pending_guard(self, pending, pre, saved):
        world = self.world
        if (world._pending is not pending or pending.pre is not pre
                or pending.control_step_before != pre.control_step
                or pending.control_step_after != pre.control_step + 1
                or pending.physics_step_before != pre.physics_step
                or pending.physics_step_after != pre.physics_step + 4
                or world.control_step != pending.control_step_after
                or world._physics_step != pending.physics_step_after
                or world._counter() != pending.physics_step_after):
            raise RuntimeError("Pending transition no longer has the unique four-substep identity")
        now = self._observation(pending.pre)
        if any(not torch.equal(now[key], value) for key, value in saved.items()):
            raise RuntimeError("Archived pre-control observations/history/episodes changed")
        post = self._observation(pending.post)
        if (pending.post.control_step != pending.control_step_after or pending.post.physics_step != pending.physics_step_after
                or not torch.equal(post["episode_ids"], saved["episode_ids"])
                or not torch.equal(world.episode_ids, saved["episode_ids"])
                or not torch.equal(post["age"], saved["age"] + 1)
                or not torch.equal(post["plan_age"], saved["plan_age"] + 1)
                or not torch.equal(world.age, post["age"])
                or not torch.equal(world.plan_age, post["plan_age"])
                or not torch.equal(world.amp_history, post["amp_history"])
                or not torch.equal(world.amp_history_episode_ids, post["history_episode_ids"])):
            raise RuntimeError("Original-episode terminal observations/history were changed or reset early")
        return post

    @torch.no_grad()
    def collect(self, controls, *, external_archive=None):
        """Return {rollout, diagnostics, metadata}; caller supplies expert/update.

        Optional external_archive receives independent {rollout,diagnostics}
        step dictionaries inside World.complete_transition, before any reset.
        Its return is not independent evidence of durable external storage.
        Any collection failure preserves partial local steps, propagates and
        faults this collector and World; no partial success or retry is returned.
        """
        if type(controls) is not int or controls < 1 or external_archive is not None and not callable(external_archive):
            raise ValueError("Positive controls and an optional callable external archive are required")
        if self._faulted or self._collecting or self.world._faulted or self.world._pending is not None or self.world._prepared is not None:
            raise RuntimeError("Collector and World must be exclusive, idle and nonfaulted")
        self.trainer.config.validate()
        if (self.world.device != self.device or self.trainer.device != self.device
                or self.world.num_envs != self.actors or self.trainer.config.actor_count != self.actors
                or self.world.include_ego != self.trainer.config.include_ego
                or dict(self.world.grid_geometry_contract) != {
                    name: getattr(self.trainer.config, name) for name in ("grid_center", "grid_heading", "height_reference")}):
            raise RuntimeError("Collector's original World/trainer dimensions and device must remain unchanged")
        frozen = self._freeze()
        self._collecting = True
        training_steps, diagnostic_steps = [], []
        start = self.world.control_step, self.world._physics_step
        try:
            for _ in range(controls):
                self._guard_rms(frozen)
                pre = self.world.prepare_control()
                saved = self._observation(pre)
                self._guard_rms(frozen)
                action = self.trainer.act(pre.self_obs, pre.task_obs, deterministic=False)
                action = _owned({key: _tensor(action[key], (self.actors, 69) if key in {"actions", "applied_actions", "mean_actions"} else (self.actors,),
                                               torch.float32, self.device, key)
                                 for key in ("actions", "applied_actions", "mean_actions", "log_prob", "values")})
                self._guard_rms(frozen)
                if not torch.equal(action["applied_actions"], action["actions"].clamp(-1, 1) * frozen["mask"]):
                    raise ValueError("Gaussian and applied57 action projection must remain separate and exact")
                # Eq2 state_action uses h[t] and applied a[t], never the post history.
                amp_reward = _tensor(self.trainer.amp_reward(saved["amp_history"], action["applied_actions"]),
                                     (self.actors,), torch.float32, self.device, "pre-state_action AMP reward").clone()
                self._guard_rms(frozen)
                pending = self.world.step(action["applied_actions"])
                post = self._pending_guard(pending, pre, saved)
                self._guard_rms(frozen)
                next_values = _tensor(self.trainer.value(pending.post.self_obs, pending.post.task_obs),
                                      (self.actors,), torch.float32, self.device, "all-actor terminal next values").clone()
                self._guard_rms(frozen)
                self._pending_guard(pending, pre, saved)
                state = pending.snapshot.robot_state
                position = _tensor(state.root_pos, (self.actors, 3), torch.float32, self.device, "actual Pelvis position")
                velocity = _tensor(state.root_vel, (self.actors, 3), torch.float32, self.device, "actual Pelvis link velocity")
                quat = _tensor(state.root_rot, (self.actors, 4), torch.float32, self.device, "actual Pelvis XYZW")
                target = _tensor(pending.target_xy, (self.actors, 2), torch.float32, self.device, "aligned post-control reference")
                efforts = _tensor(state.dof_forces, (self.actors, 69), torch.float32, self.device, "last measured generalized effort")
                dof_velocity = _tensor(state.dof_vel, (self.actors, 69), torch.float32, self.device, "matching last DOF velocity")
                power = _tensor(pending.signed_power_w, (self.actors,), torch.float32, self.device, "four-substep mean signed power")
                reward_yaw = self.original_planar(None, state)[:, 4]
                forward = velocity[:, 0] * reward_yaw.cos() + velocity[:, 1] * reward_yaw.sin()
                components = self.reward_api.paper_task_reward((position[:, :2] - target).square().sum(-1), efforts,
                    dof_velocity, forward, direction_mode=self.trainer.config.direction_mode, signed_power_w=power)
                reward = self.reward_api.paper_total_reward(amp_reward, components,
                    amp_weight=self.trainer.config.amp_reward_weight, task_weight=self.trainer.config.task_reward_weight)
                x, y, z, w = quat.unbind(-1)
                reward_yaw_defined = torch.hypot(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square())) > 1e-12

                def archive(transition, supplied_next_values):
                    if transition is not pending or not torch.equal(supplied_next_values, next_values):
                        raise RuntimeError("Archive must consume this unique terminal value/transition")
                    self._guard_rms(frozen)
                    post_now = self._pending_guard(pending, pre, saved)
                    if any(not torch.equal(post_now[key], value) for key, value in post.items()):
                        raise RuntimeError("Terminal observation changed between critic and archive")
                    if not torch.equal(pending.applied_actions, action["applied_actions"]):
                        raise RuntimeError("Physics applied actions differ from the AMP/PPO action")
                    fields = {"self_obs": saved["self_obs"], "task_obs": saved["task_obs"], "amp_history": saved["amp_history"],
                        "episode_ids": saved["episode_ids"], "history_episode_ids": saved["history_episode_ids"],
                        **{key: action[key] for key in ("actions", "applied_actions", "log_prob", "values")},
                        "next_values": supplied_next_values, "rewards": reward, "amp_rewards": amp_reward,
                        "trajectory_rewards": components["trajectory"], "energy_rewards": components["energy"],
                        "direction_rewards": components["direction"], "signed_power_w": power, "forward_velocity": forward,
                        "terminated": pending.terminated, "truncated": pending.truncated}
                    for key in ("terminated", "truncated"):
                        _tensor(fields[key], (self.actors,), torch.bool, self.device, key)
                    for key in ("rewards", "trajectory_rewards", "energy_rewards", "direction_rewards", "forward_velocity"):
                        _tensor(fields[key], (self.actors,), torch.float32, self.device, key)
                    diagnostic = {"next_self_obs": post["self_obs"], "next_task_obs": post["task_obs"],
                        "post_amp_history": post["amp_history"], "post_history_episode_ids": post["history_episode_ids"],
                        "post_episode_ids": post["episode_ids"], "pre_age": saved["age"], "post_age": post["age"],
                        "pre_plan_age": saved["plan_age"], "post_plan_age": post["plan_age"], "mean_actions": action["mean_actions"],
                        "target_xy": target, "software_reward_yaw": reward_yaw, "software_reward_yaw_defined": reward_yaw_defined,
                        "actual_root_quat_xyzw": pending.actual_root_quat_xyzw, "used_task_heading": pending.used_task_heading,
                        "task_heading_defined": pending.task_heading_defined, "singular_heading_fallback": pending.singular_heading_fallback,
                        "reset_mask": pending.reset_mask, "physical_fall": pending.physical_fall,
                        "nonfoot_normal_contact": pending.nonfoot_normal_contact.contact_flag,
                        "nonfoot_resultant_normal_force_w": pending.nonfoot_normal_contact.resultant_force_w,
                        "nonfoot_resultant_magnitude_n": pending.nonfoot_normal_contact.magnitude_n,
                        "source_end_truncated": pending.source_end_truncated, "cap_truncated": pending.cap_truncated,
                        "collateral_truncated": pending.collateral_truncated}
                    for prefix, snapshot, names in (
                        ("common_", pending.snapshot.robot_state, ("rigid_body_pos", "rigid_body_rot", "rigid_body_vel", "rigid_body_ang_vel", "dof_pos", "dof_vel", "dof_forces")),
                        ("native_root_link_", pending.snapshot.root_state, ("root_pos", "root_rot", "root_vel", "root_ang_vel"))):
                        diagnostic.update({prefix + name: getattr(snapshot, name) for name in names})
                    for name in ("control_step_before", "control_step_after", "physics_step_before", "physics_step_after"):
                        diagnostic[name] = torch.full((self.actors,), getattr(pending, name), dtype=torch.int64, device=self.device)
                    # Full owned step exists locally before any optional external callback/reset.
                    training_steps.append(_owned(fields))
                    diagnostic_steps.append(_owned(diagnostic))
                    if external_archive is not None:
                        self._guard_model(frozen)
                        external_archive({"rollout": _owned(training_steps[-1]), "diagnostics": _owned(diagnostic_steps[-1])})
                        self._guard_model(frozen)
                        after = self._pending_guard(pending, pre, saved)
                        if any(not torch.equal(after[key], value) for key, value in post.items()):
                            raise RuntimeError("External archive changed terminal observations")
                    self._guard_rms(frozen)

                self.world.complete_transition(next_values, archive)
                self._guard_rms(frozen)
            self._guard_model(frozen)
            rollout = {key: torch.stack([step[key] for step in training_steps]) for key in training_steps[0]}
            diagnostics = {key: torch.stack([step[key] for step in diagnostic_steps]) for key in diagnostic_steps[0]}
            self.contracts.validate_rollout(rollout, self_dim=357, task_dim=12308, history_frames=10, amp_frame_dim=195,
                                            action_mask=frozen["mask"])
            if (self.world.control_step != start[0] + controls or self.world._physics_step != start[1] + 4 * controls
                    or len(training_steps) != controls or self.world._pending is not None or self.world._prepared is not None):
                raise RuntimeError("Completed rollout lost a control or retained an unconsumed transition")
            return {"rollout": rollout, "diagnostics": diagnostics,
                    "metadata": {"schema": "local.neutral_smpl.social_rollout_collection.v1", "controls": controls,
                        "actors": self.actors, "control_step_start": start[0], "physics_step_start": start[1],
                        "control_step_end": self.world.control_step, "physics_step_end": self.world._physics_step,
                        "training_update_count": frozen["counters"][0], "training_physical_transition_count": frozen["counters"][1],
                        "configuration": frozen["configuration"],
                        "grid_geometry_contract": dict(frozen["grid_geometry_contract"]),
                        "grid_callback_semantics_independently_verified": False,
                        "frozen_rms": {name: _owned(values) for name, (_, values, _) in frozen["normalizers"].items()},
                        "external_archive_returned_controls": controls if external_archive else 0,
                        "archive_independently_verified": False, "SDK_checkpoint_recovery_implemented": False,
                        "power_quantity": "four-substep mean signed generalized projected joint effort times synchronized DOF velocity, W",
                        "reward_heading": "exact original AMPWorld.planar actual XYZW atan2; undefined software result retained separately",
                        "task_heading": "World explicit fallen-singular per-episode continuation; actual quaternion unchanged",
                        "terminal_bootstrap": "unchanged finite next_values; trainer GAE masks true terminated only",
                        "exclusive_collection": True}}
        except BaseException:
            self._faulted = self.world._faulted = True
            self.failed_steps = {"rollout_steps": training_steps, "diagnostic_steps": diagnostic_steps}
            raise
        finally:
            self._collecting = False
