"""Direct IsaacLab SMPL physics for Diffusion + AMP, without MaskedMimic.

Import this module after launching Isaac Sim. Public PACER defines observation
and PD action conventions; current ProtoMotions provides named SMPL physics.
The physical rig and solver are recorded as reconstruction assumptions.
"""
from __future__ import annotations

from dataclasses import asdict
import math

import torch

from CrowdSim.humanoid.amp import build_amp_observation, build_self_observation, build_task_observation, action_to_pd
from CrowdSim.humanoid.amp.observations import PacerSMPLMapping
from protomotions.components.scene_lib import SceneLib, SceneLibConfig
from protomotions.components.terrains.config import TerrainConfig
from protomotions.components.terrains.terrain import Terrain
from protomotions.robot_configs.smpl import SmplRobotConfig
from protomotions.simulator.base_simulator.simulator_state import ResetState, RobotState, StateConversion
from protomotions.simulator.isaaclab.config import IsaacLabSimulatorConfig
from protomotions.simulator.isaaclab.simulator import IsaacLabSimulator


class PaperPowerSimulator(IsaacLabSimulator):
    """Measure projected active joint effort after each PhysX substep.

    The public tensor API reports the previous simulation step's joint wrench
    projected onto each DOF. This is measured generalized joint effort, not an
    isolated motor-torque channel. ``applied_torque`` from ImplicitActuator is
    instead a pre-step PD estimate and must not be called measured torque.
    """

    power_source = "projected_joint_effort_substep_mean"
    mean_projected_joint_power_w = None

    def _projected_effort_native(self):
        import warp as wp
        return wp.to_torch(self._robot.root_view.get_dof_projected_joint_forces()).clone()

    def _physics_step(self):
        monitor = getattr(self, "pedestrian_contacts", None)
        if monitor is not None:
            monitor.begin_control_step()
        if self.power_source == "legacy_pd_proxy_misaligned":
            if monitor is not None:
                raise ValueError("Substep contact reporting requires the measured-power physics loop")
            return super()._physics_step()
        import warp as wp
        power = torch.zeros(self.num_envs, device=self.device)
        for idx in range(self.decimation):
            self._apply_control()
            self._scene.write_data_to_sim()
            self._sim.step(render=False)
            if (idx + 1) % self.decimation == 0 and not self.headless:
                self._sim.render()
            self._scene.update(dt=self._sim.get_physics_dt())
            if monitor is not None:
                monitor.sample(self._sim.get_physics_dt())
            effort = self._projected_effort_native()
            velocity = wp.to_torch(self._robot.data.joint_vel)
            power += (effort * velocity).sum(-1)
        # Equal 120 Hz substeps: time average in watts, not work in joules.
        self.mean_projected_joint_power_w = power / self.decimation

    def _get_simulator_dof_forces(self, env_ids=None):
        if self.power_source == "legacy_pd_proxy_misaligned":
            return super()._get_simulator_dof_forces(env_ids)
        effort = self._projected_effort_native()
        if env_ids is not None:
            effort = effort[env_ids]
        return RobotState(dof_forces=effort, state_conversion=StateConversion.SIMULATOR)


class AMPWorld:
    def __init__(self, app, corpus, *, num_envs=64, device="cuda:0", shared_world=False,
                 history_frames=10, episode_steps=300, scene_description=None,
                 power_source="projected_joint_effort_substep_mean", terrain_extent_m=64.,
                 contact_reporting=False, spawn_margin_m=6., amass_root_training=False, amass_root_tail_policy="hold"):
        if type(amass_root_training) is not bool or (amass_root_training and (shared_world or scene_description is not None or history_frames != 10)):
            raise ValueError("AMASS root curriculum is an explicit isolated training-only source")
        if power_source not in {"projected_joint_effort_substep_mean", "legacy_pd_proxy_misaligned"}:
            raise ValueError("Unknown joint power measurement source")
        self.power_source = power_source
        if not math.isfinite(terrain_extent_m) or terrain_extent_m < 32:
            raise ValueError("Terrain extent must be finite and at least 32 metres")
        self.terrain_extent_m = float(terrain_extent_m)
        if not math.isfinite(spawn_margin_m) or spawn_margin_m < 0:
            raise ValueError("Spawn margin must be finite and nonnegative")
        self.device = torch.device(device)
        self.corpus = corpus
        self.expert_reference_sampler = None
        self.num_envs = num_envs
        self.history_frames = history_frames
        self.episode_steps = episode_steps
        self.shared_world = shared_world
        self.validation_scene = None
        if scene_description is not None:
            from CrowdSim.humanoid.reconstruction_scene import ReconstructionScene
            self.validation_scene = ReconstructionScene(scene_description, device=self.device)
            if len(self.validation_scene.origins) != num_envs or not shared_world:
                raise ValueError("Declared shared scene requires one spawn/route per actor and shared physics")
        self.robot_config = SmplRobotConfig()
        if contact_reporting:
            # ContactReportAPI must be authored by the USD spawner before
            # PhysX initialization; adding tensor views afterward is too late.
            self.robot_config.contact_bodies = list(self.robot_config.kinematic_info.body_names)
        self.mapping = PacerSMPLMapping(tuple(self.robot_config.kinematic_info.body_names), tuple(self.robot_config.kinematic_info.dof_names))
        config = IsaacLabSimulatorConfig(num_envs=num_envs, headless=True,
                                        experiment_name="diffusion_amp_reconstruction")
        config.filter_collisions = not shared_world
        config.sim.fps = 120
        config.sim.decimation = 4
        config.sim.physx.num_position_iterations = 4
        config.sim.physx.num_velocity_iterations = 4
        self.dt = 1/30
        terrain_config = TerrainConfig(map_length=self.terrain_extent_m, map_width=self.terrain_extent_m, border_size=5,
                                       num_levels=1, num_terrains=1, spacing_between_scenes=4)
        self.terrain = Terrain(terrain_config, num_envs, self.device)
        scenes = SceneLib(SceneLibConfig(), num_envs, device=self.device, terrain=self.terrain)
        self.simulator = PaperPowerSimulator(config, self.robot_config, self.terrain,
                                          device=self.device, simulation_app=app, scene_lib=scenes)
        self.simulator.power_source = power_source
        if self.validation_scene is not None:
            import isaaclab.sim as sim_utils
            for index, box in enumerate(scene_description.get("cuboids", [])):
                shape = sim_utils.CuboidCfg(size=tuple(box["size"]),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.002, rest_offset=0.),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.2, .3, .4)),
                    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=1., restitution=0.))
                sim_utils.spawn_cuboid(f"/World/PaperValidation/Obstacle_{index}", shape,
                                      translation=tuple(box["center"]))
        self.simulator._initialize_with_markers(None)
        self.lower = self.robot_config.kinematic_info.dof_limits_lower.to(self.device)
        self.upper = self.robot_config.kinematic_info.dof_limits_upper.to(self.device)
        side = math.ceil(math.sqrt(num_envs))
        ids = torch.arange(num_envs, device=self.device)
        self.origins = torch.stack((spawn_margin_m + (ids // side) * 4, spawn_margin_m + (ids % side) * 4), -1).float()
        if scene_description is None and float(self.origins.max()) >= self.terrain_extent_m:
            raise ValueError("Isolated spawn grid exceeds the declared terrain extent")
        if self.validation_scene is not None:
            self.origins = self.validation_scene.origins.clone()
        self.age = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.episode_ids = torch.zeros_like(self.age)
        self.amp_history = torch.zeros(num_envs, history_frames, 195, device=self.device)
        self.amp_history_episode_ids = torch.zeros(num_envs, history_frames, dtype=torch.long, device=self.device)
        self.planar_history = torch.zeros(num_envs, 91, 5, device=self.device)
        self.history_available = torch.zeros(num_envs, 91, dtype=torch.bool, device=self.device)
        self.reference_ids = torch.zeros_like(self.age)
        self.amass_root_curriculum = None
        if amass_root_training:
            from CrowdSim.humanoid.training.amass_trajectories import AMASSRootTrajectoryCurriculum
            self.amass_root_curriculum = AMASSRootTrajectoryCurriculum(corpus, num_envs, device=self.device,
                bounds_xy=(0., 0., self.terrain_extent_m, self.terrain_extent_m), tail_policy=amass_root_tail_policy)
        self.reset(torch.arange(num_envs, device=self.device))

    def height_at(self, xy):
        ground = self.ground_height_at(xy)
        return ground if self.validation_scene is None else torch.maximum(ground, self.validation_scene.obstacle_height(xy))

    def ground_height_at(self, xy):
        shape = xy.shape[:-1]
        return self.terrain.get_ground_heights(xy.reshape(-1, 2)).reshape(shape)

    def state(self):
        return self.simulator.get_robot_state()

    def planar(self, state=None):
        state = self.state() if state is None else state
        q = state.root_rot
        yaw = torch.atan2(2*(q[:, 3]*q[:, 2]+q[:, 0]*q[:, 1]), 1-2*(q[:, 1].square()+q[:, 2].square()))
        return torch.cat((state.root_pos[:, :2], state.root_vel[:, :2], yaw[:, None]), -1)

    def reset(self, ids):
        if not len(ids):
            return
        curriculum = getattr(self, "amass_root_curriculum", None)
        if curriculum is None:
            reference, history, _, motion_ids, frames = self.corpus.sample_reference(len(ids), self.history_frames)
        else:
            reference, history, _, motion_ids, frames = self.corpus.sample_reference(len(ids), self.history_frames,
                future_frames=curriculum.FUTURE_FRAMES)
        positions = reference.root_pos.clone()
        positions[:, :2] = self.origins[ids]
        positions[:, 2] += self.ground_height_at(positions[:, :2])
        self.simulator.reset_envs(ResetState(root_pos=positions, root_rot=reference.root_rot.clone(),
            root_vel=reference.root_vel.clone(), root_ang_vel=reference.root_ang_vel.clone(),
            dof_pos=reference.dof_pos.clone(), dof_vel=reference.dof_vel.clone(),
            state_conversion=StateConversion.COMMON), env_ids=ids)
        # Physics manager refreshes FK at reset; no extra simulation step.
        actual = self.state()
        self.amp_history[ids] = build_amp_observation(history, mapping=self.mapping).reshape(len(ids), self.history_frames, 195)
        self.amp_history[ids, 0] = build_amp_observation(actual, mapping=self.mapping)[ids]
        planar = self.planar(actual)[ids]
        self.planar_history[ids] = planar[:, None]
        self.history_available[ids] = False
        self.history_available[ids, -1] = True
        self.age[ids] = 0
        self.episode_ids[ids] += 1
        self.amp_history_episode_ids[ids] = self.episode_ids[ids, None]
        self.reference_ids[ids] = motion_ids.to(self.device)
        if curriculum is not None:
            curriculum.bind_reset(ids, motion_ids, frames, reference.root_pos[:, :2], positions[:, :2], self.episode_ids[ids])

    def observations(self, trajectory, state=None):
        state = self.state() if state is None else state
        return (build_self_observation(state, mapping=self.mapping),
                build_task_observation(state, trajectory, self, mapping=self.mapping))

    def expert_batch(self, batch):
        source = getattr(self, "expert_reference_sampler", None) or self.corpus
        _, history, next_position, _, _ = source.sample_reference(batch, self.history_frames)
        features = build_amp_observation(history, mapping=self.mapping).reshape(batch, self.history_frames, 195)
        scale = action_to_pd(torch.ones(batch, 69, device=self.device), self.lower, self.upper,
                             dof_names=self.mapping.dof_names)
        # For excluded hand/toe joints, action_to_pd always emits zero. Those
        # expert actions must also be zero, rather than divide by zero.
        scale = scale[:, self.mapping.dof_indices]
        next_position = next_position[:, self.mapping.dof_indices]
        raw = torch.where(scale.abs() > 1e-8, next_position / scale.clamp_min(1e-8), torch.zeros_like(next_position)).clamp(-1, 1)
        return features, raw

    def configure_expert_sampling(self, config):
        """Install an explicit training-only prior after the initial reset."""
        if getattr(self, "expert_reference_sampler", None) is not None:
            raise ValueError("Expert sampling is already configured; do not silently reseed it")
        if config is None:
            return None
        from CrowdSim.humanoid.training.expert_sampling import ExpertReferenceSampler
        self.expert_reference_sampler = ExpertReferenceSampler(self.corpus, config, history_frames=self.history_frames)
        return self.expert_reference_sampler.describe()

    def expert_sampling_state_dict(self):
        sampler = getattr(self, "expert_reference_sampler", None)
        return None if sampler is None else sampler.state_dict()

    def load_expert_sampling_state_dict(self, state):
        sampler = getattr(self, "expert_reference_sampler", None)
        if sampler is None:
            if state is not None:
                raise ValueError("A disabled expert sampler cannot resume enabled sampling state")
            return
        if state is None:
            raise ValueError("Enabled expert sampling resume requires complete saved state; no reseeding fallback")
        sampler.load_state_dict(state)

    def step(self, applied_actions):
        targets = action_to_pd(applied_actions, self.lower, self.upper, dof_names=self.mapping.dof_names)
        self.simulator.step(targets)
        self.age += 1
        state = self.state()
        if not all(torch.isfinite(value).all() for value in (state.root_pos, state.root_rot, state.dof_pos, state.dof_vel, state.dof_forces)):
            raise FloatingPointError("Nonfinite physical SMPL state")
        self.amp_history = torch.cat((build_amp_observation(state, mapping=self.mapping)[:, None], self.amp_history[:, :-1]), 1)
        self.amp_history_episode_ids = torch.cat((self.episode_ids[:, None], self.amp_history_episode_ids[:, :-1]), 1)
        self.planar_history = torch.cat((self.planar_history[:, 1:], self.planar(state)[:, None]), 1)
        self.history_available = torch.cat((self.history_available[:, 1:], torch.ones(self.num_envs, 1, dtype=torch.bool, device=self.device)), 1)
        ground = self.ground_height_at(state.root_pos[:, :2])
        q = state.root_rot
        up_z = 1 - 2*q[:, :2].square().sum(-1)
        terminated = ((state.root_pos[:, 2] - ground) < .45) | (up_z < .1)
        truncated = self.age >= self.episode_steps
        curriculum = getattr(self, "amass_root_curriculum", None)
        if curriculum is not None:
            truncated = truncated | curriculum.completed_step(self.age, terminated, truncated)
        return state, terminated, truncated

    def measured_signed_power_w(self):
        """Four synchronized solver-effort/end-velocity samples per control step."""
        if self.power_source == "legacy_pd_proxy_misaligned":
            return None
        power = self.simulator.mean_projected_joint_power_w
        if power is None or not torch.isfinite(power).all():
            raise FloatingPointError("Missing/nonfinite projected joint-power measurement")
        return power

    def enable_pairwise_contacts(self, *, threshold_n=1., batched_sensor_bodies=True):
        if self.robot_config.contact_bodies is None:
            raise ValueError("Create AMPWorld(contact_reporting=True) before enabling tensor contact views")
        from CrowdSim.humanoid.contact_monitor import PedestrianContactMonitor
        self.simulator.pedestrian_contacts = PedestrianContactMonitor(self, threshold_n=threshold_n,
            batched_sensor_bodies=batched_sensor_bodies)
        return self.simulator.pedestrian_contacts.manifest

    def planner_inputs(self, ids=None):
        if ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        history = self.planar_history[ids, ::3]
        available = self.history_available[ids, ::3]
        if self.shared_world and self.num_envs > 1:
            others = torch.stack([torch.cat((torch.arange(int(i), device=self.device), torch.arange(int(i)+1, self.num_envs, device=self.device))) for i in ids])
            neighbors = self.planar_history[others, ::3]
            neighbor_mask = self.history_available[others, ::3]
            distance = torch.linalg.vector_norm(neighbors[:, :, -1, :2]-history[:, None, -1, :2], dim=-1)
            neighbor_mask = neighbor_mask & (distance <= 15.)[..., None]
        else:
            neighbors = torch.zeros(len(ids), 1, 31, 5, device=self.device)
            neighbor_mask = torch.zeros(len(ids), 1, 31, dtype=torch.bool, device=self.device)
        raster = torch.zeros(len(ids), 2, 224, 224, device=self.device)
        raster[:, 0] = 1  # This environment is explicitly a flat unobstructed plane.
        if self.validation_scene is not None:
            raster = self.validation_scene.raster(history[:, -1])
        return {"states": history[:, -1], "history": history, "neighbors": neighbors,
                "raster": raster, "history_mask": available, "neighbor_mask": neighbor_mask}

    def physics_manifest(self):
        import warp as wp
        result = {"engine": "PhysX", "physics_hz": 120, "control_hz": 30,
                  "isolated_environments": not self.shared_world,
                  "fall_definition": {"pelvis_ground_height_less_than_m": .45,
                                      "root_up_z_less_than": .1},
                  "rig": "current ProtoMotions neutral SMPL MJCF/USD",
                  "body_names": list(self.mapping.body_names), "dof_names": list(self.mapping.dof_names),
                  "dof_limits_lower": self.lower.cpu().tolist(), "dof_limits_upper": self.upper.cpu().tolist(),
                  "joint_control": {key: asdict(value) for key, value in self.robot_config.control.control_info.items()},
                  "terrain_material": asdict(self.terrain.sim_config), "author_physics_settings": False}
        result["terrain_extent_m"] = self.terrain_extent_m
        result["isolated_spawn_margin_m"] = None if self.validation_scene is not None else float(self.origins.min())
        result["scene_geometry"] = None if self.validation_scene is None else self.validation_scene.description
        # Native live PhysX mass readings, not authored USD transforms.
        masses = wp.to_torch(self.simulator._robot.root_view.get_masses()).cpu()
        result["terrain_material"]["combine_mode"] = self.terrain.sim_config.combine_mode.value
        result["live_native_body_names"] = list(self.simulator._robot.body_names)
        result["live_native_body_masses_kg"] = masses[0].tolist()
        indices = self.simulator.data_conversion.body_convert_to_common.cpu()
        result["live_body_masses_kg"] = masses[0, indices].tolist()
        result["live_total_mass_kg"] = float(masses[0].sum())
        return result

    def close(self):
        self.simulator.close()
