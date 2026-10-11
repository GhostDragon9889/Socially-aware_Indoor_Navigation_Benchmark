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
                provider=getattr(self,"scene_provider",None)
                if provider is not None:
                    from CrowdSim.humanoid.runtime.native_contact_report import read_static_report
                    force=read_static_report(provider,simulator=self,monitor=monitor,dt=self._sim.get_physics_dt(),
                        physics_step=self._sim.get_physics_step_count())
                    monitor.peer.static_max_force_n=torch.maximum(monitor.peer.static_max_force_n,
                        torch.as_tensor(force,device=self.device,dtype=monitor.peer.static_max_force_n.dtype))
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


class NativeScenePowerSimulator(PaperPowerSimulator):
    """One SDK world; provider injection happens before physics initialization."""

    def __init__(self, *args, scene_provider, **kwargs):
        self.scene_provider = scene_provider
        super().__init__(*args, **kwargs)

    def _prepare_simulation_cfg(self, configuration):
        return self.scene_provider.prepare_simulation_cfg(configuration)

    def _get_scene_cfg(self):
        return self.scene_provider.prepare_scene_cfg(super()._get_scene_cfg())


class AMPWorld:
    def __init__(self, app, corpus, *, num_envs=64, device="cuda:0", shared_world=False,
                 history_frames=10, episode_steps=300, scene_description=None,
                 power_source="projected_joint_effort_substep_mean", terrain_extent_m=64.,
                 contact_reporting=False, spawn_margin_m=6., amass_root_training=False, amass_root_tail_policy="hold",
                 mixed_trajectory_training=False, mixed_procedural_config=None, mixed_seed=7,
                 native_navigation_scene=None, align_spawn_heading_to_route=False, initial_reference_max_speed_m_s=None, initialize_at_rest=False, scene_physics_replication=None, initial_reference_arm_cone_deg=None, training_walking_course=None):
        if native_navigation_scene is not None and (
                not shared_world or scene_description is not None or amass_root_training
                or mixed_trajectory_training or history_frames != 10):
            raise ValueError("Native navigation requires one explicit shared world without another scene/curriculum")
        if type(amass_root_training) is not bool or (amass_root_training and (shared_world or scene_description is not None or history_frames != 10)):
            raise ValueError("AMASS root curriculum is an explicit isolated training-only source")
        if type(mixed_trajectory_training) is not bool or (mixed_trajectory_training and
                (amass_root_training or shared_world or scene_description is not None or history_frames != 10
                 or type(num_envs) is not int or num_envs < 2 or num_envs % 2)):
            raise ValueError("Mixed fixed-slot curriculum requires even isolated training actors and history10")
        if not mixed_trajectory_training and mixed_procedural_config is not None:
            raise ValueError("Mixed procedural config requires the explicit mixed training source")
        if power_source not in {"projected_joint_effort_substep_mean", "legacy_pd_proxy_misaligned"}:
            raise ValueError("Unknown joint power measurement source")
        if type(align_spawn_heading_to_route) is not bool or (align_spawn_heading_to_route and native_navigation_scene is None):
            raise ValueError("Spawn heading alignment requires an explicit native route scene")
        if initial_reference_max_speed_m_s is not None and (native_navigation_scene is None or
                type(initial_reference_max_speed_m_s) not in (int,float) or not 0<initial_reference_max_speed_m_s<=.5):
            raise ValueError("Low-speed real-reference initialization requires an explicit native route and speed in(0,.5]")
        if type(initialize_at_rest) is not bool or (initialize_at_rest and native_navigation_scene is None):raise ValueError("Rest initialization requires an explicit native scene")
        if initial_reference_arm_cone_deg is not None and (native_navigation_scene is None or initial_reference_max_speed_m_s is None):raise ValueError("Arm-cone initialization requires the explicit low-speed native-scene prior")
        self.initial_reference_arm_cone_deg=initial_reference_arm_cone_deg
        self.initialize_at_rest=initialize_at_rest
        self.initial_reference_max_speed_m_s=initial_reference_max_speed_m_s
        self.align_spawn_heading_to_route=align_spawn_heading_to_route
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
        self.validation_scene = native_navigation_scene
        self.scene_provider = None if native_navigation_scene is None else native_navigation_scene.provider
        if native_navigation_scene is not None and len(native_navigation_scene.origins) != num_envs:
            raise ValueError("Native scene requires one reviewed spawn/route per actor")
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
        if scene_physics_replication is not None and type(scene_physics_replication) is not bool:
            raise ValueError("Explicit scene_physics_replication must be boolean or None")
        config.scene_physics_replication = scene_physics_replication
        config.sim.fps = 120
        config.sim.decimation = 4
        config.sim.physx.num_position_iterations = 4
        config.sim.physx.num_velocity_iterations = 4
        self.dt = 1/30
        terrain_config = TerrainConfig(map_length=self.terrain_extent_m, map_width=self.terrain_extent_m, border_size=5,
                                       num_levels=1, num_terrains=1, spacing_between_scenes=4)
        self.terrain = Terrain(terrain_config, num_envs, self.device)
        scenes = SceneLib(SceneLibConfig(), num_envs, device=self.device, terrain=self.terrain)
        if self.scene_provider is None:
            self.simulator = PaperPowerSimulator.__new__(PaperPowerSimulator)
            PaperPowerSimulator.__init__(self.simulator, config, self.robot_config, self.terrain,
                                              device=self.device, simulation_app=app, scene_lib=scenes)
        else:
            self.simulator = NativeScenePowerSimulator.__new__(NativeScenePowerSimulator)
            NativeScenePowerSimulator.__init__(self.simulator, config, self.robot_config, self.terrain,
                device=self.device, simulation_app=app, scene_lib=scenes, scene_provider=self.scene_provider)
        self.simulator.power_source = power_source
        if scene_description is not None:
            import isaaclab.sim as sim_utils
            for index, box in enumerate(scene_description.get("cuboids", [])):
                shape = sim_utils.CuboidCfg(size=tuple(box["size"]),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.002, rest_offset=0.),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.2, .3, .4)),
                    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1., dynamic_friction=1., restitution=0.))
                sim_utils.spawn_cuboid(f"/World/PaperValidation/Obstacle_{index}", shape,
                                      translation=tuple(box["center"]))
        self.simulator._initialize_with_markers(None)
        if self.scene_provider is not None:
            import omni.physx
            self.scene_provider.bind_native_query(omni.physx.get_physx_scene_query_interface(),
                physics_step_count=self.simulator._sim.get_physics_step_count,
                dynamic_robot_roots=[f"/World/envs/env_{actor}/Robot" for actor in range(num_envs)])
        self.lower = self.robot_config.kinematic_info.dof_limits_lower.to(self.device)
        self.upper = self.robot_config.kinematic_info.dof_limits_upper.to(self.device)
        side = math.ceil(math.sqrt(num_envs))
        ids = torch.arange(num_envs, device=self.device)
        self.origins = torch.stack((spawn_margin_m + (ids // side) * 4, spawn_margin_m + (ids % side) * 4), -1).float()
        if self.validation_scene is None and float(self.origins.max()) >= self.terrain_extent_m:
            raise ValueError("Isolated spawn grid exceeds the declared terrain extent")
        if self.validation_scene is not None:
            self.origins = self.validation_scene.origins.clone()
        self.native_scene_live_observation = None
        if self.scene_provider is not None:
            self.native_scene_live_observation = self.scene_provider.verify_live_runtime(
                origins_xy=self.origins, simulation_context=self.simulator._sim,
                free_cell_limit=128)
        self.age = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.episode_ids = torch.zeros_like(self.age)
        self.amp_history = torch.zeros(num_envs, history_frames, 195, device=self.device)
        self.amp_history_episode_ids = torch.zeros(num_envs, history_frames, dtype=torch.long, device=self.device)
        self.planar_history = torch.zeros(num_envs, 91, 5, device=self.device)
        self.history_available = torch.zeros(num_envs, 91, dtype=torch.bool, device=self.device)
        self.reference_ids = torch.zeros_like(self.age)
        self.amass_root_curriculum = None
        self.mixed_trajectory_curriculum = None
        if amass_root_training:
            from CrowdSim.humanoid.training.amass_trajectories import AMASSRootTrajectoryCurriculum
            self.amass_root_curriculum = AMASSRootTrajectoryCurriculum(corpus, num_envs, device=self.device,
                bounds_xy=(0., 0., self.terrain_extent_m, self.terrain_extent_m), tail_policy=amass_root_tail_policy)
        if mixed_trajectory_training:
            from CrowdSim.humanoid.training.mixed_trajectories import MixedTrajectoryCurriculum
            self.mixed_trajectory_curriculum = MixedTrajectoryCurriculum(corpus, num_envs, device=self.device,
                bounds_xy=(0., 0., self.terrain_extent_m, self.terrain_extent_m), tail_policy=amass_root_tail_policy,
                procedural_config=mixed_procedural_config, seed=mixed_seed)
        if training_walking_course is not None and (not amass_root_training or training_walking_course.corpus is not corpus or training_walking_course.num_envs!=num_envs):
            raise ValueError('Walking course must bind this exact isolated AMASS training world')
        self.training_walking_course=training_walking_course
        if training_walking_course is not None:
            self.amass_root_curriculum.config['reset_sampling']='explicit source-bound real walking windows; expert sampling unchanged'
        self.reset(torch.arange(num_envs, device=self.device))

    def height_at(self, xy):
        if self.scene_provider is not None:
            return self.scene_provider.height_at(xy)
        ground = self.ground_height_at(xy)
        return ground if self.validation_scene is None else torch.maximum(ground, self.validation_scene.obstacle_height(xy))

    def ground_height_at(self, xy):
        if self.scene_provider is not None:
            return self.scene_provider.ground_height_at(xy)
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
        mixed = getattr(self, "mixed_trajectory_curriculum", None)
        walking=getattr(self,'training_walking_course',None)
        if walking is not None:
            reference,history,_,motion_ids,frames=walking.sample_reference(len(ids))
        elif mixed is not None:
            reference, history, _, motion_ids, frames = mixed.sample_reference(ids, self.history_frames)
        elif curriculum is None:
            if self.initial_reference_max_speed_m_s is None:
                reference, history, _, motion_ids, frames = self.corpus.sample_reference(len(ids), self.history_frames)
            else:
                if not hasattr(self,"_low_speed_initial_pool"):
                    from .initial_reference_pool import build_initial_reference_pool
                    self._low_speed_initial_pool,self.initial_reference_pool_report=build_initial_reference_pool(
                        self.corpus,self.history_frames,self.initial_reference_max_speed_m_s,self.initial_reference_arm_cone_deg)
                index,source_ids,source_frames=self._low_speed_initial_pool
                chosen=torch.randint(len(index),(len(ids),),generator=self.corpus.generator)
                absolute=index[chosen];motion_ids=source_ids[chosen];frames=source_frames[chosen]
                self.initial_reference_selected_frames={"motion_ids":motion_ids.tolist(),"local_frames":frames.tolist(),"absolute_frames":absolute.tolist()}
                reference=self.corpus.state_at(absolute)
                history=self.corpus.state_at((absolute[:,None]-torch.arange(self.history_frames)).reshape(-1))
        else:
            reference, history, _, motion_ids, frames = self.corpus.sample_reference(len(ids), self.history_frames,
                future_frames=curriculum.FUTURE_FRAMES)
        positions = reference.root_pos.clone()
        positions[:, :2] = self.origins[ids]
        positions[:, 2] += self.ground_height_at(positions[:, :2])
        root_rot,root_vel,root_ang=reference.root_rot.clone(),reference.root_vel.clone(),reference.root_ang_vel.clone()
        if self.align_spawn_heading_to_route:
            # A rigid yaw transform only at initial/reset placement. Actual PD
            # controls determine every subsequent pose, speed and turn.
            from protomotions.utils.rotations import quat_mul
            tangent=self.validation_scene.routes[ids,0]-positions[:,:2]
            desired=torch.atan2(tangent[:,1],tangent[:,0])
            current=torch.atan2(2*(root_rot[:,3]*root_rot[:,2]+root_rot[:,0]*root_rot[:,1]),
                1-2*(root_rot[:,1].square()+root_rot[:,2].square()))
            angle=desired-current;c,s=angle.cos(),angle.sin()
            z=torch.zeros_like(angle);turn=torch.stack((z,z,(angle/2).sin(),(angle/2).cos()),-1)
            root_rot=quat_mul(turn,root_rot,w_last=True)
            for velocity in (root_vel,root_ang):
                x,y=velocity[:,0].clone(),velocity[:,1].clone()
                velocity[:,0],velocity[:,1]=c*x-s*y,s*x+c*y
        joint_velocity=reference.dof_vel.clone()
        if self.initialize_at_rest:
            root_vel.zero_();root_ang.zero_();joint_velocity.zero_()
        self.simulator.reset_envs(ResetState(root_pos=positions, root_rot=root_rot,
            root_vel=root_vel, root_ang_vel=root_ang,
            dof_pos=reference.dof_pos.clone(), dof_vel=joint_velocity,
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
        if walking is not None:
            walking.bind_reset(self,ids)
        if mixed is not None:
            mixed.bind_reset(ids, motion_ids, frames, reference.root_pos[:, :2], positions[:, :2], self.episode_ids[ids])

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
        mixed = getattr(self, "mixed_trajectory_curriculum", None)
        if mixed is not None:
            truncated = truncated | mixed.completed_step(self.age, terminated, truncated)
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
        result["scene_instantiation"] = getattr(self.simulator._scene,"physics_replication_report",{"mode":"existing SDK replicate session"})
        result["terrain_extent_m"] = self.terrain_extent_m
        result["isolated_spawn_margin_m"] = None if self.validation_scene is not None else float(self.origins.min())
        result["scene_geometry"] = None if self.validation_scene is None else self.validation_scene.description
        # Native live PhysX mass readings, not authored USD transforms.
        masses = wp.to_torch(self.simulator._robot.root_view.get_masses()).cpu()
        result["terrain_material"]["combine_mode"] = self.terrain.sim_config.combine_mode.value
        if self.scene_provider is not None:
            # The provider enforces this authored MaterialAPI/PhysxMaterialAPI
            # recipe. The Terrain object is retained for empty SceneLib plumbing.
            result["terrain_material"] = {"static_friction": 1., "dynamic_friction": 1.,
                "restitution": 0., "height_offset": self.scene_provider.ground_height_m, "combine_mode": "average"}
            result["physical_ground_source"] = {"kind": "session_only_finite_reference_ground",
                "floor_paths": list(self.scene_provider.floor_roots), "generated_Terrain_spawned": False,
                "material_contract": "provider source-bound MaterialAPI/PhysxMaterialAPI recipe",
                "native_contact_quality_verified_by_this_manifest": False}
        result["live_native_body_names"] = list(self.simulator._robot.body_names)
        result["live_native_body_masses_kg"] = masses[0].tolist()
        indices = self.simulator.data_conversion.body_convert_to_common.cpu()
        result["live_body_masses_kg"] = masses[0, indices].tolist()
        result["live_total_mass_kg"] = float(masses[0].sum())
        return result

    def close(self):
        self.simulator.close()
