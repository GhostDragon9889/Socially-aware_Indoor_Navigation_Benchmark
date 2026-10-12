"""CPU map/drive harness: only articulation transport replaces the simulator."""

from types import SimpleNamespace as NS

import numpy as np
from PIL import Image
import torch

from CrowdSim.nav_manager import CrowdNavigationConfig, CrowdNavigationManager
from CrowdSim.control.drive import yaw_to_quaternion


class MemoryArticulation:
    """Store IsaacLab-shaped writes without claiming a PhysX integration test."""

    joint_names = ["joint_wheel_left", "joint_wheel_right"]

    def __init__(self, count):
        poses = torch.zeros(count, 7)
        poses[:, 6] = 1.0
        self.data = NS(
            root_pos_w=poses[:, :3],
            root_quat_w=poses[:, 3:7],
            root_lin_vel_w=torch.zeros(count, 3),
            root_ang_vel_w=torch.zeros(count, 3),
            default_root_state=torch.cat((poses.clone(), torch.zeros(count, 6)), dim=1),
            default_joint_pos=torch.zeros(count, 2),
            default_joint_vel=torch.zeros(count, 2),
        )
        self.writes = []

    def find_joints(self, names, preserve_order=True):
        return [self.joint_names.index(name) for name in names], names

    def write_root_pose_to_sim(self, poses, env_ids=None):
        ids = torch.arange(len(poses)) if env_ids is None else env_ids
        self.data.root_pos_w[ids] = poses[:, :3]
        self.data.root_quat_w[ids] = poses[:, 3:7]
        self.writes.append(("pose", ids.clone(), poses.clone()))

    def write_root_velocity_to_sim(self, velocities, env_ids=None):
        ids = torch.arange(len(velocities)) if env_ids is None else env_ids
        self.data.root_lin_vel_w[ids] = velocities[:, :3]
        self.data.root_ang_vel_w[ids] = velocities[:, 3:]
        self.writes.append(("velocity", ids.clone(), velocities.clone()))

    def write_joint_state_to_sim(self, positions, velocities, env_ids):
        self.writes.append(("joint", env_ids.clone(), positions.clone()))

    def set_joint_velocity_target(self, targets, joint_ids, env_ids):
        self.writes.append(("wheel", env_ids.clone(), targets.clone()))

    def reset(self, env_ids):
        self.writes.append(("reset", env_ids.clone(), None))


def make_manager(tmp_path, **overrides):
    tmp_path.mkdir(parents=True, exist_ok=True)
    image = np.full((128, 128), 255, dtype=np.uint8)
    image[[0, -1], :] = 0
    image[:, [0, -1]] = 0
    path = tmp_path / "map.png"
    Image.fromarray(image).save(path)
    arguments = dict(
        map_path=path,
        map_resolution=0.1,
        free_threshold=200,
        num_humanoids=2,
        num_robots=2,
        device=torch.device("cpu"),
        min_start_goal_distance=2.0,
        max_start_goal_distance=4.0,
        min_spawn_spacing=0.5,
        planning_clearance=0.1,
        planning_step_size=0.2,
        agent_radius=0.1,
        collision_distance=0.25,
        humanoid_target_enabled=False,
        rl_depth_enabled=False,
        rl_rgb_enabled=False,
        rl_map_size=8,
        rl_map_extent=2.0,
        trajectory_output_dir=tmp_path / "run",
        update_hz=12.5,
    )
    arguments.update(overrides)
    return CrowdNavigationManager(CrowdNavigationConfig(**arguments))


def attach_memory_environment(manager, count=4):
    state = NS(root_pos=torch.zeros(count, 3), root_vel=torch.zeros(count, 3))
    n_h = manager.config.num_humanoids
    state.root_pos[:n_h, :2] = manager.humanoid_starts_xy
    robot = MemoryArticulation(count)
    env = NS(
        device=torch.device("cpu"),
        num_envs=count,
        dt=0.04,
        simulator=NS(
            headless=True, native_quaternion_order="xyzw", get_root_state=lambda: state
        ),
        crowdsim_robot=robot,
        reset_buf=torch.zeros(count, dtype=torch.bool),
        progress_buf=torch.zeros(count, dtype=torch.long),
        extras={},
        respawn_root_offset=torch.zeros(count, 3),
        _crowdsim_desired_root_xy=manager.humanoid_spawn_xy_for_all_envs(count),
    )
    xy_yaw = manager.robot_spawn_poses_for_all_envs(count)
    env._crowdsim_robot_spawn_poses = torch.zeros(count, 7)
    env._crowdsim_robot_spawn_poses[:, :2] = xy_yaw[:, :2]
    env._crowdsim_robot_spawn_poses[:, 3:] = yaw_to_quaternion(xy_yaw[:, 2])
    env.respawn_root_offset[:, :2] = env._crowdsim_desired_root_xy

    def reset(env_ids=None):
        ids = (
            torch.arange(count)
            if env_ids is None
            else torch.as_tensor(env_ids).flatten()
        )
        state.root_pos[ids, :2] = env._crowdsim_desired_root_xy[ids]
        state.root_vel[ids] = 0
        env.reset_buf[ids] = False
        return {"transport": "reset"}, {}

    def step(action):
        env.progress_buf += 1
        return {"transport": "step"}, None, None, None, {}

    env.reset, env.step = reset, step
    manager.attach(env)
    if manager.drive is not None:
        manager.drive.enable_prim_sync(False)
    env.reset()
    return env, state
