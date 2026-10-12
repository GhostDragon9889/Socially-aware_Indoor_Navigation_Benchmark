import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import cv2


class Social_Force:
    def __init__(self, edt, cfg):
        self.name = 'sfm'
        self.dt = cfg['env']['dt']
        self.rad = cfg['agent']['radius']
        self.max_speed = cfg['agent']['max_vel']      

        self.gain_k = 5
        self.gain_a_static = 15
        self.gain_b_static = 0.25

        self.gain_a_agent = 25
        self.gain_b_agent = 0.5
        self.gain_ttc = float(cfg['env'].get('ttc_gain', 12.0))
        
        self.distance_field = edt
        self.map_resolution = float(cfg['map'].get('resolution', 1.0))
        self.grad_map = self.compute_gradient(self.distance_field, self.map_resolution)

        self.safe_dis = cfg['env']['safe_distance']
        self.avoid_dis = float(cfg['env'].get('neighbor_radius', 4.0))
        self.reach_dis = float(cfg['env'].get('reach_distance', 2*self.rad))
        self.prediction_horizon = float(
            cfg['env'].get('prediction_horizon', 2.0)
        )
        self.ttc_threshold = float(cfg['env'].get('ttc_threshold', 1.5))

        self.fov = False
        

    @staticmethod
    def compute_gradient(distance_field, map_resolution):
        """计算EDT的梯度场"""
        grad_y, grad_x = np.gradient(distance_field, map_resolution, map_resolution)
        grad_world_x = grad_x
        grad_world_y = -grad_y

        # 计算梯度模长
        magnitude = np.sqrt(grad_world_x**2 + grad_world_y**2)

        # 处理零梯度区域
        magnitude[magnitude == 0] = 1e-6  # 避免除以零

        # 归一化梯度
        grad_norm = np.stack([grad_world_x / magnitude, grad_world_y / magnitude], axis=0)

        return grad_norm
    
    @staticmethod
    def fov_filter(curr_dir, rel_obs):
        near_nbrs_idx, nbrs_dis, nbrs_relpos, nbrs_relvel = rel_obs

        if len(near_nbrs_idx) == 0:
            nbrs_idx = []
            nbrs_dis = None
            nbrs_pos = None

            return nbrs_idx, nbrs_dis, nbrs_pos, nbrs_relvel

        # relpos is neighbor-minus-ego; positive projection is in front.
        cos_angles = np.dot(nbrs_relpos, curr_dir)

        # 筛选在视野范围内的邻居（夹角在±90度内，即cos值>0）
        in_fov = cos_angles >= 0

        fov_nbrs_idx = near_nbrs_idx[in_fov]
        fov_nbrs_dis = nbrs_dis[in_fov]
        fov_nbrs_relpos = nbrs_relpos[in_fov]
        fov_nbrs_relvel = nbrs_relvel[in_fov]
        
        if len(fov_nbrs_idx) > 5:
            sorted_indices = np.argsort(fov_nbrs_dis)[:5]          
            nbrs_idx = fov_nbrs_idx[sorted_indices]
            nbrs_dis = fov_nbrs_dis[sorted_indices]
            nbrs_pos = fov_nbrs_relpos[sorted_indices]
            nbrs_vel = fov_nbrs_relvel[sorted_indices]
        else:
            nbrs_idx = fov_nbrs_idx
            nbrs_dis = fov_nbrs_dis
            nbrs_pos = fov_nbrs_relpos
            nbrs_vel = fov_nbrs_relvel

        return nbrs_idx, nbrs_dis, nbrs_pos, nbrs_vel


    def compute_forces(self, agent_state, rel_obs):
        pos, cord_int, vel, goal = agent_state
        nbrs_idx, nbrs_dis, nbrs_relpos, nbrs_relvel = (
            self.fov_filter(vel, rel_obs) if self.fov else rel_obs
        )
        pixel_y, pixel_x = cord_int
        
        # ===== 1. 计算目标吸引力 =====
        direction_to_goal = goal - pos
        distance_to_goal = np.linalg.norm(direction_to_goal)
        
        if distance_to_goal > self.reach_dis:
            attractive_force = self.max_speed * direction_to_goal/distance_to_goal
        else:
            # print('[SFM] !!! Reach')
            attractive_force = np.zeros(2)
        
        
        # ===== 2. 计算障碍排斥力 =====
        dist = self.distance_field[pixel_y, pixel_x]
        
        if dist < self.avoid_dis:
            # 获取梯度方向
            grad = np.array([self.grad_map[0, pixel_y, pixel_x], self.grad_map[1, pixel_y, pixel_x]])
            
            # 排斥力计算（距离越近力越大）
            # repulsive_force = self.repulsive_gain * (1/dist - 1/self.safety_radius) * grad
            repulsive_force = self.gain_a_static * np.exp( (0.5*self.safe_dis - dist)/ self.gain_b_static ) * grad
        else:
            repulsive_force = np.zeros(2)
        
        if len(nbrs_idx) != 0:
            relative_positions = np.asarray(nbrs_relpos, dtype=np.float32)
            safe_distances = np.maximum(nbrs_dis, 1e-6).reshape(-1, 1)
            away_now = -relative_positions / safe_distances
            interact_force = self.gain_a_agent * np.exp(
                (self.safe_dis - safe_distances) / self.gain_b_agent
            ) * away_now
            interact_force = np.sum(interact_force, axis=0)
        else:
            interact_force = np.zeros(2)

        # Predict closest approach using relative velocity.  relpos is
        # neighbor-minus-ego and relvel is neighbor-minus-ego velocity, so
        # relpos + relvel*t is the future separation vector.
        if len(nbrs_idx) != 0:
            relative_positions = np.asarray(nbrs_relpos, dtype=np.float32)
            relvel_sq = np.sum(nbrs_relvel * nbrs_relvel, axis=1)
            approaching = np.sum(
                relative_positions * nbrs_relvel, axis=1
            ) < 0.0
            closest_time = np.zeros(len(nbrs_idx), dtype=np.float32)
            moving = relvel_sq > 1e-6
            closest_time[moving] = np.clip(
                -np.sum(
                    relative_positions[moving] * nbrs_relvel[moving], axis=1
                )
                / relvel_sq[moving],
                0.0,
                self.prediction_horizon,
            )
            closest = relative_positions + nbrs_relvel * closest_time[:, None]
            closest_distance = np.linalg.norm(closest, axis=1)
            distance_risk = np.clip(
                (self.safe_dis - closest_distance) / max(self.safe_dis, 1e-6),
                0.0,
                1.0,
            )
            time_risk = np.clip(
                (self.ttc_threshold - closest_time)
                / max(self.ttc_threshold, 1e-6),
                0.0,
                1.0,
            )
            risk = distance_risk * time_risk * approaching.astype(np.float32)
            away = -closest / np.maximum(closest_distance[:, None], 1e-6)
            preferred_norm = max(float(np.linalg.norm(direction_to_goal)), 1e-6)
            preferred_direction = direction_to_goal / preferred_norm
            pass_right = np.asarray(
                [preferred_direction[1], -preferred_direction[0]],
                dtype=np.float32,
            )
            away[closest_distance < 1e-4] = pass_right
            ttc_force = self.gain_ttc * np.sum(risk[:, None] * away, axis=0)
        else:
            ttc_force = np.zeros(2)

        # ===== 合力合成 =====
        # Sum of push & pull forces
        d_vel = self.gain_k * (attractive_force - vel)
        interaction_vel = repulsive_force + interact_force + ttc_force
        total_d_vel = (d_vel + interaction_vel) * self.dt
        new_vel = vel + total_d_vel
        # Clamp speed to max_speed to prevent explosion when agents are very
        # close (exponential repulsion can produce huge forces).
        speed = np.linalg.norm(new_vel)
        if speed > self.max_speed:
            new_vel = new_vel * (self.max_speed / speed)

        return new_vel, [interact_force, repulsive_force, d_vel, ttc_force]
    
    def get_action(self, ego_state, env_obs):

        return self.compute_forces(ego_state, env_obs)


def load_image(img_path,scale=1/3):
    """加载并预处理地图"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  
    _, binary_img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
    binary_img = cv2.resize(binary_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    return binary_img

# 可视化函数
def visualize_simulation(distance_field, path, start, goal, show=True):
    plt.imshow(distance_field, cmap='gray_r')
    plt.plot(start[1], start[0], 'gx')
    plt.plot(goal[1], goal[0], 'rx')
   
    # 绘制路径
    path = np.array(path)
    plt.plot(path[:,1], path[:,0])
    
    if show:
        plt.show()

def visualize_gradient(distance_field, grad_map, stride=10, scale=30):
    """Plot world-space gradients in the image's downward-positive row frame."""
    rows, cols = distance_field.shape
    if stride <= 0 or grad_map.shape != (2, rows, cols):
        raise ValueError("Gradient diagnostic requires positive stride and shape (2, H, W)")
    x_grid, y_grid = np.mgrid[0:rows:1, 0:cols:1]
    mask = (x_grid % stride == 0) & (y_grid % stride == 0)
    x_sub, y_sub = x_grid[mask], y_grid[mask]
    gx_sub = grad_map[0, x_sub, y_sub]
    gy_sub = grad_map[1, x_sub, y_sub]
    plt.imshow(distance_field, cmap='gray_r')
    return plt.quiver(y_sub, x_sub, gx_sub, -gy_sub,
                      angles='xy', scale_units='xy', scale=scale, color='red')


def main(argv=None):
    """Run the current SFM API on an occupancy map; this is a CPU diagnostic."""
    import argparse
    import json
    import torch
    from CrowdSim.nav_manager import CrowdNavigationConfig
    from CrowdSim.world.map import NavigationTask

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--map", type=Path, default=Path(__file__).resolve().parents[1] / "maps/World0.png")
    parser.add_argument("--resolution", type=float, default=.1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.steps <= 0:
        parser.error("steps must be positive")
    if not np.isfinite(args.resolution) or args.resolution <= 0:
        parser.error("resolution must be positive and finite")
    if args.headless:
        plt.switch_backend("Agg")
    config = CrowdNavigationConfig(args.map, args.resolution, 206, 1, 0, torch.device("cpu"),
                                   min_start_goal_distance=2., max_start_goal_distance=4.)
    task = NavigationTask(config, 1)
    distance = cv2.distanceTransform(task.free_mask.astype(np.uint8), cv2.DIST_L2, 5) * args.resolution
    controller = Social_Force(distance, {
        "env": {"dt": .1, "safe_distance": .9, "neighbor_radius": 4., "reach_distance": .2},
        "agent": {"radius": config.agent_radius, "max_vel": config.max_speed},
        "map": {"resolution": args.resolution},
    })
    position, goal = task.starts_xy[0].copy(), task.goals_xy[0]
    velocity = np.zeros(2, dtype=np.float32)
    path = [position.copy()]
    waypoint = 1
    empty = (np.empty(0, dtype=int), np.empty(0), np.empty((0, 2)), np.empty((0, 2)))
    for _ in range(args.steps):
        while waypoint < len(task.paths_xy[0]) - 1 and np.linalg.norm(position - task.paths_xy[0][waypoint]) <= .45:
            waypoint += 1
        pixel = np.clip(task.world_to_pixel(position), [0, 0], [task.height - 1, task.width - 1])
        velocity, _ = controller.get_action((position, pixel, velocity, task.paths_xy[0][waypoint]), empty)
        position = position + velocity * controller.dt
        path.append(position.copy())
        if np.linalg.norm(position - goal) <= config.goal_tolerance:
            break
    fig = plt.figure()
    try:
        pixels = task.world_to_pixel(np.asarray(path))
        visualize_simulation(distance, pixels, pixels[0], task.world_to_pixel(goal), show=not args.headless)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.output)
    finally:
        plt.close(fig)
    print(json.dumps({"simulated_steps": len(path) - 1,
                      "reached": bool(np.linalg.norm(position - goal) <= config.goal_tolerance),
                      "physics_executed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
