from types import SimpleNamespace
import numpy as np
import pytest
import torch
from deployment.motion_utils import MotionPlayer
from crowdsim_ppo.point_goal_env import PointGoalBatchEnv, PointGoalEnvConfig
from CrowdSim.control.drive import DriveController


def motion_file(tmp_path, frames=2, fps=10.0):
    time = torch.arange(frames, dtype=torch.float32) / fps
    body_pos = torch.zeros(frames, 1, 3)
    body_pos[:, 0, 0] = time
    body_rot = torch.zeros(frames, 1, 4)
    body_rot[..., 3] = 1
    data = {'fps': fps, 'dof_pos': time[:, None], 'dof_vel': torch.ones(frames, 1),
            'rigid_body_pos': body_pos, 'rigid_body_rot': body_rot,
            'rigid_body_vel': torch.zeros(frames, 1, 3),
            'rigid_body_ang_vel': torch.zeros(frames, 1, 3)}
    path = tmp_path / 'walk.motion'
    torch.save(data, path)
    return path


def test_motion_player_queries_match_training_uniform_control_clock(tmp_path):
    player = MotionPlayer(motion_file(tmp_path), control_dt=.03)
    # The final 0.12s query clamps to the last 0.1s source sample, as in training.
    assert player.total_frames == 5
    actual = [player.get_state_at_frame(i)['body_pos'][0, 0] for i in range(5)]
    np.testing.assert_allclose(actual, [0, .03, .06, .09, .1], atol=1e-7)
    assert player.control_dt == .03


def test_single_source_frame_can_be_played_at_any_positive_control_rate(tmp_path):
    player = MotionPlayer(motion_file(tmp_path, frames=1), control_dt=.03)
    assert player.total_frames == 1
    np.testing.assert_array_equal(player.get_state_at_frame(100)['body_pos'], [[0, 0, 0]])


@pytest.mark.parametrize('dt', [0., -.01, float('nan'), float('inf')])
def test_motion_player_rejects_invalid_control_clock(tmp_path, dt):
    with pytest.raises(ValueError, match='control_dt'):
        MotionPlayer(motion_file(tmp_path), control_dt=dt)


@pytest.mark.parametrize('field,value', [('dt', 0.), ('dt', float('nan')),
    ('max_episode_steps', 0), ('max_linear_velocity', float('inf')),
    ('neighbor_speed', -.1)])
def test_point_goal_rejects_invalid_dynamics_config(field, value):
    cfg = PointGoalEnvConfig(num_envs=1, num_neighbors=0)
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match=field):
        PointGoalBatchEnv(cfg, torch.device('cpu'))


def test_point_goal_rejects_nonfinite_action_without_mutating_episode():
    env = PointGoalBatchEnv(PointGoalEnvConfig(num_envs=1, num_neighbors=0), torch.device('cpu'))
    position, heading = env.position.clone(), env.heading.clone()
    with pytest.raises(ValueError, match='finite'):
        env.step(torch.tensor([[float('nan'), 0.]]))
    torch.testing.assert_close(env.position, position)
    torch.testing.assert_close(env.heading, heading)
    assert env.episode_step.item() == 0


def test_point_goal_rejects_action_batch_shape_without_mutation():
    env = PointGoalBatchEnv(PointGoalEnvConfig(num_envs=2, num_neighbors=0), torch.device('cpu'))
    with pytest.raises(ValueError, match='shape'):
        env.step(torch.zeros(1, 2))
    assert env.episode_step.sum().item() == 0


def test_robot_drive_rejects_nonfinite_action_without_mutating_previous_command():
    drive = DriveController(None, SimpleNamespace(num_robots=1), torch.device('cpu'))
    drive.set_action(np.array([[.4, .2]], dtype=np.float32))
    previous, current = drive._prev_actions.copy(), drive._actions.copy()
    with pytest.raises(ValueError, match='finite'):
        drive.set_action(np.array([[np.inf, np.nan]], dtype=np.float32))
    np.testing.assert_array_equal(drive._actions, current)
    np.testing.assert_array_equal(drive._prev_actions, previous)


def test_native_sfm_source_version_keeps_exact_historical_method_inverse():
    from CrowdSim.humanoid.trajectory.native_social_force_runtime import (
        ROOT, PINS, HISTORICAL_PINS, derive_actor_methods, source_bindings,
    )
    source = (ROOT / PINS['fixture_bridge'][0]).read_text()
    _, proof = derive_actor_methods(source)
    assert proof['fixture_location_exact_inverse']
    assert proof['historical_actor_method_bytes_equal']
    assert proof['historical_fixture_sha256'] == HISTORICAL_PINS['fixture_bridge'][1]
    assert source_bindings()['source_version'] == 'native_sfm.current_scene_sources.v2'
    with pytest.raises(ValueError, match='exact source-bound'):
        derive_actor_methods(source + '\n# unregistered source change\n')


def test_native_sfm_refuses_unregistered_current_source_hash(monkeypatch):
    from CrowdSim.humanoid.trajectory import native_social_force_runtime as native
    real = native.file_identity
    def changed_identity(path):
        value = real(path)
        return dict(value, sha256='0' * 64) if str(path).endswith('dynbench_scene.py') else value
    monkeypatch.setattr(native, 'file_identity', changed_identity)
    with pytest.raises(ValueError, match='dependency changed: provider'):
        native.source_bindings()


@pytest.mark.parametrize('native_proxy', [False, True])
def test_robot_camera_recorder_accepts_sdk_torch_and_proxy_arrays(tmp_path, native_proxy):
    from PIL import Image
    from CrowdSim.utils.sensor_stream import RobotCameraRecorder
    def wrap(tensor):
        return SimpleNamespace(torch=tensor) if native_proxy else tensor
    rgb = torch.full((1, 2, 3, 4), 42, dtype=torch.uint8)
    depth = torch.full((1, 2, 3, 1), 1.5)
    intrinsics = torch.eye(3)[None]
    camera = SimpleNamespace(data=SimpleNamespace(image_shape=(2, 3),
        intrinsic_matrices=wrap(intrinsics), output={
        'rgb': wrap(rgb), 'distance_to_image_plane': wrap(depth)}))
    recorder = RobotCameraRecorder(SimpleNamespace(crowdsim_robot_camera=camera), tmp_path, 10., [0])
    recorder.session_dir = tmp_path
    (tmp_path / 'env_0000').mkdir()
    recorder._write_metadata()
    recorder._save_frame()
    stored = torch.load(tmp_path / 'metadata.pt', weights_only=True)
    torch.testing.assert_close(stored['intrinsic_matrices'], intrinsics)
    torch.testing.assert_close(torch.load(tmp_path / 'env_0000/depth_000000.pt', weights_only=True), depth[0])
    np.testing.assert_array_equal(np.asarray(Image.open(tmp_path / 'env_0000/rgb_000000.png')), rgb[0, ..., :3].numpy())


def test_motion_player_cache_roundtrip_preserves_control_clock_and_all_states(tmp_path):
    raw = MotionPlayer(motion_file(tmp_path), control_dt=.03)
    cache = tmp_path / 'walk.cached.pt'
    raw.cache_to_file(cache)
    cached = MotionPlayer(cache)
    assert cached.control_dt == raw.control_dt and cached.total_frames == raw.total_frames
    for frame in range(raw.total_frames):
        for key, value in raw.get_state_at_frame(frame).items():
            np.testing.assert_array_equal(cached.get_state_at_frame(frame)[key], value)


@pytest.mark.parametrize('field,value', [('control_dt', 0.), ('control_dt', float('nan')),
    ('num_frames', 0), ('num_frames', 1.5), ('body_pos', np.zeros(3)),
    ('dof_vel', np.zeros((1, 2))), ('body_rot', np.full((1, 1, 4), np.nan))])
def test_motion_player_rejects_invalid_cache_metadata_or_arrays(tmp_path, field, value):
    raw = MotionPlayer(motion_file(tmp_path, frames=1), control_dt=.03)
    path = tmp_path / 'walk.cached.pt'
    raw.cache_to_file(path)
    data = torch.load(path, weights_only=False)
    data[field] = value
    torch.save(data, path)
    with pytest.raises(ValueError, match='Cached|cached'):
        MotionPlayer(path)
