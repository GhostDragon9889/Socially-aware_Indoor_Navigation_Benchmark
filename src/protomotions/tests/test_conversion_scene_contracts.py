"""Data and scene shape/time validation regression cases."""

from protomotions.repository import repository_root
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from protomotions.components.scene_lib import SceneObject
from protomotions.components.motion_lib import MotionLib
from protomotions.utils.retargeting_fps import downsample_factor


@pytest.fixture
def csv_converter(monkeypatch):
    scripts = repository_root(__file__) / 'data/scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('g1_csv_contract', scripts / 'convert_g1_csv_to_proto.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _csv(tmp_path, frames=6):
    data = np.concatenate([np.arange(frames)[:, None], np.zeros((frames, 6)), np.ones((frames, 2))], axis=-1)
    path = tmp_path / 'motion.csv'
    np.savetxt(path, data, delimiter=',', header='frame,x,y,z,rx,ry,rz,j1,j2', comments='')
    return path


@pytest.mark.parametrize('input_fps,output_fps', [(30,20),(60,24),(30,60),(0,30),(30,0),(float('nan'),30),(30,float('inf'))])
def test_exact_fps_helper_rejects_time_distortion(input_fps, output_fps):
    with pytest.raises(ValueError):
        downsample_factor(input_fps, output_fps)


def test_g1_csv_single_frame_and_exact_decimation(csv_converter, tmp_path):
    one = csv_converter.process_csv_file(_csv(tmp_path, 1), 30, 30, 'cpu', torch.float32, 'xyz')
    assert [tuple(t.shape) for t in one] == [(1,3), (1,4), (1,2)]
    many = csv_converter.process_csv_file(_csv(tmp_path, 6), 60, 30, 'cpu', torch.float32, 'xyz')
    assert all(len(t) == 3 for t in many)
    with pytest.raises(ValueError, match='integer multiple'):
        csv_converter.process_csv_file(_csv(tmp_path, 6), 30, 20, 'cpu', torch.float32, 'xyz')


@pytest.mark.parametrize('argument,value', [('pos_units','mm'),('joint_units','degrees')])
def test_g1_csv_unknown_units_fail(csv_converter, tmp_path, argument, value):
    with pytest.raises(ValueError, match='Unknown'):
        csv_converter.process_csv_file(_csv(tmp_path), 30, 30, 'cpu', torch.float32, 'xyz', **{argument:value})


def test_g1_contact_decimation_matches_motion_and_rejects_inexact_fps(csv_converter, tmp_path):
    np.savez(tmp_path/'motion_contacts.npz', foot_contacts=np.array([[1,0],[0,1],[0,1],[1,0],[1,1],[0,0]]))
    motion = SimpleNamespace(rigid_body_pos=torch.zeros(3, 2, 3))
    csv_converter.apply_contact_labels_to_motion(motion, tmp_path, 'motion.csv', 60,30,0,1,'cpu')
    torch.testing.assert_close(motion.rigid_body_contacts, torch.tensor([[True,False],[False,True],[True,True]]))
    with pytest.raises(ValueError, match='integer multiple'):
        csv_converter.apply_contact_labels_to_motion(motion, tmp_path, 'motion.csv',30,20,0,1,'cpu')


@pytest.mark.parametrize('label', [-1.,2.,float('nan'),float('inf'),.3])
def test_contact_smoothing_rejects_values_outside_binary_set(label):
    lib = MotionLib.empty(device='cpu')
    lib.contacts = torch.tensor([[label]])
    with pytest.raises(ValueError, match='must be binary'):
        lib.smooth_contacts(3)


def test_scene_rejects_translation_rotation_frame_mismatch_both_directions():
    with pytest.raises(AssertionError, match='same number of frames'):
        SceneObject(translation=[(0,0,0)], rotation=[(0,0,0,1),(0,0,0,1)], fps=30)


@pytest.mark.parametrize('fps', [0.,-1.,float('nan'),float('inf')])
def test_motion_scene_fps_must_be_positive_finite(fps):
    with pytest.raises(ValueError, match='finite and positive'):
        SceneObject(translation=[(0,0,0),(1,0,0)], rotation=[(0,0,0,1),(0,0,0,1)], fps=fps)
