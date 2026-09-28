import importlib.util
from pathlib import Path

import numpy as np


def _rotation():
    path = Path(__file__).resolve().parents[1] / "metaworld" / "utils" / "rotation.py"
    spec = importlib.util.spec_from_file_location("metaworld_rotation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_half_turn_quaternion_round_trip():
    rotation = _rotation()
    quat = np.array([0.0, 0.0, 0.0, 1.0])
    back = rotation.point_quat2quat(rotation.quat2point_quat(quat))
    np.testing.assert_allclose(np.squeeze(back), quat, atol=1e-8)


def test_two_thirds_turn_quaternion_round_trip():
    rotation = _rotation()
    theta = 2 * np.pi / 3
    quat = np.array([np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)])
    back = rotation.point_quat2quat(rotation.quat2point_quat(quat))
    np.testing.assert_allclose(np.squeeze(back), quat, atol=1e-8)


def test_quarter_turn_quaternion_still_round_trips():
    rotation = _rotation()
    quat = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    back = rotation.point_quat2quat(rotation.quat2point_quat(quat))
    np.testing.assert_allclose(np.squeeze(back), quat, atol=1e-8)


def test_third_quadrant_euler_round_trip():
    rotation = _rotation()
    euler = np.array([-3 * np.pi / 4, 0.2, 1.0])
    back = rotation.point_euler2euler(rotation.euler2point_euler(euler))
    np.testing.assert_allclose(np.squeeze(back), euler, atol=1e-8)
