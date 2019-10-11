import pytest
import numpy as np

from pyquaternion import Quaternion
import metaworld.envs.env_util as env_util

def quat_to_zangle_old(quat):
    angle = (Quaternion(axis = [0,1,0], angle = (np.pi/2)).inverse * Quaternion(quat)).angle
    return angle
    
def zangle_to_quat_old(zangle):
    """
    :param zangle in rad
    :return: quaternion
    """
    return (Quaternion(axis = [0,1,0], angle = (np.pi/2)) * Quaternion(axis=[-1, 0, 0], angle= zangle)).elements

def random_axis():
    bases = [[0, 0, 1], [0, 1, 0], [1, 0, 0],
             [0, 0, -1], [0, -1, 0], [-1, 0, 0]]
    axis_idx = np.random.randint(len(bases))
    return np.array(bases[axis_idx], dtype='float')

def random_angle():
    return np.random.rand()*2*np.pi - np.pi

def assert_close(q1, q2, tol=1e-6):
    diff = q1 - q2
    max_diff = np.abs(diff).max()
    if max_diff > tol:
        msg = 'Quaternions differ by {}, q1={}, q2={}'.format(max_diff, q1, q2)
        raise AssertionError(msg)

def test_quaternions():
        
    num_samps = 1000
    for k in range(num_samps):

        # test construction from axis+angle
        ax1 = random_axis()
        ang1 = random_angle()

        q11 = Quaternion(axis=ax1, angle=ang1)
        q12 = env_util.quat_create(ax1, ang1)

        assert_close(q11.elements, q12)

        # test multiplication
        ax2 = random_axis()
        ang2 = random_angle()
        q21 = Quaternion(axis=ax2, angle=ang2)
        q22 = env_util.quat_create(ax2, ang2)

        prod1 = q11 * q21
        prod2 = env_util.quat_mul(q12, q22)
        
        assert_close(prod1.elements, prod2)

        # test inversion
        assert_close(q11.inverse.elements, env_util.quat_inv(q12))
        assert_close(q21.inverse.elements, env_util.quat_inv(q22))

        # test conversion to angle
        exp_ang = quat_to_zangle_old(q12)
        ang = env_util.quat_to_zangle(q12)
        assert np.abs(exp_ang - ang) < 1e-6, "exp_ang={}, ang={}".format(exp_ang, ang)

        # test conversion from angle
        exp_q = zangle_to_quat_old(ang1)
        q = env_util.zangle_to_quat(ang1)
        assert_close(exp_q, q)
