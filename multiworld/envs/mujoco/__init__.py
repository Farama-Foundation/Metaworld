import gym
from gym.envs.registration import register
import logging

from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import sawyer_xyz_reacher_camera

LOGGER = logging.getLogger(__name__)

_REGISTERED = False


def register_custom_envs():
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    LOGGER.info("Registering multiworld mujoco gym environments")
    register(
        id='SawyerReachXYEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_reach:SawyerReachXYEnv',
        tags={
            'git-commit-hash': 'c5e15f7',
            'author': 'Vitchyr'
        },
        kwargs={
            'hide_goal_markers': False,
        },
    )
    register(
        id='image48sawyerreachxyenv-v0',
        entry_point=create_image_48_sawyer_reach_xy_env_v0,
        tags={
            'git-commit-hash': 'c5e15f7',
            'author': 'vitchyr'
        },
    )
    register(
        id='image84sawyerreachxyenv-v0',
        entry_point=create_image_84_sawyer_reach_xy_env_v0,
        tags={
            'git-commit-hash': 'c5e15f7',
            'author': 'vitchyr'
        },
    )


def create_image_48_sawyer_reach_xy_env_v0():
    wrapped_env = gym.make('SawyerReachXYEnv-v0')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=sawyer_xyz_reacher_camera,
        transpose=True,
        normalize=True,
    )


def create_image_84_sawyer_reach_xy_env_v0():
    wrapped_env = gym.make('SawyerReachXYEnv-v0')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=sawyer_xyz_reacher_camera,
        transpose=True,
        normalize=True,
    )


register_custom_envs()
