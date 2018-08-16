import gym
from gym.envs.registration import register
import logging

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
            'author': 'vitchyr'
        },
        kwargs={
            'hide_goal_markers': False,
        },
    )
    register(
        id='Image48SawyerReachXYEnv-v0',
        entry_point=create_image_48_sawyer_reach_xy_env_v0,
        tags={
            'git-commit-hash': 'c5e15f7',
            'author': 'vitchyr'
        },
    )
    register(
        id='Image84SawyerReachXYEnv-v0',
        entry_point=create_image_84_sawyer_reach_xy_env_v0,
        tags={
            'git-commit-hash': 'c5e15f7',
            'author': 'vitchyr'
        },
    )
    register(
        id='SawyerPushAndReacherXYEnv-ResetFree-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': '3d4adbe',
            'author': 'vitchyr'
        },
        kwargs=dict(
            reward_type='puck_distance',
            reset_free=True,
            hand_low=(-0.28, 0.3, 0.05),
            hand_high=(0.28, 0.9, 0.3),
            puck_low=(-.4, .2),
            puck_high=(.4, 1),
            goal_low=(-0.25, 0.3, 0.02, -.2, .4),
            goal_high=(0.25, 0.875, 0.02, .2, .8),
        )
    )
    register(
        id='SawyerPushAndReacherXYEnv-StateDistance-ResetFree-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': '3d4adbe',
            'author': 'vitchyr'
        },
        kwargs=dict(
            reward_type='state_distance',
            reset_free=True,
            hand_low=(-0.28, 0.3, 0.05),
            hand_high=(0.28, 0.9, 0.3),
            puck_low=(-.4, .2),
            puck_high=(.4, 1),
            goal_low=(-0.25, 0.3, 0.02, -.2, .4),
            goal_high=(0.25, 0.875, 0.02, .2, .8),
        )
    )


def create_image_48_sawyer_reach_xy_env_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_xyz_reacher_camera

    wrapped_env = gym.make('SawyerReachXYEnv-v0')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=sawyer_xyz_reacher_camera,
        transpose=True,
        normalize=True,
    )


def create_image_84_sawyer_reach_xy_env_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_xyz_reacher_camera

    wrapped_env = gym.make('SawyerReachXYEnv-v0')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=sawyer_xyz_reacher_camera,
        transpose=True,
        normalize=True,
    )


register_custom_envs()
