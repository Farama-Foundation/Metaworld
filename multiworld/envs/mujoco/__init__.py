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

    """
    Reaching tasks
    """
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

    """
    Pushing tasks, XY, With Reset
    """
    register(
        id='SawyerPushAndReacherXYEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': '3503e9f',
            'author': 'vitchyr'
        },
        kwargs=dict(
            hide_goal_markers=True,
            action_scale=.02,
            puck_low=[-0.25, .4],
            puck_high=[0.25, .8],
            mocap_low=[-0.2, 0.45, 0.],
            mocap_high=[0.2, 0.75, 0.5],
            goal_low=[-0.2, 0.45, 0.02, -0.25, 0.4],
            goal_high=[0.2, 0.75, 0.02, 0.25, 0.8],
        )
    )
    register(
        id='Image48SawyerPushAndReacherXYEnv-v0',
        entry_point=create_Image48SawyerPushAndReacherXYEnv_v0,
        tags={
            'git-commit-hash': '3503e9f',
            'author': 'vitchyr'
        },
    )
    register(
        id='SawyerPushAndReachXYEasyEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': 'fec148f',
            'author': 'vitchyr'
        },
        kwargs=dict(
            reward_type='puck_distance',
            reset_free=False,
            hand_low=(-0.28, 0.3, 0.05),
            hand_high=(0.28, 0.9, 0.3),
            puck_low=(-.4, .2),
            puck_high=(.4, 1),
            goal_low=(-0.05, 0.4, 0.02, -.1, .5),
            goal_high=(0.05, 0.7, 0.02, .1, .7),
        )
    )
    register(
        id='Image48SawyerPushAndReachXYEasyEnv-v0',
        entry_point=create_image_48_sawyer_reach_and_reach_xy_easy_env_v0,
        tags={
            'git-commit-hash': 'fec148f',
            'author': 'vitchyr'
        },
    )

    """
    Pushing tasks, XY, Reset Free
    """
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
        id='SawyerPushXYEnv-ResetFree-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': '33c6b71',
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
        id='SawyerPushXYEnv-ResetFree-v1',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': '33c6b71',
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
        id='SawyerPushXYEnv-CompleteResetFree-v1',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': 'b9b5ce0',
            'author': 'murtaza'
        },
        kwargs=dict(
            reward_type='puck_distance',
            hand_low=(-0.28, 0.3, 0.05),
            hand_high=(0.28, 0.9, 0.3),
            puck_low=(-.4, .2),
            puck_high=(.4, 1),
            goal_low=(-0.25, 0.3, 0.02, -.2, .4),
            goal_high=(0.25, 0.875, 0.02, .2, .8),
            num_resets_before_puck_reset=int(1e6),
            num_resets_before_hand_reset=int(1e6),
        )
    )

    register(
        id='SawyerPushAndReachXYEnv-ResetFree-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': '33c6b71',
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
            num_resets_before_puck_reset=1,
        )
    )
    register(
        id='SawyerPushAndReachXYEnv-ResetFree-Every1B-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': '33c6b71',
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
            num_resets_before_puck_reset=int(1e9),
        )
    )
    register(
        id='SawyerPushAndReachXYEnv-ResetFree-Every2-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': '33c6b71',
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
            num_resets_before_puck_reset=2,
        )
    )
    register(
        id='SawyerPushAndReachXYEnv-ResetFree-Every3-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env:SawyerPushAndReachXYEnv',
        tags={
            'git-commit-hash': '33c6b71',
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
            num_resets_before_puck_reset=3,
        )
    )

    """
    Push XYZ
    """
    register(
        id='SawyerPushXyzEasyEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_and_reach_env:SawyerPushAndReachXYZEnv',
        tags={
            'git-commit-hash': 'f7d1e91',
            'author': 'vitchyr'
        },
        kwargs=dict(
            reward_type='puck_distance',
            reset_free=False,
            hand_low=(-0.28, 0.3, 0.05),
            hand_high=(0.28, 0.9, 0.3),
            puck_low=(-.4, .2),
            puck_high=(.4, 1),
            goal_low=(-0.05, 0.4, 0.02, -.1, .5),
            goal_high=(0.05, 0.7, 0.02, .1, .7),
        )
    )
    register(
        id='SawyerPushAndReachXyzEasyEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_and_reach_env:SawyerPushAndReachXYZEnv',
        tags={
            'git-commit-hash': 'f7d1e91',
            'author': 'vitchyr'
        },
        kwargs=dict(
            reward_type='state_distance',
            reset_free=False,
            hand_low=(-0.28, 0.3, 0.05),
            hand_high=(0.28, 0.9, 0.3),
            puck_low=(-.4, .2),
            puck_high=(.4, 1),
            goal_low=(-0.05, 0.4, 0.02, -.1, .5),
            goal_high=(0.05, 0.7, 0.02, .1, .7),
        )
    )
    register(
        id='SawyerPushXyzFullArenaEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_and_reach_env:SawyerPushAndReachXYZEnv',
        tags={
            'git-commit-hash': 'f7d1e91',
            'author': 'vitchyr'
        },
        kwargs=dict(
            reward_type='puck_distance',
            reset_free=False,
            hand_low=(-0.28, 0.3, 0.05),
            hand_high=(0.28, 0.9, 0.3),
            puck_low=(-.4, .2),
            puck_high=(.4, 1),
            goal_low=(-0.25, 0.3, 0.02, -.2, .4),
            goal_high=(0.25, 0.875, 0.02, .2, .8),
        )
    )
    register(
        id='SawyerPushAndReachXyzFullArenaEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_push_and_reach_env:SawyerPushAndReachXYZEnv',
        tags={
            'git-commit-hash': 'f7d1e91',
            'author': 'vitchyr'
        },
        kwargs=dict(
            reward_type='state_distance',
            reset_free=False,
            hand_low=(-0.28, 0.3, 0.05),
            hand_high=(0.28, 0.9, 0.3),
            puck_low=(-.4, .2),
            puck_high=(.4, 1),
            goal_low=(-0.25, 0.3, 0.02, -.2, .4),
            goal_high=(0.25, 0.875, 0.02, .2, .8),
        )
    )
    register(
        id='SawyerDoorPullEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_door:SawyerDoorEnv',
        tags={
            'git-commit-hash': '19f2be6',
            'author': 'vitchyr'
        },
        kwargs=dict(
            goal_low=(-.25, .3, .12, -1.5708),
            goal_high=(.25, .6, .12, 0),
            action_reward_scale=0,
            reward_type='angle_difference',
            indicator_threshold=(.02, .03),
            fix_goal=False,
            fixed_goal=(0, .45, .12, -.25),
            num_resets_before_door_and_hand_reset=1,
            fixed_hand_z=0.12,
            hand_low=(-0.25, 0.3, .12),
            hand_high=(0.25, 0.6, .12),
            target_pos_scale=1,
            target_angle_scale=1,
            min_angle=-1.5708,
            max_angle=0,
            xml_path='sawyer_xyz/sawyer_door_pull.xml',
        )
    )

    """
    Door Hook Env
    """
    register(
        id='SawyerDoorHookEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_door_hook:SawyerDoorHookEnv',
        tags={
            'git-commit-hash': 'b5ac6f9',
            'author': 'vitchyr',
        },
        kwargs=dict(
            goal_low=(-0.1, 0.42, 0.05, 0),
            goal_high=(0.0, 0.65, .075, 1.0472),
            hand_low=(-0.1, 0.42, 0.05),
            hand_high=(0., 0.65, .075),
            max_angle=1.0472,
            xml_path='sawyer_xyz/sawyer_door_pull_hook.xml',
        )
    )
    register(
        id='Image48SawyerDoorHookEnv-v0',
        entry_point=create_Image48SawyerDoorHookEnv_v0,
        tags={
            'git-commit-hash': 'b5ac6f9',
            'author': 'vitchyr',
        },
    )
    register(
        id='SawyerDoorHookResetFreeEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_door_hook:SawyerDoorHookEnv',
        tags={
            'git-commit-hash': 'b5ac6f9',
            'author': 'vitchyr',
        },
        kwargs=dict(
            goal_low=(-0.1, 0.42, 0.05, 0),
            goal_high=(0.0, 0.65, .075, 1.0472),
            hand_low=(-0.1, 0.42, 0.05),
            hand_high=(0., 0.65, .075),
            max_angle=1.0472,
            xml_path='sawyer_xyz/sawyer_door_pull_hook.xml',
            reset_free=True,
        )
    )
    register(
        id='Image48SawyerDoorHookResetFreeEnv-v0',
        entry_point=create_Image48SawyerDoorHookResetFreeEnv_v0,
        tags={
            'git-commit-hash': 'b5ac6f9',
            'author': 'vitchyr',
        },
    )

    register(
        id='SawyerDoorHookResetFreeEnv-v1',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_door_hook:SawyerDoorHookEnv',
        tags={
            'git-commit-hash': '333776f',
            'author': 'murtaza',
        },
        kwargs=dict(
            goal_low=(-0.1, 0.45, 0.15, 0),
            goal_high=(0.0, 0.65, .225, 1.0472),
            hand_low=(-0.1, 0.45, 0.15),
            hand_high=(0., 0.65, .225),
            max_angle=1.0472,
            xml_path='sawyer_xyz/sawyer_door_pull_hook.xml',
            reset_free=True,
        )
    )
    register(
        id='Image48SawyerDoorHookResetFreeEnv-v1',
        entry_point=create_Image48SawyerDoorHookResetFreeEnv_v1,
        tags={
            'git-commit-hash': '333776f',
            'author': 'murtaza',
        },
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


def create_image_48_sawyer_reach_and_reach_xy_easy_env_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2

    wrapped_env = gym.make('SawyerPushAndReachXYEasyEnv-v0')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=sawyer_pusher_camera_upright_v2,
        transpose=True,
        normalize=True,
    )


def create_Image48SawyerPushAndReacherXYEnv_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_top_down

    wrapped_env = gym.make('SawyerPushAndReacherXYEnv-v0')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=sawyer_pusher_camera_top_down,
        transpose=True,
        normalize=True,
    )


def create_Image48SawyerDoorHookEnv_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_door_env_camera_v3

    wrapped_env = gym.make('SawyerDoorHookEnv-v0')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=sawyer_door_env_camera_v3,
        transpose=True,
        normalize=True,
    )


def create_Image48SawyerDoorHookResetFreeEnv_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_door_env_camera_v3

    wrapped_env = gym.make('SawyerDoorHookResetFreeEnv-v0')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=sawyer_door_env_camera_v3,
        transpose=True,
        normalize=True,
    )

def create_Image48SawyerDoorHookResetFreeEnv_v1():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_door_env_camera_v3

    wrapped_env = gym.make('SawyerDoorHookResetFreeEnv-v1')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=sawyer_door_env_camera_v3,
        transpose=True,
        normalize=True,
    )


register_custom_envs()
