'''
This file provide lists of environment for multitask learning.
'''

from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place_6dof import SawyerReachPushPickPlace6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door_6dof import SawyerDoor6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_stack_6dof import SawyerStack6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_hand_insert import SawyerHandInsert6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_assembly_peg_6dof import SawyerNutAssembly6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_sweep import SawyerSweep6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_window_open_6dof import SawyerWindowOpen6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_hammer_6dof import SawyerHammer6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_window_close_6dof import SawyerWindowClose6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_dial_turn_6dof import SawyerDialTurn6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_lever_pull import SawyerLeverPull6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_drawer_open_6dof import SawyerDrawerOpen6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_button_press_topdown_6dof import SawyerButtonPressTopdown6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_drawer_close_6dof import SawyerDrawerClose6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_box_open_6dof import SawyerBoxOpen6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_box_close_6dof import SawyerBoxClose6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_peg_insertion_side_6dof import SawyerPegInsertionSide6DOFEnv


# easy mode for CoRL
EASY_MODE_LIST = [
    SawyerReachPushPickPlace6DOFEnv,
    SawyerDoor6DOFEnv,
    SawyerDrawerOpen6DOFEnv,
    SawyerDrawerClose6DOFEnv,
    SawyerButtonPressTopdown6DOFEnv,
    SawyerPegInsertionSide6DOFEnv,
    SawyerWindowOpen6DOFEnv,
    SawyerWindowClose6DOFEnv,
]


def verify_env_list_space(env_list):
    '''
    This method verifies the action_space and observation_space
    of all environments in env_list are the same.
    '''
    prev_action_space = None
    prev_obs_space = None
    for env_cls in env_list:
        env = env_cls()
        if prev_action_space is None or prev_obs_space is None:
            prev_action_space = env.action_space
            prev_obs_space = env.observation_space
            continue
        assert env.action_space.shape == prev_action_space.shape,\
            '{}, {}, {}'.format(env, env.action_space.shape, prev_action_space)
        assert env.observation_space.shape == prev_obs_space.shape,\
            '{}, {}, {}'.format(env, env.observation_space.shape, prev_obs_space)
        prev_action_space = env.action_space
        prev_obs_space = env.observation_space
