import pytest

# from tests.helpers import close_env
from multiworld.envs.mujoco.sawyer_xyz.base import OBS_TYPE
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
from multiworld.envs.mujoco.sawyer_xyz.sawyer_button_press_6dof import SawyerButtonPress6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_bin_picking_6dof import SawyerBinPicking6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_sweep_into_goal import SawyerSweepIntoGoal6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door_close import SawyerDoorClose6DOFEnv

env_cls = [
    SawyerReachPushPickPlace6DOFEnv,
    SawyerDoor6DOFEnv,
    SawyerStack6DOFEnv,
    SawyerHandInsert6DOFEnv,
    SawyerNutAssembly6DOFEnv,
    SawyerSweep6DOFEnv,
    SawyerWindowOpen6DOFEnv,
    SawyerHammer6DOFEnv,
    SawyerWindowClose6DOFEnv,
    SawyerDialTurn6DOFEnv,
    SawyerLeverPull6DOFEnv,
    SawyerDrawerOpen6DOFEnv,
    SawyerDrawerClose6DOFEnv,
    SawyerButtonPressTopdown6DOFEnv,
    SawyerBoxOpen6DOFEnv,
    SawyerBoxClose6DOFEnv,
    SawyerPegInsertionSide6DOFEnv,
    SawyerButtonPress6DOFEnv,
    SawyerBinPicking6DOFEnv,
    SawyerSweepIntoGoal6DOFEnv,
    SawyerDoorClose6DOFEnv,
]


@pytest.mark.parametrize('env_cls', env_cls)
def test_obs_type(env_cls):
    for c in env_cls:
        for t in OBS_TYPE:
            if t == 'with_goal_and_id':
                env = c(obs_type=t, multitask_num=2, multitask=True)
            else:
                env = c(obs_type=t)
            o = env.reset()
            o_g = env._get_obs()
            space = env.observation_space
            assert space.shape == o.shape, 'type: {}, env: {}'.format(t, env)
            assert space.shape == o_g.shape, 'type: {}, env: {}'.format(t, env)
