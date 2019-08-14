import numpy as np


from metaworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place_6dof import SawyerReachPushPickPlace6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_door_6dof import SawyerDoor6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_hand_insert import SawyerHandInsert6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_assembly_peg_6dof import SawyerNutAssembly6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_sweep import SawyerSweep6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_window_open_6dof import SawyerWindowOpen6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_hammer_6dof import SawyerHammer6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_window_close_6dof import SawyerWindowClose6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_dial_turn_6dof import SawyerDialTurn6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_lever_pull import SawyerLeverPull6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_drawer_open_6dof import SawyerDrawerOpen6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_button_press_topdown_6dof import SawyerButtonPressTopdown6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_drawer_close_6dof import SawyerDrawerClose6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_box_close_6dof import SawyerBoxClose6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_peg_insertion_side_6dof import SawyerPegInsertionSide6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_bin_picking_6dof import SawyerBinPicking6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_drawer_close_6dof import SawyerDrawerClose6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_box_close_6dof import SawyerBoxClose6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_stick_push_6dof import SawyerStickPush6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_stick_pull_6dof import SawyerStickPull6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_button_press_6dof import SawyerButtonPress6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_shelf_place_6dof import SawyerShelfPlace6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_door_close import SawyerDoorClose6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_sweep_into_goal import SawyerSweepIntoGoal6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_coffee_button_6dof import SawyerCoffeeButton6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_coffee_push_6dof import SawyerCoffeePush6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_coffee_pull_6dof import SawyerCoffeePull6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_faucet_open import SawyerFaucetOpen6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_faucet_close import SawyerFaucetClose6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_peg_unplug_side_6dof import SawyerPegUnplugSide6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_soccer import SawyerSoccer6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_basketball import SawyerBasketball6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place_wall_6dof import SawyerReachPushPickPlaceWall6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_push_back_6dof import SawyerPushBack6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_pick_out_of_hole import SawyerPickOutOfHole6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_shelf_remove_6dof import SawyerShelfRemove6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_disassemble_peg_6dof import SawyerNutDisassemble6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_door_lock import SawyerDoorLock6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_door_unlock import SawyerDoorUnlock6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_sweep_tool import SawyerSweepTool6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_button_press_wall_6dof import SawyerButtonPressWall6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_button_press_topdown_wall_6dof import SawyerButtonPressTopdownWall6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_handle_press_6dof import SawyerHandlePress6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_handle_pull_6dof import SawyerHandlePull6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_handle_press_side_6dof import SawyerHandlePressSide6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_handle_pull_side_6dof import SawyerHandlePullSide6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_plate_slide_6dof import SawyerPlateSlide6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_plate_slide_back_6dof import SawyerPlateSlideBack6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_plate_slide_side_6dof import SawyerPlateSlideSide6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_plate_slide_back_side_6dof import SawyerPlateSlideBackSide6DOFEnv


SHARE_CLS_TASKS = ['reach', 'pick_place', 'push']
SHARE_CLS_DEFAULT_GOALS = {
    'pick_place': np.array([0.1, 0.8, 0.2]),
    'reach': np.array([-0.1, 0.8, 0.2]),
    'push': np.array([0.1, 0.8, 0.02]),
}

EASY_MODE_CLS_DICT= {
    'reach': SawyerReachPushPickPlace6DOFEnv,
    'push': SawyerReachPushPickPlace6DOFEnv,
    'pick_place': SawyerReachPushPickPlace6DOFEnv,
    'door': SawyerDoor6DOFEnv,
    'drawer_open': SawyerDrawerOpen6DOFEnv,
    'drawer_close': SawyerDrawerClose6DOFEnv,
    'button_press_topdown': SawyerButtonPressTopdown6DOFEnv,
    'ped_insert_side': SawyerPegInsertionSide6DOFEnv,
    'window_open': SawyerWindowOpen6DOFEnv,
    'window_close': SawyerWindowClose6DOFEnv,
}


'''
    MT10 environments and arguments.
    Example usage:

        from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv
        from metaworld.envs.mujoco.env_dict import EASY_MODE_CLS_DICT, EASY_MODE_ARGS_KWARGS, SHARE_CLS_DEFAULT_GOALS

        env = MultiClassMultiTaskEnv(
            task_env_cls_dict=EASY_MODE_CLS_DICT,
            task_args_kwargs=EASY_MODE_ARGS_KWARGS,
            sample_goals=False,
            obs_type='with_goal_idx',
        )
        goals_dict = {
            t: [e.goal.copy()]
            for t, e in zip(env._task_names, env._task_envs)
        }
        for t, g in SHARE_CLS_DEFAULT_GOALS:
            goals_dict[t] = [g.copy()]
        env.discretize_goal_space(goals_dict)
'''
EASY_MODE_ARGS_KWARGS = {
    key: dict(args=[], kwargs={'obs_type': 'plain'})
    for key, _ in EASY_MODE_CLS_DICT.items()
}
for t in SHARE_CLS_TASKS:
    EASY_MODE_ARGS_KWARGS[t]['kwargs']['task_type'] = t


'''
    ML10 environments and arguments
    Example usage for meta-training:

        from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv
        from metaworld.envs.mujoco.env_dict import MEDIUM_MODE_CLS_DICT, MEDIUM_MODE_ARGS_KWARGS

        # goals are sampled and set anyways so we don't care about the default goal of reach
        # pick_place, push are the same.
        env = MultiClassMultiTaskEnv(
            task_env_cls_dict=MEDIUM_MODE_CLS_DICT['train'],
            task_args_kwargs=MEDIUM_MODE_ARGS_KWARGS['train'],
            sample_goals=True,
            obs_type='plain',
        )
'''

MEDIUM_MODE_CLS_DICT = dict(
    train={
        'reach': SawyerReachPushPickPlace6DOFEnv,
        'push': SawyerReachPushPickPlace6DOFEnv,
        'pick_place': SawyerReachPushPickPlace6DOFEnv,
        'door': SawyerDoor6DOFEnv,
        'drawer_close': SawyerDrawerClose6DOFEnv,
        'button_press_topdown': SawyerButtonPressTopdown6DOFEnv,
        'ped_insert_side': SawyerPegInsertionSide6DOFEnv,
        'window_open': SawyerWindowOpen6DOFEnv,
        'sweep': SawyerSweep6DOFEnv,
        'basketball': SawyerBasketball6DOFEnv,
    },
    test={
        'drawer_close': SawyerDrawerClose6DOFEnv,
        'door_close': SawyerDoorClose6DOFEnv,
        'shelf_place': SawyerShelfPlace6DOFEnv,
        'sweep': SawyerSweep6DOFEnv,
        'lever_pull': SawyerLeverPull6DOFEnv,
    }
)
medium_mode_train_args_kwargs = {
    key: dict(args=[], kwargs={'obs_type': 'plain', 'random_init': True})
    for key, _ in MEDIUM_MODE_CLS_DICT['train'].items()
}

medium_mode_test_args_kwargs = {
    key: dict(args=[], kwargs={'obs_type': 'plain'})
    for key, _ in MEDIUM_MODE_CLS_DICT['test'].items()
}

for t in SHARE_CLS_TASKS:
    medium_mode_train_args_kwargs[t]['kwargs']['task_type'] = t

MEDIUM_MODE_ARGS_KWARGS = dict(
    train=medium_mode_train_args_kwargs,
    test=medium_mode_test_args_kwargs,
)
