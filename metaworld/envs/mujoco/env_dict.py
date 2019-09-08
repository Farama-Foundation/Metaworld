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


EASY_MODE_CLS_DICT= {
    'reach-v1': SawyerReachPushPickPlace6DOFEnv,
    'push-v1': SawyerReachPushPickPlace6DOFEnv,
    'pick-place-v1': SawyerReachPushPickPlace6DOFEnv,
    'door-v1': SawyerDoor6DOFEnv,
    'drawer-open-v1': SawyerDrawerOpen6DOFEnv,
    'drawer-close-v1': SawyerDrawerClose6DOFEnv,
    'button-press-topdown-v1': SawyerButtonPressTopdown6DOFEnv,
    'ped-insert-side-v1': SawyerPegInsertionSide6DOFEnv,
    'window-open-v1': SawyerWindowOpen6DOFEnv,
    'window-close-v1': SawyerWindowClose6DOFEnv,
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
            obs_type='with_goal_id',
        )
        goals_dict = {
            t: [e.goal.copy()]
            for t, e in zip(env._task_names, env._task_envs)
        }
        env.discretize_goal_space(goals_dict)
'''
EASY_MODE_ARGS_KWARGS = {
    key: dict(args=[], kwargs={'obs_type': 'plain'})
    for key, _ in EASY_MODE_CLS_DICT.items()
}
EASY_MODE_ARGS_KWARGS['reach-v1']['kwargs']['task_type'] = 'reach'
EASY_MODE_ARGS_KWARGS['push-v1']['kwargs']['task_type'] = 'push'
EASY_MODE_ARGS_KWARGS['pick-place-v1']['kwargs']['task_type'] = 'pick_place'

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
        'reach-v1': SawyerReachPushPickPlace6DOFEnv,
        'push-v1': SawyerReachPushPickPlace6DOFEnv,
        'pick-place-v1': SawyerReachPushPickPlace6DOFEnv,
        'door-v1': SawyerDoor6DOFEnv,
        'drawer-close-v1': SawyerDrawerClose6DOFEnv,
        'button-press-topdown-v1': SawyerButtonPressTopdown6DOFEnv,
        'ped-insert-side-v1': SawyerPegInsertionSide6DOFEnv,
        'window-open-v1': SawyerWindowOpen6DOFEnv,
        'sweep-v1': SawyerSweep6DOFEnv,
        'basketball-v1': SawyerBasketball6DOFEnv,
    },
    test={
        'drawer-open-v1': SawyerDrawerOpen6DOFEnv,
        'door-close-v1': SawyerDoorClose6DOFEnv,
        'shelf-place-v1': SawyerShelfPlace6DOFEnv,
        'sweep-into-v1': SawyerSweepIntoGoal6DOFEnv,
        'lever-pull-v1': SawyerLeverPull6DOFEnv,
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

medium_mode_train_args_kwargs['reach-v1']['kwargs']['task_type'] = 'reach'
medium_mode_train_args_kwargs['push-v1']['kwargs']['task_type'] = 'push'
medium_mode_train_args_kwargs['pick-place-v1']['kwargs']['task_type'] = 'pick_place'

MEDIUM_MODE_ARGS_KWARGS = dict(
    train=medium_mode_train_args_kwargs,
    test=medium_mode_test_args_kwargs,
)


'''
    ML45 environments and arguments
'''
HARD_MODE_CLS_DICT = dict(
    train={
        'reach-v1': SawyerReachPushPickPlace6DOFEnv,
        'push-v1': SawyerReachPushPickPlace6DOFEnv,
        'pick-place-v1': SawyerReachPushPickPlace6DOFEnv,
        'reach-wall-v1': SawyerReachPushPickPlaceWall6DOFEnv,
        'pick-place-wall-v1': SawyerReachPushPickPlaceWall6DOFEnv,
        'push-wall-v1': SawyerReachPushPickPlaceWall6DOFEnv,
        'door-open-v1': SawyerDoor6DOFEnv,
        'door-close-v1': SawyerDoorClose6DOFEnv,
        'drawer-open-v1': SawyerDrawerOpen6DOFEnv,
        'drawer-close-v1': SawyerDrawerClose6DOFEnv,
        'button-press_topdown-v1': SawyerButtonPressTopdown6DOFEnv,
        'button-press-v1': SawyerButtonPress6DOFEnv,
        'button-press-topdown-wall-v1': SawyerButtonPressTopdownWall6DOFEnv,
        'button-press-wall-v1': SawyerButtonPressWall6DOFEnv,
        'peg-insert-side-v1': SawyerPegInsertionSide6DOFEnv,
        'peg-unplug-side-v1': SawyerPegUnplugSide6DOFEnv,
        'window-open-v1': SawyerWindowOpen6DOFEnv,
        'window-close-v1': SawyerWindowClose6DOFEnv,
        'dissassemble-v1': SawyerNutDisassemble6DOFEnv,
        'hammer-v1': SawyerHammer6DOFEnv,
        'plate-slide-v1': SawyerPlateSlide6DOFEnv,
        'plate-slide-side-v1': SawyerPlateSlideSide6DOFEnv,
        'plate-slide-back-v1': SawyerPlateSlideBack6DOFEnv, 
        'plate-slide-back-side-v1': SawyerPlateSlideBackSide6DOFEnv,
        'handle-press-v1': SawyerHandlePress6DOFEnv,
        'handle-pull-v1': SawyerHandlePull6DOFEnv,
        'handle-press-side-v1': SawyerHandlePressSide6DOFEnv,
        'handle-pull-side-v1': SawyerHandlePullSide6DOFEnv,
        'stick-push-v1': SawyerStickPush6DOFEnv,
        'stick-pull-v1': SawyerStickPull6DOFEnv,
        'basket-ball-v1': SawyerBasketball6DOFEnv,
        'soccer-v1': SawyerSoccer6DOFEnv,
        'faucet-open-v1': SawyerFaucetOpen6DOFEnv,
        'faucet-close-v1': SawyerFaucetClose6DOFEnv,
        'coffee-push-v1': SawyerCoffeePush6DOFEnv,
        'coffee-pull-v1': SawyerCoffeePull6DOFEnv,
        'coffee-button-v1': SawyerCoffeeButton6DOFEnv,
        'sweep-v1': SawyerSweep6DOFEnv,
        'sweep-into-v1': SawyerSweepIntoGoal6DOFEnv,
        'pick-out-of-hole-v1': SawyerPickOutOfHole6DOFEnv,
        'assembly-v1': SawyerNutAssembly6DOFEnv,
        'shelf-place-v1': SawyerShelfPlace6DOFEnv,
        'push-back-v1': SawyerPushBack6DOFEnv,
        'lever-pull-v1': SawyerLeverPull6DOFEnv,
        'dial-turn-v1': SawyerDialTurn6DOFEnv,},
    test={
        'bin-picking-v1': SawyerBinPicking6DOFEnv,
        'box-close-v1': SawyerBoxClose6DOFEnv,
        'hand-insert-v1': SawyerHandInsert6DOFEnv,
        'door-lock-v1': SawyerDoorLock6DOFEnv,
        'door-unlock-v1': SawyerDoorUnlock6DOFEnv,},
)


def _hard_mode_args_kwargs(env_cls, key):
    kwargs = dict(random_init=True, obs_type='plain')
    if key == 'reach-v1' or key == 'reach-wall-v1':
        kwargs['task_type'] = 'reach'
    elif key == 'push-v1' or key == 'push-wall-v1':
        kwargs['task_type'] = 'push'
    elif key == 'pick-place-v1' or key == 'pick-place-wall-v1':
        kwargs['task_type'] = 'pick_place'
    return dict(args=[], kwargs=kwargs)


HARD_MODE_ARGS_KWARGS = dict(train={}, test={})
for key, env_cls in HARD_MODE_CLS_DICT['train'].items():
    HARD_MODE_ARGS_KWARGS['train'][key] = _hard_mode_args_kwargs(env_cls, key)
for key, env_cls in HARD_MODE_CLS_DICT['test'].items():
    HARD_MODE_ARGS_KWARGS['test'][key] = _hard_mode_args_kwargs(env_cls, key)
