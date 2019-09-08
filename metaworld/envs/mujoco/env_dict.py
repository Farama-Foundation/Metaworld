import numpy as np


from metaworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place import SawyerReachPushPickPlaceEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_door import SawyerDoorEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_hand_insert import SawyerHandInsertEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_assembly_peg import SawyerNutAssemblyEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_sweep import SawyerSweepEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_window_open import SawyerWindowOpenEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_hammer import SawyerHammerEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_window_close import SawyerWindowCloseEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_dial_turn import SawyerDialTurnEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_lever_pull import SawyerLeverPullEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_drawer_open import SawyerDrawerOpenEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_button_press_topdown import SawyerButtonPressTopdownEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_drawer_close import SawyerDrawerCloseEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_box_close import SawyerBoxCloseEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_peg_insertion_side import SawyerPegInsertionSideEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_bin_picking import SawyerBinPickingEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_drawer_close import SawyerDrawerCloseEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_box_close import SawyerBoxCloseEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_stick_push import SawyerStickPushEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_stick_pull import SawyerStickPullEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_button_press import SawyerButtonPressEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_shelf_place import SawyerShelfPlaceEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_door_close import SawyerDoorCloseEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_sweep_into_goal import SawyerSweepIntoGoalEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_coffee_button import SawyerCoffeeButtonEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_coffee_push import SawyerCoffeePushEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_coffee_pull import SawyerCoffeePullEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_faucet_open import SawyerFaucetOpenEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_faucet_close import SawyerFaucetCloseEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_peg_unplug_side import SawyerPegUnplugSideEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_soccer import SawyerSoccerEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_basketball import SawyerBasketballEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place_wall import SawyerReachPushPickPlaceWallEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_push_back import SawyerPushBackEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_pick_out_of_hole import SawyerPickOutOfHoleEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_shelf_remove import SawyerShelfRemoveEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_disassemble_peg import SawyerNutDisassembleEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_door_lock import SawyerDoorLockEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_door_unlock import SawyerDoorUnlockEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_sweep_tool import SawyerSweepToolEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_button_press_wall import SawyerButtonPressWallEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_button_press_topdown_wall import SawyerButtonPressTopdownWallEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_handle_press import SawyerHandlePressEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_handle_pull import SawyerHandlePullEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_handle_press_side import SawyerHandlePressSideEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_handle_pull_side import SawyerHandlePullSideEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_plate_slide import SawyerPlateSlideEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_plate_slide_back import SawyerPlateSlideBackEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_plate_slide_side import SawyerPlateSlideSideEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_plate_slide_back_side import SawyerPlateSlideBackSideEnv


EASY_MODE_CLS_DICT= {
    'reach-v1': SawyerReachPushPickPlaceEnv,
    'push-v1': SawyerReachPushPickPlaceEnv,
    'pick-place-v1': SawyerReachPushPickPlaceEnv,
    'door-v1': SawyerDoorEnv,
    'drawer-open-v1': SawyerDrawerOpenEnv,
    'drawer-close-v1': SawyerDrawerCloseEnv,
    'button-press-topdown-v1': SawyerButtonPressTopdownEnv,
    'ped-insert-side-v1': SawyerPegInsertionSideEnv,
    'window-open-v1': SawyerWindowOpenEnv,
    'window-close-v1': SawyerWindowCloseEnv,
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
        'reach-v1': SawyerReachPushPickPlaceEnv,
        'push-v1': SawyerReachPushPickPlaceEnv,
        'pick-place-v1': SawyerReachPushPickPlaceEnv,
        'door-v1': SawyerDoorEnv,
        'drawer-close-v1': SawyerDrawerCloseEnv,
        'button-press-topdown-v1': SawyerButtonPressTopdownEnv,
        'ped-insert-side-v1': SawyerPegInsertionSideEnv,
        'window-open-v1': SawyerWindowOpenEnv,
        'sweep-v1': SawyerSweepEnv,
        'basketball-v1': SawyerBasketballEnv,
    },
    test={
        'drawer-open-v1': SawyerDrawerOpenEnv,
        'door-close-v1': SawyerDoorCloseEnv,
        'shelf-place-v1': SawyerShelfPlaceEnv,
        'sweep-into-v1': SawyerSweepIntoGoalEnv,
        'lever-pull-v1': SawyerLeverPullEnv,
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
        'reach-v1': SawyerReachPushPickPlaceEnv,
        'push-v1': SawyerReachPushPickPlaceEnv,
        'pick-place-v1': SawyerReachPushPickPlaceEnv,
        'reach-wall-v1': SawyerReachPushPickPlaceWallEnv,
        'pick-place-wall-v1': SawyerReachPushPickPlaceWallEnv,
        'push-wall-v1': SawyerReachPushPickPlaceWallEnv,
        'door-open-v1': SawyerDoorEnv,
        'door-close-v1': SawyerDoorCloseEnv,
        'drawer-open-v1': SawyerDrawerOpenEnv,
        'drawer-close-v1': SawyerDrawerCloseEnv,
        'button-press_topdown-v1': SawyerButtonPressTopdownEnv,
        'button-press-v1': SawyerButtonPressEnv,
        'button-press-topdown-wall-v1': SawyerButtonPressTopdownWallEnv,
        'button-press-wall-v1': SawyerButtonPressWallEnv,
        'peg-insert-side-v1': SawyerPegInsertionSideEnv,
        'peg-unplug-side-v1': SawyerPegUnplugSideEnv,
        'window-open-v1': SawyerWindowOpenEnv,
        'window-close-v1': SawyerWindowCloseEnv,
        'dissassemble-v1': SawyerNutDisassembleEnv,
        'hammer-v1': SawyerHammerEnv,
        'plate-slide-v1': SawyerPlateSlideEnv,
        'plate-slide-side-v1': SawyerPlateSlideSideEnv,
        'plate-slide-back-v1': SawyerPlateSlideBackEnv, 
        'plate-slide-back-side-v1': SawyerPlateSlideBackSideEnv,
        'handle-press-v1': SawyerHandlePressEnv,
        'handle-pull-v1': SawyerHandlePullEnv,
        'handle-press-side-v1': SawyerHandlePressSideEnv,
        'handle-pull-side-v1': SawyerHandlePullSideEnv,
        'stick-push-v1': SawyerStickPushEnv,
        'stick-pull-v1': SawyerStickPullEnv,
        'basket-ball-v1': SawyerBasketballEnv,
        'soccer-v1': SawyerSoccerEnv,
        'faucet-open-v1': SawyerFaucetOpenEnv,
        'faucet-close-v1': SawyerFaucetCloseEnv,
        'coffee-push-v1': SawyerCoffeePushEnv,
        'coffee-pull-v1': SawyerCoffeePullEnv,
        'coffee-button-v1': SawyerCoffeeButtonEnv,
        'sweep-v1': SawyerSweepEnv,
        'sweep-into-v1': SawyerSweepIntoGoalEnv,
        'pick-out-of-hole-v1': SawyerPickOutOfHoleEnv,
        'assembly-v1': SawyerNutAssemblyEnv,
        'shelf-place-v1': SawyerShelfPlaceEnv,
        'push-back-v1': SawyerPushBackEnv,
        'lever-pull-v1': SawyerLeverPullEnv,
        'dial-turn-v1': SawyerDialTurnEnv,},
    test={
        'bin-picking-v1': SawyerBinPickingEnv,
        'box-close-v1': SawyerBoxCloseEnv,
        'hand-insert-v1': SawyerHandInsertEnv,
        'door-lock-v1': SawyerDoorLockEnv,
        'door-unlock-v1': SawyerDoorUnlockEnv,},
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
