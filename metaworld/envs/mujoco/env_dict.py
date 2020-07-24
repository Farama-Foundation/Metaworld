from collections import OrderedDict

from metaworld.envs.mujoco.sawyer_xyz import (
    SawyerNutAssemblyEnv,
    SawyerBasketballEnv,
    SawyerBinPickingEnv,
    SawyerBinPickingEnvV2,
    SawyerBoxCloseEnv,
    SawyerButtonPressEnv,
    SawyerButtonPressTopdownEnv,
    SawyerButtonPressTopdownWallEnv,
    SawyerButtonPressWallEnv,
    SawyerCoffeeButtonEnv,
    SawyerCoffeePullEnv,
    SawyerCoffeePushEnv,
    SawyerDialTurnEnv,
    SawyerNutDisassembleEnv,
    SawyerDoorEnv,
    SawyerDoorCloseEnv,
    SawyerDoorLockEnv,
    SawyerDoorUnlockEnv,
    SawyerDrawerCloseEnv,
    SawyerDrawerOpenEnv,
    SawyerFaucetCloseEnv,
    SawyerFaucetOpenEnv,
    SawyerHammerEnv,
    SawyerHandInsertEnv,
    SawyerHandlePressEnv,
    SawyerHandlePressSideEnv,
    SawyerHandlePressSideEnvV2,
    SawyerHandlePullEnv,
    SawyerHandlePullSideEnv,
    SawyerLeverPullEnv,
    SawyerLeverPullEnvV2,
    SawyerPegInsertionSideEnv,
    SawyerPegInsertionSideEnvV2,
    SawyerPegUnplugSideEnv,
    SawyerPickOutOfHoleEnv,
    SawyerPickPlaceEnvV2,
    SawyerPickPlaceWallEnvV2,
    SawyerPlateSlideEnv,
    SawyerPlateSlideBackEnv,
    SawyerPlateSlideBackSideEnv,
    SawyerPlateSlideBackSideEnvV2,
    SawyerPlateSlideSideEnv,
    SawyerPushBackEnv,
    SawyerPushEnvV2,
    SawyerPushWallEnvV2,
    SawyerReachPushPickPlaceEnv,
    SawyerReachPushPickPlaceWallEnv,
    SawyerReachEnvV2,
    SawyerReachWallEnvV2,
    SawyerShelfPlaceEnv,
    SawyerSoccerEnv,
    SawyerStickPullEnv,
    SawyerStickPushEnv,
    SawyerSweepEnv,
    SawyerSweepIntoGoalEnv,
    SawyerWindowCloseEnv,
    SawyerWindowCloseEnvV2,
    SawyerWindowOpenEnv,
    SawyerWindowOpenEnvV2,
)


ALL_V1_ENVIRONMENTS = OrderedDict((
    ('reach-v1', SawyerReachPushPickPlaceEnv),
    ('push-v1', SawyerReachPushPickPlaceEnv),
    ('pick-place-v1', SawyerReachPushPickPlaceEnv),
    ('door-open-v1', SawyerDoorEnv),
    ('drawer-open-v1', SawyerDrawerOpenEnv),
    ('drawer-close-v1', SawyerDrawerCloseEnv),
    ('button-press-topdown-v1', SawyerButtonPressTopdownEnv),
    ('peg-insert-side-v1', SawyerPegInsertionSideEnv),
    ('window-open-v1', SawyerWindowOpenEnv),
    ('window-close-v1', SawyerWindowCloseEnv),
    ('door-close-v1', SawyerDoorCloseEnv),
    ('reach-wall-v1', SawyerReachPushPickPlaceWallEnv),
    ('pick-place-wall-v1', SawyerReachPushPickPlaceWallEnv),
    ('push-wall-v1', SawyerReachPushPickPlaceWallEnv),
    ('button-press-v1', SawyerButtonPressEnv),
    ('button-press-topdown-wall-v1', SawyerButtonPressTopdownWallEnv),
    ('button-press-wall-v1', SawyerButtonPressWallEnv),
    ('peg-unplug-side-v1', SawyerPegUnplugSideEnv),
    ('disassemble-v1', SawyerNutDisassembleEnv),
    ('hammer-v1', SawyerHammerEnv),
    ('plate-slide-v1', SawyerPlateSlideEnv),
    ('plate-slide-side-v1', SawyerPlateSlideSideEnv),
    ('plate-slide-back-v1', SawyerPlateSlideBackEnv),
    ('plate-slide-back-side-v1', SawyerPlateSlideBackSideEnv),
    ('handle-press-v1', SawyerHandlePressEnv),
    ('handle-pull-v1', SawyerHandlePullEnv),
    ('handle-press-side-v1', SawyerHandlePressSideEnv),
    ('handle-pull-side-v1', SawyerHandlePullSideEnv),
    ('stick-push-v1', SawyerStickPushEnv),
    ('stick-pull-v1', SawyerStickPullEnv),
    ('basketball-v1', SawyerBasketballEnv),
    ('soccer-v1', SawyerSoccerEnv),
    ('faucet-open-v1', SawyerFaucetOpenEnv),
    ('faucet-close-v1', SawyerFaucetCloseEnv),
    ('coffee-push-v1', SawyerCoffeePushEnv),
    ('coffee-pull-v1', SawyerCoffeePullEnv),
    ('coffee-button-v1', SawyerCoffeeButtonEnv),
    ('sweep-v1', SawyerSweepEnv),
    ('sweep-into-v1', SawyerSweepIntoGoalEnv),
    ('pick-out-of-hole-v1', SawyerPickOutOfHoleEnv),
    ('assembly-v1', SawyerNutAssemblyEnv),
    ('shelf-place-v1', SawyerShelfPlaceEnv),
    ('push-back-v1', SawyerPushBackEnv),
    ('lever-pull-v1', SawyerLeverPullEnv),
    ('dial-turn-v1', SawyerDialTurnEnv),
    ('bin-picking-v1', SawyerBinPickingEnv),
    ('box-close-v1', SawyerBoxCloseEnv),
    ('hand-insert-v1', SawyerHandInsertEnv),
    ('door-lock-v1', SawyerDoorLockEnv),
    ('door-unlock-v1', SawyerDoorUnlockEnv),))

ALL_V2_ENVIRONMENTS = OrderedDict((
    ('bin-picking-v2', SawyerBinPickingEnvV2),
    ('handle-press-side-v2', SawyerHandlePressSideEnvV2),
    ('lever-pull-v2', SawyerLeverPullEnvV2),
    ('reach-v2', SawyerReachEnvV2),
    ('push-v2', SawyerPushEnvV2),
    ('pick-place-v2', SawyerPickPlaceEnvV2),
    ('plate-slide-back-side-v2', SawyerPlateSlideBackSideEnvV2),
    ('peg-insert-side-v2', SawyerPegInsertionSideEnvV2),
    ('window-open-v2', SawyerWindowOpenEnvV2),
    ('window-close-v2', SawyerWindowCloseEnvV2),
    ('reach-wall-v2', SawyerReachWallEnvV2),
    ('push-wall-v2', SawyerPushWallEnvV2),
    ('pick-place-wall-v2', SawyerPickPlaceWallEnvV2),
))

_NUM_METAWORLD_ENVS = len(ALL_V1_ENVIRONMENTS)

EASY_MODE_CLS_DICT = OrderedDict((
    ('reach-v1', SawyerReachPushPickPlaceEnv),
    ('push-v1', SawyerReachPushPickPlaceEnv),
    ('pick-place-v1', SawyerReachPushPickPlaceEnv),
    ('door-open-v1', SawyerDoorEnv),
    ('drawer-open-v1', SawyerDrawerOpenEnv),
    ('drawer-close-v1', SawyerDrawerCloseEnv),
    ('button-press-topdown-v1', SawyerButtonPressTopdownEnv),
    ('peg-insert-side-v1', SawyerPegInsertionSideEnv),
    ('window-open-v1', SawyerWindowOpenEnv),
    ('window-close-v1', SawyerWindowCloseEnv)),)


EASY_MODE_ARGS_KWARGS = {
    key: dict(args=[],
              kwargs={
                  'task_id': list(ALL_V1_ENVIRONMENTS.keys()).index(key)
              })
    for key, _ in EASY_MODE_CLS_DICT.items()
}

EASY_MODE_ARGS_KWARGS['reach-v1']['kwargs']['task_type'] = 'reach'
EASY_MODE_ARGS_KWARGS['push-v1']['kwargs']['task_type'] = 'push'
EASY_MODE_ARGS_KWARGS['pick-place-v1']['kwargs']['task_type'] = 'pick_place'


MEDIUM_MODE_CLS_DICT = OrderedDict((
    ('train',
        OrderedDict((
            ('reach-v1', SawyerReachPushPickPlaceEnv),
            ('push-v1', SawyerReachPushPickPlaceEnv),
            ('pick-place-v1', SawyerReachPushPickPlaceEnv),
            ('door-open-v1', SawyerDoorEnv),
            ('drawer-close-v1', SawyerDrawerCloseEnv),
            ('button-press-topdown-v1', SawyerButtonPressTopdownEnv),
            ('peg-insert-side-v1', SawyerPegInsertionSideEnv),
            ('window-open-v1', SawyerWindowOpenEnv),
            ('sweep-v1', SawyerSweepEnv),
            ('basketball-v1', SawyerBasketballEnv)))
    ),
    ('test',
        OrderedDict((
            ('drawer-open-v1', SawyerDrawerOpenEnv),
            ('door-close-v1', SawyerDoorCloseEnv),
            ('shelf-place-v1', SawyerShelfPlaceEnv),
            ('sweep-into-v1', SawyerSweepIntoGoalEnv),
            ('lever-pull-v1', SawyerLeverPullEnv,)))
    )
))
medium_mode_train_args_kwargs = {
    key: dict(args=[], kwargs={
        'task_id' : list(ALL_V1_ENVIRONMENTS.keys()).index(key),
    })
    for key, _ in MEDIUM_MODE_CLS_DICT['train'].items()
}

medium_mode_test_args_kwargs = {
    key: dict(args=[], kwargs={'task_id' : list(ALL_V1_ENVIRONMENTS.keys()).index(key)})
    for key, _ in MEDIUM_MODE_CLS_DICT['test'].items()
}

medium_mode_train_args_kwargs['reach-v1']['kwargs']['task_type'] = 'reach'
medium_mode_train_args_kwargs['push-v1']['kwargs']['task_type'] = 'push'
medium_mode_train_args_kwargs['pick-place-v1']['kwargs'][
    'task_type'] = 'pick_place'

MEDIUM_MODE_ARGS_KWARGS = dict(
    train=medium_mode_train_args_kwargs,
    test=medium_mode_test_args_kwargs,
)
'''
    ML45 environments and arguments
'''
HARD_MODE_CLS_DICT = OrderedDict((
    ('train',
        OrderedDict((
            ('reach-v1', SawyerReachPushPickPlaceEnv),
            ('push-v1', SawyerReachPushPickPlaceEnv),
            ('pick-place-v1', SawyerReachPushPickPlaceEnv),
            ('door-open-v1', SawyerDoorEnv),
            ('drawer-open-v1', SawyerDrawerOpenEnv),
            ('drawer-close-v1', SawyerDrawerCloseEnv),
            ('button-press-topdown-v1', SawyerButtonPressTopdownEnv),
            ('peg-insert-side-v1', SawyerPegInsertionSideEnv),
            ('window-open-v1', SawyerWindowOpenEnv),
            ('window-close-v1', SawyerWindowCloseEnv),
            ('door-close-v1', SawyerDoorCloseEnv),
            ('reach-wall-v1', SawyerReachPushPickPlaceWallEnv),
            ('pick-place-wall-v1', SawyerReachPushPickPlaceWallEnv),
            ('push-wall-v1', SawyerReachPushPickPlaceWallEnv),
            ('button-press-v1', SawyerButtonPressEnv),
            ('button-press-topdown-wall-v1', SawyerButtonPressTopdownWallEnv),
            ('button-press-wall-v1', SawyerButtonPressWallEnv),
            ('peg-unplug-side-v1', SawyerPegUnplugSideEnv),
            ('disassemble-v1', SawyerNutDisassembleEnv),
            ('hammer-v1', SawyerHammerEnv),
            ('plate-slide-v1', SawyerPlateSlideEnv),
            ('plate-slide-side-v1', SawyerPlateSlideSideEnv),
            ('plate-slide-back-v1', SawyerPlateSlideBackEnv),
            ('plate-slide-back-side-v1', SawyerPlateSlideBackSideEnv),
            ('handle-press-v1', SawyerHandlePressEnv),
            ('handle-pull-v1', SawyerHandlePullEnv),
            ('handle-press-side-v1', SawyerHandlePressSideEnv),
            ('handle-pull-side-v1', SawyerHandlePullSideEnv),
            ('stick-push-v1', SawyerStickPushEnv),
            ('stick-pull-v1', SawyerStickPullEnv),
            ('basketball-v1', SawyerBasketballEnv),
            ('soccer-v1', SawyerSoccerEnv),
            ('faucet-open-v1', SawyerFaucetOpenEnv),
            ('faucet-close-v1', SawyerFaucetCloseEnv),
            ('coffee-push-v1', SawyerCoffeePushEnv),
            ('coffee-pull-v1', SawyerCoffeePullEnv),
            ('coffee-button-v1', SawyerCoffeeButtonEnv),
            ('sweep-v1', SawyerSweepEnv),
            ('sweep-into-v1', SawyerSweepIntoGoalEnv),
            ('pick-out-of-hole-v1', SawyerPickOutOfHoleEnv),
            ('assembly-v1', SawyerNutAssemblyEnv),
            ('shelf-place-v1', SawyerShelfPlaceEnv),
            ('push-back-v1', SawyerPushBackEnv),
            ('lever-pull-v1', SawyerLeverPullEnv),
            ('dial-turn-v1', SawyerDialTurnEnv),
        ))
    ),
    ('test',
        OrderedDict((
            ('bin-picking-v1', SawyerBinPickingEnv),
            ('box-close-v1', SawyerBoxCloseEnv),
            ('hand-insert-v1', SawyerHandInsertEnv),
            ('door-lock-v1', SawyerDoorLockEnv),
            ('door-unlock-v1', SawyerDoorUnlockEnv),
        ))
    )
))


def _hard_mode_args_kwargs(env_cls_, key_):
    del env_cls_

    kwargs = dict(task_id=list(ALL_V1_ENVIRONMENTS.keys()).index(key_))
    if key_ == 'reach-v1' or key_ == 'reach-wall-v1':
        kwargs['task_type'] = 'reach'
    elif key_ == 'push-v1' or key_ == 'push-wall-v1':
        kwargs['task_type'] = 'push'
    elif key_ == 'pick-place-v1' or key_ == 'pick-place-wall-v1':
        kwargs['task_type'] = 'pick_place'
    return dict(args=[], kwargs=kwargs)


HARD_MODE_ARGS_KWARGS = dict(train={}, test={})
for key, env_cls in HARD_MODE_CLS_DICT['train'].items():
    HARD_MODE_ARGS_KWARGS['train'][key] = _hard_mode_args_kwargs(env_cls, key)
for key, env_cls in HARD_MODE_CLS_DICT['test'].items():
    HARD_MODE_ARGS_KWARGS['test'][key] = _hard_mode_args_kwargs(env_cls, key)
