import pytest

from metaworld import MT1
from metaworld.policies import (
    SawyerAssemblyV2Policy,
    SawyerBasketballV2Policy,
    SawyerBinPickingV2Policy,
    SawyerBoxCloseV2Policy,
    SawyerButtonPressTopdownV2Policy,
    SawyerButtonPressTopdownWallV2Policy,
    SawyerButtonPressV2Policy,
    SawyerButtonPressWallV2Policy,
    SawyerCoffeeButtonV2Policy,
    SawyerCoffeePullV2Policy,
    SawyerCoffeePushV2Policy,
    SawyerDialTurnV2Policy,
    SawyerDisassembleV2Policy,
    SawyerDoorCloseV2Policy,
    SawyerDoorLockV2Policy,
    SawyerDoorOpenV2Policy,
    SawyerDoorUnlockV2Policy,
    SawyerDrawerCloseV2Policy,
    SawyerDrawerOpenV2Policy,
    SawyerFaucetCloseV2Policy,
    SawyerFaucetOpenV2Policy,
    SawyerHammerV2Policy,
    SawyerHandInsertV2Policy,
    SawyerHandlePressSideV2Policy,
    SawyerHandlePressV2Policy,
    SawyerHandlePullSideV2Policy,
    SawyerHandlePullV2Policy,
    SawyerLeverPullV2Policy,
    SawyerPegInsertionSideV2Policy,
    SawyerPegUnplugSideV2Policy,
    SawyerPickOutOfHoleV2Policy,
    SawyerPickPlaceV2Policy,
    SawyerPickPlaceWallV2Policy,
    SawyerPlateSlideBackSideV2Policy,
    SawyerPlateSlideBackV2Policy,
    SawyerPlateSlideSideV2Policy,
    SawyerPlateSlideV2Policy,
    SawyerPushBackV2Policy,
    SawyerPushV2Policy,
    SawyerPushWallV2Policy,
    SawyerReachV2Policy,
    SawyerReachWallV2Policy,
    SawyerShelfPlaceV2Policy,
    SawyerSoccerV2Policy,
    SawyerStickPullV2Policy,
    SawyerStickPushV2Policy,
    SawyerSweepIntoV2Policy,
    SawyerSweepV2Policy,
    SawyerWindowCloseV2Policy,
    SawyerWindowOpenV2Policy,
)

policies = dict(
    {
        "assembly-v2": SawyerAssemblyV2Policy,
        "basketball-v2": SawyerBasketballV2Policy,
        "bin-picking-v2": SawyerBinPickingV2Policy,
        "box-close-v2": SawyerBoxCloseV2Policy,
        "button-press-topdown-v2": SawyerButtonPressTopdownV2Policy,
        "button-press-topdown-wall-v2": SawyerButtonPressTopdownWallV2Policy,
        "button-press-v2": SawyerButtonPressV2Policy,
        "button-press-wall-v2": SawyerButtonPressWallV2Policy,
        "coffee-button-v2": SawyerCoffeeButtonV2Policy,
        "coffee-pull-v2": SawyerCoffeePullV2Policy,
        "coffee-push-v2": SawyerCoffeePushV2Policy,
        "dial-turn-v2": SawyerDialTurnV2Policy,
        "disassemble-v2": SawyerDisassembleV2Policy,
        "door-close-v2": SawyerDoorCloseV2Policy,
        "door-lock-v2": SawyerDoorLockV2Policy,
        "door-open-v2": SawyerDoorOpenV2Policy,
        "door-unlock-v2": SawyerDoorUnlockV2Policy,
        "drawer-close-v2": SawyerDrawerCloseV2Policy,
        "drawer-open-v2": SawyerDrawerOpenV2Policy,
        "faucet-close-v2": SawyerFaucetCloseV2Policy,
        "faucet-open-v2": SawyerFaucetOpenV2Policy,
        "hammer-v2": SawyerHammerV2Policy,
        "hand-insert-v2": SawyerHandInsertV2Policy,
        "handle-press-side-v2": SawyerHandlePressSideV2Policy,
        "handle-press-v2": SawyerHandlePressV2Policy,
        "handle-pull-v2": SawyerHandlePullV2Policy,
        "handle-pull-side-v2": SawyerHandlePullSideV2Policy,
        "peg-insert-side-v2": SawyerPegInsertionSideV2Policy,
        "lever-pull-v2": SawyerLeverPullV2Policy,
        "peg-unplug-side-v2": SawyerPegUnplugSideV2Policy,
        "pick-out-of-hole-v2": SawyerPickOutOfHoleV2Policy,
        "pick-place-v2": SawyerPickPlaceV2Policy,
        "pick-place-wall-v2": SawyerPickPlaceWallV2Policy,
        "plate-slide-back-side-v2": SawyerPlateSlideBackSideV2Policy,
        "plate-slide-back-v2": SawyerPlateSlideBackV2Policy,
        "plate-slide-side-v2": SawyerPlateSlideSideV2Policy,
        "plate-slide-v2": SawyerPlateSlideV2Policy,
        "reach-v2": SawyerReachV2Policy,
        "reach-wall-v2": SawyerReachWallV2Policy,
        "push-back-v2": SawyerPushBackV2Policy,
        "push-v2": SawyerPushV2Policy,
        "push-wall-v2": SawyerPushWallV2Policy,
        "shelf-place-v2": SawyerShelfPlaceV2Policy,
        "soccer-v2": SawyerSoccerV2Policy,
        "stick-pull-v2": SawyerStickPullV2Policy,
        "stick-push-v2": SawyerStickPushV2Policy,
        "sweep-into-v2": SawyerSweepIntoV2Policy,
        "sweep-v2": SawyerSweepV2Policy,
        "window-close-v2": SawyerWindowCloseV2Policy,
        "window-open-v2": SawyerWindowOpenV2Policy,
    }
)


@pytest.mark.parametrize("env_name", MT1.ENV_NAMES)
def test_policy(env_name):
    mt1 = MT1(env_name)
    env = mt1.train_classes[env_name]()
    p = policies[env_name]()
    completed = 0
    for task in mt1.train_tasks:
        env.set_task(task)
        obs, info = env.reset()
        done = False
        count = 0
        while count < 500 and not done:
            count += 1
            a = p.get_action(obs)
            next_obs, _, trunc, termn, info = env.step(a)
            done = trunc or termn
            obs = next_obs
            if int(info["success"]) == 1:
                completed += 1
                break
    print(float(completed) / 50)
    assert (float(completed) / 50) > 0.80
