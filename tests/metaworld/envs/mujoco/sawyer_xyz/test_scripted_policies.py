import pytest

from metaworld.envs.mujoco.env_dict import ALL_V1_ENVIRONMENTS, ALL_V2_ENVIRONMENTS
from metaworld.policies import *
from tests.metaworld.envs.mujoco.sawyer_xyz.utils import trajectory_summary


test_cases_old_nonoise = [
    # This should contain configs where a V2 policy is running in a V1 env.
    # name, policy, action noise pct, success rate
    ['bin-picking-v1', SawyerBinPickingV2Policy(), .0, .50],
    ['handle-press-side-v1', SawyerHandlePressSideV2Policy(), .0, .05],
    ['lever-pull-v1', SawyerLeverPullV2Policy(), .0, .0],
    ['peg-insert-side-v1', SawyerPegInsertionSideV2Policy(), .0, .0],
    ['plate-slide-back-side-v1', SawyerPlateSlideBackSideV2Policy(), .0, 1.],
    ['window-open-v1', SawyerWindowOpenV2Policy(), .0, 0.85],
    ['window-close-v1', SawyerWindowCloseV2Policy(), .0, 0.37],
]

test_cases_old_noisy = [
    # This should contain configs where a V2 policy is running in a V1 env.
    # name, policy, action noise pct, success rate
    ['bin-picking-v1', SawyerBinPickingV2Policy(), .1, .40],
    ['handle-press-side-v1', SawyerHandlePressSideV2Policy(), .1, .77],
    ['lever-pull-v1', SawyerLeverPullV2Policy(), .1, .0],
    ['peg-insert-side-v1', SawyerPegInsertionSideV2Policy(), .1, .0],
    ['plate-slide-back-side-v1', SawyerPlateSlideBackSideV2Policy(), .1, 0.30],
    ['window-open-v1', SawyerWindowOpenV2Policy(), .1, 0.81],
    ['window-close-v1', SawyerWindowCloseV2Policy(), .1, 0.37],
]

test_cases_latest_nonoise = [
    # name, policy, action noise pct, success rate
    ['assembly-v1', SawyerAssemblyV1Policy(), .0, 1.],
    ['assembly-v2', SawyerAssemblyV2Policy(), .0, 1.],
    ['basketball-v1', SawyerBasketballV1Policy(), .0, .98],
    ['basketball-v2', SawyerBasketballV2Policy(), .0, .98],
    ['bin-picking-v2', SawyerBinPickingV2Policy(), .0, .98],
    ['box-close-v1', SawyerBoxCloseV1Policy(), .0, .85],
    ['box-close-v2', SawyerBoxCloseV2Policy(), .0, .90],
    ['button-press-topdown-v1', SawyerButtonPressTopdownV1Policy(), .0, 1.],
    ['button-press-topdown-v2', SawyerButtonPressTopdownV2Policy(), .0, .95],
    ['button-press-topdown-wall-v1', SawyerButtonPressTopdownWallV1Policy(), .0, 1.],
    ['button-press-topdown-wall-v2', SawyerButtonPressTopdownWallV2Policy(), .0, .95],
    ['button-press-v1', SawyerButtonPressV1Policy(), .0, 1.],
    ['button-press-v2', SawyerButtonPressV2Policy(), .0, 1.],
    ['button-press-wall-v1', SawyerButtonPressWallV1Policy(), .0, 1.],
    ['button-press-wall-v2', SawyerButtonPressWallV2Policy(), .0, .93],
    ['coffee-button-v1', SawyerCoffeeButtonV1Policy(), .0, 1.],
    ['coffee-button-v2', SawyerCoffeeButtonV2Policy(), .0, 1.],
    ['coffee-pull-v1', SawyerCoffeePullV1Policy(), .0, .96],
    ['coffee-pull-v2', SawyerCoffeePullV2Policy(), .0, .94],
    ['coffee-push-v1', SawyerCoffeePushV1Policy(), .0, .93],
    ['coffee-push-v2', SawyerCoffeePushV2Policy(), .0, .93],
    ['dial-turn-v1', SawyerDialTurnV1Policy(), .0, 0.96],
    ['dial-turn-v2', SawyerDialTurnV2Policy(), .0, 0.96],
    ['disassemble-v1', SawyerDisassembleV1Policy(), .0, .96],
    ['disassemble-v2', SawyerDisassembleV2Policy(), .0, .92],
    ['door-close-v1', SawyerDoorCloseV1Policy(), .0, .99],
    ['door-close-v2', SawyerDoorCloseV2Policy(), .0, .99],
    ['door-lock-v1', SawyerDoorLockV1Policy(), .0, 1.],
    ['door-lock-v2', SawyerDoorLockV2Policy(), .0, 1.],
    ['door-open-v1', SawyerDoorOpenV1Policy(), .0, .98],
    ['door-open-v2', SawyerDoorOpenV2Policy(), .0, .94],
    ['door-unlock-v1', SawyerDoorUnlockV1Policy(), .0, 1.],
    ['door-unlock-v2', SawyerDoorUnlockV2Policy(), .0, 1.],
    ['drawer-close-v1', SawyerDrawerCloseV1Policy(), .0, .99],
    ['drawer-close-v2', SawyerDrawerCloseV2Policy(), .0, .99],
    ['drawer-open-v1', SawyerDrawerOpenV1Policy(), .0, .99],
    ['drawer-open-v2', SawyerDrawerOpenV2Policy(), .0, .99],
    ['faucet-close-v1', SawyerFaucetCloseV1Policy(), .0, 1.],
    ['faucet-close-v2', SawyerFaucetCloseV2Policy(), .0, 1.],
    ['faucet-open-v1', SawyerFaucetOpenV1Policy(), .0, 1.],
    ['faucet-open-v2', SawyerFaucetOpenV2Policy(), .0, 1.],
    ['hammer-v1', SawyerHammerV1Policy(), .0, 1.],
    ['hammer-v2', SawyerHammerV2Policy(), .0, 1.],
    ['hand-insert-v1', SawyerHandInsertV1Policy(), .0, 0.96],
    ['hand-insert-v2', SawyerHandInsertV2Policy(), .0, 0.96],
    ['handle-press-side-v2', SawyerHandlePressSideV2Policy(), .0, .99],
    ['handle-press-v1', SawyerHandlePressV1Policy(), .0, 1.],
    ['handle-press-v2', SawyerHandlePressV2Policy(), .0, 1.],
    ['handle-pull-v1', SawyerHandlePullV1Policy(), .0, 1.],
    ['handle-pull-v2', SawyerHandlePullV2Policy(), .0, 0.93],
    ['handle-pull-side-v1', SawyerHandlePullSideV1Policy(), .0, .92],
    ['handle-pull-side-v2', SawyerHandlePullSideV2Policy(), .0, 1.],
    ['peg-insert-side-v2', SawyerPegInsertionSideV2Policy(), .0, .89],
    ['lever-pull-v2', SawyerLeverPullV2Policy(), .0, .94],
    ['peg-unplug-side-v1', SawyerPegUnplugSideV1Policy(), .0, .99],
    ['peg-unplug-side-v2', SawyerPegUnplugSideV2Policy(), .0, .99],
    ['pick-out-of-hole-v1', SawyerPickOutOfHoleV1Policy(), .0, 1.],
    ['pick-out-of-hole-v2', SawyerPickOutOfHoleV2Policy(), .0, 1.],
    ['pick-place-v2', SawyerPickPlaceV2Policy(), .0, .95],
    ['pick-place-wall-v2', SawyerPickPlaceWallV2Policy(), .0, .95],
    ['plate-slide-back-side-v2', SawyerPlateSlideBackSideV2Policy(), .0, 1.],
    ['plate-slide-back-v1', SawyerPlateSlideBackV1Policy(), .0, 1.],
    ['plate-slide-back-v2', SawyerPlateSlideBackV2Policy(), .0, 1.],
    ['plate-slide-side-v1', SawyerPlateSlideSideV1Policy(), .0, 1.],
    ['plate-slide-side-v2', SawyerPlateSlideSideV2Policy(), .0, 1.],
    ['plate-slide-v1', SawyerPlateSlideV1Policy(), .0, 1.],
    ['plate-slide-v2', SawyerPlateSlideV2Policy(), .0, 1.],
    ['reach-v2', SawyerReachV2Policy(), .0, .99],
    ['reach-wall-v2', SawyerReachWallV2Policy(), 0.0, .98],
    ['push-back-v1', SawyerPushBackV1Policy(), .0, .97],
    ['push-back-v2', SawyerPushBackV2Policy(), .0, .97],
    ['push-v2', SawyerPushV2Policy(), .0, .97],
    ['push-wall-v2', SawyerPushWallV2Policy(), .0, .97],
    ['shelf-place-v1', SawyerShelfPlaceV1Policy(), .0, .96],
    ['shelf-place-v2', SawyerShelfPlaceV2Policy(), .0, .96],
    ['soccer-v1', SawyerSoccerV1Policy(), .0, .88],
    ['soccer-v2', SawyerSoccerV2Policy(), .0, .88],
    ['stick-pull-v1', SawyerStickPullV1Policy(), .0, 0.95],
    ['stick-pull-v2', SawyerStickPullV2Policy(), .0, 0.96],
    ['stick-push-v1', SawyerStickPushV1Policy(), .0, 0.98],
    ['stick-push-v2', SawyerStickPushV2Policy(), .0, 0.98],
    ['sweep-into-v1', SawyerSweepIntoV1Policy(), .0, 1.],
    ['sweep-into-v2',  SawyerSweepIntoV2Policy(), .0, 0.98],
    ['sweep-v1', SawyerSweepV1Policy(), .0, 1.],
    ['sweep-v2', SawyerSweepV2Policy(), .0, 0.99],
    ['window-close-v2', SawyerWindowCloseV2Policy(), 0., .98],
    ['window-open-v2', SawyerWindowOpenV2Policy(), 0., .94],
]

test_cases_latest_noisy = [
    # name, policy, action noise pct, success rate
    ['assembly-v1', SawyerAssemblyV1Policy(), .1, .69],
    ['assembly-v2', SawyerAssemblyV2Policy(), .1, .70],
    ['basketball-v1', SawyerBasketballV1Policy(), .1, .97],
    ['basketball-v2', SawyerBasketballV2Policy(), .1, .96],
    ['bin-picking-v2', SawyerBinPickingV2Policy(), .1, .96],
    ['box-close-v1', SawyerBoxCloseV1Policy(), .1, .84],
    ['box-close-v2', SawyerBoxCloseV2Policy(), .1, .82],
    ['button-press-topdown-v1', SawyerButtonPressTopdownV1Policy(), .1, .98],
    ['button-press-topdown-v2', SawyerButtonPressTopdownV2Policy(), .1, .93],
    ['button-press-topdown-wall-v1', SawyerButtonPressTopdownWallV1Policy(), .1, .99],
    ['button-press-topdown-wall-v2', SawyerButtonPressTopdownWallV2Policy(), .1, .95],
    ['button-press-v1', SawyerButtonPressV1Policy(), .1, .98],
    ['button-press-v2', SawyerButtonPressV2Policy(), .1, .98],
    ['button-press-wall-v1', SawyerButtonPressWallV1Policy(), .1, .94],
    ['button-press-wall-v2', SawyerButtonPressWallV2Policy(), .1, .92],
    ['coffee-button-v1', SawyerCoffeeButtonV1Policy(), .1, .99],
    ['coffee-button-v2', SawyerCoffeeButtonV2Policy(), .1, .99],
    ['coffee-pull-v1', SawyerCoffeePullV1Policy(), .1, .95],
    ['coffee-pull-v2', SawyerCoffeePullV2Policy(), .1, .82],
    ['coffee-push-v1', SawyerCoffeePushV1Policy(), .1, .86],
    ['coffee-push-v2', SawyerCoffeePushV2Policy(), .1, .88],
    ['dial-turn-v1', SawyerDialTurnV1Policy(), .1, 0.84],
    ['dial-turn-v2', SawyerDialTurnV2Policy(), .1, 0.84],
    ['disassemble-v1', SawyerDisassembleV1Policy(), .1, .91],
    ['disassemble-v2', SawyerDisassembleV2Policy(), .1, .88],
    ['door-close-v1', SawyerDoorCloseV1Policy(), .1, .99],
    ['door-close-v2', SawyerDoorCloseV2Policy(), .1, .97],
    ['door-lock-v1', SawyerDoorLockV1Policy(), .1, 1.],
    ['door-lock-v2', SawyerDoorLockV2Policy(), .1, .96],
    ['door-open-v1', SawyerDoorOpenV1Policy(), .1, .93],
    ['door-open-v2', SawyerDoorOpenV2Policy(), .1, .92],
    ['door-unlock-v1', SawyerDoorUnlockV1Policy(), .1, .96],
    ['door-unlock-v2', SawyerDoorUnlockV2Policy(), .1, .97],
    ['drawer-close-v1', SawyerDrawerCloseV1Policy(), .1, .64],
    ['drawer-close-v2', SawyerDrawerCloseV2Policy(), .1, .99],
    ['drawer-open-v1', SawyerDrawerOpenV1Policy(), .1, .97],
    ['drawer-open-v2', SawyerDrawerOpenV2Policy(), .1, .97],
    ['faucet-close-v1', SawyerFaucetCloseV1Policy(), .1, .93],
    ['faucet-close-v2', SawyerFaucetCloseV2Policy(), .1, 1.],
    ['faucet-open-v1', SawyerFaucetOpenV1Policy(), .1, .99],
    ['faucet-open-v2', SawyerFaucetOpenV2Policy(), .1, .99],
    ['hammer-v1', SawyerHammerV1Policy(), .1, .97],
    ['hammer-v2', SawyerHammerV2Policy(), .1, .96],
    ['hand-insert-v1', SawyerHandInsertV1Policy(), .1, 0.95],
    ['hand-insert-v2', SawyerHandInsertV2Policy(), .1, 0.86],
    ['handle-press-side-v2', SawyerHandlePressSideV2Policy(), .1, .98],
    ['handle-press-v1', SawyerHandlePressV1Policy(), .1, 1.],
    ['handle-press-v2', SawyerHandlePressV2Policy(), .1, 1.],
    ['handle-pull-v1', SawyerHandlePullV1Policy(), .1, 1.],
    ['handle-pull-v2', SawyerHandlePullV2Policy(), .1, .99],
    ['handle-pull-side-v1', SawyerHandlePullSideV1Policy(), .1, .75],
    ['handle-pull-side-v2', SawyerHandlePullSideV2Policy(), .1, .71],
    ['peg-insert-side-v2', SawyerPegInsertionSideV2Policy(), .1, .87],
    ['lever-pull-v2', SawyerLeverPullV2Policy(), .1, .90],
    ['peg-unplug-side-v1', SawyerPegUnplugSideV1Policy(), .1, .97],
    ['peg-unplug-side-v2', SawyerPegUnplugSideV2Policy(), .1, .80],
    ['pick-out-of-hole-v1', SawyerPickOutOfHoleV1Policy(), .1, .87],
    ['pick-out-of-hole-v2', SawyerPickOutOfHoleV2Policy(), .1, .89],
    ['pick-place-v2', SawyerPickPlaceV2Policy(), .1, .83],
    ['pick-place-wall-v2', SawyerPickPlaceWallV2Policy(), .1, .83],
    ['plate-slide-back-side-v2', SawyerPlateSlideBackSideV2Policy(), .1, .95],
    ['plate-slide-back-v1', SawyerPlateSlideBackV1Policy(), .1, .95],
    ['plate-slide-back-v2', SawyerPlateSlideBackV2Policy(), .1, .94],
    ['plate-slide-side-v1', SawyerPlateSlideSideV1Policy(), .1, .76],
    ['plate-slide-side-v2', SawyerPlateSlideSideV2Policy(), .1, .78],
    ['plate-slide-v1', SawyerPlateSlideV1Policy(), .1, .97],
    ['plate-slide-v2', SawyerPlateSlideV2Policy(), .1, .97],
    ['reach-v2', SawyerReachV2Policy(), .1, .98],
    ['reach-wall-v2', SawyerReachWallV2Policy(), .1, .96],
    ['push-back-v1', SawyerPushBackV1Policy(), .1, .90],
    ['push-back-v2', SawyerPushBackV2Policy(), .0, .91],
    ['push-v2', SawyerPushV2Policy(), .1, .88],
    ['push-wall-v2', SawyerPushWallV2Policy(), .1, .82],
    ['shelf-place-v1', SawyerShelfPlaceV1Policy(), .1, .90],
    ['shelf-place-v2', SawyerShelfPlaceV2Policy(), .1, .89],
    ['soccer-v1', SawyerSoccerV1Policy(), .1, .91],
    ['soccer-v2', SawyerSoccerV2Policy(), .1, .81],
    ['stick-pull-v1', SawyerStickPullV1Policy(), .1, 0.81],
    ['stick-pull-v2', SawyerStickPullV2Policy(), .1, 0.81],
    ['stick-push-v1', SawyerStickPushV1Policy(), .1, 0.95],
    ['stick-push-v2', SawyerStickPushV2Policy(), .1, 0.95],
    ['sweep-into-v1', SawyerSweepIntoV1Policy(), .1, 1.],
    ['sweep-into-v2',  SawyerSweepIntoV2Policy(), .1, 0.86],
    ['sweep-v1', SawyerSweepV1Policy(), .1, 1.],
    ['sweep-v2', SawyerSweepV2Policy(), .0, 0.99],
    ['window-close-v2', SawyerWindowCloseV2Policy(), .1, .95],
    ['window-open-v2', SawyerWindowOpenV2Policy(), .1, .93],
]

# Combine test cases into a single array to pass to parameterized test function
test_cases = []
for row in test_cases_old_nonoise:
    test_cases.append(pytest.param(*row, marks=pytest.mark.skip))
for row in test_cases_old_noisy:
    test_cases.append(pytest.param(*row, marks=pytest.mark.skip))
for row in test_cases_latest_nonoise:
    test_cases.append(pytest.param(*row, marks=pytest.mark.skip))
for row in test_cases_latest_noisy:
    test_cases.append(pytest.param(*row, marks=pytest.mark.basic))

ALL_ENVS = {**ALL_V1_ENVIRONMENTS, **ALL_V2_ENVIRONMENTS}


@pytest.fixture(scope='function')
def env(request):
    e = ALL_ENVS[request.param]()
    e._partially_observable = False
    e._freeze_rand_vec = False
    e._set_task_called = True
    return e


@pytest.mark.parametrize(
    'env,policy,act_noise_pct,expected_success_rate',
    test_cases,
    indirect=['env']
)
def test_scripted_policy(env, policy, act_noise_pct, expected_success_rate, iters=100):
    """Tests whether a given policy solves an environment in a stateless manner
    Args:
        env (metaworld.envs.MujocoEnv): Environment to test
        policy (metaworld.policies.policy.Policy): Policy that's supposed to
            succeed in env
        act_noise_pct (np.ndarray): Decimal value(s) indicating std deviation of
            the noise as a % of action space
        expected_success_rate (float): Decimal value indicating % of runs that
            must be successful
        iters (int): How many times the policy should be tested
    """
    assert len(vars(policy)) == 0, \
        '{} has state variable(s)'.format(policy.__class__.__name__)

    successes = 0
    for _ in range(iters):
        successes += float(trajectory_summary(env, policy, act_noise_pct, render=False)[0])
    print(successes)
    assert successes >= expected_success_rate * iters
