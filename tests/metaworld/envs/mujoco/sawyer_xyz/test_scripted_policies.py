import pytest

from metaworld.envs.mujoco.env_dict import ALL_V1_ENVIRONMENTS, ALL_V2_ENVIRONMENTS
from metaworld.policies import *
from tests.metaworld.envs.mujoco.sawyer_xyz.utils import check_success


test_cases = [
    # name, policy, action noise pct, success rate, env kwargs
    ['button-press-topdown-v1', SawyerButtonPressTopdownV1Policy(), .0, 1., {}],
    ['button-press-topdown-v1', SawyerButtonPressTopdownV1Policy(), .1, .99, {}],
    ['door-open-v1', SawyerDoorOpenV1Policy(), .0, 0.99, {}],
    ['door-open-v1', SawyerDoorOpenV1Policy(), .1, 0.97, {}],
    ['door-close-v1', SawyerDoorCloseV1Policy(), .0, 0.99, {}],
    ['door-close-v1', SawyerDoorCloseV1Policy(), .1, 0.99, {}],
    ['drawer-open-v1', SawyerDrawerOpenV1Policy(), .0, 0.99, {}],
    ['drawer-open-v1', SawyerDrawerOpenV1Policy(), .1, 0.98, {}],
    ['drawer-close-v1', SawyerDrawerCloseV1Policy(), .0, 0.99, {}],
    ['drawer-close-v1', SawyerDrawerCloseV1Policy(), .1, 0.75, {}],
    ['lever-pull-v2', SawyerLeverPullV2Policy(), .0, 1., {}],
    ['lever-pull-v2', SawyerLeverPullV2Policy(), .1, 1., {}],
    ['plate-slide-back-side-v1', SawyerPlateSlideBackSideV2Policy(), .0, 1., {}],
    ['plate-slide-back-side-v1', SawyerPlateSlideBackSideV2Policy(), .1, 0.30, {}],
    ['plate-slide-back-side-v2', SawyerPlateSlideBackSideV2Policy(), .0, 1., {}],
    ['plate-slide-back-side-v2', SawyerPlateSlideBackSideV2Policy(), .1, 0.97, {}],
    ['plate-slide-back-v1', SawyerPlateSlideBackV1Policy(), .0, 1., {}],
    ['plate-slide-back-v1', SawyerPlateSlideBackV1Policy(), .1, .96, {}],
    ['plate-slide-side-v1', SawyerPlateSlideSideV1Policy(), .0, 1., {}],
    ['plate-slide-side-v1', SawyerPlateSlideSideV1Policy(), .1, .80, {}],
    ['plate-slide-v2', SawyerPlateSlideV2Policy(), .0, 1., {}],
    ['plate-slide-v2', SawyerPlateSlideV2Policy(), .1, .99, {}],
    ['reach-v2', SawyerReachV2Policy(), .0, .99, {}],
    ['reach-v2', SawyerReachV2Policy(), .1, .99, {}],
    ['push-v2', SawyerPushV2Policy(), .0, .99, {}],
    ['push-v2', SawyerPushV2Policy(), .1, .97, {}],
    ['pick-place-v2', SawyerPickPlaceV2Policy(), .0, .96, {}],
    ['pick-place-v2', SawyerPickPlaceV2Policy(), .1, .92, {}],
    ['basketball-v2', SawyerBasketballV2Policy(), .0, .99, {}],
    ['basketball-v2', SawyerBasketballV2Policy(), .1, .99, {}],
    ['peg-insert-side-v2', SawyerPegInsertionSideV2Policy(), .0, .94, {}],
    ['peg-insert-side-v2', SawyerPegInsertionSideV2Policy(), .1, .92, {}],
    ['peg-unplug-side-v1', SawyerPegUnplugSideV1Policy(), .0, .99, {}],
    ['peg-unplug-side-v1', SawyerPegUnplugSideV1Policy(), .1, .98, {}],
    ['sweep-into-v1', SawyerSweepIntoV1Policy(), .0, 1., {}],
    ['sweep-into-v1', SawyerSweepIntoV1Policy(), .1, 1., {}],
    ['sweep-v1', SawyerSweepV1Policy(), .0, 1., {}],
    ['sweep-v1', SawyerSweepV1Policy(), .1, 1., {}],
    # drop the success rate threshold of this env by 0.05 due to its flakiness
    ['window-open-v1', SawyerWindowOpenV2Policy(), .0, 0.80, {}],
    ['window-open-v1', SawyerWindowOpenV2Policy(), .1, 0.81, {}],
    ['window-open-v2', SawyerWindowOpenV2Policy(), 0., 0.96, {}],
    ['window-open-v2', SawyerWindowOpenV2Policy(), .1, 0.96, {}],
    ['window-close-v1', SawyerWindowCloseV2Policy(), .0, 0.37, {}],
    ['window-close-v1', SawyerWindowCloseV2Policy(), .1, 0.37, {}],
    # drop the success rate threshold of this env by 0.05 due to its flakiness
    ['window-close-v2', SawyerWindowCloseV2Policy(), 0., 0.93, {}],
    # drop the success rate threshold of this env by 0.05 due to its flakiness
    ['window-close-v2', SawyerWindowCloseV2Policy(), .1, 0.92, {}],
    ['button-press-v1', SawyerButtonPressV1Policy(), 0., 0.94, {}],
    ['shelf-place-v2', SawyerShelfPlaceV2Policy(), 0.1, 0.93, {}],
    ['reach-wall-v2', SawyerReachWallV2Policy(), 0.0, 0.98, {}],
    ['reach-wall-v2', SawyerReachWallV2Policy(), 0.1, 0.98, {}],
    ['pick-place-wall-v2', SawyerPickPlaceWallV2Policy(), 0.0, 0.95, {}],
    ['pick-place-wall-v2', SawyerPickPlaceWallV2Policy(), 0.1, 0.92, {}],
    ['push-wall-v2', SawyerPushWallV2Policy(), 0.0, 0.95, {}],
    ['push-wall-v2', SawyerPushWallV2Policy(), 0.1, 0.85, {}],
]

test_cases_old_nonoise = [
    # This should contain configs where a V2 policy is compatible with a V1 env.
    # name, policy, action noise pct, success rate
    ['plate-slide-back-side-v1', SawyerPlateSlideBackSideV2Policy(), .0, 1.],
    ['window-open-v1', SawyerWindowOpenV2Policy(), .0, 0.85],
    ['window-close-v1', SawyerWindowCloseV2Policy(), .0, 0.37],
]

test_cases_old_noisy = [
    # This should contain configs where a V2 policy is compatible with a V1 env.
    # name, policy, action noise pct, success rate
    ['plate-slide-back-side-v1', SawyerPlateSlideBackSideV2Policy(), .1, 0.30],
    ['window-open-v1', SawyerWindowOpenV2Policy(), .1, 0.81],
    ['window-close-v1', SawyerWindowCloseV2Policy(), .1, 0.37],
]

test_cases_latest_nonoise = [
    # name, policy, action noise pct, success rate
    ['basketball-v2', SawyerBasketballV2Policy(), .0, .98],
    ['button-press-topdown-v1', SawyerButtonPressTopdownV1Policy(), .0, 1.],
    ['button-press-v1', SawyerButtonPressV1Policy(), 0., 0.94],
    ['coffee-button-v1', SawyerCoffeeButtonV1Policy(), .0, 1.],
    ['coffee-pull-v1', SawyerCoffeePullV1Policy(), .0, .98],
    ['coffee-push-v1', SawyerCoffeePushV1Policy(), .0, 1.],
    ['door-close-v1', SawyerDoorCloseV1Policy(), .0, 0.99],
    ['door-open-v1', SawyerDoorOpenV1Policy(), .0, 0.99],
    ['drawer-close-v1', SawyerDrawerCloseV1Policy(), .0, 0.99],
    ['drawer-open-v1', SawyerDrawerOpenV1Policy(), .0, 0.99],
    ['lever-pull-v2', SawyerLeverPullV2Policy(), .0, 1.],
    ['peg-insert-side-v2', SawyerPegInsertionSideV2Policy(), .0, .94],
    ['peg-unplug-side-v1', SawyerPegUnplugSideV1Policy(), .0, .99],
    ['pick-place-v2', SawyerPickPlaceV2Policy(), .0, .96],
    ['pick-place-wall-v2', SawyerPickPlaceWallV2Policy(), .0, .95],
    ['plate-slide-back-side-v2', SawyerPlateSlideBackSideV2Policy(), .0, 1.],
    ['plate-slide-back-v1', SawyerPlateSlideBackV1Policy(), .0, 1.],
    ['plate-slide-side-v1', SawyerPlateSlideSideV1Policy(), .0, 1.],
    ['plate-slide-v2', SawyerPlateSlideV2Policy(), .0, 1.],
    ['reach-v2', SawyerReachV2Policy(), .0, .99],
    ['push-v2', SawyerPushV2Policy(), .0, .99],
    ['shelf-place-v2', SawyerShelfPlaceV2Policy(), .0, .97],
    ['sweep-into-v1', SawyerSweepIntoV1Policy(), .0, 1.],
    ['sweep-v1', SawyerSweepV1Policy(), .0, 1.],
    ['window-close-v2', SawyerWindowCloseV2Policy(), 0., 0.98],
    ['window-open-v2', SawyerWindowOpenV2Policy(), 0., 0.96],
]

test_cases_latest_noisy = [
    # name, policy, action noise pct, success rate
    ['basketball-v2', SawyerBasketballV2Policy(), .1, .98],
    ['button-press-topdown-v1', SawyerButtonPressTopdownV1Policy(), .1, .98],
    ['button-press-v1', SawyerButtonPressV1Policy(), 0., 0.94],
    ['coffee-button-v1', SawyerCoffeeButtonV1Policy(), .1, 1.],
    ['coffee-pull-v1', SawyerCoffeePullV1Policy(), .1, .96],
    ['coffee-push-v1', SawyerCoffeePushV1Policy(), .1, .99],
    ['door-close-v1', SawyerDoorCloseV1Policy(), .1, 0.99],
    ['door-open-v1', SawyerDoorOpenV1Policy(), .1, 0.96],
    ['drawer-close-v1', SawyerDrawerCloseV1Policy(), .1, 0.75],
    ['drawer-open-v1', SawyerDrawerOpenV1Policy(), .1, 0.97],
    ['lever-pull-v2', SawyerLeverPullV2Policy(), .1, 1.],
    ['peg-insert-side-v2', SawyerPegInsertionSideV2Policy(), .1, .93],
    ['peg-unplug-side-v1', SawyerPegUnplugSideV1Policy(), .1, .98],
    ['pick-place-v2', SawyerPickPlaceV2Policy(), .1, .91],
    ['pick-place-wall-v2', SawyerPickPlaceWallV2Policy(), .1, .91],
    ['plate-slide-back-side-v2', SawyerPlateSlideBackSideV2Policy(), .1, 0.96],
    ['plate-slide-back-v1', SawyerPlateSlideBackV1Policy(), .1, .95],
    ['plate-slide-side-v1', SawyerPlateSlideSideV1Policy(), .1, .78],
    ['plate-slide-v2', SawyerPlateSlideV2Policy(), .1, .99],
    ['reach-v2', SawyerReachV2Policy(), .1, .99],
    ['push-v2', SawyerPushV2Policy(), .1, .94],
    ['shelf-place-v2', SawyerShelfPlaceV2Policy(), .1, 0.92],
    ['sweep-into-v1', SawyerSweepIntoV1Policy(), .1, 1.],
    ['sweep-v1', SawyerSweepV1Policy(), .1, 1.],
    ['window-close-v2', SawyerWindowCloseV2Policy(), .1, 0.96],
    ['window-open-v2', SawyerWindowOpenV2Policy(), .1, 0.95],
]


# Combine test cases into a single array to pass to parameterized test function
test_cases = []
for row in test_cases_old_nonoise:
    test_cases.append(pytest.param(*row, marks=pytest.mark.skip))
for row in test_cases_old_noisy:
    test_cases.append(pytest.param(*row, marks=pytest.mark.skip))
for row in test_cases_latest_nonoise:
    test_cases.append(pytest.param(*row, marks=pytest.mark.skip_on_ci))
for row in test_cases_latest_noisy:
    test_cases.append(pytest.param(*row, marks=pytest.mark.basic))

ALL_ENVS = {**ALL_V1_ENVIRONMENTS, **ALL_V2_ENVIRONMENTS}


@pytest.fixture(scope='function')
def env(request):
    return ALL_ENVS[request.param](random_init=True)


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
        act_noise_pct (float): Decimal value indicating std deviation of the
            noise as a % of action space
        expected_success_rate (float): Decimal value indicating % of runs that
            must be successful
        iters (int): How many times the policy should be tested
    """
    assert len(vars(policy)) == 0, \
        '{} has state variable(s)'.format(policy.__class__.__name__)

    successes = 0
    for _ in range(iters):
        successes += float(check_success(env, policy, act_noise_pct, render=False)[0])
    print(successes)
    assert successes >= expected_success_rate * iters
