import pytest

from metaworld.envs.mujoco.env_dict import ALL_V1_ENVIRONMENTS, ALL_V2_ENVIRONMENTS
from metaworld.policies import *
from tests.metaworld.envs.mujoco.sawyer_xyz.utils import check_success


test_cases = [
    # name, policy, action noise pct, success rate, env kwargs
    # ['button-press-topdown-v1', SawyerButtonPressTopdownV1Policy(), .0, 1., {}],
    # ['button-press-topdown-v1', SawyerButtonPressTopdownV1Policy(), .1, .99, {}],
    # ['door-open-v1', SawyerDoorOpenV1Policy(), .0, 0.99, {}],
    # ['door-open-v1', SawyerDoorOpenV1Policy(), .1, 0.97, {}],
    # ['door-close-v1', SawyerDoorCloseV1Policy(), .0, 0.99, {}],
    # ['door-close-v1', SawyerDoorCloseV1Policy(), .1, 0.99, {}],
    # ['drawer-open-v1', SawyerDrawerOpenV1Policy(), .0, 0.99, {}],
    # ['drawer-open-v1', SawyerDrawerOpenV1Policy(), .1, 0.98, {}],
    # ['drawer-close-v1', SawyerDrawerCloseV1Policy(), .0, 0.99, {}],
    # ['drawer-close-v1', SawyerDrawerCloseV1Policy(), .1, 0.75, {}],
    # ['lever-pull-v2', SawyerLeverPullV2Policy(), .0, 1., {}],
    # ['lever-pull-v2', SawyerLeverPullV2Policy(), .1, 1., {}],
    # ['plate-slide-back-side-v1', SawyerPlateSlideBackSideV2Policy(), .0, 1., {}],
    # ['plate-slide-back-side-v1', SawyerPlateSlideBackSideV2Policy(), .1, 0.30, {}],
    # ['plate-slide-back-side-v2', SawyerPlateSlideBackSideV2Policy(), .0, 1., {}],
    # ['plate-slide-back-side-v2', SawyerPlateSlideBackSideV2Policy(), .1, 0.97, {}],
    # ['plate-slide-back-v1', SawyerPlateSlideBackV1Policy(), .0, 1., {}],
    # ['plate-slide-back-v1', SawyerPlateSlideBackV1Policy(), .1, .96, {}],
    # ['plate-slide-side-v1', SawyerPlateSlideSideV1Policy(), .0, 1., {}],
    # ['plate-slide-side-v1', SawyerPlateSlideSideV1Policy(), .1, .82, {}],
    # ['plate-slide-v2', SawyerPlateSlideV2Policy(), .0, 1., {}],
    # ['plate-slide-v2', SawyerPlateSlideV2Policy(), .1, .99, {}],
    # ['reach-v2', SawyerReachV2Policy(), .0, .99, {}],
    # ['reach-v2', SawyerReachV2Policy(), .1, .99, {}],
    # ['push-v2', SawyerPushV2Policy(), .0, .99, {}],
    # ['push-v2', SawyerPushV2Policy(), .1, .97, {}],
    # ['pick-place-v2', SawyerPickPlaceV2Policy(), .0, .96, {}],
    # ['pick-place-v2', SawyerPickPlaceV2Policy(), .1, .92, {}],
    # ['basketball-v2', SawyerBasketballV2Policy(), .0, .99, {}],
    # ['basketball-v2', SawyerBasketballV2Policy(), .1, .99, {}],
    # ['peg-insert-side-v2', SawyerPegInsertionSideV2Policy(), .0, .94, {}],
    # ['peg-insert-side-v2', SawyerPegInsertionSideV2Policy(), .1, .94, {}],
    # ['peg-unplug-side-v1', SawyerPegUnplugSideV1Policy(), .0, .99, {}],
    # ['peg-unplug-side-v1', SawyerPegUnplugSideV1Policy(), .1, .98, {}],
    # ['sweep-into-v1', SawyerSweepIntoV1Policy(), .0, 1., {}],
    # ['sweep-into-v1', SawyerSweepIntoV1Policy(), .1, 1., {}],
    # ['sweep-v1', SawyerSweepV1Policy(), .0, 1., {}],
    # ['sweep-v1', SawyerSweepV1Policy(), .1, 1., {}],
    # # drop the success rate threshold of this env by 0.05 due to its flakiness
    # ['window-open-v1', SawyerWindowOpenV2Policy(), .0, 0.80, {}],
    # ['window-open-v1', SawyerWindowOpenV2Policy(), .1, 0.81, {}],
    # ['window-open-v2', SawyerWindowOpenV2Policy(), 0., 0.96, {}],
    # ['window-open-v2', SawyerWindowOpenV2Policy(), .1, 0.96, {}],
    # ['window-close-v1', SawyerWindowCloseV2Policy(), .0, 0.37, {}],
    # ['window-close-v1', SawyerWindowCloseV2Policy(), .1, 0.37, {}],
    # # drop the success rate threshold of this env by 0.05 due to its flakiness
    # ['window-close-v2', SawyerWindowCloseV2Policy(), 0., 0.93, {}],
    # # drop the success rate threshold of this env by 0.05 due to its flakiness
    # ['window-close-v2', SawyerWindowCloseV2Policy(), .1, 0.92, {}],
    # ['button-press-v1', SawyerButtonPressV1Policy(), 0., 0.94, {}],
    # ['shelf-place-v2', SawyerShelfPlaceV2Policy(), 0.1, 0.93, {}],
    ['push-wall-v2', SawyerPushWallV2Policy(), 0.0, 0.8, {}],
    ['push-wall-v2', SawyerPushWallV2Policy(), 0.1, 0.8, {}]
]

ALL_ENVS = {**ALL_V1_ENVIRONMENTS, **ALL_V2_ENVIRONMENTS}

for row in test_cases:
    # row[-1] contains env kwargs. This instantiates an env with those kwargs,
    # then replaces row[-1] with the env object (kwargs are no longer needed)
    row[-1] = ALL_ENVS[row[0]](random_init=True, **row[-1])
    # now remove env names from test_data, as they aren't needed in parametrize
    row.pop(0)


@pytest.mark.parametrize(
    'policy,act_noise_pct,expected_success_rate,env',
    test_cases
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
