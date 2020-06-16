import pytest

from metaworld.envs.mujoco.env_dict import ALL_V1_ENVIRONMENTS, ALL_V2_ENVIRONMENTS
from metaworld.policies import *
from tests.metaworld.envs.mujoco.sawyer_xyz.utils import check_success


test_cases = [
    # name, policy, action noise pct, success rate, env kwargs
    ['peg-insert-side-v2', SawyerPegInsertionSideV2Policy(), .0, .96, {}],
    ['peg-insert-side-v2', SawyerPegInsertionSideV2Policy(), .1, .96, {}],
    ['peg-unplug-side-v1', SawyerPegUnplugSideV1Policy(), .0, .99, {}],
    ['peg-unplug-side-v1', SawyerPegUnplugSideV1Policy(), .1, .99, {}],
    ['window-open-v1', SawyerWindowOpenV2Policy(), .0, 0.86, {}],
    ['window-open-v1', SawyerWindowOpenV2Policy(), .1, 0.84, {}],
    ['window-open-v2', SawyerWindowOpenV2Policy(), 0., 0.97, {}],
    ['window-open-v2', SawyerWindowOpenV2Policy(), .1, 0.97, {}],
    ['window-close-v1', SawyerWindowCloseV2Policy(), .0, 0.44, {}],
    ['window-close-v1', SawyerWindowCloseV2Policy(), .1, 0.44, {}],
    ['window-close-v2', SawyerWindowCloseV2Policy(), 0., 0.98, {}],
    ['window-close-v2', SawyerWindowCloseV2Policy(), .1, 0.98, {}],
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
def test_scripted_policy(env, policy, act_noise_pct, expected_success_rate, iters=1000):
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
    for i in range(iters):
        successes += float(check_success(env, policy, act_noise_pct)[0])

    assert successes >= expected_success_rate * iters
