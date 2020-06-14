import pytest

from metaworld.envs.mujoco.env_dict import ALL_ENVIRONMENTS, ALL_V2_ENVIRONMENTS
from metaworld.policies import *
from tests.metaworld.envs.mujoco.sawyer_xyz.utils import check_success

test_data_v1 = []

test_data_v2 = [
    # name,      policy,      success rate,   environment kwargs
    ['push-wall-v2', SawyerPushWallV2Policy(), 0.8, {}]
]

for row in test_data_v1:
    row[-1] = ALL_ENVIRONMENTS[row[0]](random_init=True, **row[-1])

for row in test_data_v2:
    row[-1] = ALL_V2_ENVIRONMENTS[row[0]](random_init=True, **row[-1])

test_data = test_data_v1 + test_data_v2


@pytest.mark.parametrize('label,policy,expected_success_rate,env', test_data)
def test_scripted_policy(env, policy, label, expected_success_rate, iters=1000):
    """Tests whether a given policy solves an environment in a stateless manner

    Args:
        env (metaworld.envs.MujocoEnv): Environment to test
        policy (metaworld.policies.policy.Policy): Policy that's supposed to succeed in env
        label (string): A string to use when printing out errors
        expected_success_rate (float): Decimal value indicating % of runs that must be successful
        iters (int): How many times the policy should be tested

    """
    assert len(vars(policy)) == 0, label + ' policy has state variable(s)'

    successes = 0
    for i in range(iters):
        successes += float(check_success(env, policy, noisiness=.1)[0])
        if successes >= expected_success_rate * iters:
            break
    assert successes >= expected_success_rate * iters
