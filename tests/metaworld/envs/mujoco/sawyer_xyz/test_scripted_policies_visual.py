import pytest

from metaworld.envs.mujoco.env_dict import ALL_VISUAL_ENVS
from metaworld.policies import *
from tests.metaworld.envs.mujoco.sawyer_xyz.utils import trajectory_summary


test_cases_nonoise = [
    # name, policy, action noise pct, success rate
    ['visual-sandbox-small', SawyerAssemblyV2Policy(), .0, .0],
]

test_cases_noisy = [
    # name, policy, action noise pct, success rate
    ['visual-sandbox-small', SawyerAssemblyV2Policy(), .1, .0],
]


# Combine test cases into a single array to pass to parameterized test function
test_cases = []
for row in test_cases_nonoise:
    test_cases.append(pytest.param(*row, marks=pytest.mark.skip_on_ci))
for row in test_cases_noisy:
    test_cases.append(pytest.param(*row, marks=pytest.mark.basic))

ALL_ENVS = {**ALL_VISUAL_ENVS}


@pytest.fixture(scope='function')
def env(request):
    e = ALL_ENVS[request.param]()
    e._partially_observable = False
    e._freeze_rand_vec = False
    e._set_task_called = True
    return e


@pytest.mark.skip
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
