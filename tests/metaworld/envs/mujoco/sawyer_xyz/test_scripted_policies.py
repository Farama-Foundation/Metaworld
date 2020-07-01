import pytest

from metaworld.envs.mujoco.env_dict import ALL_V1_ENVIRONMENTS, ALL_V2_ENVIRONMENTS
from metaworld.policies import *
from tests.metaworld.envs.mujoco.sawyer_xyz.utils import check_success
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()


test_cases = [
    # name, policy, action noise pct, success rate, env kwargs
    ['button-press-topdown-v1', SawyerButtonPressTopdownV1Policy(), .0, 1., {}],
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
]

ALL_ENVS = {**ALL_V1_ENVIRONMENTS, **ALL_V2_ENVIRONMENTS}

for row in test_cases:
    # row[-1] contains env kwargs. This instantiates an env with those kwargs,
    # then replaces row[-1] with the env object (kwargs are no longer needed)
    row[-1] = ALL_ENVS[row[0]](random_init=True, **row[-1])
    # now remove env names from test_data, as they aren't needed in parametrize


@pytest.mark.parametrize(
    'name,policy,act_noise_pct,expected_success_rate,env',
    test_cases
)
def test_scripted_policy(name, env, policy, act_noise_pct, expected_success_rate, iters=100):
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
    rewards = []
    returns = []
    first_successes = []
    for _ in range(iters):
        success, _, _reward, _return, first_success = check_success(env,
                                                     policy,
                                                     act_noise_pct,
                                                     render=False)
        successes += float(success)
        rewards.append(_reward)
        returns.append(_return)
        first_successes.append(first_success)
    if act_noise_pct > 0:
            name = name + "_with_10_percent_noise"
    plot_rewards_and_returns(rewards, returns, name, first_successes)

    print(successes)
    # assert successes >= expected_success_rate * iters


def plot_rewards_and_returns(rewards, returns, name, first_successess):
    """Plot the rewards and returns time series.
    save under data/rewards/name.png.
    """
    first_success_time_step = int(np.mean(first_successess))
    first_success_reward = np.mean(rewards, axis=0)[first_success_time_step]
    first_success_return = np.mean(returns, axis=0)[first_success_time_step]
    fig, ax = plt.subplots(
                        1,
                        2,
                        figsize=(6.75,4))
    rewards = [pd.DataFrame({"rewards":reward, "Time Steps":np.arange(len(reward))}) for reward in rewards]
    reward_df = pd.concat(rewards)
    ax[0] = sns.lineplot(x='Time Steps', y='rewards', ax=ax[0], data=reward_df, ci=95, lw=0.5)
    ax[0].set_xlabel("Time Steps")
    ax[0].set_ylabel("Reward")
    ax[0].set_title("Rewards")
    ax[0].vlines(first_success_time_step, ymin=0, ymax=first_success_reward, linestyle='--', color= "green")
    ax[0].hlines(first_success_reward, xmin=0, xmax=first_success_time_step, linestyle='--', color= "green")
    returns = [pd.DataFrame({"returns":_return, "Time Steps":np.arange(len(_return))}) for _return in returns]
    returns_df = pd.concat(returns)
    ax[1] = sns.lineplot(x='Time Steps', y='returns', ax=ax[1], color='coral', data=returns_df, ci=95, lw=0.5)
    ax[1].set_xlabel("Time Steps")
    ax[1].set_ylabel("Returns")
    ax[1].set_title(f"Returns")
    ax[1].vlines(first_success_time_step, ymin=0, ymax=first_success_return, linestyle='--', color= "green", label=f"first success = {first_success_time_step}")
    ax[1].hlines(first_success_return, xmin=0, xmax=first_success_time_step, linestyle='--', color= "green")
    plt.subplots_adjust(top=0.85)
    fig.suptitle(f"{name} (n=100)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(f'figures/{name}_rewards_returns.jpg')


