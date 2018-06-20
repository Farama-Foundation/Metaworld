from railrl.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
import numpy as np
from railrl.policies.simple import RandomPolicy

def generate_goal_data_set(env=None, num_goals=10000, goal_generation_dict=None):
    goal_dict = dict()
    for goal_key in goal_generation_dict:
        size, obs_to_goal_fn, obs_key = goal_generation_dict[goal_key]
        goals = np.zeros((num_goals, size))
        policy = RandomPolicy(env.action_space)
        es = OUStrategy(action_space=env.action_space, theta=0)
        exploration_policy = PolicyWrappedWithExplorationStrategy(
            exploration_strategy=es,
            policy=policy,
        )
        print('Generating Random Goals')
        for i in range(num_goals):
            if i % 50 == 0:
                print('Reset')
                env.reset(resample_on_reset=False)
                exploration_policy.reset()
            action = exploration_policy.get_action()[0] * 10
            obs, _, _, _ = env.step(
                action
            )
            print(i)
            goals[i, :] = obs_to_goal_fn(obs[obs_key])
        goal_dict[goal_key] = goals
    return goal_dict
