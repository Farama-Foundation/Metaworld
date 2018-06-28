from railrl.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.ou_strategy import OUStrategy
import numpy as np
from railrl.policies.simple import RandomPolicy
import os.path as osp
def generate_goal_data_set(env=None, num_goals=10000, goal_generation_dict=None, use_cached_dataset=False, action_scale=1/10):
    if use_cached_dataset and osp.isfile('/tmp/goals' + str(num_goals) + '.npy'):
        goal_dict = np.load('/tmp/goals' + str(num_goals) + '.npy').item()
        print("loaded data from saved file")
        return goal_dict
    goal_dict = dict()
    policy = RandomPolicy(env.action_space)
    es = OUStrategy(action_space=env.action_space, theta=0)
    exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )
    for goal_key in goal_generation_dict:
        goal_size, obs_to_goal_fn, obs_key = goal_generation_dict[goal_key]
        goal_dict[goal_key] = np.zeros((num_goals, goal_size))
    print('Generating Random Goals')
    for i in range(num_goals):
        if i % 50 == 0:
            print('Reset')
            env.reset(_resample_on_reset=False)
            exploration_policy.reset()
        action = exploration_policy.get_action()[0] * action_scale
        obs, _, _, _ = env.step(
            action
        )
        print(i)
        for goal_key in goal_generation_dict:
            goal_size, obs_to_goal_fn, obs_key = goal_generation_dict[goal_key]
            goal_dict[goal_key][i, :] = obs_to_goal_fn(obs[obs_key])
    np.save('/tmp/goals' + str(num_goals) +'.npy', goal_dict)
    return goal_dict