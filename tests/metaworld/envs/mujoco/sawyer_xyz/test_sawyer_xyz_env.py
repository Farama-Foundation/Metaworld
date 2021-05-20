import numpy as np

import random

import metaworld

def test_reset_returns_same_obj_and_goal():
    benchmark = metaworld.MT50()
    env_dict = benchmark.train_classes
    tasks = benchmark.train_tasks
    initial_obj_poses = {name: [] for name in env_dict.keys()}
    goal_poses = {name: [] for name in env_dict.keys()}

    # Execute rollout for each environment in benchmark.
    for env_name, env_cls in env_dict.items():

        # Create environment and set task.
        env = env_cls()
        env_tasks = [t for t in tasks if t.env_name == env_name]
        env.set_task(random.choice(env_tasks))

        # Step through environment for a fixed number of episodes.
        for _ in range(2):
            # Reset environment and extract initial object position.
            obs = env.reset()
            goal = obs[-3:]
            goal_poses[env_name].append(goal)
            initial_obj_pos = obs[3:9]
            initial_obj_poses[env_name].append(initial_obj_pos)

# Display initial object positions and find environments with non-unique positions.
    violating_envs_obs = []
    for env_name, task_initial_pos in initial_obj_poses.items():
        if len(np.unique(np.array(task_initial_pos), axis=0)) > 1:
            violating_envs_obs.append(env_name)
    violating_envs_goals = []
    for env_name, target_pos in goal_poses.items():
        if len(np.unique(np.array(target_pos), axis=0)) > 1:
            violating_envs_goals.append(env_name)
    assert not violating_envs_obs
    assert not violating_envs_goals