import numpy as np


def check_multiple_env_steps(env, max_episode_steps=100, iterations=1, render=True):
    """
    Resets and steps through the environment multiple times, checking that
    the observations are consistent with the internal state of the environment.

    Args:
        env: The environment to test.
        max_episode_steps: Maximum number of steps per episode.
        iterations: Number of episodes to run. To test consistency across resets.
        render: Whether to render the environment during stepping.
    """
    for _ in range(iterations):
        obs, info = env.reset()
        for _ in range(max_episode_steps):
            next_obs, _, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            if not env._goal_observable:
                assert (next_obs[-3:] == np.zeros(3)).all()
            else:
                assert (next_obs[-3:] == env._get_pos_goal()).all()
            assert (next_obs[:3] == env.get_endeff_pos()).all()
            internal_obs = env._get_pos_objects()
            internal_quat = env._get_quat_objects()
            assert (next_obs[4:7] == internal_obs[:3]).all()
            assert (next_obs[7:11] == internal_quat[:4]).all()
            if internal_obs.shape == (6,):
                assert internal_quat.shape == (8,)
                assert (next_obs[11:14] == internal_obs[3:]).all()
                assert (next_obs[14:18] == internal_quat[4:]).all()
            else:
                assert (next_obs[11:14] == np.zeros(3)).all()
                assert (next_obs[14:18] == np.zeros(4)).all()
            assert (obs[:18] == next_obs[18:-3]).all()
            obs = next_obs
            if render:
                env.render()
            if truncated or terminated:
                break
