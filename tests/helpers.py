import numpy as np
import glfw


def step_env(env, max_path_length=100, iterations=1,
             render=True, check_obs_space=True):
    """Step env helper."""
    for _ in range(iterations):
        env.reset()
        for _ in range(max_path_length):
            obs, _, done, info = env.step(env.action_space.sample())
            assert 'goal' in info
            if hasattr(env, 'active_env'):
                assert np.all(info['goal'].shape == env.active_env.goal_space.shape)
            else:
                assert np.all(info['goal'].shape == env.goal_space.shape)
            if check_obs_space:
                assert env.active_env.observation_space.contains(
                    obs[:env.active_env.observation_space.low.size]
                )
            if render:
                env.render()
            if done:
                break
