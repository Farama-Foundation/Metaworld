import numpy as np
import glfw

def step_env(env, max_path_length=100, iterations=1):
    """Step env helper."""
    for _ in range(iterations):
        env.reset()
        for _ in range(max_path_length):
            _, _, done, info = env.step(env.action_space.sample())
            assert 'goal' in info
            assert np.all(info['goal'].shape == env.goal_space.shape)
            # env.render()
            if done:
                break

def close_env(env):
    """Close env helper."""
    if env.viewer is not None:
        glfw.destroy_window(env.viewer.window)
    env.viewer = None
