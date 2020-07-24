import numpy as np
import glfw

def step_env(env, max_path_length=100, iterations=1, render=True):
    """Step env helper."""
    for _ in range(iterations):
        env.reset()
        for _ in range(max_path_length):
            _, _, done, info = env.step(env.action_space.sample())
            if render:
                env.render()
            if done:
                break
