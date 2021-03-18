import numpy as np
import glfw

def step_env(env, max_path_length=100, iterations=1, render=True):
    """Step env helper."""
    for _ in range(iterations):
        obs = env.reset()
        for _ in range(max_path_length):
            next_obs, _, done, info = env.step(env.action_space.sample())
            if env._partially_observable:
                assert (next_obs[-3:] == np.zeros(3)).all()
            else:
                assert (next_obs[-3:] == env._get_pos_goal()).all()
            assert (next_obs[:3] == env.get_endeff_pos()).all()
            internal_obs = env._get_pos_objects()
            internal_quat = env._get_quat_objects()
            assert (next_obs[4:7] == internal_obs[:3]).all()
            assert (next_obs[7:11] == internal_quat[:4]).all()
            if internal_obs.shape == (6,):
                assert internal_quat.shape == (8, )
                assert (next_obs[11:14] == internal_obs[3:]).all()
                assert (next_obs[14:18] == internal_quat[4:]).all()
            else:
                assert (next_obs[11:14] == np.zeros(3)).all()
                assert (next_obs[14:18] == np.zeros(4)).all()
            assert (obs[:18] == next_obs[18:-3]).all()
            obs = next_obs
            if render:
                env.render()
            if done:
                break
