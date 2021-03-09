import numpy as np


def trajectory_summary(env, policy, act_noise_pct, render=False, end_on_success=True):
    """Tests whether a given policy solves an environment
    Args:
        env (metaworld.envs.MujocoEnv): Environment to test
        policy (metaworld.policies.policies.Policy): Policy that's supposed to
            succeed in env
        act_noise_pct (np.ndarray): Decimal value(s) indicating std deviation of
            the noise as a % of action space
        render (bool): Whether to render the env in a GUI
        end_on_success (bool): Whether to stop stepping after first success
    Returns:
        (bool, np.ndarray, np.ndarray, int): Success flag, Rewards, Returns,
            Index of first success
    """
    success = False
    first_success = 0
    rewards = []

    for t, (r, done, info) in enumerate(trajectory_generator(env, policy, act_noise_pct, render)):
        rewards.append(r)
        assert not env.isV2 or set(info.keys()) == {
            'success',
            'near_object',
            'grasp_success',
            'grasp_reward',
            'in_place_reward',
            'obj_to_target',
            'unscaled_reward'
        }
        success |= bool(info['success'])
        if not success:
            first_success = t
        if (success or done) and end_on_success:
            break

    rewards = np.array(rewards)
    returns = np.cumsum(rewards)

    return success, rewards, returns, first_success


def trajectory_generator(env, policy, act_noise_pct, render=False):
    """Tests whether a given policy solves an environment
    Args:
        env (metaworld.envs.MujocoEnv): Environment to test
        policy (metaworld.policies.policies.Policy): Policy that's supposed to
            succeed in env
        act_noise_pct (np.ndarray): Decimal value(s) indicating std deviation of
            the noise as a % of action space
        render (bool): Whether to render the env in a GUI
    Yields:
        (float, bool, dict): Reward, Done flag, Info dictionary
    """
    action_space_ptp = env.action_space.high - env.action_space.low

    env.reset()
    env.reset_model()
    o = env.reset()
    assert o.shape == env.observation_space.shape
    assert env.observation_space.contains(o), obs_space_error_text(env, o)

    for _ in range(env.max_path_length):
        a = policy.get_action(o)
        a = np.random.normal(a, act_noise_pct * action_space_ptp)

        o, r, done, info = env.step(a)
        assert env.observation_space.contains(o), obs_space_error_text(env, o)
        if render:
            env.render()

        yield r, done, info


def obs_space_error_text(env, obs):
    return "Obs Out of Bounds\n\tlow: {}, \n\tobs: {}, \n\thigh: {}".format(
        env.observation_space.low[[0, 1, 2, -3, -2, -1]],
        obs[[0, 1, 2, -3, -2, -1]],
        env.observation_space.high[[0, 1, 2, -3, -2, -1]]
    )
