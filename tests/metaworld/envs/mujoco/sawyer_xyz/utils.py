import numpy as np


def check_success(env, policy, act_noise_pct, render=False):
    """Tests whether a given policy solves an environment
    Args:
        env (metaworld.envs.MujocoEnv): Environment to test
        policy (metaworld.policies.policies.Policy): Policy that's supposed to
            succeed in env
        act_noise_pct (float): Decimal value indicating std deviation of the
            noise as a % of action space
        render (bool): Whether to render the env in a GUI
    Returns:
        (bool, int): Success flag, Trajectory length
    """
    action_space_ptp = env.action_space.high - env.action_space.low

    env.reset()
    env.reset_model()
    o = env.reset()
    assert o.shape == env.observation_space.shape

    t = 0
    done = False
    success = False
    while not success and not done:
        a = policy.get_action(o)
        a = np.random.normal(a, act_noise_pct * action_space_ptp)
        try:
            o, r, done, info = env.step(a)
            if render:
                env.render()

            t += 1
            success |= bool(info['success'])

        except ValueError:
            break

    return success, t
