import numpy as np


def check_success(env, policy, noisiness, render=False):
    """Tests whether a given policy solves an environment
    Args:
        env (metaworld.envs.MujocoEnv): Environment to test
        policy (metaworld.policies.policies.Policy): Policy that's supposed to succeed in env
        noisiness (float): Decimal value indicating std deviation of the noise as a % of action space
        render (bool): Whether to render the env in a GUI
    Returns:
        (bool, int): Success flag, Trajectory length
    """
    action_space_ptp = env.action_space.high - env.action_space.low

    env.reset_model()
    o = env.reset()

    t = 0
    done = False
    success = False
    while not success and not done:
        assert len(o) == sum([len(i) for i in policy.parse_obs(o).values()]), 'Observation not fully parsed'

        a = policy.get_action(o)
        a = np.random.normal(a, noisiness * action_space_ptp)
        try:
            o, r, done, info = env.step(a)

            if render:
                env.render()

            t += 1
            success |= bool(info['success'])

        except ValueError as e:
            print(e)
            env.reset()
            done = True

    return success, t
