def check_success(env, policy, render=False):
    """Tests whether a given policy solves an environment

    Args:
        env (metaworld.envs.MujocoEnv): Environment to test
        policy (metaworld.policies.policies.Policy): Policy that's supposed to succeed in env
        render (bool): Whether to render the env in a GUI

    Returns:
        (bool, int): Success flag, Trajectory length
    """
    env.reset_model()
    o = env.reset()

    t = 0
    done = False
    success = False
    while not success and not done:
        assert len(o) == sum([len(i) for i in policy.parse_obs(o).values()]), 'Observation not fully parsed'
        a = policy.get_action(o)
        try:
            o, r, done, info = env.step(a)

            if render:
                env.render()

            t += 1
            success |= bool(info['success'])

        except ValueError:
            env.reset()
            done = True

    return success, t
