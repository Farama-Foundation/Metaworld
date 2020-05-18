from metaworld.benchmarks import ML45


def test_random_init_train():
    """Test that random_init == True for all envs."""
    env = ML45(env_type='train')
    assert len(env._task_envs) == 45
    for task_env in env._task_envs:
        assert task_env.random_init


def test_random_init_test():
    """Test that random_init == True for all envs."""
    env = ML45(env_type='test')
    assert len(env._task_envs) == 5
    for task_env in env._task_envs:
        assert task_env.random_init
