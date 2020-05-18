from metaworld.benchmarks import ML10


def test_random_init_train():
    """Test that random_init == True for all envs."""
    env = ML10(env_type='train')
    assert len(env._task_envs) == 10
    for task_env in env._task_envs:
        assert task_env.random_init


def test_random_init_test():
    """Test that random_init == True for all envs."""
    env = ML10(env_type='test')
    assert len(env._task_envs) == 5
    for task_env in env._task_envs:
        assert task_env.random_init
