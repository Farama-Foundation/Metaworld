from metaworld.benchmarks import MT50


def test_random_init():
    """Test that random_init == False for all envs."""
    env = MT50()
    assert len(env._task_envs) == 50
    for task_env in env._task_envs:
        assert not task_env.random_init
