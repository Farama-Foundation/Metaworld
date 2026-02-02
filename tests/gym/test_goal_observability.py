import pytest
import numpy as np

import gymnasium as gym

from metaworld.env_dict import ENV_NAMES, ALL_V3_ENVIRONMENTS, MT_BENCHMARKS_TRAIN_ENV_NAMES, ML_BENCHMARKS


def _assert_goal_observability(obs, env: gym.Env, goal_observable: bool):
    env_name = env.unwrapped.ENV_NAME
    zero_pos = np.zeros(3)
    goal_pos = obs[-3:]
    if goal_observable:
        assert not np.array_equal(goal_pos, zero_pos), \
            f"Goal position appears to be hidden in env {env_name} when it should be observable"
    else:
        assert np.array_equal(goal_pos, zero_pos), \
            f"Goal position appears to be observable in env {env_name} when it should be hidden"


@pytest.mark.parametrize("goal_observable", [True, False])
@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_env_goal_observability(env_name, goal_observable):
    env = ALL_V3_ENVIRONMENTS[env_name](goal_observable=goal_observable)
    obs, info = env.reset()
    _assert_goal_observability(obs, env, goal_observable)
    env.close()


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_env_goal_observability_default(env_name):
    env = ALL_V3_ENVIRONMENTS[env_name]()
    obs, info = env.reset()
    _assert_goal_observability(obs, env, goal_observable=True)
    env.close()


def test_env_goal_observability_mtX():
    # Test MT1
    env = gym.make("Meta-World/MT1",
                   env_name="reach-v3",)
    obs, info = env.reset()
    _assert_goal_observability(obs, env, goal_observable=True)
    env.close()
    # Test MT1 user override
    env = gym.make("Meta-World/MT1",
                   env_name="reach-v3",
                   goal_observable=False)
    obs, info = env.reset()
    _assert_goal_observability(obs, env, goal_observable=False)
    env.close()

    # Test MTX
    for mt_bench in MT_BENCHMARKS_TRAIN_ENV_NAMES.keys():
        envs = gym.make_vec(
            f"Meta-World/{mt_bench}")
        obs_vec, info_vec = envs.reset()
        for obs, env in zip(obs_vec, envs.envs):
            _assert_goal_observability(obs, env, goal_observable=True)
        envs.close()
    # Test MTX user override
    for mt_bench in MT_BENCHMARKS_TRAIN_ENV_NAMES.keys():
        envs = gym.make_vec(
            f"Meta-World/{mt_bench}",
            goal_observable=False)
        obs_vec, info_vec = envs.reset()
        for obs, env in zip(obs_vec, envs.envs):
            _assert_goal_observability(obs, env, goal_observable=False)
        envs.close()

    # Test MTCustom
    env = gym.make_vec(
        "Meta-World/custom-mt-envs",
        train_env_names=["reach-v3", "push-v3", "pick-place-v3"],
    )
    obs_vec, info_vec = env.reset()
    for obs, env in zip(obs_vec, env.envs):
        _assert_goal_observability(obs, env, goal_observable=True)
    env.close()
    # Test MTCustom user override
    env = gym.make_vec(
        "Meta-World/custom-mt-envs",
        train_env_names=["reach-v3", "push-v3", "pick-place-v3"],
        goal_observable=False)
    obs_vec, info_vec = env.reset()
    for obs, env in zip(obs_vec, env.envs):
        _assert_goal_observability(obs, env, goal_observable=False)
    env.close()

    # Test ML1
    envs = gym.make_vec("Meta-World/ML1-train",
                        env_name="reach-v3",)
    obs_vec, info_vec = envs.reset()
    for obs, env in zip(obs_vec, envs.envs):
        _assert_goal_observability(obs, env, goal_observable=False)
    envs.close()
    # Test ML1 user override
    envs = gym.make_vec("Meta-World/ML1-train",
                        env_name="reach-v3",
                        goal_observable=True)
    obs_vec, info_vec = envs.reset()
    for obs, env in zip(obs_vec, envs.envs):
        _assert_goal_observability(obs, env, goal_observable=True)
    envs.close()

    # Test MLX
    for ml_bench in ML_BENCHMARKS.keys():
        envs = gym.make_vec(
            f"Meta-World/{ml_bench}-train")
        obs_vec, info_vec = envs.reset()
        for obs, env in zip(obs_vec, envs.envs):
            _assert_goal_observability(obs, env, goal_observable=False)
        envs.close()
    # Test MLX user override
    for ml_bench in ML_BENCHMARKS.keys():
        envs = gym.make_vec(
            f"Meta-World/{ml_bench}-train",
            goal_observable=True)
        obs_vec, info_vec = envs.reset()
        for obs, env in zip(obs_vec, envs.envs):
            _assert_goal_observability(obs, env, goal_observable=True)
        envs.close()

    # Test MLCustom
    env = gym.make_vec(
        "Meta-World/custom-ml-envs",
        train_env_names=["reach-v3", "push-v3"],
        test_env_names=["pick-place-v3"],
        split="train",
    )
    obs_vec, info_vec = env.reset()
    for obs, env in zip(obs_vec, env.envs):
        _assert_goal_observability(obs, env, goal_observable=False)
    env.close()
    # Test MLCustom user override
    env = gym.make_vec(
        "Meta-World/custom-ml-envs",
        train_env_names=["reach-v3", "push-v3"],
        test_env_names=["pick-place-v3"],
        goal_observable=True,
        split="train",
    )
    obs_vec, info_vec = env.reset()
    for obs, env in zip(obs_vec, env.envs):
        _assert_goal_observability(obs, env, goal_observable=True)
    env.close()
