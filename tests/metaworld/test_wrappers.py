from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

import metaworld  # noqa: F401
from metaworld.wrappers import OneHotWrapper, RNNBasedMetaRLWrapper


def _make_env() -> gym.Env:
    return gym.make("Meta-World/MT1", env_name="reach-v3", seed=0)


@pytest.mark.parametrize(
    "wrap",
    [
        lambda env: OneHotWrapper(env, task_idx=0, num_tasks=10),
        lambda env: RNNBasedMetaRLWrapper(env),
    ],
    ids=["OneHotWrapper", "RNNBasedMetaRLWrapper"],
)
def test_wrapper_observation_space_keeps_dtype(wrap):
    """A wrapper must declare the dtype of the observations it hands back.

    Both wrappers built their `Box` without a dtype, which falls back to
    float32, while the Sawyer observation space is float64 and both wrappers
    concatenate float64 arrays. The declared space could not hold its own
    observations.
    """
    env = _make_env()
    wrapped = wrap(env)

    assert wrapped.observation_space.dtype == env.observation_space.dtype

    obs, _ = wrapped.reset(seed=0)
    assert np.can_cast(obs.dtype, wrapped.observation_space.dtype)

    obs, _, _, _, _ = wrapped.step(wrapped.action_space.sample())
    assert np.can_cast(obs.dtype, wrapped.observation_space.dtype)

    env.close()


def test_one_hot_wrapper_keeps_the_wrapped_bounds():
    """The bounds of the wrapped space must survive unrounded.

    Casting them to float32 moved every finite bound the environment declared.
    """
    env = _make_env()
    wrapped = OneHotWrapper(env, task_idx=0, num_tasks=10)

    env_dim = env.observation_space.shape[0]
    assert np.array_equal(
        wrapped.observation_space.low[:env_dim], env.observation_space.low
    )
    assert np.array_equal(
        wrapped.observation_space.high[:env_dim], env.observation_space.high
    )

    env.close()


def test_rnn_wrapper_observation_is_inside_its_space():
    """The recurrent observation is unbounded, so it must lie in its own space."""
    env = _make_env()
    wrapped = RNNBasedMetaRLWrapper(env)

    obs, _ = wrapped.reset(seed=0)
    assert wrapped.observation_space.contains(obs)

    obs, _, _, _, _ = wrapped.step(wrapped.action_space.sample())
    assert wrapped.observation_space.contains(obs)

    env.close()
