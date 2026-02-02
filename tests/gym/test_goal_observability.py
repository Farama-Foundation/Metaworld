import pytest
import numpy as np
import gymnasium as gym

from metaworld.env_dict import (
    ENV_NAMES,
    ENV_CLASS_MAP,
    MT_BENCHMARKS_TRAIN_ENV_NAMES,
    ML_BENCHMARKS
)

# --- Helper Functions ---


def _assert_goal_observability(obs, env: gym.Env, goal_observable: bool):
    """Checks a single observation/env pair."""
    env_name = env.unwrapped.ENV_NAME
    zero_pos = np.zeros(3)
    goal_pos = obs[-3:]

    if goal_observable:
        assert not np.array_equal(goal_pos, zero_pos), \
            f"Goal position appears to be hidden in env {env_name} when it should be observable"
    else:
        assert np.array_equal(goal_pos, zero_pos), \
            f"Goal position appears to be observable in env {env_name} when it should be hidden"


def _verify_goal_observability(env_instance, expected_observable: bool):
    """
    Handles env.reset, checking vector vs scalar envs, and closing.
    """
    try:
        obs, _ = env_instance.reset()

        # Check if it is a VectorEnv (has attribute 'envs')
        if hasattr(env_instance, 'envs'):
            # Iterate through vector environments
            for single_obs, single_env in zip(obs, env_instance.envs):
                _assert_goal_observability(
                    single_obs, single_env, expected_observable)
        else:
            # Standard single environment
            _assert_goal_observability(obs, env_instance, expected_observable)

    finally:
        env_instance.close()


# --- Individual Environment Tests ---

@pytest.mark.parametrize("goal_observable", [True, False])
@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_v3_env_explicit_goal_observability(env_name, goal_observable):
    """Test explicit goal observability flags on individual V3 environments."""
    env = ENV_CLASS_MAP[env_name](goal_observable=goal_observable)
    _verify_goal_observability(env, goal_observable)


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_v3_env_default_goal_observability(env_name):
    """Test default goal observability behavior (should be observable) on V3 environments."""
    env = ENV_CLASS_MAP[env_name]()
    _verify_goal_observability(env, expected_observable=True)


# --- MT (Multi-Task) Tests (Default: Observable) ---

@pytest.mark.parametrize("override_setting, expected", [
    (None, True),   # Default behavior
    (False, False),  # User override
    (True, True)    # Explicit True
])
def test_mt1_goal_observability(override_setting, expected):
    """Test MT1 specific loading."""
    kwargs = {"env_name": "reach-v3"}
    if override_setting is not None:
        kwargs["goal_observable"] = override_setting

    env = gym.make("Meta-World/MT1", **kwargs)
    _verify_goal_observability(env, expected)


@pytest.mark.parametrize("benchmark_name", MT_BENCHMARKS_TRAIN_ENV_NAMES.keys())
@pytest.mark.parametrize("override_setting, expected", [
    (None, True),   # Default behavior for MT is Visible
    (False, False),  # Override to Hidden
])
def test_mt_benchmarks_goal_observability(benchmark_name, override_setting, expected):
    """Test standard MT benchmarks (MT10, MT50, etc)."""
    kwargs = {}
    if override_setting is not None:
        kwargs["goal_observable"] = override_setting

    envs = gym.make_vec(f"Meta-World/{benchmark_name}", **kwargs)
    _verify_goal_observability(envs, expected)


@pytest.mark.parametrize("override_setting, expected", [
    (None, True),   # Default
    (False, False)  # Override
])
def test_mt_custom_goal_observability(override_setting, expected):
    """Test Custom MT environment construction."""
    kwargs = {"train_env_names": ["reach-v3"]}
    if override_setting is not None:
        kwargs["goal_observable"] = override_setting

    envs = gym.make_vec("Meta-World/custom-mt-envs", **kwargs)
    _verify_goal_observability(envs, expected)


# --- ML (Meta-Learning) Tests (Default: Hidden) ---

@pytest.mark.parametrize("override_setting, expected", [
    (None, False),  # Default behavior for ML is Hidden
    (True, True),   # User override
    (False, False)  # Explicit False
])
def test_ml1_goal_observability(override_setting, expected):
    """Test ML1 specific loading."""
    kwargs = {"env_name": "reach-v3"}
    if override_setting is not None:
        kwargs["goal_observable"] = override_setting

    envs = gym.make_vec("Meta-World/ML1-train", **kwargs)
    _verify_goal_observability(envs, expected)


@pytest.mark.parametrize("benchmark_name", ML_BENCHMARKS.keys())
@pytest.mark.parametrize("override_setting, expected", [
    (None, False),  # Default behavior for ML is Hidden
    (True, True),   # Override to Visible
])
def test_ml_benchmarks_goal_observability(benchmark_name, override_setting, expected):
    """Test standard ML benchmarks (ML10, ML45, etc)."""
    kwargs = {}
    if override_setting is not None:
        kwargs["goal_observable"] = override_setting

    envs = gym.make_vec(f"Meta-World/{benchmark_name}-train", **kwargs)
    _verify_goal_observability(envs, expected)


@pytest.mark.parametrize("override_setting, expected", [
    (None, False),  # Default
    (True, True)    # Override
])
def test_ml_custom_goal_observability(override_setting, expected):
    """Test Custom ML environment construction."""
    kwargs = {
        "train_env_names": ["reach-v3"],
        "test_env_names": ["pick-place-v3"],
        "split": "train"
    }
    if override_setting is not None:
        kwargs["goal_observable"] = override_setting

    envs = gym.make_vec("Meta-World/custom-ml-envs", **kwargs)
    _verify_goal_observability(envs, expected)
