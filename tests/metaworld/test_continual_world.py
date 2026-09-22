from __future__ import annotations

from functools import partial

import gymnasium as gym
import numpy as np
import pytest

import metaworld
from metaworld.wrappers import ContinualWorldEnv


class _PhaseEnv(gym.Env):
    def __init__(self, phase: int, truncate_each_step: bool = False):
        self.phase = phase
        self.truncate_each_step = truncate_each_step
        self.steps = 0
        self.observation_space = gym.spaces.Box(0, 100, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.array([10 * self.phase], dtype=np.float32), {
            "reset_phase": self.phase,
            "reset_seed": -1 if seed is None else seed,
        }

    def step(self, action):
        self.steps += 1
        return (
            np.array([10 * self.phase + self.steps], dtype=np.float32),
            1.0,
            False,
            self.truncate_each_step,
            {"step_phase": self.phase},
        )


def _make_phase(
    phase: int,
    truncate_each_step: bool = False,
    vectorizer=gym.vector.SyncVectorEnv,
):
    return vectorizer(
        [partial(_PhaseEnv, phase, truncate_each_step) for _ in range(2)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )


def test_phase_boundary_preserves_terminal_data_and_reset_info():
    env = ContinualWorldEnv([_make_phase(0), _make_phase(1)], steps_per_task=2)
    try:
        obs, info = env.reset(seed=7)
        np.testing.assert_array_equal(obs, [[0], [0]])
        np.testing.assert_array_equal(info["reset_seed"], [7, 8])

        obs, _, _, truncated, info = env.step(np.zeros((2, 1)))
        np.testing.assert_array_equal(obs, [[1], [1]])
        assert not truncated.any()
        np.testing.assert_array_equal(info["task_idx"], [0, 0])

        obs, rewards, terminated, truncated, info = env.step(np.zeros((2, 1)))
        np.testing.assert_array_equal(obs, [[10], [10]])
        np.testing.assert_array_equal(rewards, [1, 1])
        assert not terminated.any()
        assert truncated.all()
        np.testing.assert_array_equal(info["reset_phase"], [1, 1])
        np.testing.assert_array_equal(info["reset_seed"], [9, 10])
        np.testing.assert_array_equal(info["task_idx"], [1, 1])
        np.testing.assert_array_equal(info["final_obs"].tolist(), [[2], [2]])
        np.testing.assert_array_equal(info["final_info"]["step_phase"], [0, 0])
        np.testing.assert_array_equal(info["final_info"]["task_idx"], [0, 0])

        env.step(np.zeros((2, 1)))
        _, _, _, truncated, info = env.step(np.zeros((2, 1)))
        assert truncated.all()
        assert info["sequence_complete"].all()
        np.testing.assert_array_equal(info["final_obs"].tolist(), [[12], [12]])
        np.testing.assert_array_equal(info["final_info"]["step_phase"], [1, 1])
        np.testing.assert_array_equal(info["final_info"]["task_idx"], [1, 1])
        with pytest.raises(RuntimeError, match="reset"):
            env.step(np.zeros((2, 1)))

        obs, _ = env.reset(seed=7)
        np.testing.assert_array_equal(obs, [[0], [0]])
    finally:
        env.close()


def test_boundary_uses_true_final_obs_when_inner_env_autoresets():
    env = ContinualWorldEnv(
        [_make_phase(0, truncate_each_step=True), _make_phase(1)], steps_per_task=1
    )
    try:
        env.reset(seed=2)
        obs, _, _, truncated, info = env.step(np.zeros((2, 1)))
        np.testing.assert_array_equal(obs, [[10], [10]])
        assert truncated.all()
        np.testing.assert_array_equal(info["final_obs"].tolist(), [[1], [1]])
        np.testing.assert_array_equal(info["final_info"]["step_phase"], [0, 0])
    finally:
        env.close()


def test_async_phase_switch():
    env = ContinualWorldEnv(
        [
            _make_phase(0, vectorizer=gym.vector.AsyncVectorEnv),
            _make_phase(1, vectorizer=gym.vector.AsyncVectorEnv),
        ],
        steps_per_task=1,
    )
    try:
        env.reset(seed=3)
        obs, _, _, truncated, info = env.step(np.zeros((2, 1)))
        np.testing.assert_array_equal(obs, [[10], [10]])
        assert truncated.all()
        np.testing.assert_array_equal(info["final_obs"].tolist(), [[1], [1]])
        np.testing.assert_array_equal(info["final_info"]["task_idx"], [0, 0])
    finally:
        env.close()


def test_reset_targets_active_phase_without_restarting_budget():
    env = ContinualWorldEnv([_make_phase(0), _make_phase(1)], steps_per_task=2)
    try:
        env.reset(seed=5)
        env.step(np.zeros((2, 1)))
        obs, info = env.reset()
        np.testing.assert_array_equal(obs, [[0], [0]])
        np.testing.assert_array_equal(info["task_idx"], [0, 0])
        obs, _, _, truncated, _ = env.step(np.zeros((2, 1)))
        assert truncated.all()
        np.testing.assert_array_equal(obs, [[10], [10]])

        obs, info = env.reset()
        np.testing.assert_array_equal(obs, [[10], [10]])
        np.testing.assert_array_equal(info["task_idx"], [1, 1])
        assert env.phase_step == 0
        assert env.get_attr("phase") == (1, 1)
    finally:
        env.close()


def test_invalid_step_budget():
    envs = [_make_phase(0), _make_phase(1)]
    with pytest.raises(ValueError, match="steps_per_task"):
        ContinualWorldEnv(envs, steps_per_task=0)
    for env in envs:
        env.close()


def test_standard_sequences_use_current_metaworld_names():
    assert metaworld.CW10_TASK_NAMES == (
        "hammer-v3",
        "push-wall-v3",
        "faucet-close-v3",
        "push-back-v3",
        "stick-pull-v3",
        "handle-press-side-v3",
        "push-v3",
        "shelf-place-v3",
        "window-close-v3",
        "peg-unplug-side-v3",
    )
    assert metaworld.CW20_TASK_NAMES == metaworld.CW10_TASK_NAMES * 2


def test_cw10_registration_smoke():
    env = gym.make_vec(
        "Meta-World/CW10",
        num_envs=1,
        steps_per_task=1,
        seed=11,
        max_episode_steps=5,
    )
    try:
        obs, _ = env.reset(seed=11)
        assert obs.shape[0] == 1
        assert np.argmax(obs[0, -10:]) == 0
        obs, _, _, truncated, info = env.step(env.action_space.sample())
        assert truncated.tolist() == [True]
        assert np.argmax(obs[0, -10:]) == 1
        assert np.argmax(info["final_obs"][0][-10:]) == 0
    finally:
        env.close()


def test_cw20_revisits_first_task_at_phase_ten():
    env = gym.make_vec(
        "Meta-World/CW20",
        num_envs=1,
        steps_per_task=1,
        seed=11,
        max_episode_steps=5,
    )
    try:
        env.reset(seed=11)
        for _ in range(10):
            obs, _, _, truncated, info = env.step(env.action_space.sample())
            assert truncated.tolist() == [True]
        assert info["task_idx"].tolist() == [10]
        assert info["final_info"]["task_idx"].tolist() == [9]
        assert np.argmax(obs[0, -20:]) == 10
        first_task = env.envs[0].envs[0].unwrapped.__class__
        repeated_task = env.envs[10].envs[0].unwrapped.__class__
        assert repeated_task is first_task
    finally:
        env.close()
