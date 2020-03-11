from gym.spaces import Box

import numpy as np

class ProxyEnv():
    def __init__(self, wrapped_env):
        self._wrapped_env = wrapped_env

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def __str__(self):
        return '{}({})'.format(type(self).__name__, self.wrapped_env)


class NormalizedBoxEnv(ProxyEnv):
    """
    Normalize action to in [-1, 1].

    Optionally normalize observations.
    """
    def __init__(
            self,
            env,
            obs_means=None,
            obs_stds=None,
            obs_to_normalize_keys=['observation'],
    ):
        super().__init__(env)
        self._should_normalize = not (obs_means is None and obs_stds is None)
        num_obs_types = len(obs_to_normalize_keys)
        if self._should_normalize:
            if obs_means is None:
                obs_means = dict()
                for key in self.obs_to_normalize_keys:
                    obs_means[key] = np.zeros_like(env.observation_space[key].low)
            else:
                obs_means = dict()
                for key in self.obs_to_normalize_keys:
                    obs_means[key] = np.array(obs_means[key])
            if obs_stds is None:
                obs_stds = dict()
                for key in self.obs_to_normalize_keys:
                    obs_stds[key] = np.zeros_like(env.observation_space[key].low)
            else:
                obs_stds = dict()
                for key in self.obs_to_normalize_keys:
                    obs_stds[key] = np.array(obs_stds[key])
        self._obs_means = obs_means
        self._obs_stds = obs_stds
        ub = np.ones(self._wrapped_env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)
        self.obs_to_normalize_keys=obs_to_normalize_keys

    def estimate_obs_stats(self, obs_batch, override_values=False):
        raise NotImplementedError()

    def _apply_normalize_obs(self, obs):
        for key in self.obs_to_normalize_keys:
            obs[key]= (obs[key] - self._obs_means[key]) / (self._obs_stds[key] + 1e-8)

    def step(self, action):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)
        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step
        if self._should_normalize:
            next_obs = self._apply_normalize_obs(next_obs)
        return next_obs, reward, done, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env
