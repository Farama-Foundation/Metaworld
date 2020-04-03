from copy import deepcopy
from types import MethodType

import gym
from gym.spaces import Box
import numpy as np

from metaworld.envs.mujoco.env_dict import _NUM_METAWORLD_ENVS


class MultiClassMultiTaskEnv(gym.Env):

    # TODO maybe we should add a task_space to this
    # environment. In that case we can just do a `task_space.sample()`
    # and have a single task sampling API accros this repository. 

    def __init__(self,
                 task_env_cls_dict,
                 task_args_kwargs,
                 sample_all=True,
                 sample_goals=False,
                 obs_type='plain'):
        assert len(task_env_cls_dict.keys()) == len(task_args_kwargs.keys())
        assert len(task_env_cls_dict.keys()) >= 1
        for k in task_env_cls_dict.keys():
            assert k in task_args_kwargs

        self._task_envs = []
        self._task_names = []
        self._sampled_all = sample_all
        self._sample_goals = sample_goals
        self._obs_type = obs_type

        # hardcoded so we don't have to iterate over all envs and check the maximum
        # this is the maximum observation dimension after augmenting observation
        # e.g. adding goal
        self._max_obs_dim = 15
        self._env_discrete_index = {}
        for task, env_cls in task_env_cls_dict.items():
            task_args = task_args_kwargs[task]['args']
            task_kwargs = deepcopy(task_args_kwargs[task]['kwargs'])
            self._env_discrete_index[task] = task_kwargs['task_id']
            del task_kwargs['task_id']
            task_env = env_cls(*task_args, **task_kwargs)
            assert np.prod(task_env.observation_space.shape) <= self._max_obs_dim
            # this multitask env only accept plain observations
            # since it handles all the observation augmentations
            assert task_env.obs_type == 'plain'
            self._task_envs.append(task_env)
            self._task_names.append(task)
        # If key (taskname) is in this `self._discrete_goals`, then this task are seen
        # to be using a discrete goal space. This wrapper will
        # set the property discrete_goal_space as True, update the goal_space
        # and the sample_goals method will sample from a discrete space.
        self._discrete_goals = dict()
        self._fully_discretized = True if not sample_goals else False
        self._n_discrete_goals = len(task_env_cls_dict.keys())
        self._active_task = 0
        self._check_env_list()
    
    @property
    def all_task_names(self):
        """list[str]: Name of all available tasks. Note that two envs of a task can have different goals."""
        return self._task_names

    def discretize_goal_space(self, discrete_goals):
        for task, goals in discrete_goals.items():
            if task in self._task_names:
                idx = self._task_names.index(task)
                self._discrete_goals[task] = discrete_goals[task]
                self._task_envs[idx].discretize_goal_space(
                    self._discrete_goals[task]
                )
        # if obs_type include task id, then all the tasks have
        # to use a discrete goal space and we hash indexes for tasks.
        self._fully_discretized = True
        for env in self._task_envs:
            if not env.discrete_goal_space:
                self._fully_discretized = False

        start = 0
        if self._fully_discretized:
            self._env_discrete_index = dict()
            for task, env in zip(self._task_names, self._task_envs):
                self._env_discrete_index[task] = start
                start += env.discrete_goal_space.n
            self._n_discrete_goals = start

    def _check_env_list(self):
        assert len(self._task_envs) >= 1
        first_obs_type = self._task_envs[0].obs_type
        first_action_space = self._task_envs[0].action_space

        for env in self._task_envs:
            assert env.obs_type == first_obs_type, "All the environment should use the same observation type!"
            assert env.action_space.shape == first_action_space.shape, "All the environment should have the same action space!"

        # get the greatest observation space
        # currently only support 1-dimensional Box
        max_flat_dim = np.prod(self._task_envs[0].observation_space.shape)
        for i, env in enumerate(self._task_envs):
            assert len(env.observation_space.shape) == 1
            if np.prod(env.observation_space.shape) >= max_flat_dim:
                self.observation_space_index = i
                max_flat_dim = np.prod(env.observation_space.shape)
        self._max_plain_dim = max_flat_dim

    @property
    def observation_space(self):
        """ TODO not complete, needs support for different observation types"""
        if self._obs_type == "plain":
            obs_dim = self._max_obs_dim
        elif self._obs_type == "with_goal_id":
            obs_dim = self._max_obs_dim + _NUM_METAWORLD_ENVS
        elif self._obs_type == "with_goal_and_id":
            obs_dim = self._max_obs_dim + _NUM_METAWORLD_ENVS
        elif self._obs_type == "with_goal":
            obs_dim = self._max_obs_dim
        else:
            raise Exception("invalid obs_type, obs_type was {}".format(self._obs_type))
        return Box(low=-np.inf, high=np.inf, shape=(obs_dim,))

    def set_task(self, task):
        if self._sample_goals:
            assert isinstance(task, dict)
            t = task['task']
            g = task['goal']
            self._active_task = t % len(self._task_envs)
            # TODO: remove underscore
            self.active_env.set_goal_(g)
        else:
            self._active_task = task % len(self._task_envs)

    def sample_tasks(self, meta_batch_size):
        if self._sampled_all:
            assert meta_batch_size >= len(self._task_envs)
            tasks = [i for i in range(meta_batch_size)]
        else:
            tasks = np.random.randint(
                0, self.num_tasks, size=meta_batch_size).tolist()
        if self._sample_goals:
            goals = [
                self._task_envs[t % len(self._task_envs)].sample_goals_(1)[0]
                for t in tasks
            ]
            tasks_with_goal = [
                dict(task=t, goal=g)
                for t, g in zip(tasks, goals)
            ]
            return tasks_with_goal
        else:
            return tasks

    def step(self, action):
        import ipdb; ipdb.set_trace()
        obs, reward, done, info = self.active_env.step(action)
        obs = self._augment_observation(obs)
        info['task_name'] = self._task_names[self._active_task]
        return obs, reward, done, info

    def _augment_observation(self, obs):
        # optionally zero-pad observation, before augmenting observation
        if np.prod(obs.shape) < self._max_obs_dim:
            zeros = np.zeros(
                shape=(self._max_obs_dim - np.prod(obs.shape),)
            )
            obs = np.concatenate([obs, zeros])

        # augment the observation based on obs_type:
        if self._obs_type == 'with_goal_id' or self._obs_type == 'with_goal_and_id':
            if self._obs_type == 'with_goal_and_id':
                obs = np.concatenate([obs, self.active_env._state_goal])
            obs = np.concatenate([obs, self.active_task_one_hot])
        elif self._obs_type == 'with_goal':
            obs = np.concatenate([obs, self.active_env._state_goal])
        return obs

    def reset(self, **kwargs):
        obs = self._augment_observation(self.active_env.reset(**kwargs))
        return obs

    # Utils for ImageEnv
    # Not using the `get_image` from the base class since
    # `sim.render()` is extremely slow with mujoco_py.
    # Ref: https://github.com/openai/mujoco-py/issues/58 
    def get_image(self, width=84, height=84, camera_name=None):
        self.active_env._get_viewer(mode='rgb_array').render(width, height)
        data = self.active_env._get_viewer(mode='rgb_array').read_pixels(width, height, depth=False)
        # original image is upside-down, so flip it
        return data[::-1, :, :]

    # This method is kinda dirty but this offer setting camera
    # angle programatically. You can easily select a good camera angle
    # by firing up a python interactive session then render an
    # environment and use the mouse to select a view. To retrive camera
    # information, just run `print(env.viewer.cam.lookat, env.viewer.cam.distance,
    # env.viewer.cam.elevation, env.viewer.cam.azimuth)`
    def _configure_viewer(self, setting):
        def _viewer_setup(env):
            env.viewer.cam.trackbodyid = 0
            env.viewer.cam.lookat[0] = setting['lookat'][0]
            env.viewer.cam.lookat[1] = setting['lookat'][1]
            env.viewer.cam.lookat[2] = setting['lookat'][2]
            env.viewer.cam.distance = setting['distance']
            env.viewer.cam.elevation = setting['elevation']
            env.viewer.cam.azimuth = setting['azimuth']
            env.viewer.cam.trackbodyid = -1
        self.active_env.viewer_setup = MethodType(_viewer_setup, self.active_env)

    @property
    def num_tasks(self):
        """Number of tasks contained within this environment instance
            e.g. MT10 has 10 available 
        note: This might be different than the number in the benchmark if
            the from_task api is used.
        """
        return len(self._task_envs)

    @property
    def active_task(self):
        """The current task being exposed."""
        return ((self._env_discrete_index[self._task_names[self._active_task]] +
                (self.active_env.active_discrete_goal or 0)) % _NUM_METAWORLD_ENVS)

    @property
    def active_task_one_hot(self):
        task_onehot = np.zeros(shape=(_NUM_METAWORLD_ENVS,), dtype=np.float32)
        task_onehot[self.active_task] = 1.
        return task_onehot

    @property
    def active_env(self):
        return self._task_envs[self._active_task or 0]

    @property
    def action_space(self):
        return self.active_env.action_space

    def render(self, *args, **kwargs):
        return self.active_env.render(*args, **kwargs)

    def close(self):
        for env in self._task_envs:
            env.close()
