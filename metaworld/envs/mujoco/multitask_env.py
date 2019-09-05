import gym
from gym.spaces import Box
import numpy as np

from metaworld.core.serializable import Serializable


class MultiTaskEnv(gym.Env, Serializable):
    def __init__(self,
                 task_env_cls=None,
                 task_args=None,
                 task_kwargs=None,):
        Serializable.quick_init(self, locals())
        self._task_envs = [
            task_env_cls(*t_args, **t_kwargs)
            for t_args, t_kwargs in zip(task_args, task_kwargs)
        ]
        self._active_task = None

    def reset(self, **kwargs):
        return self.active_env.reset(**kwargs)

    @property
    def action_space(self):
        return self.active_env.action_space

    @property
    def observation_space(self):
        return self.active_env.observation_space

    def step(self, action):
        obs, reward, done, info = self.active_env.step(action)
        info['task'] = self.active_task_one_hot
        return obs, reward, done, info

    def render(self, *args, **kwargs):
        return self.active_env.render(*args, **kwargs)

    def close(self):
        for env in self._task_envs:
            env.close()

    @property
    def task_space(self):
        n = len(self._task_envs)
        one_hot_ub = np.ones(n)
        one_hot_lb = np.zeros(n)
        return gym.spaces.Box(one_hot_lb, one_hot_ub, dtype=np.float32)

    @property
    def active_task(self):
        return self._active_task

    @property
    def active_task_one_hot(self):
        one_hot = np.zeros(self.task_space.shape)
        t = self.active_task or 0
        one_hot[t] = self.task_space.high[t]
        return one_hot

    @property
    def active_env(self):
        return self._task_envs[self.active_task or 0]

    @property
    def num_tasks(self):
        return len(self._task_envs)

    '''
    API's for MAML Sampler
    '''
    def sample_tasks(self, meta_batch_size):
        return np.random.randint(0, self.num_tasks, size=meta_batch_size)
    
    def set_task(self, task):
        self._active_task = task

    def log_diagnostics(self, paths, prefix):
        pass


class MultiClassMultiTaskEnv(MultiTaskEnv):

    # TODO maybe we should add a task_space to this
    # environment. In that case we can just do a `task_space.sample()`
    # and have a single task sampling API accros this repository. 

    def __init__(self,
                 task_env_cls_dict,
                 task_args_kwargs,
                 sample_all=True,
                 sample_goals=False,
                 obs_type='plain',):
        Serializable.quick_init(self, locals())
        assert len(task_env_cls_dict.keys()) == len(task_args_kwargs.keys())
        assert len(task_env_cls_dict.keys()) >= 1
        for k in task_env_cls_dict.keys():
            assert k in task_args_kwargs

        self._task_envs = []
        self._task_names = []
        self._sampled_all = sample_all
        self._sample_goals = sample_goals
        self._obs_type = obs_type

        for task, env_cls in task_env_cls_dict.items():
            task_args = task_args_kwargs[task]['args']
            task_kwargs = task_args_kwargs[task]['kwargs']
            task_env = env_cls(*task_args, **task_kwargs)

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
        self._env_discrete_index = {
            task: i
            for i, task in enumerate(self._task_names)
        }
        self._fully_discretized = True if not sample_goals else False
        self._n_discrete_goals = len(task_env_cls_dict.keys())
        self._active_task = 0
        self._check_env_list()

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
        if self._obs_type == 'plain':
            return self._task_envs[self.observation_space_index].observation_space
        else:
            plain_high = self._task_envs[self.observation_space_index].observation_space.high
            plain_low = self._task_envs[self.observation_space_index].observation_space.low
            goal_high = self.active_env.goal_space.high
            goal_low = self.active_env.goal_space.low
            if self._obs_type == 'with_goal':
                return Box(
                    high=np.concatenate([plain_high, goal_high]),
                    low=np.concatenate([plain_low, goal_low]))
            elif self._obs_type == 'with_goal_id' and self._fully_discretized:
                goal_id_low = np.zeros(shape=(self._n_discrete_goals,))
                goal_id_high = np.ones(shape=(self._n_discrete_goals,))
                return Box(
                    high=np.concatenate([plain_high, goal_id_low,]),
                    low=np.concatenate([plain_low, goal_id_high,]))
            elif self._obs_type == 'with_goal_and_id' and self._fully_discretized:
                goal_id_low = np.zeros(shape=(self._n_discrete_goals,))
                goal_id_high = np.ones(shape=(self._n_discrete_goals,))
                return Box(
                    high=np.concatenate([plain_high, goal_id_low, goal_high]),
                    low=np.concatenate([plain_low, goal_id_high, goal_low]))
            else:
                raise NotImplementedError

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
        obs, reward, done, info = self.active_env.step(action)
        obs = self._augment_observation(obs)
        if 'task_type' in dir(self.active_env):
            name = '{}-{}'.format(str(self.active_env.__class__.__name__), self.active_env.task_type)
        else:
            name = str(self.active_env.__class__.__name__)
        info['task_name'] = name
        return obs, reward, done, info

    def _augment_observation(self, obs):
        # optionally zero-pad observation
        if np.prod(obs.shape) < self._max_plain_dim:
            zeros = np.zeros(
                shape=(self._max_plain_dim - np.prod(obs.shape),)
            )
            obs = np.concatenate([obs, zeros])

        # augment the observation based on obs_type:
        if self._obs_type == 'with_goal_id' or self._obs_type == 'with_goal_and_id':
            if self._obs_type == 'with_goal_and_id':
                obs = np.concatenate([obs, self.active_env._state_goal])
            task_id = self._env_discrete_index[self._task_names[self.active_task]] + (self.active_env.active_discrete_goal or 0)
            task_onehot = np.zeros(shape=(self._n_discrete_goals,), dtype=np.float32)
            task_onehot[task_id] = 1.
            obs = np.concatenate([obs, task_onehot])
        elif self._obs_type == 'with_goal':
            obs = np.concatenate([obs, self.active_env._state_goal])
        return obs

    def reset(self, **kwargs):
        return self._augment_observation(self.active_env.reset(**kwargs))

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
