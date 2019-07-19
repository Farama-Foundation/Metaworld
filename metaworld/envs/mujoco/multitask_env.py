import gym
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

    def __init__(self,
                 task_env_cls_dict=None,
                 task_args_kwargs=None,
                 sample_all=True,):
        Serializable.quick_init(self, locals())

        assert len(task_env_cls_dict.keys()) == len(task_args_kwargs.keys())
        for k in task_env_cls_dict.keys():
            assert k in task_args_kwargs

        self._task_envs = []
        self._task_names = []
        self._sampled_all = sample_all

        for task, env_cls in task_env_cls_dict.items():
            task_args = task_args_kwargs[task]['args']
            task_kwargs = task_args_kwargs[task]['kwargs']
            task_env = env_cls(*task_args, **task_kwargs)
            self._task_envs.append(task_env)
            self._task_names.append(task)
        self._active_task = 0
        self._check_env_list()

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
            
    @property
    def observation_space(self):
        return self._task_envs[self.observation_space_index].observation_space

    def set_task(self, task):
        self._active_task = task % len(self._task_envs)

    def sample_tasks(self, meta_batch_size):
        if self._sampled_all:
            assert meta_batch_size >= len(self._task_envs)
            return [i for i in range(meta_batch_size)]
        else:
            return np.random.randint(0, self.num_tasks, size=meta_batch_size)

    def step(self, action):
        obs, reward, done, info = self.active_env.step(action)
        # optionally zero-pad observation
        # I know using np.prod is overkilling but maybe we
        # want to expand this to higher dimension..?
        if np.prod(obs.shape) < np.prod(self.observation_space.shape):
            obs_type = self.active_env.obs_type
            zeros = np.zeros(
                shape=(np.prod(self.observation_space.shape) - np.prod(obs.shape),)
            )
            if obs_type == 'plain':
                obs = np.concatenate([obs, zeros])
            elif obs_type == 'with_goal_idx':
                id_len = self.active_env._state_goal_idx.shape[0]
                obs = np.concatenate([obs[:-id_len], zeros, obs[-id_len:]])
            elif obs_type == 'with_goal_and_idx':
                # this assumes that the environment has a goal space
                id_len = self.active_env._state_goal_idx.shape[0]
                goal_len = np.prod(self.active_env.goal_space.low.shape)
                obs = np.concatenate([obs[:-id_len - goal_len], zeros, obs[-id_len - goal_len:]])
            else:
                # with goal
                # this assumes that the environment has a goal space
                goal_len = np.prod(self.active_env.goal_space.low.shape)
                obs = np.concatenate([obs[:-goal_len], zeros, obs[-goal_len:]])
        if 'task_type' in dir(self.active_env):
            name = '{}-{}'.format(str(self.active_env.__class__.__name__), self.active_env.task_type)
        else:
            name = str(self.active_env.__class__.__name__)
        info['task_name'] = name
        return obs, reward, done, info
