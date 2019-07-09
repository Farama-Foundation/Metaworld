import gym
import gym.spaces
import numpy as np

from metaworld.core.multitask_env import MultitaskEnv


class ContinuousMultitask(MultitaskEnv, gym.Wrapper):
    """
    A MultitaskEnv wrapper which represents a World of continously-
    parameterized tasks.

    Every call to `reset()`, the wrapper samples a new task uniformly from the
    underlying task parameterization.

    Args:
        env(:obj:`TaskBased`): A `TaskBased` World to wrap
    """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        # Augment the observation space with a state goal
        base = self.env.observation_space.spaces.copy()
        base['task'] = self.env.task_schema
        self.observation_space = gym.spaces.Dict(base)

    def sample_goals(self, batch_size):
        tasks = [self.env.task_schema.sample() for _ in range(batch_size)]
        return {
            'state_desired_goal': [self.env.goal_from_task(t) for t in tasks],
        }

    def get_goal(self):
        return self.env.get_goal()

    def compute_rewards(self, actions, obs):
        return self.env.compute_rewards(actions, obs)

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        return statistics

    def log_diagnostics(self, paths=None, logger=None):
        pass

    def step(self, action):
        o, r, d, i = self.env.step(action)
        return self._augment_observation(o), r, d, i

    def reset(self):
        self.env.task = self.env.task_schema.sample()
        return self._augment_observation(self.env.reset())

    def _augment_observation(self, o):
        o['task'] = self.env.task
        return o


if __name__ == '__main__':
    import time

    from .sawyer_window_open_6dof import SawyerWindowOpen6DOFEnv

    world = SawyerWindowOpen6DOFEnv()
    env = ContinuousMultitask(world)
    for _ in range(1000):
        env.reset()
        for _ in range(100):
            env.render()
            step = env.step(np.array([1, 0, 0, 1]))
            time.sleep(0.05)
