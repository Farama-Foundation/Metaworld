import gym
import gym.spaces
import numpy as np

from metaworld.core.multitask_env import MultitaskEnv


class GoalConditioned(MultitaskEnv, gym.Wrapper):
    """
    A MultitaskEnv wrapper which represents a World of tasks which are
    continously-parameterized by a single goal state.

    Args:
        env(:obj:`TaskBased`): A `TaskBased` World to wrap
    """

    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

        # Augment the observation space with a state goal
        base = self.env.observation_space.spaces.copy()
        base['goal'] = self.env.task_schema.spaces['obj_init_pos']
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
        return self._augment_observation(self.env.reset())

    def _augment_observation(self, o):
        o['goal'] = self.env.get_goal()
        return o


if __name__ == '__main__':
    import time

    from .sawyer_window_open_6dof import SawyerWindowOpen6DOFEnv

    world = SawyerWindowOpen6DOFEnv()
    env = GoalConditioned(world)
    for _ in range(1000):
        t = world.task_schema.sample()
        world.task = t
        env.reset()
        for _ in range(100):
            env.render()
            step = env.step(np.array([1, 0, 0, 1]))
            time.sleep(0.05)
