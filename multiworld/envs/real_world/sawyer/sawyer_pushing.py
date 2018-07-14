from collections import OrderedDict
import numpy as np
from gym.spaces import Dict
import sawyer_control.envs.sawyer_pushing as sawyer_pushing
from sawyer_control.core.serializable import Serializable

class SawyerPushXYEnv(sawyer_pushing.SawyerPushXYEnv):
    ''' Must Wrap with Image Env to use!'''
    def __init__(self,
                 **kwargs
                ):
        Serializable.quick_init(self, locals())
        sawyer_pushing.SawyerPushXYEnv.__init__(self, **kwargs)
        self.observation_space = Dict([
            ('observation', self.observation_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.goal_space),
            ('state_observation', self.observation_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.goal_space),
        ])

    def step(self, action):
        self._act(action)
        observation = self._get_obs()
        reward = self.compute_reward(action, observation)
        info = self._get_info()
        done = False
        return observation, reward, done, info

    def compute_rewards(self, actions, obs):
        pass

    def _get_obs(self):
        achieved_goal = None
        state_obs = self._get_env_obs()
        return dict(
            observation=state_obs,
            desired_goal=self._state_goal,
            achieved_goal=achieved_goal,

            state_observation=state_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=achieved_goal,
        )

    def reset(self):
        self._reset_robot()
        self._state_goal = self.sample_goal()['state_desired_goal']
        return self._get_obs()

    """
    Multitask functions
    """

    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

    def sample_goals(self, batch_size):
        goals = super().sample_goals(batch_size)
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def set_to_goal(self, goal):
        goal = goal['state_desired_goal']
        super().set_to_goal(goal)