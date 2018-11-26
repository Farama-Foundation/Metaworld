import numpy as np
import sawyer_control.envs.sawyer_reaching as sawyer_reaching

from multiworld.core.multitask_env import MultitaskEnv
from multiworld.core.serializable import Serializable
from gym.spaces import Dict

class SawyerReachXYZEnv(sawyer_reaching.SawyerReachXYZEnv, MultitaskEnv):
    def __init__(self,
                 **kwargs
                 ):
        Serializable.quick_init(self, locals())
        sawyer_reaching.SawyerReachXYZEnv.__init__(self, **kwargs)
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
        reward = MultitaskEnv.compute_reward(self, action, observation)
        info = self._get_info()
        done = False
        return observation, reward, done, info

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        hand_pos = achieved_goals
        goals = desired_goals
        distances = np.linalg.norm(hand_pos - goals, axis=1)
        if self.reward_type == 'hand_distance':
            r = -distances
        elif self.reward_type == 'hand_success':
            r = -(distances > self.indicator_threshold).astype(float)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def _get_obs(self):
        ee_pos = self._get_endeffector_pose()
        state_obs = super()._get_obs()
        return dict(
            observation=state_obs,
            desired_goal=self._state_goal,
            achieved_goal=ee_pos,

            state_observation=state_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=ee_pos,
        )

    def reset(self):
        if self.action_mode == "position":
            self._position_act(self.reset_pos - self._get_endeffector_pose(), in_reset=True)
        else:
            self._reset_robot()
        goal = self.sample_goal()
        self._state_goal = goal['state_desired_goal']
        return self._get_obs()

    """
    Multitask functions
    """

    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

    def sample_goal(self):
        return MultitaskEnv.sample_goal(self)

    def sample_goals(self, batch_size):
        goals = super().sample_goals(batch_size)
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def set_to_goal(self, goal):
        goal = goal['state_desired_goal']
        super().set_to_goal(goal)

if __name__=="__main__":
    env = SawyerReachXYZEnv()
    env.reset()
