from collections import OrderedDict
import numpy as np
from sawyer_control.envs.sawyer_env_base import SawyerEnvBase
from multiworld.core.serializable import Serializable
from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict
from multiworld.core.multitask_env import MultitaskEnv
from gym.spaces import Dict

class SawyerReachXYZEnv(SawyerEnvBase, MultitaskEnv):
    def __init__(self,
                 fixed_goal=(1, 1, 1),
                 indicator_threshold=.05,
                 reward_type='hand_distance',
                 **kwargs
                 ):
        Serializable.quick_init(self, locals())
        SawyerEnvBase.__init__(self, **kwargs)
        if self.action_mode=='torque':
            self.goal_space = self.config.TORQUE_SAFETY_BOX
        else:
            self.goal_space = self.config.POSITION_SAFETY_BOX
        self.indicator_threshold=indicator_threshold
        self.reward_type = reward_type
        self._state_goal = np.array(fixed_goal)
        self.observation_space = Dict([
            ('observation', self._observation_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.goal_space),
            ('state_observation', self._observation_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.goal_space),
        ])
        self.reset()

    def step(self, action):
        self._act(action)
        observation = self._get_obs()
        reward = self.compute_reward(action, observation)
        info = self._get_info()
        done = False
        return observation, reward, done, info

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['achieved_goal']
        desired_goals = obs['desired_goal']
        hand_pos = achieved_goals
        goals = desired_goals
        distances = np.linalg.norm(hand_pos - goals, axis=1)
        if self.reward_type == 'hand_distance':
            r = -distances
        elif self.reward_type == 'hand_success':
            r = -(distances < self.indicator_threshold).astype(float)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def _get_obs(self):
        ee_pos = self.get_endeffector_pose()
        state_obs = self._get_env_obs()
        return dict(
            observation=state_obs,
            desired_goal=self._state_goal,
            achieved_goal=ee_pos,

            state_observation=state_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=ee_pos,
        )

    def _get_info(self):
        hand_distance = np.linalg.norm(self._state_goal - self._get_endeffector_pose())
        return dict(
            hand_distance=hand_distance,
            hand_success=(hand_distance<self.indicator_threshold).astype(float)
        )

    def reset(self):
        self._reset_robot()
        goal = self.sample_goal()
        self._state_goal = goal['state_desired_goal']
        return self._get_obs()

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'hand_distance',
            'hand_success',
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
                ))
            statistics.update(create_stats_ordered_dict(
                'Final %s%s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
                ))
        return statistics

    #Image Env Functions
    def get_env_state(self):
        #this should be implemented for position control, but just return None for torque control
        return None


    def set_env_state(self, state):
        # this should be implemented for position control, but just return None for torque control
        return None

    """
    Multitask functions
    """

    @property
    def goal_dim(self) -> int:
        return 3

    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

    def sample_goals(self, batch_size):
        if self.fix_goal:
            goals = np.repeat(
                self.fixed_goal.copy()[None],
                batch_size,
                0
            )
        else:
            goals = np.random.uniform(
                self.goal_space.low,
                self.goal_space.high,
                size=(batch_size, self.goal_space.low.size),
            )
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def set_to_goal(self, goal):
        #for position control this should be implemented, for torque control this should raise an error
        raise NotImplementedError()

    def convert_obs_to_goals(self, obs):
        return obs[:, -3:]

