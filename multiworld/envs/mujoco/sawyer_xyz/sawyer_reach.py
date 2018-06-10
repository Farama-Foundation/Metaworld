from collections import OrderedDict
import numpy as np
from gym.spaces import Box

from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv


class SawyerReachXYZEnv(MultitaskEnv, SawyerXYZEnv):
    def __init__(
            self,
            reward_type='hand_distance',
            indicator_threshold=0.06,

            fix_goal=False,
            fixed_goal=(0.15, 0.6, 0.3),
            goal_low=None,
            goal_high=None,

            **kwargs
    ):
        self.quick_init(locals())
        MultitaskEnv.__init__(self)
        SawyerXYZEnv.__init__(self, model_name=self.model_name, **kwargs)

        if goal_low is None:
            goal_low = self.hand_low
        if goal_high is None:
            goal_high = self.hand_high

        self.reward_type = reward_type
        self.indicator_threshold = indicator_threshold

        self.fix_goal = fix_goal
        self.fixed_goal = np.array(fixed_goal)
        self.goal_space = Box(goal_low, goal_high)
        self._goal = self.sample_goal()

        self.action_space = Box(np.array([-1, -1, -1]), np.array([1, 1, 1]))
        self.observation_space = Box(self.hand_low, self.hand_high)

    def step(self, action):
        self.set_xyz_action(action)
        self.do_simulation(None)
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._goal)
        info = self.get_info()
        obs = info['observation']
        reward = self.compute_reward(obs, action, obs, info)
        done = False
        return obs, reward, done, info

    def _get_obs(self):
        return self.get_endeff_pos()

    def get_info(self):
        hand_goal = self._goal
        hand_distance = np.linalg.norm(hand_goal - self.get_endeff_pos())
        obs = self._get_obs()
        return dict(
            observation=obs,
            desired_goal=self._goal,
            achieved_goal=obs,
            state_observation=obs,
            state_desired_goal=self._goal,
            state_achieved_goal=obs,
            hand_distance=hand_distance,
            hand_success=float(hand_distance < self.indicator_threshold),
        )

    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('hand-goal-site')] = (
            goal
        )

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_reach.xml')

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 1.0
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.distance = 0.3
        self.viewer.cam.elevation = -45
        self.viewer.cam.azimuth = 270
        self.viewer.cam.trackbodyid = -1

    def reset_model(self):
        self._reset_hand()
        goal = self.sample_goal()
        self._set_goal(goal)
        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', np.array([0, 0.5, 0.02]))
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation(None, self.frame_skip)

    def _set_goal(self, goal):
        self._goal = goal
        self._set_goal_marker(self._goal)

    """
    Multitask functions
    """
    def get_goal(self):
        return self._goal

    def sample_goals(self, batch_size):
        if self.fix_goal:
            return np.repeat(
                self.fixed_goal.copy()[None],
                batch_size,
                0
            )
        else:
            return np.random.uniform(
                self.goal_space.low,
                self.goal_space.high,
                size=(batch_size, self.goal_space.low.size),
            )

    def compute_rewards(self, obs, actions, next_obs, env_infos):
        hand_pos = next_obs
        goals = env_infos['desired_goal']
        distances = np.linalg.norm(hand_pos - goals, axis=1)
        if self.reward_type == 'hand_distance':
            r = -distances
        elif self.reward_type == 'hand_success':
            r = -(distances < self.indicator_threshold).astype(float)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

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


class SawyerReachXYEnv(SawyerReachXYZEnv):
    def __init__(self, *args,
                 fixed_goal=(0.15, 0.6),
                 hand_z_position=0.055, **kwargs):
        self.quick_init(locals())
        SawyerReachXYZEnv.__init__(
            self,
            *args,
            fixed_goal=(fixed_goal[0], fixed_goal[1], hand_z_position),
            **kwargs
        )
        self.hand_z_position = hand_z_position
        self.action_space = Box(np.array([-1, -1]), np.array([1, 1]))
        self.goal_space = Box(
            np.hstack((self.goal_space.low[:2], self.hand_z_position)),
            np.hstack((self.goal_space.high[:2], self.hand_z_position))
        )
        self._goal = self.sample_goal()

    def step(self, action):
        delta_z = self.hand_z_position - self.data.mocap_pos[0, 2]
        action = np.hstack((action, delta_z))
        return super().step(action)
