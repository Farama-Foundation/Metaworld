from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict

from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv


class SawyerPickAndPlaceEnv(MultitaskEnv, SawyerXYZEnv):
    def __init__(
            self,
            block_low=None,
            block_high=None,

            reward_type='hand_and_block_distance',
            indicator_threshold=0.06,

            fix_goal=False,
            fixed_goal=(0.15, 0.6, 0.055, -0.15, 0.6),
            goal_low=None,
            goal_high=None,

            **kwargs
    ):
        self.quick_init(locals())
        MultitaskEnv.__init__(self)
        SawyerXYZEnv.__init__(
            self,
            model_name=self.model_name,
            **kwargs
        )
        if block_low is None:
            block_low = self.hand_low
        if block_high is None:
            block_high = self.hand_high

        if goal_low is None:
            goal_low = np.hstack((self.hand_low, block_low))
        if goal_high is None:
            goal_high = np.hstack((self.hand_high, block_high))

        self.reward_type = reward_type
        self.indicator_threshold = indicator_threshold

        self.fix_goal = fix_goal
        self.fixed_goal = np.array(fixed_goal)
        self._goal = None

        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
        )
        self.hand_and_block_space = Box(
            np.hstack((self.hand_low, block_low)),
            np.hstack((self.hand_high, block_high)),
        )
        self.observation_space = Dict([
            ('observation', self.hand_and_block_space),
            ('desired_goal', self.hand_and_block_space),
            ('achieved_goal', self.hand_and_block_space),
            ('state_observation', self.hand_and_block_space),
            ('state_desired_goal', self.hand_and_block_space),
            ('state_achieved_goal', self.hand_and_block_space),
        ])

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_pick_and_place.xml')

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 1.0
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.distance = 0.3
        self.viewer.cam.elevation = -45
        self.viewer.cam.azimuth = 270
        self.viewer.cam.trackbodyid = -1

    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation(action[3:])
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._goal)
        obs = self._get_obs()
        info = self._get_info()
        reward = self.compute_reward(
            obs['achieved_goal'],
            obs['desired_goal'],
            info,
        )
        done = False
        return obs, reward, done, info

    def _get_obs(self):
        e = self.get_endeff_pos()
        b = self.get_block_pos()
        flat_obs = np.concatenate((e, b))

        return dict(
            observation=flat_obs,
            desired_goal=self._goal,
            achieved_goal=flat_obs,
            state_observation=flat_obs,
            state_desired_goal=self._goal,
            state_achieved_goal=flat_obs,
        )

    def _get_info(self):
        hand_goal = self._goal[:3]
        block_goal = self._goal[3:]
        hand_distance = np.linalg.norm(hand_goal - self.get_endeff_pos())
        block_distance = np.linalg.norm(block_goal - self.get_block_pos())
        touch_distance = np.linalg.norm(
            self.get_endeff_pos() - self.get_block_pos()
        )
        return dict(
            hand_distance=hand_distance,
            block_distance=block_distance,
            hand_and_block_distance=hand_distance+block_distance,
            touch_distance=touch_distance,
            hand_success=float(hand_distance < self.indicator_threshold),
            block_success=float(block_distance < self.indicator_threshold),
            hand_and_block_success=float(
                hand_distance+block_distance < self.indicator_threshold
            ),
            touch_success=float(touch_distance < self.indicator_threshold),
        )

    def get_block_pos(self):
        return self.data.get_body_xpos('block').copy()

    def sample_block_xy(self):
        return np.array([0, 0.6])

    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('hand-goal-site')] = (
            goal[:3]
        )
        self.data.site_xpos[self.model.site_name2id('block-goal-site')] = (
            goal[3:]
        )

    def _set_block_xy(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[8:11] = np.hstack((pos.copy(), np.array([0.02])))
        qvel[8:11] = [0, 0, 0]
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()
        goal = self.sample_goal()
        self._set_goal(goal)

        self._set_block_xy(self.sample_block_xy())
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
                self.hand_and_block_space.low,
                self.hand_and_block_space.high,
                size=(batch_size, self.hand_and_block_space.low.size),
            )

    def compute_rewards(self, achieved_goals, desired_goals, info):
        hand_pos = achieved_goals[:, :3]
        block_pos = achieved_goals[:, 3:]
        hand_goals = desired_goals[:, :3]
        block_goals = desired_goals[:, 3:]

        hand_distances = np.linalg.norm(hand_goals - hand_pos, axis=1)
        block_distances = np.linalg.norm(block_goals - block_pos, axis=1)
        hand_and_block_distances = hand_distances + block_distances
        touch_distances = np.linalg.norm(hand_pos - block_pos, axis=1)

        if self.reward_type == 'hand_distance':
            r = -hand_distances
        elif self.reward_type == 'hand_success':
            r = -(hand_distances < self.indicator_threshold).astype(float)
        elif self.reward_type == 'block_distance':
            r = -block_distances
        elif self.reward_type == 'block_success':
            r = -(block_distances < self.indicator_threshold).astype(float)
        elif self.reward_type == 'hand_and_block_distance':
            r = -hand_and_block_distances
        elif self.reward_type == 'hand_and_block_success':
            r = -(
                hand_and_block_distances < self.indicator_threshold
            ).astype(float)
        elif self.reward_type == 'touch_distance':
            r = -touch_distances
        elif self.reward_type == 'touch_success':
            r = -(touch_distances < self.indicator_threshold).astype(float)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'hand_distance',
            'block_distance',
            'hand_and_block_distance',
            'touch_distance',
            'hand_success',
            'block_success',
            'hand_and_block_success',
            'touch_success',
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
