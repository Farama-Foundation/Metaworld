from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict

from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv


class SawyerPushAndReachXYZEnv(MultitaskEnv, SawyerXYZEnv):
    def __init__(
            self,
            puck_low=None,
            puck_high=None,

            reward_type='hand_and_puck_distance',
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
        if puck_low is None:
            puck_low = self.hand_low[:2]
        if puck_high is None:
            puck_high = self.hand_high[:2]

        if goal_low is None:
            goal_low = np.hstack((self.hand_low, puck_low))
        if goal_high is None:
            goal_high = np.hstack((self.hand_high, puck_high))

        self.reward_type = reward_type
        self.indicator_threshold = indicator_threshold

        self.fix_goal = fix_goal
        self.fixed_goal = np.array(fixed_goal)
        self._goal = None

        self.action_space = Box(np.array([-1, -1, -1]), np.array([1, 1, 1]))
        self.hand_and_puck_space = Box(
            np.hstack((self.hand_low, puck_low)),
            np.hstack((self.hand_high, puck_high)),
        )
        self.observation_space = Dict([
            ('observation', self.hand_and_puck_space),
            ('desired_goal', self.hand_and_puck_space),
            ('achieved_goal', self.hand_and_puck_space),
            ('state_observation', self.hand_and_puck_space),
            ('state_desired_goal', self.hand_and_puck_space),
            ('state_achieved_goal', self.hand_and_puck_space),
        ])
        self.init_puck_z = self.get_puck_pos()[2]

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_push_puck.xml')

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
        self.set_xyz_action(action)
        self.do_simulation(None)
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
        b = self.get_puck_pos()[:2]
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
        puck_goal = self._goal[3:]
        hand_distance = np.linalg.norm(hand_goal - self.get_endeff_pos())
        puck_distance = np.linalg.norm(puck_goal - self.get_puck_pos()[:2])
        touch_distance = np.linalg.norm(
            self.get_endeff_pos() - self.get_puck_pos()
        )
        return dict(
            hand_distance=hand_distance,
            puck_distance=puck_distance,
            hand_and_puck_distance=hand_distance+puck_distance,
            touch_distance=touch_distance,
            hand_success=float(hand_distance < self.indicator_threshold),
            puck_success=float(puck_distance < self.indicator_threshold),
            hand_and_puck_success=float(
                hand_distance+puck_distance < self.indicator_threshold
            ),
            touch_success=float(touch_distance < self.indicator_threshold),
        )

    def get_puck_pos(self):
        return self.data.get_body_xpos('puck').copy()

    def sample_puck_xy(self):
        return np.array([0, 0.6])

    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('hand-goal-site')] = (
            goal[:3]
        )
        self.data.site_xpos[self.model.site_name2id('puck-goal-site')][:2] = (
            goal[3:]
        )

    def _set_puck_xy(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[7:10] = np.hstack((pos.copy(), np.array([0.02])))
        qvel[7:10] = [0, 0, 0]
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()
        goal = self.sample_goal()
        self._set_goal(goal)

        self._set_puck_xy(self.sample_puck_xy())
        # self.reset_mocap_welds()
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
                self.hand_and_puck_space.low,
                self.hand_and_puck_space.high,
                size=(batch_size, self.hand_and_puck_space.low.size),
            )

    def compute_rewards(self, achieved_goals, desired_goals, info):
        hand_pos = achieved_goals[:, :3]
        puck_pos = achieved_goals[:, 3:]
        hand_goals = desired_goals[:, :3]
        puck_goals = desired_goals[:, 3:]

        hand_distances = np.linalg.norm(hand_goals - hand_pos, axis=1)
        puck_distances = np.linalg.norm(puck_goals - puck_pos, axis=1)
        hand_and_puck_distances = hand_distances + puck_distances
        puck_zs = self.init_puck_z * np.ones((desired_goals.shape[0], 1))
        touch_distances = np.linalg.norm(
            hand_pos - np.hstack((puck_pos, puck_zs)),
            axis=1,
        )

        if self.reward_type == 'hand_distance':
            r = -hand_distances
        elif self.reward_type == 'hand_success':
            r = -(hand_distances < self.indicator_threshold).astype(float)
        elif self.reward_type == 'puck_distance':
            r = -puck_distances
        elif self.reward_type == 'puck_success':
            r = -(puck_distances < self.indicator_threshold).astype(float)
        elif self.reward_type == 'hand_and_puck_distance':
            r = -hand_and_puck_distances
        elif self.reward_type == 'hand_and_puck_success':
            r = -(hand_and_puck_distances < self.indicator_threshold).astype(float)
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
            'puck_distance',
            'hand_and_puck_distance',
            'touch_distance',
            'hand_success',
            'puck_success',
            'hand_and_puck_success',
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


class SawyerPushAndReachXYEnv(SawyerPushAndReachXYZEnv):
    def __init__(self, *args, hand_z_position=0.055, **kwargs):
        self.quick_init(locals())
        SawyerPushAndReachXYZEnv.__init__(self, *args, **kwargs)
        self.hand_z_position = hand_z_position
        self.action_space = Box(np.array([-1, -1]), np.array([1, 1]))
        self.fixed_goal[2] = hand_z_position
        hand_and_puck_low = self.hand_and_puck_space.low.copy()
        hand_and_puck_low[2] = hand_z_position
        hand_and_puck_high = self.hand_and_puck_space.high.copy()
        hand_and_puck_high[2] = hand_z_position
        self.hand_and_puck_space = Box(hand_and_puck_low, hand_and_puck_high)
        self.observation_space = Dict([
            ('observation', self.hand_and_puck_space),
            ('desired_goal', self.hand_and_puck_space),
            ('achieved_goal', self.hand_and_puck_space),
            ('state_observation', self.hand_and_puck_space),
            ('state_desired_goal', self.hand_and_puck_space),
            ('state_achieved_goal', self.hand_and_puck_space),
        ])

    def step(self, action):
        delta_z = self.hand_z_position - self.data.mocap_pos[0, 2]
        action = np.hstack((action, delta_z))
        return super().step(action)
