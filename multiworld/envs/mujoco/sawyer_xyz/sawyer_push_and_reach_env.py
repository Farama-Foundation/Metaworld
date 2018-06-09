from collections import OrderedDict
import numpy as np
from gym.spaces import Box

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
        self.goal_low = goal_low
        self.goal_high = goal_high
        self._goal = self.sample_goal()
        self.goal_space = Box(
            np.hstack((self.hand_low, puck_low)),
            np.hstack((self.hand_high, puck_high)),
        )

        self.action_space = Box(np.array([-1, -1, -1]), np.array([1, 1, 1]))
        self.observation_space = Box(
            np.hstack((self.hand_low, puck_low)),
            np.hstack((self.hand_high, puck_high)),
        )

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
        info = self.get_info()
        reward = self.compute_reward(obs, action, obs, self._goal, info)
        done = False
        return obs, reward, done, info

    def _get_obs(self):
        e = self.get_endeff_pos()
        b = self.get_puck_pos()[:2]
        return np.concatenate((e, b))

    def get_info(self):
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
        return self.data.get_body_xpos('puck')

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
                self.goal_low,
                self.goal_high,
                size=(batch_size, self.goal_low.size),
            )

    def compute_rewards(self, obs, actions, next_obs, goals, env_infos):
        if self.reward_type == 'hand_and_puck_distance':
            r = -np.hstack([info['hand_and_puck_distance'] for info in env_infos])
        elif self.reward_type == 'hand_distance':
            r = -np.hstack([info['hand_distance'] for info in env_infos])
        elif self.reward_type == 'puck_distance':
            r = -np.hstack([info['puck_distance'] for info in env_infos])
        elif self.reward_type == 'touch_distance':
            r = -np.hstack([info['touch_distance'] for info in env_infos])
        elif self.reward_type == 'hand_and_puck_success':
            r = -np.hstack([info['hand_and_puck_success'] for info in env_infos])
        elif self.reward_type == 'hand_success':
            r = -np.hstack([info['hand_success'] for info in env_infos])
        elif self.reward_type == 'puck_success':
            r = -np.hstack([info['puck_success'] for info in env_infos])
        elif self.reward_type == 'touch_success':
            r = -np.hstack([info['touch_success'] for info in env_infos])
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
        goal_low = self.goal_space.low.copy()
        goal_low[2] = hand_z_position
        goal_high = self.goal_space.high.copy()
        goal_high[2] = hand_z_position
        self.goal_space = Box(goal_low, goal_high)

    def step(self, action):
        delta_z = self.hand_z_position - self.data.mocap_pos[0, 2]
        action = np.hstack((action, delta_z))
        return super().step(action)
