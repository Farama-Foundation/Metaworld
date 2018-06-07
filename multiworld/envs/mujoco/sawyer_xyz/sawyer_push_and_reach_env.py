from collections import OrderedDict
import numpy as np
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box
import mujoco_py

from multiworld.core.multitask_env import MultitaskEnv


def get_asset_full_path(param):
    pass


class SawyerPushAndReachXYEnv(MujocoEnv, MultitaskEnv):
    INIT_BLOCK_LOW = np.array([-0.05, 0.55])
    INIT_BLOCK_HIGH = np.array([0.05, 0.65])
    PUCK_GOAL_LOW = INIT_BLOCK_LOW
    PUCK_GOAL_HIGH = INIT_BLOCK_HIGH
    HAND_GOAL_LOW = INIT_BLOCK_LOW
    HAND_GOAL_HIGH = INIT_BLOCK_HIGH
    FIXED_PUCK_GOAL = np.array([0.05, 0.6])
    FIXED_HAND_GOAL = np.array([-0.05, 0.6])
    INIT_HAND_POS = np.array([0, 0.4, 0.02])

    def __init__(
            self,
            reward_info=None,
            frame_skip=50,
            pos_action_scale=2. / 100,
            randomize_goals=True,
            hide_goal=False,
            puck_low=(-0.2, 0.5),
            puck_high=(0.2, 0.7),
    ):
        self.quick_init(locals())
        self.reward_info = reward_info
        self.randomize_goals = randomize_goals
        self._pos_action_scale = pos_action_scale
        self.hide_goal = hide_goal
        self._goal_xyxy = self.sample_goal_xyxy()
        MultitaskEnv.__init__(self)
        MujocoEnv.__init__(self, self.model_name, frame_skip=frame_skip)

        self.action_space = Box(
            np.array([-1, -1]),
            np.array([1, 1]),
        )
        self.observation_space = Box(
            np.array([-0.2, 0.5, -0.2, 0.5]),
            np.array([0.2, 0.7, 0.2, 0.7]),
        )
        self.goal_space = Box(
            self.observation_space.low,
            self.observation_space.high,
        )
        self.reset()
        self.reset_mocap_welds()

    @property
    def model_name(self):
        if self.hide_goal:
            # TODO: make new verseion
            return get_asset_full_path('sawyer_push_puck.xml')
        else:
            return get_asset_full_path('sawyer_push_puck.xml')

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

        self.viewer.cam.lookat[0] = 0
        self.viewer.cam.lookat[1] = 1.0
        self.viewer.cam.lookat[2] = 0.5
        self.viewer.cam.distance = 0.3
        self.viewer.cam.elevation = -45
        self.viewer.cam.azimuth = 270
        self.viewer.cam.trackbodyid = -1

    def step(self, a):
        a = np.clip(a, -1, 1)
        mocap_delta_z = 0.06 - self.data.mocap_pos[0, 2]
        new_mocap_action = np.hstack((
            a,
            np.array([mocap_delta_z])
        ))
        self.mocap_set_action(new_mocap_action[:3] * self._pos_action_scale)
        u = np.zeros(7)
        self.do_simulation(u, self.frame_skip)
        obs = self._get_obs()
        reward = self.compute_reward(obs, u, obs, self._goal_xyxy)
        done = False

        hand_distance = np.linalg.norm(
            self.get_hand_goal_pos() - self.get_endeff_pos()
        )
        puck_distance = np.linalg.norm(
            self.get_puck_goal_pos() - self.get_puck_pos())
        touch_distance = np.linalg.norm(
            self.get_endeff_pos() - self.get_puck_pos())
        info = dict(
            hand_distance=hand_distance,
            puck_distance=puck_distance,
            sum_distance=hand_distance+puck_distance,
            touch_distance=touch_distance,
            success=float(hand_distance + puck_distance < 0.06),
        )
        return obs, reward, done, info

    def mocap_set_action(self, action):
        pos_delta = action[None]
        new_mocap_pos = self.data.mocap_pos + pos_delta
        new_mocap_pos[0, 0] = np.clip(
            new_mocap_pos[0, 0],
            -0.1,
            0.1,
        )
        new_mocap_pos[0, 1] = np.clip(
            new_mocap_pos[0, 1],
            -0.1 + 0.6,
            0.1 + 0.6,
            )
        new_mocap_pos[0, 2] = np.clip(
            new_mocap_pos[0, 2],
            0,
            0.5,
        )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

    def _get_obs(self):
        e = self.get_endeff_pos()[:2]
        b = self.get_puck_pos()[:2]
        return np.concatenate((e, b))

    def get_puck_pos(self):
        return self.data.body_xpos[self.puck_id].copy()

    def get_endeff_pos(self):
        return self.data.body_xpos[self.endeff_id].copy()

    def get_hand_goal_pos(self):
        return self.data.body_xpos[self.hand_goal_id].copy()

    def get_puck_goal_pos(self):
        return self.data.body_xpos[self.puck_goal_id].copy()

    @property
    def endeff_id(self):
        return self.model.body_names.index('leftclaw')

    @property
    def puck_id(self):
        return self.model.body_names.index('puck')

    @property
    def puck_goal_id(self):
        return self.model.body_names.index('puck-goal')

    @property
    def hand_goal_id(self):
        return self.model.body_names.index('hand-goal')

    def sample_goal_xyxy(self):
        if self.randomize_goals:
            hand = np.random.uniform(self.HAND_GOAL_LOW, self.HAND_GOAL_HIGH)
            puck = np.random.uniform(self.PUCK_GOAL_LOW, self.PUCK_GOAL_HIGH)
        else:
            hand = self.FIXED_HAND_GOAL.copy()
            puck = self.FIXED_PUCK_GOAL.copy()
        return np.hstack((hand, puck))

    def sample_puck_xy(self):
        raise NotImplementedError("Shouldn't you use "
                                  "SawyerPushAndReachXYEasyEnv? Ask Vitchyr")
        pos = np.random.uniform(self.INIT_BLOCK_LOW, self.INIT_BLOCK_HIGH)
        while np.linalg.norm(self.get_endeff_pos()[:2] - pos) < 0.035:
            pos = np.random.uniform(self.INIT_BLOCK_LOW, self.INIT_BLOCK_HIGH)
        return pos

    def set_puck_xy(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[7:10] = np.hstack((pos.copy(), np.array([0.02])))
        qvel[7:10] = [0, 0, 0]
        self.set_state(qpos, qvel)

    def set_goal_xyxy(self, xyxy):
        self._goal_xyxy = xyxy
        hand_goal = xyxy[:2]
        puck_goal = xyxy[-2:]
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[14:17] = np.hstack((hand_goal.copy(), np.array([0.02])))
        qvel[14:17] = [0, 0, 0]
        qpos[21:24] = np.hstack((puck_goal.copy(), np.array([0.02])))
        qvel[21:24] = [0, 0, 0]
        self.set_state(qpos, qvel)

    def reset_mocap_welds(self):
        """Resets the mocap welds that we use for actuation."""
        sim = self.sim
        if sim.model.nmocap > 0 and sim.model.eq_data is not None:
            for i in range(sim.model.eq_data.shape[0]):
                if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.])
        sim.forward()

    def reset_mocap2body_xpos(self):
        # move mocap to weld joint
        self.data.set_mocap_pos(
            'mocap',
            np.array([self.data.body_xpos[self.endeff_id]]),
        )
        self.data.set_mocap_quat(
            'mocap',
            np.array([self.data.body_xquat[self.endeff_id]]),
        )

    def reset(self):
        velocities = self.data.qvel.copy()
        angles = np.array(self.init_angles)
        self.set_state(angles.flatten(), velocities.flatten())
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.INIT_HAND_POS)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        # set_state resets the goal xy, so we need to explicit set it again
        self.set_goal_xyxy(self._goal_xyxy)
        self.set_puck_xy(self.sample_puck_xy())
        self.reset_mocap_welds()
        return self._get_obs()

    def compute_reward(self, ob, action, next_ob, goal, env_info):
        hand_xy = next_ob[:2]
        puck_xy = next_ob[-2:]
        hand_goal_xy = goal[:2]
        puck_goal_xy = goal[-2:]
        hand_dist = np.linalg.norm(hand_xy - hand_goal_xy)
        puck_dist = np.linalg.norm(puck_xy - puck_goal_xy)
        if not self.reward_info or self.reward_info["type"] == "euclidean":
            r = - hand_dist - puck_dist
        elif self.reward_info["type"] == "hand_only":
            r = - hand_dist
        elif self.reward_info["type"] == "puck_only":
            r = - puck_dist
        elif self.reward_info["type"] == "sparse":
            t = self.reward_info["threshold"]
            r = ((
                hand_dist + puck_dist < t
            ) - 1).astype(float)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    @property
    def init_angles(self):
        return [1.78026069e+00, - 6.84415781e-01, - 1.54549231e-01,
                2.30672090e+00, 1.93111471e+00,  1.27854012e-01,
                1.49353907e+00, 1.80196716e-03, 7.40415706e-01,
                2.09895360e-02,  9.99999990e-01,  3.05766105e-05,
                - 3.78462492e-06, 1.38684523e-04, - 3.62518873e-02,
                6.13435141e-01, 2.09686080e-02,  7.07106781e-01,
                1.48979724e-14, 7.07106781e-01, - 1.48999170e-14,
                        0, 0.6, 0.02,
                        1, 0, 1, 0,
                ]

    def diagnostics(self, paths, logger=logger, prefix=""):
        super().log_diagnostics(paths)
        statistics = OrderedDict()
        for stat_name in [
            'hand_distance',
            'puck_distance',
            'sum_distance',
            'touch_distance',
            'success',
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s %s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'Final %s %s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
            ))
        return statistics

        for key, value in statistics.items():
            logger.record_tabular(key, value)

    """
    Multitask functions
    """

    @property
    def goal_dim(self) -> int:
        return 4

    def sample_goal_for_rollout(self):
        return self.sample_goal_xyxy()

    def set_goal(self, goal):
        MultitaskEnv.set_goal(self, goal)
        self.set_goal_xyxy(goal)

    def set_to_goal(self, goal):
        self.set_hand_xy(goal[:2])
        self.set_puck_xy(goal[-2:])

    def convert_obs_to_goals(self, obs):
        return obs

    def sample_goals(self, batch_size):
        raise NotImplementedError()

    def set_hand_xy(self, xy):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', np.array([xy[0], xy[1], 0.02]))
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            u = np.zeros(7)
            self.do_simulation(u, self.frame_skip)


class SawyerPushAndReachXYEasyEnv(SawyerPushAndReachXYEnv):
    """
    Always start the block in the same position
    """
    PUCK_GOAL_LOW = np.array([-0.2, 0.5])
    PUCK_GOAL_HIGH = np.array([0.2, 0.7])

    def sample_puck_xy(self):
        return np.array([0, 0.6])
