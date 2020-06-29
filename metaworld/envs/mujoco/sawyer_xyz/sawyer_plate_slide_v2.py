import numpy as np
from gym.spaces import Box

from metaworld.envs.env_util import get_asset_full_path
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv


class SawyerPlateSlideEnvV2(SawyerXYZEnv):
    """
    Motivation for V2:
        V1 was very difficult to solve because the observation didn't say where
        the cabinet was along the X axis
    Changelog from V1 to V2:
        - (6/22/20) Added a 1 element vector to the observation. This vector
            points from the end effector to the cabinet in the X direction.
            i.e. (self._state_goal - pos_hand)[0]
    """
    def __init__(self, random_init=False):

        goal_low = (-0.1, 0.85, 0.02)
        goal_high = (0.1, 0.9, 0.02)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (0., 0.6, 0.015)
        obj_high = (0., 0.6, 0.015)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.random_init = random_init

        self.init_config = {
            'obj_init_angle': 0.3,
            'obj_init_pos': np.array([0., 0.6, 0.015], dtype=np.float32),
            'hand_init_pos': np.array((0, 0.6, 0.2), dtype=np.float32),
        }
        self.goal = np.array([0., 0.85, 0.02])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.max_path_length = 150

        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
        )

        self.obj_and_goal_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        hand_to_goal_max_x = self.hand_high[0] - np.array(goal_low)[0]
        self.observation_space = Box(
            np.hstack((self.hand_low, obj_low, -hand_to_goal_max_x)),
            np.hstack((self.hand_high, obj_high, hand_to_goal_max_x)),
        )

        self.reset()

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_plate_slide.xml')

    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation
        ob = self._get_obs()
        obs_dict = self._get_obs_dict()
        reward, reachDist, pullDist = self.compute_reward(action, obs_dict)
        self.curr_path_length += 1

        info = {
            'reachDist': reachDist,
            'goalDist': pullDist,
            'epRew': reward,
            'pickRew': None,
            'success': float(pullDist <= 0.08),
            'goal': self.goal
        }

        return ob, reward, False, info

    def _get_obs(self):
        pos_hand = self.get_endeff_pos()
        pos_obj = self.data.get_geom_xpos('objGeom')
        pos_cabinet = (self._state_goal - pos_hand)[0]

        flat_obs = np.hstack((pos_hand, pos_obj, pos_cabinet))
        return np.concatenate([flat_obs, ])

    def _get_obs_dict(self):
        return dict(
            state_observation=self._get_obs(),
            state_desired_goal=self._state_goal,
            state_achieved_goal=self.data.get_geom_xpos('objGeom'),
        )

    def _set_goal_marker(self, goal):
        self.data.site_xpos[self.model.site_name2id('goal')] = (
            goal[:3]
        )

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:11] = pos
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()
        self._state_goal = self.goal.copy()
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.objHeight = self.data.get_geom_xpos('objGeom')[2]

        if self.random_init:
            obj_pos = np.random.uniform(
                self.obj_and_goal_space.low,
                self.obj_and_goal_space.high,
                size=(self.obj_and_goal_space.low.size),
            )
            self.obj_init_pos = obj_pos[:3]
            goal_pos = obj_pos[3:]
            self._state_goal = goal_pos

        self.sim.model.body_pos[
            self.model.body_name2id('cabinet')] = self._state_goal
        self._set_obj_xyz(np.zeros(2))
        self.maxDist = np.linalg.norm(
            self.obj_init_pos[:-1] - self._state_goal[:-1])
        self.target_reward = 1000 * self.maxDist + 1000 * 2

        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1, 1], self.frame_skip)

        rightFinger, leftFinger = self.get_site_pos(
            'rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM = (rightFinger + leftFinger) / 2

    def compute_reward(self, actions, obs):
        del actions

        obs = obs['state_observation']

        objPos = obs[3:6]

        rightFinger, leftFinger = self.get_site_pos(
            'rightEndEffector'), self.get_site_pos('leftEndEffector')
        fingerCOM = (rightFinger + leftFinger) / 2

        pullGoal = self._state_goal

        reachDist = np.linalg.norm(objPos - fingerCOM)

        pullDist = np.linalg.norm(objPos[:-1] - pullGoal[:-1])

        c1 = 1000
        c2 = 0.01
        c3 = 0.001
        if reachDist < 0.05:
            pullRew = 1000 * (self.maxDist - pullDist) + c1 * (
                        np.exp(-(pullDist ** 2) / c2) + np.exp(
                    -(pullDist ** 2) / c3))
            pullRew = max(pullRew, 0)
        else:
            pullRew = 0
        reward = -reachDist + pullRew

        return [reward, reachDist, pullDist]
