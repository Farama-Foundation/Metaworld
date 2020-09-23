import numpy as np
from gym.spaces import Box

from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerHandInsertEnvV2(SawyerXYZEnv):
    def __init__(self):

        hand_low = (-0.5, 0.40, -0.15)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.6, 0.05)
        obj_high = (0.1, 0.7, 0.05)
        goal_low = (-0.04, 0.8, -0.0801)
        goal_high = (0.04, 0.88, -0.0799)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([0, 0.6, 0.05]),
            'obj_init_angle': 0.3,
            'hand_init_pos': np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0., 0.84, -0.08], dtype=np.float32)
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.max_path_length = 200

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_table_with_hole.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        reward, reachDist = self.compute_reward(action, ob)
        self.curr_path_length += 1

        info = {
            'reachDist': reachDist,
            'goalDist': None,
            'epRew': reward,
            'pickRew': None,
            'success': float(reachDist <= 0.05)
        }

        return ob, reward, False, info

    @property
    def _target_site_config(self):
        return [('goal', np.hstack(
            (*self._target_pos[:2], self.obj_init_pos[2])
        ))]

    def _get_pos_objects(self):
        return self.get_body_com('obj')

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.objHeight = self.get_body_com('obj')[2]

        # if self.random_init:
        goal_pos = self._get_state_rand_vec()
        while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.15:
            goal_pos = self._get_state_rand_vec()
        self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))
        self._target_pos = goal_pos[-3:]

        self._set_obj_xyz(self.obj_init_pos)
        self.maxReachDist = np.abs(self.hand_init_pos[-1] - self._target_pos[-1])

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand()
        self.pickCompleted = False

    def compute_reward(self, actions, obs):
        del actions
        del obs

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        goal = self._target_pos

        c1 = 1000
        c2 = 0.01
        c3 = 0.001
        reachDist = np.linalg.norm(fingerCOM[:-1] - goal[:-1])
        reachRew = -reachDist
        reachDist_z = np.abs(fingerCOM[-1] - goal[-1])

        if reachDist < 0.05:
            reachNearRew = 1000*(self.maxReachDist - reachDist_z) + c1*(np.exp(-(reachDist_z**2)/c2) + np.exp(-(reachDist_z**2)/c3))
        else:
            reachNearRew = 0.

        reachNearRew = max(reachNearRew,0)
        reward = reachRew + reachNearRew

        return [reward, reachDist]
