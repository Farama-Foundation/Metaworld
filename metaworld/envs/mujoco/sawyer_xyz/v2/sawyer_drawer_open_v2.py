import numpy as np
from gym.spaces import Box

from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerDrawerOpenEnvV2(SawyerXYZEnv):
    def __init__(self):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.9, 0.0)
        obj_high = (0.1, 0.9, 0.0)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': np.array([0.3, ], dtype=np.float32),
            'obj_init_pos': np.array([0., 0.9, 0.0], dtype=np.float32),
            'hand_init_pos': np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        goal_low = self.hand_low
        goal_high = self.hand_high

        self.max_path_length = 150

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.maxDist = 0.2
        self.target_reward = 1000 * self.maxDist + 1000 * 2

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_drawer.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        reward, reachDist, pullDist = self.compute_reward(action, ob)
        self.curr_path_length += 1
        info = {
            'reachDist': reachDist,
            'goalDist': pullDist,
            'epRew': reward,
            'pickRew': None,
            'success': float(pullDist <= 0.03),
        }

        return ob, reward, False, info

    def _get_pos_objects(self):
        return self.get_body_com('drawer_link') + np.array([.0, -.16, .05])

    def reset_model(self):
        self._reset_hand()

        # Compute nightstand position
        self.obj_init_pos = self._get_state_rand_vec() if self.random_init \
            else self.init_config['obj_init_pos']
        # Set mujoco body to computed position
        self.sim.model.body_pos[self.model.body_name2id(
            'drawer'
        )] = self.obj_init_pos
        # Set _target_pos to current drawer position (closed) minus an offset
        self._target_pos = self.obj_init_pos + np.array([.0, -.16 - self.maxDist, .09])

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand()
        self.reachCompleted = False

    def compute_reward(self, actions, obs):
        del actions

        objPos = obs[3:6]
        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2
        pullGoal = self._target_pos
        pullDist = np.abs(objPos[1] - pullGoal[1])
        reachDist = np.linalg.norm(objPos - fingerCOM)
        reachRew = -reachDist

        self.reachCompleted = reachDist < 0.05

        def pullReward():
            c1 = 1000
            c2 = 0.01
            c3 = 0.001

            if self.reachCompleted:
                pullRew = 1000 * (self.maxDist - pullDist) + c1 * (np.exp(-(pullDist ** 2) / c2) + np.exp(-(pullDist ** 2) / c3))
                pullRew = max(pullRew,0)
                return pullRew
            else:
                return 0

            pullRew = max(pullRew,0)

            return pullRew


        pullRew = pullReward()
        reward = reachRew + pullRew

        return [reward, reachDist, pullDist]
