import numpy as np
from gym.spaces import Box

from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerSweepEnvV2(SawyerXYZEnv):

    def __init__(self):

        init_puck_z = 0.1
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1.0, 0.5)
        obj_low = (-0.1, 0.6, 0.02)
        obj_high = (0.1, 0.7, 0.02)
        goal_low = (.49, .6, 0.00)
        goal_high = (0.51, .7, 0.02)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos':np.array([0., 0.6, 0.02]),
            'obj_init_angle': 0.3,
            'hand_init_pos': np.array([0., .6, .2]),
        }
        self.goal = np.array([0.5, 0.65, 0.01])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.max_path_length = 200
        self.init_puck_z = init_puck_z

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_sweep_v2.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        reward, reachDist, pushDist = self.compute_reward(action, ob)
        self.curr_path_length += 1

        info = {
            'reachDist': reachDist,
            'goalDist': pushDist,
            'epRew': reward,
            'pickRew': None,
            'success': float(pushDist <= 0.05)
        }

        return ob, reward, False, info

    def _get_pos_objects(self):
        return self.get_body_com('obj')

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.objHeight = self.get_body_com('obj')[2]

        if self.random_init:
            obj_pos = self._get_state_rand_vec()
            self.obj_init_pos = np.concatenate((obj_pos[:2], [self.obj_init_pos[-1]]))
            self._target_pos[1] = obj_pos.copy()[1]
            
        self._set_obj_xyz(self.obj_init_pos)
        self.maxPushDist = np.linalg.norm(self.get_body_com('obj')[:-1] - self._target_pos[:-1])
        self.target_reward = 1000*self.maxPushDist + 1000*2

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand()

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.reachCompleted = False

    def compute_reward(self, actions, obs):
        del actions

        objPos = obs[3:6]

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        pushGoal = self._target_pos

        reachDist = np.linalg.norm(objPos - fingerCOM)
        pushDistxy = np.linalg.norm(objPos[:-1] - pushGoal[:-1])
        reachRew = -reachDist

        self.reachCompleted = reachDist < 0.05

        if objPos[-1] < self.obj_init_pos[-1] - 0.05:
            reachRew = 0
            pushDistxy = 0
            reachDist = 0

        def pushReward():
            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            if self.reachCompleted:
                pushRew = 1000*(self.maxPushDist - pushDistxy) + c1*(np.exp(-(pushDistxy**2)/c2) + np.exp(-(pushDistxy**2)/c3))
                pushRew = max(pushRew,0)
                return pushRew
            else:
                return 0

        pushRew = pushReward()
        reward = reachRew + pushRew

        return [reward, reachDist, pushDistxy]
