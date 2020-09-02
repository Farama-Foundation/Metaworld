import numpy as np
from gym.spaces import Box

from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set

class SawyerCoffeePullEnvV2(SawyerXYZEnv):

    def __init__(self):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.05, 0.7, -.001)
        obj_high = (0.05, 0.75, +.001)
        goal_low = (-0.1, 0.55, -.001)
        goal_high = (0.1, 0.65, +.001)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([0, 0.75, 0.]),
            'obj_init_angle': 0.3,
            'hand_init_pos': np.array([0., .4, .2]),
        }
        self.goal = np.array([0., 0.6, 0])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_coffee.xml')

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
            'success': float(pullDist <= 0.07)
        }

        return ob, reward, False, info

    @property
    def _target_site_config(self):
        return [('mug_goal', self._target_pos)]

    def _get_pos_objects(self):
        return self.get_body_com('obj')

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flatten()
        qvel = self.data.qvel.flatten()
        qpos[0:3] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()

        pos_mug_init = self.init_config['obj_init_pos']
        pos_mug_goal = self.goal

        if self.random_init:
            pos_mug_init, pos_mug_goal = np.split(self._get_state_rand_vec(), 2)
            while np.linalg.norm(pos_mug_init[:2] - pos_mug_goal[:2]) < 0.15:
                pos_mug_init, pos_mug_goal = np.split(
                    self._get_state_rand_vec(),
                    2
                )

        self._set_obj_xyz(pos_mug_init)
        self.obj_init_pos = pos_mug_init

        pos_machine = pos_mug_init + np.array([.0, .22, .0])
        self.sim.model.body_pos[self.model.body_name2id(
            'coffee_machine'
        )] = pos_machine

        self._target_pos = pos_mug_goal

        self.maxPullDist = np.linalg.norm(pos_mug_init[:2] - pos_mug_goal[:2])

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand()

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        self.init_fingerCOM = (rightFinger + leftFinger)/2
        self.reachCompleted = False

    def compute_reward(self, actions, obs):
        objPos = obs[3:6]

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        goal = self._target_pos

        c1 = 1000
        c2 = 0.01
        c3 = 0.001
        assert np.all(goal == self._get_site_pos('mug_goal'))
        reachDist = np.linalg.norm(fingerCOM - objPos)
        pullDist = np.linalg.norm(objPos[:2] - goal[:2])
        reachRew = -reachDist
        reachDistxy = np.linalg.norm(np.concatenate((objPos[:-1], [self.init_fingerCOM[-1]])) - fingerCOM)

        if reachDistxy < 0.05: #0.02
            reachRew = -reachDist + 0.1
            if reachDist < 0.05:
                reachRew += max(actions[-1],0)/50
        else:
            reachRew =  -reachDistxy

        if reachDist < 0.05:
            pullRew = 1000*(self.maxPullDist - pullDist) + c1*(np.exp(-(pullDist**2)/c2) + np.exp(-(pullDist**2)/c3))
            pullRew = max(pullRew, 0)
        else:
            pullRew = 0

        reward = reachRew + pullRew

        return [reward, reachDist, pullDist]
