import numpy as np
from gym.spaces import  Box

from metaworld.envs.asset_path_utils import full_v1_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerWindowCloseEnv(SawyerXYZEnv):

    def __init__(self):

        liftThresh = 0.02
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (0., 0.75, 0.15)
        obj_high = (0., 0.9, 0.15)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': 0.3,
            'obj_init_pos': np.array([0.1, 0.785, 0.15], dtype=np.float32),
            'hand_init_pos': np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.goal = np.array([-0.08, 0.785, 0.15])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        goal_low = self.hand_low
        goal_high = self.hand_high

        self.liftThresh = liftThresh

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v1_path_for('sawyer_xyz/sawyer_window_horizontal.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        reward, reachDist, pickrew, pullDist = self.compute_reward(action, ob)

        info = {
            'reachDist': reachDist,
            'goalDist': pullDist,
            'epRew': reward,
            'pickRew': pickrew,
            'success': float(pullDist <= 0.05)
        }

        return ob, reward, False, info

    def _get_pos_objects(self):
        return self._get_site_pos('handleCloseStart')

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.objHeight = self.data.get_geom_xpos('handle')[2]
        self.heightTarget = self.objHeight + self.liftThresh

        if self.random_init:
            obj_pos = self._get_state_rand_vec()
            self.obj_init_pos = obj_pos
            goal_pos = obj_pos.copy()
            goal_pos[0] -= 0.18
            self._target_pos = goal_pos

        wall_pos = self.obj_init_pos.copy() - np.array([0.1, 0, 0.12])
        window_another_pos = self.obj_init_pos.copy() + np.array([0, 0.03, 0])
        self.sim.model.body_pos[self.model.body_name2id('window')] = self.obj_init_pos
        self.sim.model.body_pos[self.model.body_name2id('window_another')] = window_another_pos
        self.sim.model.body_pos[self.model.body_name2id('wall')] = wall_pos
        self.sim.model.site_pos[self.model.site_name2id('goal')] = self._target_pos
        self.maxPullDist = 0.2
        self.target_reward = 1000*self.maxPullDist + 1000*2

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand(10)

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.reachCompleted = False

    def compute_reward(self, actions, obs):
        del actions

        objPos = obs[3:6]

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        pullGoal = self._target_pos

        pullDist = np.abs(objPos[0] - pullGoal[0])
        reachDist = np.linalg.norm(objPos - fingerCOM)

        self.reachCompleted = reachDist < 0.05

        c1 = 1000
        c2 = 0.01
        c3 = 0.001
        reachRew = -reachDist

        if self.reachCompleted:
            pullRew = 1000*(self.maxPullDist - pullDist) + c1*(np.exp(-(pullDist**2)/c2) + np.exp(-(pullDist**2)/c3))
        else:
            pullRew = 0
        reward = reachRew + pullRew

        return [reward, reachDist, None, pullDist]
