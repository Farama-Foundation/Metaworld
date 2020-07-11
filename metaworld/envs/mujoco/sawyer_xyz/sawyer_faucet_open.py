import numpy as np
from gym.spaces import Box

from metaworld.envs.env_util import get_asset_full_path
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv, _assert_task_is_set


class SawyerFaucetOpenEnv(SawyerXYZEnv):
    def __init__(self):

        hand_low = (-0.5, 0.40, -0.15)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.05, 0.8, 0.05)
        obj_high = (0.05, 0.85, 0.05)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([0, 0.8, 0.05]),
            'hand_init_pos': np.array([0., .6, .2]),
        }
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']
        self.goal = np.array([0.1, 0.8, 0.115])
        goal_low = self.hand_low
        goal_high = self.hand_high

        self.max_path_length = 150

        self.obj_and_goal_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.observation_space = Box(
            np.hstack((self.hand_low, obj_low, obj_low, goal_low)),
            np.hstack((self.hand_high, obj_high, obj_high, goal_high)),
        )

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_faucet.xml')

    @_assert_task_is_set
    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        reward, reachDist, pullDist = self.compute_reward(action, ob)
        self.curr_path_length +=1

        info = {'reachDist': reachDist, 'goalDist': pullDist, 'epRew' : reward, 'pickRew':None, 'success': float(pullDist <= 0.05)}
        info['goal'] = self.goal

        return ob, reward, False, info

    def _get_pos_objects(self):
        return self.get_site_pos('handleStartOpen')

    def _set_goal_marker(self, goal):
        self.data.site_xpos[self.model.site_name2id('goal_open')] = (
            goal[:3]
        )
        self.data.site_xpos[self.model.site_name2id('goal_close')] = (
            np.array([10.0, 10.0, 10.0])
        )

    def reset_model(self):
        self._reset_hand()
        self._state_goal = self.goal.copy()
        self.obj_init_pos = self.init_config['obj_init_pos']

        if self.random_init:
            goal_pos = self._get_state_rand_vec()

            self.obj_init_pos = goal_pos[:3]
            final_pos = goal_pos.copy()
            final_pos += np.array([0.1, -0.015, 0.065])
            self._state_goal = final_pos

        self.sim.model.body_pos[self.model.body_name2id('faucet')] = self.obj_init_pos
        self.sim.model.body_pos[self.model.body_name2id('faucetBase')] = self.obj_init_pos
        self._set_goal_marker(self._state_goal)
        self.maxPullDist = np.linalg.norm(self._state_goal - self.obj_init_pos)

        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.reachCompleted = False

    def compute_reward(self, actions, obs):
        del actions

        objPos = obs[3:6]

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        pullGoal = self._state_goal

        pullDist = np.linalg.norm(objPos - pullGoal)
        reachDist = np.linalg.norm(objPos - fingerCOM)
        reachRew = -reachDist

        self.reachCompleted = reachDist < 0.05

        def pullReward():
            c1 = 1000
            c2 = 0.01
            c3 = 0.001

            if self.reachCompleted:
                pullRew = 1000*(self.maxPullDist - pullDist) + c1*(np.exp(-(pullDist**2)/c2) + np.exp(-(pullDist**2)/c3))
                pullRew = max(pullRew,0)
                return pullRew
            else:
                return 0

        pullRew = pullReward()
        reward = reachRew + pullRew

        return [reward, reachDist, pullDist]
