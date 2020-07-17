import numpy as np
from gym.spaces import Box

from metaworld.envs.env_util import get_asset_full_path
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv, _assert_task_is_set


class SawyerNutDisassembleEnv(SawyerXYZEnv):
    def __init__(self):

        liftThresh = 0.05
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (0.1, 0.75, 0.02)
        obj_high = (0., 0.85, 0.02)
        goal_low = (-0.1, 0.75, 0.17)
        goal_high = (0.1, 0.85, 0.17)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': 0.3,
            'obj_init_pos': np.array([0, 0.8, 0.02]),
            'hand_init_pos': np.array((0, 0.6, 0.2), dtype=np.float32),
        }
        self.goal = np.array([0, 0.8, 0.17])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.liftThresh = liftThresh
        self.max_path_length = 200

        self.obj_and_goal_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.observation_space = Box(
            np.hstack((self.hand_low, obj_low, obj_low, goal_low)),
            np.hstack((self.hand_high, obj_high, obj_high, goal_high)),
        )

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_assembly_peg.xml')

    @_assert_task_is_set
    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        obs_dict = self._get_obs_dict()
        reward, _, reachDist, pickRew, _, placingDist, success = self.compute_reward(action, obs_dict)
        self.curr_path_length += 1
        info = {'reachDist': reachDist, 'pickRew':pickRew, 'epRew' : reward, 'goalDist': placingDist, 'success': success}
        info['goal'] = self.goal

        return ob, reward, False, info

    def _get_pos_objects(self):
        return self.data.get_geom_xpos('RoundNut-8')

    def _get_obs_dict(self):
        obs_dict = super()._get_obs_dict()
        obs_dict['state_achieved_goal'] = self.get_body_com('RoundNut')
        return obs_dict

    def _set_goal_marker(self, goal):
        self.data.site_xpos[self.model.site_name2id('pegTop')] = (
            goal[:3]
        )

    def reset_model(self):
        self._reset_hand()
        self._state_goal = self.goal.copy()
        self.obj_init_pos = np.array(self.init_config['obj_init_pos'])
        self.obj_init_angle = self.init_config['obj_init_angle']

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.1:
                goal_pos = self._get_state_rand_vec()
            self.obj_init_pos = goal_pos[:3]
            self._state_goal = goal_pos[:3] + np.array([0, 0, 0.15])

        peg_pos = self.obj_init_pos + np.array([0., 0., 0.03])
        peg_top_pos = self.obj_init_pos + np.array([0., 0., 0.08])
        self.sim.model.body_pos[self.model.body_name2id('peg')] = peg_pos
        self.sim.model.site_pos[self.model.site_name2id('pegTop')] = peg_top_pos
        self._set_obj_xyz(self.obj_init_pos)
        self._set_goal_marker(self._state_goal)
        self.objHeight = self.data.get_geom_xpos('RoundNut-8')[2]
        self.heightTarget = self.objHeight + self.liftThresh
        self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) - np.array(self._state_goal)) + self.heightTarget

        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False

    def compute_reward(self, actions, obs):
        obs = obs['state_observation']

        graspPos = obs[3:6]
        objPos = graspPos

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        heightTarget = self.heightTarget
        placingGoal = self._state_goal

        reachDist = np.linalg.norm(graspPos - fingerCOM)
        reachDistxy = np.linalg.norm(graspPos[:-1] - fingerCOM[:-1])
        zDist = np.abs(fingerCOM[-1] - self.init_fingerCOM[-1])

        placingDist = np.linalg.norm(objPos - placingGoal)

        def reachReward():
            reachRew = -reachDist
            if reachDistxy < 0.04:
                reachRew = -reachDist
            else:
                reachRew =  -reachDistxy - 2*zDist

            # incentive to close fingers when reachDist is small
            if reachDist < 0.04:
                reachRew = -reachDist + max(actions[-1],0)/50
            return reachRew, reachDist

        def pickCompletionCriteria():
            tolerance = 0.01
            if objPos[2] >= (heightTarget- tolerance) and reachDist < 0.04:
                return True
            else:
                return False

        if pickCompletionCriteria():
            self.pickCompleted = True

        def objDropped():
            return (objPos[2] < (self.objHeight + 0.005)) and (placingDist >0.02) and (reachDist > 0.02)

        def orig_pickReward():
            hScale = 100
            if self.pickCompleted and not(objDropped()):
                return hScale*heightTarget
            elif (reachDist < 0.04) and (objPos[2]> (self.objHeight + 0.005)) :
                return hScale* min(heightTarget, objPos[2])
            else:
                return 0

        def placeRewardMove():
            c1 = 1000
            c2 = 0.01
            c3 = 0.001

            placeRew = 1000*(self.maxPlacingDist - placingDist) + c1*(np.exp(-(placingDist**2)/c2) + np.exp(-(placingDist**2)/c3))
            placeRew = max(placeRew,0)
            cond = self.pickCompleted and (reachDist < 0.03) and not(objDropped())
            if cond:
                return [placeRew, placingDist]
            else:
                return [0 , placingDist]


        reachRew, reachDist = reachReward()
        pickRew = orig_pickReward()

        peg_pos = self.sim.model.body_pos[self.model.body_name2id('peg')]
        nut_pos = self.get_body_com('RoundNut')
        if abs(nut_pos[0] - peg_pos[0]) > 0.05 or \
                abs(nut_pos[1] - peg_pos[1]) > 0.05:
            placingDist = 0
            reachRew = 0
            reachDist = 0
            pickRew = heightTarget*100

        placeRew , placingDist = placeRewardMove()
        assert ((placeRew >=0) and (pickRew>=0))
        reward = reachRew + pickRew + placeRew
        success = (abs(nut_pos[0] - peg_pos[0]) > 0.05 or abs(nut_pos[1] - peg_pos[1]) > 0.05) or placingDist < 0.02

        return [reward, reachRew, reachDist, pickRew, placeRew, placingDist, float(success)]
