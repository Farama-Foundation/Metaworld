import numpy as np
from gym.spaces import Box

from metaworld.envs.asset_path_utils import full_v1_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerNutAssemblyEnv(SawyerXYZEnv):

    def __init__(self):

        liftThresh = 0.1
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (0, 0.6, 0.02)
        obj_high = (0, 0.6, 0.02)
        goal_low = (-0.1, 0.75, 0.1)
        goal_high = (0.1, 0.85, 0.1)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': 0.3,
            'obj_init_pos': np.array([0, 0.6, 0.02], dtype=np.float32),
            'hand_init_pos': np.array((0, 0.6, 0.2), dtype=np.float32),
        }
        self.goal = np.array([0.1, 0.8, 0.1], dtype=np.float32)
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.liftThresh = liftThresh

        goal_low = np.array(goal_low)
        goal_high = np.array(goal_high)
        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v1_path_for('sawyer_xyz/sawyer_assembly_peg.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        reward, _, reachDist, pickRew, _, placingDist, _, success = self.compute_reward(action, ob)
        info = {
            'reachDist': reachDist,
            'pickRew': pickRew,
            'epRew': reward,
            'goalDist': placingDist,
            'success': float(success)
        }

        return ob, reward, False, info

    @property
    def _target_site_config(self):
        return [('pegTop', self._target_pos)]

    def _get_pos_objects(self):
        return self.data.get_geom_xpos('RoundNut-8')

    def _get_obs_dict(self):
        obs_dict = super()._get_obs_dict()
        obs_dict['state_achieved_goal'] = self.get_body_com('RoundNut')
        return obs_dict

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.objHeight = self.data.get_geom_xpos('RoundNut-8')[2]
        self.heightTarget = self.objHeight + self.liftThresh

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.1:
                goal_pos = self._get_state_rand_vec()
            self.obj_init_pos = goal_pos[:3]
            self._target_pos = goal_pos[-3:]

        peg_pos = self._target_pos - np.array([0., 0., 0.05])
        self._set_obj_xyz(self.obj_init_pos)
        self.sim.model.body_pos[self.model.body_name2id('peg')] = peg_pos
        self.sim.model.site_pos[self.model.site_name2id('pegTop')] = self._target_pos
        self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) - np.array(self._target_pos)) + self.heightTarget

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand(10)

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False
        self.placeCompleted = False

    def compute_reward(self, actions, obs):
        graspPos = obs[3:6]
        objPos = self.get_body_com('RoundNut')

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        heightTarget = self.heightTarget
        placingGoal = self._target_pos

        reachDist = np.linalg.norm(graspPos - fingerCOM)

        placingDist = np.linalg.norm(objPos[:2] - placingGoal[:2])
        placingDistFinal = np.abs(objPos[-1] - self.objHeight)

        def reachReward():
            reachRew = -reachDist
            reachDistxy = np.linalg.norm(graspPos[:-1] - fingerCOM[:-1])
            zRew = np.linalg.norm(fingerCOM[-1] - self.init_fingerCOM[-1])
            if reachDistxy < 0.04:
                reachRew = -reachDist
            else:
                reachRew =  -reachDistxy - zRew

            # incentive to close fingers when reachDist is small
            if reachDist < 0.04:
                reachRew = -reachDist + max(actions[-1],0)/50
            return reachRew, reachDist

        def pickCompletionCriteria():
            tolerance = 0.01
            if objPos[2] >= (heightTarget - tolerance) and reachDist < 0.03:
                return True
            else:
                return False

        if pickCompletionCriteria():
            self.pickCompleted = True

        def objDropped():
            return (objPos[2] < (self.objHeight + 0.005)) and (placingDist >0.02) and (reachDist > 0.02)

        def placeCompletionCriteria():
            if abs(objPos[0] - placingGoal[0]) < 0.03 and \
                abs(objPos[1] - placingGoal[1]) < 0.03:
                return True
            else:
                return False

        if placeCompletionCriteria():
            self.placeCompleted = True
        else:
            self.placeCompleted = False

        def orig_pickReward():
            hScale = 100
            if self.placeCompleted or (self.pickCompleted and not(objDropped())):
                return hScale*heightTarget
            elif (reachDist < 0.04) and (objPos[2]> (self.objHeight + 0.005)) :
                return hScale* min(heightTarget, objPos[2])
            else:
                return 0

        def placeRewardMove():
            c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
            placeRew = 1000*(self.maxPlacingDist - placingDist) + c1*(np.exp(-(placingDist**2)/c2) + np.exp(-(placingDist**2)/c3))
            if self.placeCompleted:
                c4 = 2000; c5 = 0.003; c6 = 0.0003
                placeRew += 2000*(heightTarget - placingDistFinal) + c4*(np.exp(-(placingDistFinal**2)/c5) + np.exp(-(placingDistFinal**2)/c6))
            placeRew = max(placeRew,0)
            cond = self.placeCompleted or (self.pickCompleted and (reachDist < 0.04) and not(objDropped()))
            if cond:
                return [placeRew, placingDist, placingDistFinal]
            else:
                return [0, placingDist, placingDistFinal]

        reachRew, reachDist = reachReward()
        pickRew = orig_pickReward()
        placeRew , placingDist, placingDistFinal = placeRewardMove()
        assert ((placeRew >=0) and (pickRew>=0))
        reward = reachRew + pickRew + placeRew
        success = (abs(objPos[0] - placingGoal[0]) < 0.03 and abs(objPos[1] - placingGoal[1]) < 0.03 and placingDistFinal <= 0.04)
        return [reward, reachRew, reachDist, pickRew, placeRew, placingDist, placingDistFinal, success]

