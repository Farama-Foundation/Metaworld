import numpy as np
from gym.spaces import Box

from metaworld.envs.asset_path_utils import full_v1_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerPegInsertionSideEnv(SawyerXYZEnv):

    def __init__(self):

        liftThresh = 0.11
        hand_init_pos = (0, 0.6, 0.2)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.5, 0.02)
        obj_high = (0.1, 0.7, 0.02)
        goal_low = (-0.35, 0.5, 0.05)
        goal_high = (-0.25, 0.8, 0.05)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([0, 0.6, 0.02]),
            'hand_init_pos': np.array([0, .6, .2]),
        }
        self.goal = np.array([-0.3, 0.6, 0.05])

        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        goal_low = self.hand_low
        goal_high = self.hand_high

        self.liftThresh = liftThresh

        self.hand_init_pos = np.array(hand_init_pos)

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v1_path_for('sawyer_xyz/sawyer_peg_insertion_side.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        reward, _, reachDist, pickRew, _, placingDist = self.compute_reward(action, ob)

        info = {
            'reachDist': reachDist,
            'pickRew': pickRew,
            'epRew': reward,
            'goalDist': placingDist,
            'success': float(placingDist <= 0.07)
        }

        return ob, reward, False, info

    def _get_pos_objects(self):
        return self.get_body_com('peg')



    def reset_model(self):
        self._reset_hand()

        self.sim.model.body_pos[self.model.body_name2id('box')] = np.array([-0.3, 0.6, 0.05])
        self._target_pos = self.sim.model.site_pos[self.model.site_name2id('hole')] + self.sim.model.body_pos[self.model.body_name2id('box')]
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.objHeight = self.get_body_com('peg').copy()[2]
        self.heightTarget = self.objHeight + self.liftThresh

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.1:
                goal_pos = self._get_state_rand_vec()
            self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))
            self.sim.model.body_pos[self.model.body_name2id('box')] = goal_pos[-3:]
            self._target_pos = self.sim.model.site_pos[self.model.site_name2id('hole')] + self.sim.model.body_pos[self.model.body_name2id('box')]

        self._set_obj_xyz(self.obj_init_pos)
        self.obj_init_pos = self.get_body_com('peg')
        self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) - np.array(self._target_pos)) + self.heightTarget
        self.target_reward = 1000*self.maxPlacingDist + 1000*2
        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand(10)

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False

    def compute_reward(self, actions, obs):

        objPos = obs[3:6]
        pegHeadPos = self._get_site_pos('pegHead')

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        heightTarget = self.heightTarget
        placingGoal = self._target_pos

        reachDist = np.linalg.norm(objPos - fingerCOM)

        placingDistHead = np.linalg.norm(pegHeadPos - placingGoal)
        placingDist = np.linalg.norm(objPos - placingGoal)


        def reachReward():
            reachRew = -reachDist
            reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
            zRew = np.linalg.norm(fingerCOM[-1] - self.init_fingerCOM[-1])

            if reachDistxy < 0.05:
                reachRew = -reachDist
            else:
                reachRew =  -reachDistxy - zRew

            # incentive to close fingers when reachDist is small
            if reachDist < 0.05:
                reachRew = -reachDist + max(actions[-1],0)/50

            return reachRew, reachDist

        def pickCompletionCriteria():
            tolerance = 0.01

            return objPos[2] >= (heightTarget- tolerance)

        self.pickCompleted = pickCompletionCriteria()

        def objDropped():
            return (objPos[2] < (self.objHeight + 0.005)) and (placingDist >0.02) and (reachDist > 0.02)
            # Object on the ground, far away from the goal, and from the gripper
            # Can tweak the margin limits

        def orig_pickReward():
            hScale = 100
            if self.pickCompleted and not(objDropped()):
                return hScale*heightTarget
            elif (reachDist < 0.1) and (objPos[2]> (self.objHeight + 0.005)) :
                return hScale* min(heightTarget, objPos[2])
            else:
                return 0

        def placeReward():
            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            cond = self.pickCompleted and (reachDist < 0.1) and not(objDropped())

            if cond:
                if placingDistHead <= 0.05:
                    placeRew = 1000*(self.maxPlacingDist - placingDist) + c1*(np.exp(-(placingDist**2)/c2) + np.exp(-(placingDist**2)/c3))
                else:
                    placeRew = 1000*(self.maxPlacingDist - placingDistHead) + c1*(np.exp(-(placingDistHead**2)/c2) + np.exp(-(placingDistHead**2)/c3))
                placeRew = max(placeRew,0)
                return [placeRew , placingDist]
            else:
                return [0 , placingDist]

        reachRew, reachDist = reachReward()
        pickRew = orig_pickReward()
        placeRew, placingDist = placeReward()
        assert ((placeRew >=0) and (pickRew>=0))
        reward = reachRew + pickRew + placeRew

        return [reward, reachRew, reachDist, pickRew, placeRew, placingDist]
