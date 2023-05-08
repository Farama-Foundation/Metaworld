import numpy as np
from gym.spaces import Box

from metaworld.envs.asset_path_utils import full_v1_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerPickOutOfHoleEnv(SawyerXYZEnv):

    def __init__(self):

        liftThresh = 0.11
        hand_low = (-0.5, 0.40, -0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (0, 0.84, -0.03)
        obj_high = (0, 0.84, -0.03)
        goal_low = (-0.1, 0.6, 0.15)
        goal_high = (0.1, 0.7, 0.3)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([0, 0.84, -0.03]),
            'obj_init_angle': 0.3,
            'hand_init_pos': np.array([0., .6, .2]),
        }
        self.goal = np.array([0., 0.6, 0.2])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.liftThresh = liftThresh

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v1_path_for('sawyer_xyz/sawyer_pick_out_of_hole.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        reward, reachDist, pickRew, placingDist = self.compute_reward(action, ob)

        info = {
            'reachDist': reachDist,
            'goalDist': placingDist,
            'epRew': reward,
            'pickRew': pickRew,
            'success': float(placingDist <= 0.08)
        }

        return ob, reward, False, info

    def _get_pos_objects(self):
        return self.data.get_geom_xpos('objGeom')

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            self._target_pos = goal_pos[-3:]
            while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
                goal_pos = self._get_state_rand_vec()
                self._target_pos = goal_pos[-3:]
            self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))

        self._set_obj_xyz(self.obj_init_pos)
        self.objHeight = self.data.get_geom_xpos('objGeom')[2]
        self.heightTarget = self.objHeight + self.liftThresh
        self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) - np.array(self._target_pos)) + self.heightTarget

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand(10)

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False

    def compute_reward(self, actions, obs):

        objPos = obs[3:6]

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        heightTarget = self.heightTarget
        goal = self._target_pos

        reachDist = np.linalg.norm(objPos - fingerCOM)
        placingDist = np.linalg.norm(objPos - goal)
        assert np.all(goal == self._get_site_pos('goal'))

        def reachReward():
            reachRew = -reachDist
            reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
            zRew = np.linalg.norm(fingerCOM[-1] - self.init_fingerCOM[-1])
            if reachDistxy < 0.05:
                reachRew = -reachDist
            else:
                reachRew =  -reachDistxy - 2*zRew
            # incentive to close fingers when reachDist is small
            if reachDist < 0.05:
                reachRew = -reachDist + max(actions[-1],0)/50

            return reachRew , reachDist

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
                return hScale*(heightTarget - self.objHeight + 0.02)
            elif (reachDist < 0.1) and (objPos[2]> (self.objHeight + 0.005)) :
                return hScale* (min(heightTarget, objPos[2]) - self.objHeight + 0.02)
            else:
                return 0

        def placeReward():
            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            cond = self.pickCompleted and (reachDist < 0.1) and not(objDropped())
            if cond:
                placeRew = 1000*(self.maxPlacingDist - placingDist) + c1*(np.exp(-(placingDist**2)/c2) + np.exp(-(placingDist**2)/c3))
                placeRew = max(placeRew,0)
                return [placeRew , placingDist]
            else:
                return [0 , placingDist]

        reachRew, reachDist = reachReward()
        pickRew = orig_pickReward()
        placeRew , placingDist = placeReward()
        assert ((placeRew >=0) and (pickRew>=0))
        reward = reachRew + pickRew + placeRew

        return [reward, reachDist, pickRew, placingDist]
