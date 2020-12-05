import numpy as np
from gym.spaces import Box

from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerBinPickingEnvV2(SawyerXYZEnv):
    """
    Motivation for V2:
        V1 was often unsolvable because the cube could be located outside of
        the starting bin. It could even be near the base of the Sawyer and out
        of reach of the gripper. V2 changes the `obj_low` and `obj_high` bounds
        to fix this.
    Changelog from V1 to V2:
        - (7/20/20) Changed object initialization space
        - (7/24/20) Added Byron's XML changes
    """
    def __init__(self):

        liftThresh = 0.1
        hand_low = (-0.5, 0.40, 0.07)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.21, 0.65, 0.02)
        obj_high = (-0.03, 0.75, 0.02)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': 0.3,
            'obj_init_pos': np.array([-0.12, 0.7, 0.02]),
            'hand_init_pos': np.array((0, 0.6, 0.2)),
        }
        self.goal = np.array([0.12, 0.7, 0.02])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        # Small bounds around the center of the target bin
        goal_low = np.array([0.1199, 0.699, -0.001])
        goal_high = np.array([0.1201, 0.701, +0.001])

        self.liftThresh = liftThresh
        self.max_path_length = 200

        self.hand_and_obj_space = Box(
            np.hstack((self.hand_low, obj_low)),
            np.hstack((self.hand_high, obj_high)),
        )

        self.goal_and_obj_space = Box(
            np.hstack((goal_low[:2], obj_low[:2])),
            np.hstack((goal_high[:2], obj_high[:2])),
        )

        self.goal_space = Box(goal_low, goal_high)
        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_bin_picking.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        reward, _, reachDist, pickRew, _, placingDist = self.compute_reward(action, ob)
        self.curr_path_length += 1

        info = {
            'reachDist': reachDist,
            'pickRew': pickRew,
            'epRew': reward,
            'goalDist': placingDist,
            'success': float(placingDist <= 0.08)
        }

        return ob, reward, False, info

    @property
    def _target_site_config(self):
        return []

    def _get_pos_objects(self):
        return self.get_body_com('obj')

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.objHeight = self.get_body_com('obj')[2]
        self.heightTarget = self.objHeight + self.liftThresh

        if self.random_init:
            self.obj_init_pos = self._get_state_rand_vec()[:2]
            self.obj_init_pos = np.concatenate((self.obj_init_pos, [self.objHeight]))

        self._set_obj_xyz(self.obj_init_pos)
        self._target_pos = self.get_body_com('bin_goal')
        self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1]]) - np.array(self._target_pos)[:-1]) + self.heightTarget

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand()
        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False
        self.placeCompleted = False

    def compute_reward(self, actions, obs):
        objPos = obs[3:6]

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        heightTarget = self.heightTarget
        placingGoal = self._target_pos

        reachDist = np.linalg.norm(objPos - fingerCOM)

        placingDist = np.linalg.norm(objPos[:2] - placingGoal[:-1])


        def reachReward():
            reachRew = -reachDist
            reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
            zRew = np.linalg.norm(fingerCOM[-1] - self.init_fingerCOM[-1])
            if reachDistxy < 0.06:
                reachRew = -reachDist
            else:
                reachRew =  -reachDistxy - zRew

            # incentive to close fingers when reachDist is small
            if reachDist < 0.05:
                reachRew = -reachDist + max(actions[-1],0)/50
            return reachRew , reachDist

        def pickCompletionCriteria():
            tolerance = 0.01
            if objPos[2] >= (heightTarget- tolerance):
                return True
            else:
                return False

        if pickCompletionCriteria():
            self.pickCompleted = True


        def objDropped():
            return (objPos[2] < (self.objHeight + 0.005)) and (placingDist >0.02) and (reachDist > 0.02)
            # Object on the ground, far away from the goal, and from the gripper
            # Can tweak the margin limits

        def placeCompletionCriteria():
            if abs(objPos[0] - placingGoal[0]) < 0.05 and \
                abs(objPos[1] - placingGoal[1]) < 0.05 and \
                objPos[2] < self.objHeight + 0.05:
                return True
            else:
                return False

        if placeCompletionCriteria():
            self.placeCompleted = True

        def orig_pickReward():
            hScale = 100
            if self.placeCompleted or (self.pickCompleted and not(objDropped())):
                return hScale*heightTarget
            elif (reachDist < 0.1) and (objPos[2]> (self.objHeight + 0.005)) :
                return hScale* min(heightTarget, objPos[2])
            else:
                return 0

        def placeReward():
            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            placeRew = 1000*(self.maxPlacingDist - placingDist) + c1*(np.exp(-(placingDist**2)/c2) + np.exp(-(placingDist**2)/c3))
            placeRew = max(placeRew,0)
            cond = self.pickCompleted and (reachDist < 0.1) and not(objDropped())

            if self.placeCompleted:
                return [-200*actions[-1] + placeRew, placingDist]
            elif cond:
                if abs(objPos[0] - placingGoal[0]) < 0.05 and \
                    abs(objPos[1] - placingGoal[1]) < 0.05:
                    return [-200*actions[-1] + placeRew, placingDist]
                else:
                    return [placeRew, placingDist]
            else:
                return [0 , placingDist]


        reachRew, reachDist = reachReward()
        pickRew = orig_pickReward()
        placeRew , placingDist = placeReward()

        if self.placeCompleted:
            reachRew = 0
            reachDist = 0
        reward = reachRew + pickRew + placeRew

        return [reward, reachRew, reachDist, pickRew, placeRew, placingDist]
