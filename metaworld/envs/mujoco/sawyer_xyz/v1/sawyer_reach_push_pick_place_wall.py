import numpy as np
from gym.spaces import Box

from metaworld.envs.asset_path_utils import full_v1_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerReachPushPickPlaceWallEnv(SawyerXYZEnv):

    def __init__(self):

        liftThresh = 0.04
        goal_low = (-0.05, 0.85, 0.05)
        goal_high = (0.05, 0.9, 0.3)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.05, 0.6, 0.015)
        obj_high = (0.05, 0.65, 0.015)

        self.task_types = ['pick_place', 'reach', 'push']

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.task_type = None
        self.init_config = {
            'obj_init_angle': .3,
            'obj_init_pos': np.array([0, 0.6, 0.02]),
            'hand_init_pos': np.array([0, .6, .2]),
        }

        self.obj_init_angle = self.init_config['obj_init_angle']
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.liftThresh = liftThresh

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.num_resets = 0

    def _set_task_inner(self, *, task_type, **kwargs):
        super()._set_task_inner(**kwargs)
        self.task_type = task_type
        # we only do one task from [pick_place, reach, push]
        # per instance of SawyerReachPushPickPlaceEnv.
        # Please only set task_type from constructor.
        if self.task_type == 'pick_place':
            self.goal = np.array([0.05, 0.8, 0.2])
        elif self.task_type == 'reach':
            self.goal = np.array([-0.05, 0.8, 0.2])
        elif self.task_type == 'push':
            self.goal = np.array([0.05, 0.8, 0.015])
        else:
            raise NotImplementedError

    @property
    def model_name(self):
        return full_v1_path_for('sawyer_xyz/sawyer_reach_push_pick_and_place_wall.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        reward, _, reachDist, _, pushDist, pickRew, _, placingDist = self.compute_reward(action, ob)
        goal_dist = placingDist if self.task_type == 'pick_place' else pushDist

        if self.task_type == 'reach':
            success = float(reachDist <= 0.05)
        else:
            success = float(goal_dist <= 0.07)

        info = {
            'reachDist': reachDist,
            'pickRew': pickRew,
            'epRew': reward,
            'goalDist': goal_dist,
            'success': success
        }

        return ob, reward, False, info

    @property
    def _target_site_config(self):
        far_away = np.array([10., 10., 10.])
        return [
            ('goal_' + t, self._target_pos if t == self.task_type else far_away)
            for t in self.task_types
        ]

    def _get_pos_objects(self):
        return self.data.get_geom_xpos('objGeom')

    def adjust_initObjPos(self, orig_init_pos):
        # This is to account for meshes for the geom and object are not aligned
        # If this is not done, the object could be initialized in an extreme position
        diff = self.get_body_com('obj')[:2] - self.data.get_geom_xpos('objGeom')[:2]
        adjustedPos = orig_init_pos[:2] + diff

        # The convention we follow is that body_com[2] is always 0, and geom_pos[2] is the object height
        return [adjustedPos[0], adjustedPos[1],self.data.get_geom_xpos('objGeom')[-1]]

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.adjust_initObjPos(self.init_config['obj_init_pos'])
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.objHeight = self.data.get_geom_xpos('objGeom')[2]
        self.heightTarget = self.objHeight + self.liftThresh

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            self._target_pos = goal_pos[3:]
            while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
                goal_pos = self._get_state_rand_vec()
                self._target_pos = goal_pos[3:]
            if self.task_type == 'push':
                self._target_pos = np.concatenate((goal_pos[-3:-1], [self.obj_init_pos[-1]]))
                self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))
            else:
                self._target_pos = goal_pos[-3:]
                self.obj_init_pos = goal_pos[:3]

        self._set_obj_xyz(self.obj_init_pos)
        self.maxReachDist = np.linalg.norm(self.init_fingerCOM - np.array(self._target_pos))
        self.maxPushDist = np.linalg.norm(self.obj_init_pos[:2] - np.array(self._target_pos)[:2])
        self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) - np.array(self._target_pos)) + self.heightTarget
        self.target_rewards = [1000*self.maxPlacingDist + 1000*2, 1000*self.maxReachDist + 1000*2, 1000*self.maxPushDist + 1000*2]

        if self.task_type == 'pick_place':
            idx = 0
        elif self.task_type == 'reach':
            idx = 1
        elif self.task_type == 'push':
            idx = 2

        self.target_reward = self.target_rewards[idx]
        self.num_resets += 1

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

        def compute_reward_reach(actions, obs):
            del actions
            del obs

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            reachDist = np.linalg.norm(fingerCOM - goal)
            reachRew = c1*(self.maxReachDist - reachDist) + c1*(np.exp(-(reachDist**2)/c2) + np.exp(-(reachDist**2)/c3))
            reachRew = max(reachRew, 0)
            reward = reachRew

            return [reward, reachRew, reachDist, None, None, None, None, None]

        def compute_reward_push(actions, obs):
            del actions
            del obs

            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            assert np.all(goal == self._get_site_pos('goal_push'))
            reachDist = np.linalg.norm(fingerCOM - objPos)
            pushDist = np.linalg.norm(objPos[:2] - goal[:2])
            reachRew = -reachDist

            if reachDist < 0.05:
                pushRew = 1000*(self.maxPushDist - pushDist) + c1*(np.exp(-(pushDist**2)/c2) + np.exp(-(pushDist**2)/c3))
                pushRew = max(pushRew, 0)
            else:
                pushRew = 0

            reward = reachRew + pushRew
            return [reward, reachRew, reachDist, pushRew, pushDist, None, None, None]

        def compute_reward_pick_place(actions, obs):
            del obs

            reachDist = np.linalg.norm(objPos - fingerCOM)
            placingDist = np.linalg.norm(objPos - goal)
            assert np.all(goal == self._get_site_pos('goal_pick_place'))

            def reachReward():
                reachRew = -reachDist
                reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
                zRew = np.linalg.norm(fingerCOM[-1] - self.init_fingerCOM[-1])

                if reachDistxy < 0.05:
                    reachRew = -reachDist
                else:
                    reachRew =  -reachDistxy - 2*zRew

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

            return [reward, reachRew, reachDist, None, None, pickRew, placeRew, placingDist]

        if self.task_type == 'reach':
            return compute_reward_reach(actions, obs)
        elif self.task_type == 'push':
            return compute_reward_push(actions, obs)
        elif self.task_type == 'pick_place':
            return compute_reward_pick_place(actions, obs)
        else:
            raise NotImplementedError
