import numpy as np
from gym.spaces import Box

from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerPickPlaceWallEnvV2(SawyerXYZEnv):
    """
    Motivation for V2:
        V1 was difficult to solve because the observation didn't say where
        to move after picking up the puck.
    Changelog from V1 to V2:
        - (7/7/20) Removed 3 element vector. Replaced with 3 element position
            of the goal (for consistency with other environments)
        - (6/24/20) Added a 3 element vector to the observation. This vector
            points from the end effector to the goal coordinate.
            i.e. (self._target_pos - pos_hand)
        - (6/24/20) Separated pick-place-wall into from
          reach-push-pick-place-wall.
    """
    def __init__(self):

        liftThresh = 0.04
        goal_low = (-0.05, 0.85, 0.05)
        goal_high = (0.05, 0.9, 0.3)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.05, 0.6, 0.015)
        obj_high = (0.05, 0.65, 0.015)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': .3,
            'obj_init_pos': np.array([0, 0.6, 0.02]),
            'hand_init_pos': np.array([0, .6, .2]),
        }

        self.goal = np.array([0.05, 0.8, 0.2])

        self.obj_init_angle = self.init_config['obj_init_angle']
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.liftThresh = liftThresh
        self.max_path_length = 200

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.num_resets = 0

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_pick_place_wall_v2.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)

        reward, reach_dist, pick_rew, placing_dist = self.compute_reward(action, ob)
        success = float(placing_dist <= 0.07)

        info = {
            'reachDist': reach_dist,
            'pickRew': pick_rew,
            'epRew': reward,
            'goalDist': placing_dist,
            'success': success
        }

        self.curr_path_length +=1
        return ob, reward, False, info

    def _get_pos_objects(self):
        return self.data.get_geom_xpos('objGeom')

    def adjust_initObjPos(self, orig_init_pos):
        # This is to account for meshes for the geom and object are not aligned
        # If this is not done, the object could be initialized in an extreme position
        diff = self.get_body_com('obj')[:2] - \
               self.data.get_geom_xpos('objGeom')[:2]
        adjustedPos = orig_init_pos[:2] + diff

        # The convention we follow is that body_com[2] is always 0, and geom_pos[2] is the object height
        return [
            adjustedPos[0],
            adjustedPos[1],
            self.data.get_geom_xpos('objGeom')[-1]
        ]

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
            self._target_pos = goal_pos[-3:]
            self.obj_init_pos = goal_pos[:3]

        self._set_obj_xyz(self.obj_init_pos)
        self.maxplacing_dist = np.linalg.norm(
            np.array([
                self.obj_init_pos[0],
                self.obj_init_pos[1],
                self.heightTarget
            ]) - np.array(self._target_pos)
        ) + self.heightTarget
        self.target_reward = 1000 * self.maxplacing_dist + 1000 * 2
        self.num_resets += 1

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand()

        rightFinger, leftFinger = (
            self._get_site_pos('rightEndEffector'),
            self._get_site_pos('leftEndEffector')
        )
        self.init_finger_center  =  (rightFinger + leftFinger) / 2
        self.pick_completed = False

    def compute_reward(self, actions, obs):
        pos_obj = obs[3:6]

        finger_right, finger_left = (
            self._get_site_pos('rightEndEffector'),
            self._get_site_pos('leftEndEffector')
        )

        finger_center = (finger_right + finger_left) / 2
        heightTarget = self.heightTarget

        goal = self._target_pos
        assert np.all(goal == self._get_site_pos('goal'))

        tolerance = 0.01
        self.pick_completed = (pos_obj[2] >= (heightTarget - tolerance))

        reach_dist = np.linalg.norm(pos_obj - finger_center)
        placing_dist = np.linalg.norm(pos_obj - goal)

        def obj_dropped():
            # Object on the ground, far away from the goal, and from the
            # gripper. Can tweak the margin limits
            return (pos_obj[2] < (self.objHeight + 0.005)) \
                    and (placing_dist > 0.02) \
                    and (reach_dist > 0.02)

        def reach_reward():
            reach_xy = np.linalg.norm(pos_obj[:-1] - finger_center[:-1])
            z_rew = np.linalg.norm(finger_center[-1] - self.init_finger_center[-1])
            reach_rew = -reach_dist if reach_xy < 0.05 else -reach_xy - 2*z_rew
            # Incentive to close fingers when reach_dist is small
            if reach_dist < 0.05:
                reach_rew = -reach_dist + max(actions[-1], 0)/50
            return reach_rew, reach_dist

        def pick_reward():
            h_scale = 100
            if self.pick_completed and not(obj_dropped()):
                return h_scale * heightTarget
            elif (reach_dist < 0.1) and (pos_obj[2] > (self.objHeight + 0.005)):
                return h_scale * min(heightTarget, pos_obj[2])
            else:
                return 0

        def place_reward():
            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            if self.pick_completed and (reach_dist < 0.1) and not(obj_dropped()):
                place_rew = c1 * (self.maxplacing_dist - placing_dist) + \
                            c1*(np.exp(-(placing_dist**2)/c2) +
                                np.exp(-(placing_dist**2)/c3))
                place_rew = max(place_rew, 0)
                return [place_rew, placing_dist]
            else:
                return [0, placing_dist]

        reach_rew, reach_dist = reach_reward()
        pick_rew = pick_reward()
        place_rew, placing_dist = place_reward()
        assert ((place_rew >= 0) and (pick_rew >= 0))
        reward = reach_rew + pick_rew + place_rew

        return [reward, reach_dist, pick_rew, placing_dist]
