import numpy as np
from gym.spaces import Box

from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerReachEnvV2(SawyerXYZEnv):
    """
    Motivation for V2:
        V1 was very difficult to solve because the observation didn't say where
        to move (where to reach).
    Changelog from V1 to V2:
        - (7/7/20) Removed 3 element vector. Replaced with 3 element position
            of the goal (for consistency with other environments)
        - (6/15/20) Added a 3 element vector to the observation. This vector
            points from the end effector to the goal coordinate.
            i.e. (self._target_pos - pos_hand)
        - (6/15/20) Separated reach-push-pick-place into 3 separate envs.
    """
    def __init__(self):
        lift_thresh = 0.04

        goal_low = (-0.1, 0.8, 0.05)
        goal_high = (0.1, 0.9, 0.3)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.6, 0.02)
        obj_high = (0.1, 0.7, 0.02)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': .3,
            'obj_init_pos': np.array([0., 0.6, 0.02]),
            'hand_init_pos': np.array([0., 0.6, 0.2]),
        }

        self.goal = np.array([-0.1, 0.8, 0.2])

        self.obj_init_angle = self.init_config['obj_init_angle']
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.liftThresh = lift_thresh
        self.max_path_length = 150

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.num_resets = 0

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_reach_v2.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)

        reward, reach_dist = self.compute_reward(action, ob)
        success = float(reach_dist <= 0.05)

        info = {
            'reachDist': reach_dist,
            'epRew': reward,
            'success': success
        }

        self.curr_path_length += 1
        return ob, reward, False, info

    def _get_pos_objects(self):
        return self.get_body_com('obj')

    def fix_extreme_obj_pos(self, orig_init_pos):
        # This is to account for meshes for the geom and object are not
        # aligned. If this is not done, the object could be initialized in an
        # extreme position
        diff = self.get_body_com('obj')[:2] - \
               self.get_body_com('obj')[:2]
        adjusted_pos = orig_init_pos[:2] + diff
        # The convention we follow is that body_com[2] is always 0,
        # and geom_pos[2] is the object height
        return [
            adjusted_pos[0],
            adjusted_pos[1],
            self.get_body_com('obj')[-1]
        ]

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.fix_extreme_obj_pos(self.init_config['obj_init_pos'])
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.objHeight = self.get_body_com('obj')[2]
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
        self.maxReachDist = np.linalg.norm(
            self.init_finger_center - np.array(self._target_pos))
        self.target_reward = 1000 * self.maxReachDist + 1000 * 2
        self.num_resets += 1

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand()

        finger_right, finger_left = (
            self._get_site_pos('rightEndEffector'),
            self._get_site_pos('leftEndEffector')
        )
        self.init_finger_center = (finger_right + finger_left) / 2
        self.pickCompleted = False

    def compute_reward(self, actions, obs):
        finger_right, finger_left = (
            self._get_site_pos('rightEndEffector'),
            self._get_site_pos('leftEndEffector')
        )
        finger_center = (finger_right + finger_left) / 2

        goal = self._target_pos
        assert np.all(goal == self._get_site_pos('goal'))

        c1 = 1000
        c2 = 0.01
        c3 = 0.001
        reach_dist = np.linalg.norm(finger_center - goal)
        reach_rew = c1 * (self.maxReachDist - reach_dist) + \
                    c1 * (np.exp(-(reach_dist ** 2) / c2) +
                          np.exp(-(reach_dist ** 2) / c3))

        reach_rew = max(reach_rew, 0)
        return [reach_rew, reach_dist]
