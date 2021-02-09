import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
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
    def evaluate_state(self, obs, action):
        obj = obs[4:7]
        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            grasp_reward,
            in_place_reward,
        ) = self.compute_reward(action, obs)

        success = float(obj_to_target <= 0.07)
        near_object = float(tcp_to_obj <= 0.03)
        grasp_success = float(self.touching_main_object and (tcp_open > 0)
                              and (obj[2] - 0.02 > self.obj_init_pos[2]))
        info = {
            'success': success,
            'near_object': near_object,
            'grasp_success': grasp_success,
            'grasp_reward': grasp_reward,
            'in_place_reward': in_place_reward,
            'obj_to_target': obj_to_target,
            'unscaled_reward': reward
        }

        return reward, info

    def _get_pos_objects(self):
        return self.data.get_geom_xpos('objGeom')

    def _get_quat_objects(self):
        return Rotation.from_matrix(
            self.data.get_geom_xmat('objGeom')
        ).as_quat()

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

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            self._target_pos = goal_pos[3:]
            while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
                goal_pos = self._get_state_rand_vec()
                self._target_pos = goal_pos[3:]
            self._target_pos = goal_pos[-3:]
            self.obj_init_pos = goal_pos[:3]

        self._set_obj_xyz(self.obj_init_pos)
        self.num_resets += 1

        return self._get_obs()

    def compute_reward(self, action, obs):
        _TARGET_RADIUS = 0.05
        tcp = self.tcp_center
        obj = obs[4:7]
        tcp_opened = obs[3]
        midpoint = np.array([self._target_pos[0], 0.77, 0.25])
        target = self._target_pos

        tcp_to_obj = np.linalg.norm(obj - tcp)

        in_place_scaling = np.array([1., 1., 3.])
        obj_to_midpoint = np.linalg.norm((obj - midpoint) * in_place_scaling)
        obj_to_midpoint_init = np.linalg.norm((self.obj_init_pos - midpoint) * in_place_scaling)

        obj_to_target = np.linalg.norm(obj - target)
        obj_to_target_init = np.linalg.norm(self.obj_init_pos - target)

        in_place_part1 = reward_utils.tolerance(obj_to_midpoint,
            bounds=(0, _TARGET_RADIUS),
            margin=obj_to_midpoint_init,
            sigmoid='long_tail',
        )

        in_place_part2 = reward_utils.tolerance(obj_to_target,
            bounds=(0, _TARGET_RADIUS),
            margin=obj_to_target_init,
            sigmoid='long_tail'
        )

        object_grasped = self._gripper_caging_reward(action=action,
                                                     obj_pos=obj,
                                                     obj_radius=0.015,
                                                     pad_success_thresh=0.05,
                                                     object_reach_radius=0.01,
                                                     xz_thresh=0.005,
                                                     high_density=False
                                                     )

        in_place_and_object_grasped = reward_utils.hamacher_product(object_grasped,
                                                                    in_place_part1)
        reward = in_place_and_object_grasped

        if tcp_to_obj < 0.02 and (tcp_opened > 0) and (obj[2] - 0.015 > self.obj_init_pos[2]):
            reward = in_place_and_object_grasped + 1. + 4. * in_place_part1
            if obj[1] > 0.75:
                reward = in_place_and_object_grasped + 1. + 4. + 3. * in_place_part2

        if obj_to_target < _TARGET_RADIUS:
            reward = 10.

        return [
            reward,
            tcp_to_obj,
            tcp_opened,
            np.linalg.norm(obj - target),
            object_grasped,
            in_place_part2
        ]
