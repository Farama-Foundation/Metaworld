import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerPickPlaceEnvV2(SawyerXYZEnv):
    """
    Motivation for V2:
        V1 was very difficult to solve because the observation didn't say where
        to move after picking up the puck.
    Changelog from V1 to V2:
        - (7/7/20) Removed 3 element vector. Replaced with 3 element position
            of the goal (for consistency with other environments)
        - (6/15/20) Added a 3 element vector to the observation. This vector
            points from the end effector to the goal coordinate.
            i.e. (self._target_pos - pos_hand)
        - (6/15/20) Separated reach-push-pick-place into 3 separate envs.
    """
    def __init__(self):
        liftThresh = 0.04

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
            'obj_init_pos': np.array([0, 0.6, 0.02]),
            'hand_init_pos': np.array([0, .6, .2]),
        }

        self.goal = np.array([0.1, 0.8, 0.2])

        self.obj_init_angle = self.init_config['obj_init_angle']
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.liftThresh = liftThresh
        self.max_path_length = 500

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.num_resets = 0
        self.obj_init_pos = None

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_pick_place_v2.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)

        obs = self._get_obs()
        obj = obs[4:7]


        reward, tcp_to_obj, tcp_open, obj_to_target, grasp_reward, in_place_reward = self.compute_reward(action, obs)
        success = float(obj_to_target <= 0.07)
        near_object = float(tcp_to_obj <= 0.03)
        grasp_success = float(self.touching_object and (tcp_open > 0) and (obj[2] - 0.02 > self.obj_init_pos[2]))
        info = {
            'success': success,
            'near_object': near_object,
            'grasp_success': grasp_success,
            'grasp_reward': grasp_reward,
            'in_place_reward': in_place_reward,
            'obj_to_target': obj_to_target
        }

        self.curr_path_length += 1
        return obs, reward, False, info

    @property
    def touching_object(self):
        object_geom_id = self.unwrapped.model.geom_name2id('objGeom')
        leftpad_geom_id = self.unwrapped.model.geom_name2id('leftpad_geom')
        rightpad_geom_id = self.unwrapped.model.geom_name2id('rightpad_geom')

        leftpad_object_contacts = [
            x for x in self.unwrapped.data.contact
            if (leftpad_geom_id in (x.geom1, x.geom2)
                and object_geom_id in (x.geom1, x.geom2))
        ]

        rightpad_object_contacts = [
            x for x in self.unwrapped.data.contact
            if (rightpad_geom_id in (x.geom1, x.geom2)
                and object_geom_id in (x.geom1, x.geom2))
        ]

        leftpad_object_contact_force = sum(
            self.unwrapped.data.efc_force[x.efc_address]
            for x in leftpad_object_contacts)

        rightpad_object_contact_force = sum(
            self.unwrapped.data.efc_force[x.efc_address]
            for x in rightpad_object_contacts)

        gripping = (0 < leftpad_object_contact_force
                    and 0 < rightpad_object_contact_force)

        return gripping

    def _get_pos_orientation_objects(self):
        position = self.get_body_com('obj')
        orientation = Rotation.from_matrix(
            self.data.get_geom_xmat('objGeom')).as_quat()
        return position, orientation, np.array([]), np.array([])

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
            self.init_tcp = self.tcp_center
            self.init_left_pad = self.get_body_com('leftpad')
            self.init_right_pad = self.get_body_com('rightpad')

        self._set_obj_xyz(self.obj_init_pos)
        self.maxPlacingDist = np.linalg.norm(
            np.array([self.obj_init_pos[0],
                      self.obj_init_pos[1],
                      self.heightTarget]) -
            np.array(self._target_pos)) + self.heightTarget
        self.target_reward = 1000*self.maxPlacingDist + 1000*2
        self.num_resets += 1

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand()

        finger_right, finger_left = (
            self._get_site_pos('rightEndEffector'),
            self._get_site_pos('leftEndEffector')
        )
        self.init_finger_center = (finger_right + finger_left) / 2
        self.pick_completed = False

    # def _gripper_caging_reward(self, action, obj_position):
    #     pad_success_margin = 0.05
    #     tcp = self.tcp_center
    #     left_pad = self.get_body_com('leftpad')
    #     right_pad = self.get_body_com('rightpad')
    #     delta_object_y_left_pad = left_pad[1] - obj_position[1]
    #     delta_object_y_right_pad = obj_position[1] - right_pad[1]
    #     right_caging_margin = abs(abs(obj_position[1] - self.init_right_pad[1]) - pad_success_margin)
    #     left_caging_margin = abs(abs(obj_position[1] - self.init_left_pad[1]) - pad_success_margin)
    #     right_caging = reward_utils.tolerance(delta_object_y_right_pad,
    #                             bounds=(0.015, pad_success_margin),
    #                             margin=right_caging_margin,
    #                             sigmoid='long_tail',)
    #     left_caging = reward_utils.tolerance(delta_object_y_left_pad,
    #                             bounds=(0.015, pad_success_margin),
    #                             margin=left_caging_margin,
    #                             sigmoid='long_tail',)
    #     assert right_caging >= 0 and right_caging <= 1
    #     assert left_caging >= 0 and left_caging <= 1
    #     # hamacher product
    #     y_caging = ((right_caging * left_caging) / (right_caging + left_caging -
    #         (right_caging * left_caging)))
    #     assert y_caging >= 0 and y_caging <= 1
    #     tcp_xz = tcp + np.array([0., -tcp[1], 0.])
    #     obj_position_x_z = np.copy(obj_position) + np.array([0., -obj_position[1], 0.])
    #     tcp_obj_norm_x_z = np.linalg.norm(tcp_xz - obj_position_x_z, ord=2)
    #     init_obj_x_z = self.obj_init_pos + np.array([0., -self.obj_init_pos[1], 0.])
    #     init_tcp_x_z = self.init_tcp + np.array([0., -self.init_tcp[1], 0.])
    #
    #     x_z_success_margin = 0.005
    #     tcp_obj_x_z_margin = np.linalg.norm(init_obj_x_z - init_tcp_x_z, ord=2) - x_z_success_margin
    #     x_z_caging = reward_utils.tolerance(tcp_obj_norm_x_z,
    #                             bounds=(0, x_z_success_margin),
    #                             margin=tcp_obj_x_z_margin,
    #                             sigmoid='long_tail',)
    #     assert right_caging >= 0 and right_caging <= 1
    #     gripper_closed = min(max(0, action[-1]), 1)
    #     assert gripper_closed >= 0 and gripper_closed <= 1
    #     caging = ((y_caging * x_z_caging) / (y_caging + x_z_caging -
    #         (y_caging * x_z_caging)))
    #     assert caging >= 0 and caging <= 1
    #     # gripping = caging * gripper_closed
    #     if caging > 0.97:
    #         gripping = gripper_closed
    #     else:
    #         gripping = 0.
    #     assert gripping >= 0 and gripping <= 1
    #     caging_and_gripping = ((caging * gripping) / (caging + gripping -
    #         (caging * gripping)))
    #     assert caging_and_gripping >= 0 and caging_and_gripping <= 1
    #     return caging_and_gripping

    def compute_reward(self, action, obs):
        _TARGET_RADIUS = 0.05
        tcp = self.tcp_center
        obj = obs[4:7]
        tcp_opened = obs[3]
        target = self._target_pos

        obj_to_target = np.linalg.norm(obj - target)
        tcp_to_obj = np.linalg.norm(obj - tcp)
        in_place_margin = (np.linalg.norm(self.obj_init_pos - target))
        in_place = reward_utils.tolerance(obj_to_target,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=in_place_margin,
                                    sigmoid='long_tail',)

        assert in_place >= 0 and in_place <= 1

        object_grasped = self._gripper_caging_reward(action, obj, 0.015)

        assert object_grasped >= 0 and object_grasped <= 1

        in_place_grasped = in_place
        if not object_grasped and not in_place_grasped:
            reward = 0
        else:
            in_place_and_object_grasped = reward_utils.hamacher_product(object_grasped, in_place_grasped)
            assert in_place_and_object_grasped >= 0 and in_place_and_object_grasped <= 1
            reward = in_place_and_object_grasped

        reward *= 10
        return [reward, tcp_to_obj, tcp_opened, obj_to_target, object_grasped, in_place]
