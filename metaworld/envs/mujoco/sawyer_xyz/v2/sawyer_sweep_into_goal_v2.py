import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerSweepIntoGoalEnvV2(SawyerXYZEnv):

    OBJ_RADIUS = 0.02

    def __init__(self):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.6, 0.02)
        obj_high = (0.1, 0.7, 0.02)
        goal_low = (-.001, 0.8399, 0.0199)
        goal_high = (+.001, 0.8401, 0.0201)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos':np.array([0., 0.6, 0.02]),
            'obj_init_angle': 0.3,
            'hand_init_pos': np.array([0., .6, .2]),
        }
        self.goal = np.array([0., 0.84, 0.02])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_table_with_hole.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        obj = obs[4:7]
        (
            reward,
            tcp_to_obj,
            tcp_opened,
            target_to_obj,
            object_grasped,
            in_place
        ) = self.compute_reward(action, obs)

        grasp_success = float(self.touching_main_object and (tcp_opened > 0))

        info = {
            'success': float(target_to_obj <= 0.05),
            'near_object': float(tcp_to_obj <= 0.03),
            'grasp_reward': object_grasped,
            'grasp_success': grasp_success,
            'in_place_reward': in_place,
            'obj_to_target': target_to_obj,
            'unscaled_reward': reward,
        }
        return reward, info

    def _get_quat_objects(self):
        return Rotation.from_matrix(
            self.data.get_geom_xmat('objGeom')
        ).as_quat()

    def _get_pos_objects(self):
        return self.get_body_com('obj')

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.get_body_com('obj')
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.objHeight = self.get_body_com('obj')[2]

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
                goal_pos = self._get_state_rand_vec()
            self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))

        self._set_obj_xyz(self.obj_init_pos)
        self.maxPushDist = np.linalg.norm(self.obj_init_pos[:2] - np.array(self._target_pos)[:2])

        return self._get_obs()

    def _gripper_caging_reward(self, action, obj_position, obj_radius):
        pad_success_margin = 0.05
        grip_success_margin = obj_radius + 0.005
        x_z_success_margin = 0.01

        tcp = self.tcp_center
        left_pad = self.get_body_com('leftpad')
        right_pad = self.get_body_com('rightpad')
        delta_object_y_left_pad = left_pad[1] - obj_position[1]
        delta_object_y_right_pad = obj_position[1] - right_pad[1]
        right_caging_margin = abs(abs(obj_position[1] - self.init_right_pad[1]) - pad_success_margin)
        left_caging_margin = abs(abs(obj_position[1] - self.init_left_pad[1]) - pad_success_margin)

        right_caging = reward_utils.tolerance(delta_object_y_right_pad,
            bounds=(obj_radius, pad_success_margin),
            margin=right_caging_margin,
            sigmoid='long_tail',
        )
        left_caging = reward_utils.tolerance(delta_object_y_left_pad,
            bounds=(obj_radius, pad_success_margin),
            margin=left_caging_margin,
            sigmoid='long_tail',
        )

        right_gripping = reward_utils.tolerance(delta_object_y_right_pad,
            bounds=(obj_radius, grip_success_margin),
            margin=right_caging_margin,
            sigmoid='long_tail',
        )
        left_gripping = reward_utils.tolerance(delta_object_y_left_pad,
            bounds=(obj_radius, grip_success_margin),
            margin=left_caging_margin,
            sigmoid='long_tail',
        )

        assert right_caging >= 0 and right_caging <= 1
        assert left_caging >= 0 and left_caging <= 1

        y_caging = reward_utils.hamacher_product(right_caging, left_caging)
        y_gripping = reward_utils.hamacher_product(right_gripping, left_gripping)

        assert y_caging >= 0 and y_caging <= 1

        tcp_xz = tcp + np.array([0., -tcp[1], 0.])
        obj_position_x_z = np.copy(obj_position) + np.array([0., -obj_position[1], 0.])
        tcp_obj_norm_x_z = np.linalg.norm(tcp_xz - obj_position_x_z, ord=2)
        init_obj_x_z = self.obj_init_pos + np.array([0., -self.obj_init_pos[1], 0.])
        init_tcp_x_z = self.init_tcp + np.array([0., -self.init_tcp[1], 0.])

        tcp_obj_x_z_margin = np.linalg.norm(init_obj_x_z - init_tcp_x_z, ord=2) - x_z_success_margin
        x_z_caging = reward_utils.tolerance(tcp_obj_norm_x_z,
                                bounds=(0, x_z_success_margin),
                                margin=tcp_obj_x_z_margin,
                                sigmoid='long_tail',)

        assert right_caging >= 0 and right_caging <= 1
        gripper_closed = min(max(0, action[-1]), 1)
        assert gripper_closed >= 0 and gripper_closed <= 1
        caging = reward_utils.hamacher_product(y_caging, x_z_caging)
        assert caging >= 0 and caging <= 1

        if caging > 0.95:
            gripping = y_gripping
        else:
            gripping = 0.
        assert gripping >= 0 and gripping <= 1

        caging_and_gripping = (caging + gripping) / 2
        assert caging_and_gripping >= 0 and caging_and_gripping <= 1

        return caging_and_gripping

    def compute_reward(self, action, obs):
        _TARGET_RADIUS = 0.05
        tcp = self.tcp_center
        obj = obs[4:7]
        tcp_opened = obs[3]
        target = np.array([self._target_pos[0], self._target_pos[1], obj[2]])

        obj_to_target = np.linalg.norm(obj - target)
        tcp_to_obj = np.linalg.norm(obj - tcp)
        in_place_margin = np.linalg.norm(self.obj_init_pos - target)

        in_place = reward_utils.tolerance(obj_to_target,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=in_place_margin,
                                    sigmoid='long_tail',)

        object_grasped = self._gripper_caging_reward(action, obj, self.OBJ_RADIUS)
        in_place_and_object_grasped = reward_utils.hamacher_product(object_grasped,
                                                                    in_place)

        reward = (2*object_grasped) + (6*in_place_and_object_grasped)

        if obj_to_target < _TARGET_RADIUS:
            reward = 10.
        return [reward, tcp_to_obj, tcp_opened, obj_to_target, object_grasped, in_place]
