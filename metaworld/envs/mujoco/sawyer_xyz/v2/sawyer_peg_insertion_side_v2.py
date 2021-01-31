import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set
from scipy.spatial.transform import Rotation


class SawyerPegInsertionSideEnvV2(SawyerXYZEnv):
    TARGET_RADIUS = 0.07
    """
    Motivation for V2:
        V1 was difficult to solve because the observation didn't say where
        to insert the peg (the hole's location). Furthermore, the hole object
        could be initialized in such a way that it severely restrained the
        sawyer's movement.
    Changelog from V1 to V2:
        - (8/21/20) Updated to Byron's XML
        - (7/7/20) Removed 1 element vector. Replaced with 3 element position
            of the hole (for consistency with other environments)
        - (6/16/20) Added a 1 element vector to the observation. This vector
            points from the end effector to the hole in the Y direction.
            i.e. (self._target_pos - pos_hand)[1]
        - (6/16/20) Used existing goal_low and goal_high values to constrain
            the hole's position, as opposed to hand_low and hand_high
    """
    def __init__(self):
        hand_init_pos = (0, 0.6, 0.2)

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (.0, 0.5, 0.02)
        obj_high = (.2, 0.7, 0.02)
        goal_low = (-0.35, 0.4, -0.001)
        goal_high = (-0.25, 0.7, 0.001)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([0, 0.6, 0.02]),
            'hand_init_pos': np.array([0, .6, .2]),
        }

        self.goal = np.array([-0.3, 0.6, 0.0])

        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.hand_init_pos = np.array(hand_init_pos)

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(
            np.array(goal_low) + np.array([.03, .0, .13]),
            np.array(goal_high) + np.array([.03, .0, .13])
        )

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_peg_insertion_side.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        obj = obs[4:7]

        reward, tcp_to_obj, tcp_open, obj_to_target, grasp_reward, in_place_reward, collision_box_front, ip_orig= (
            self.compute_reward(action, obs))
        grasp_success = float(tcp_to_obj < 0.02 and (tcp_open > 0) and (obj[2] - 0.01 > self.obj_init_pos[2]))
        success = float(obj_to_target <= 0.07)
        near_object = float(tcp_to_obj <= 0.03)

        info = {
            'success': success,
            'near_object': near_object,
            'grasp_success': grasp_success,
            'grasp_reward': grasp_reward,
            'in_place_reward': in_place_reward,
            'obj_to_target': obj_to_target,
            'unscaled_reward': reward,
        }

        return reward, info

    def _get_pos_objects(self):
        return self._get_site_pos('pegGrasp')

    def _get_quat_objects(self):
        return Rotation.from_matrix(self.data.get_site_xmat('pegGrasp')).as_quat()

    def reset_model(self):
        self._reset_hand()

        pos_peg = self.obj_init_pos
        pos_box = self.goal
        if self.random_init:
            pos_peg, pos_box = np.split(self._get_state_rand_vec(), 2)
            while np.linalg.norm(pos_peg[:2] - pos_box[:2]) < 0.1:
                pos_peg, pos_box = np.split(self._get_state_rand_vec(), 2)

        self.obj_init_pos = pos_peg
        self.peg_head_pos_init = self._get_site_pos('pegHead')
        self._set_obj_xyz(self.obj_init_pos)

        self.sim.model.body_pos[self.model.body_name2id('box')] = pos_box
        self._target_pos = pos_box + np.array([.03, .0, .13])

        return self._get_obs()

    def compute_reward(self, action, obs):
        tcp = self.tcp_center
        obj = obs[4:7]
        obj_head = self._get_site_pos('pegHead')
        tcp_opened = obs[3]
        target = self._target_pos
        tcp_to_obj = np.linalg.norm(obj - tcp)
        scale = np.array([1., 2., 2.])
        #  force agent to pick up object then insert
        obj_to_target = np.linalg.norm((obj_head - target) * scale)

        in_place_margin = np.linalg.norm((self.peg_head_pos_init - target) * scale)
        in_place = reward_utils.tolerance(obj_to_target,
                                    bounds=(0, self.TARGET_RADIUS),
                                    margin=in_place_margin,
                                    sigmoid='long_tail',)
        ip_orig = in_place
        brc_col_box_1 = self._get_site_pos('bottom_right_corner_collision_box_1')
        tlc_col_box_1 = self._get_site_pos('top_left_corner_collision_box_1')

        brc_col_box_2 = self._get_site_pos('bottom_right_corner_collision_box_2')
        tlc_col_box_2 = self._get_site_pos('top_left_corner_collision_box_2')
        collision_box_bottom_1 = reward_utils.rect_prism_tolerance(curr=obj_head,
                                                                   one=tlc_col_box_1,
                                                                   zero=brc_col_box_1)
        collision_box_bottom_2 = reward_utils.rect_prism_tolerance(curr=obj_head,
                                                                   one=tlc_col_box_2,
                                                                   zero=brc_col_box_2)
        collision_boxes = reward_utils.hamacher_product(collision_box_bottom_2,
                                                        collision_box_bottom_1)
        in_place = reward_utils.hamacher_product(in_place,
                                                 collision_boxes)

        pad_success_margin = 0.03
        object_reach_radius=0.01
        x_z_margin = 0.005
        obj_radius = 0.0075

        object_grasped = self._gripper_caging_reward(action,
                                                     obj,
                                                     object_reach_radius=object_reach_radius,
                                                     obj_radius=obj_radius,
                                                     pad_success_thresh=pad_success_margin,
                                                     xz_thresh=x_z_margin,
                                                     high_density=True)
        if tcp_to_obj < 0.08 and (tcp_opened > 0) and (obj[2] - 0.01 > self.obj_init_pos[2]):
            object_grasped = 1.
        in_place_and_object_grasped = reward_utils.hamacher_product(object_grasped,
                                                                    in_place)
        reward = in_place_and_object_grasped

        if tcp_to_obj < 0.08 and (tcp_opened > 0) and (obj[2] - 0.01 > self.obj_init_pos[2]):
            reward += 1. + 5 * in_place

        if obj_to_target <= 0.07:
            reward = 10.

        return [reward, tcp_to_obj, tcp_opened, obj_to_target, object_grasped, in_place, collision_boxes, ip_orig]

