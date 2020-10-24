import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerHandInsertEnvV2(SawyerXYZEnv):
    PAD_SUCCESS_MARGIN = 0.05
    X_Z_SUCCESS_MARGIN = 0.005
    OBJ_RADIUS = 0.015
    TARGET_RADIUS = 0.05

    def __init__(self):

        hand_low = (-0.5, 0.40, -0.15)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.6, 0.05)
        obj_high = (0.1, 0.7, 0.05)
        goal_low = (-0.04, 0.8, -0.0201)
        goal_high = (0.04, 0.88, -0.0199)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([0, 0.6, 0.05]),
            'obj_init_angle': 0.3,
            'hand_init_pos': np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0., 0.84, -0.08], dtype=np.float32)
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.max_path_length = 500

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_table_with_hole.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        obj = ob[4:7]

        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            grasp_reward,
            in_place_reward
        ) = self.compute_reward(action, ob)

        info = {
            'success': float(obj_to_target <= 0.05),
            'near_object': float(tcp_to_obj <= 0.03),
            'grasp_success': float(
                self.touching_main_object and
                (tcp_open > 0) and
                (obj[2] - 0.02 > self.obj_init_pos[2])
            ),
            'grasp_reward': grasp_reward,
            'in_place_reward': in_place_reward,
            'obj_to_target': obj_to_target,
            'unscaled_reward': reward,
        }

        self.curr_path_length += 1
        return ob, reward, False, info

    @property
    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id('objGeom')

    def _get_pos_objects(self):
        return self.get_body_com('obj')

    def _get_quat_objects(self):
        return self.sim.data.get_body_xquat('obj')

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.objHeight = self.get_body_com('obj')[2]

        # if self.random_init:
        goal_pos = self._get_state_rand_vec()
        while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.15:
            goal_pos = self._get_state_rand_vec()
        self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))
        self._target_pos = goal_pos[-3:]

        self._set_obj_xyz(self.obj_init_pos)
        self.maxReachDist = np.abs(self.hand_init_pos[-1] - self._target_pos[-1])

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand()
        self.pickCompleted = False
        self.init_tcp = self.tcp_center
        self.init_left_pad = self.get_body_com('leftpad')
        self.init_right_pad = self.get_body_com('rightpad')

    def _gripper_caging_reward(self, action, pos_main_object):
        # MARK: Left-right gripper information for caging reward----------------
        left_pad = self.get_body_com('leftpad')
        right_pad = self.get_body_com('rightpad')

        pad_y_lr = np.hstack((left_pad[1], right_pad[1]))
        pad_y_lr_init = np.hstack((self.init_left_pad[1], self.init_right_pad[1]))

        obj_to_pad_lr = np.abs(pad_y_lr - pos_main_object[1])
        obj_to_pad_lr_init = np.abs(pad_y_lr_init - pos_main_object[1])

        caging_margin_lr = np.abs(obj_to_pad_lr_init - self.PAD_SUCCESS_MARGIN)
        caging_lr = [reward_utils.tolerance(
            obj_to_pad_lr[i],
            bounds=(self.OBJ_RADIUS, self.PAD_SUCCESS_MARGIN),
            margin=caging_margin_lr[i],
            sigmoid='long_tail',
        ) for i in range(2)]
        caging_y = reward_utils.hamacher_product(*caging_lr)

        # MARK: X-Z gripper information for caging reward-----------------------
        tcp = self.tcp_center
        xz = [0, 2]
        xz_margin = np.linalg.norm(self.obj_init_pos[xz] - self.init_tcp[xz])
        xz_margin -= self.X_Z_SUCCESS_MARGIN

        caging_xz = reward_utils.tolerance(
            np.linalg.norm(tcp[xz] - pos_main_object[xz]),
            bounds=(0, self.X_Z_SUCCESS_MARGIN),
            margin=xz_margin,
            sigmoid='long_tail',
        )

        # MARK: Closed-extent gripper information for caging reward-------------
        gripper_closed = min(max(0, action[-1]), 1)

        # MARK: Combine components----------------------------------------------
        caging = reward_utils.hamacher_product(caging_y, caging_xz)
        gripping = gripper_closed if caging > 0.97 else 0.

        return reward_utils.hamacher_product(caging, gripping)

    def compute_reward(self, action, obs):
        obj = obs[4:7]

        target_to_obj = np.linalg.norm(obj - self._target_pos)
        target_to_obj_init = np.linalg.norm(obj - self.obj_init_pos)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=target_to_obj_init,
            sigmoid='long_tail',
        )

        object_grasped = self._gripper_caging_reward(action, obj)
        reward = reward_utils.hamacher_product(object_grasped, in_place)

        tcp_opened = obs[3]
        tcp_to_obj = np.linalg.norm(obj - self.tcp_center)

        if tcp_to_obj < 0.02 and tcp_opened > 0:
            reward += 1. + 5. * in_place
        if target_to_obj < self.TARGET_RADIUS:
            reward = 10.
        return (
            reward,
            tcp_to_obj,
            tcp_opened,
            target_to_obj,
            object_grasped,
            in_place
        )
