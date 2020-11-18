import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils


from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerDoorEnvV2(SawyerXYZEnv):

    OBJ_RADIUS = 0.03

    def __init__(self):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (0., 0.85, 0.15)
        obj_high = (0.1, 0.95, 0.15)
        goal_low = (-.3, 0.4, 0.1499)
        goal_high = (-.2, 0.5, 0.1501)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': np.array([0.3, ]),
            'obj_init_pos': np.array([0.1, 0.95, 0.15]),
            'hand_init_pos': np.array([0, 0.6, 0.4]),
        }

        self.goal = np.array([-0.2, 0.7, 0.15])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.max_path_length = 150

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.door_angle_idx = self.model.get_joint_qpos_addr('doorjoint')

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_door_pull.xml')

    @_assert_task_is_set
    def step(self, action):
        obs = super().step(action)
        # (
        #     reward,
        #     tcp_to_obj,
        #     tcp_opened,
        #     obj_to_target,
        #     object_grasped,
        #     in_place
        # ) = self.compute_reward(action, obs)

        (
            reward,
            gripper_error,
            gripped,
            handle_error,
            caging_reward,
            opening_reward
        ) = self.compute_reward(action, obs)

        goal_dist = np.linalg.norm(obs[4:7] - self._target_pos)

        info = {
            'success': float(goal_dist <= 0.03),
            'near_object': float(gripper_error <= 0.03),
            'grasp_success': float(gripped > 0),
            'grasp_reward': caging_reward,
            'in_place_reward': opening_reward,
            'obj_to_target': handle_error,
            'unscaled_reward': reward,
        }


        self.curr_path_length += 1
        # info = {
        #     'reward': reward,
        #     'tcp_to_obj': tcp_to_obj,
        #     'obj_to_target': obj_to_target,
        #     'in_place_reward': in_place,
        #     'success': float(obj_to_target <= 0.08)
        # }

        return obs, reward, False, info

    def _get_pos_objects(self):
        return self.data.get_geom_xpos('handle').copy()

    def _get_quat_objects(self):
        return Rotation.from_matrix(self.data.get_geom_xmat('handle')).as_quat()

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        qpos[self.door_angle_idx] = pos
        qvel[self.door_angle_idx] = 0
        self.set_state(qpos.flatten(), qvel.flatten())

    def reset_model(self):
        self._reset_hand()

        self.objHeight = self.data.get_geom_xpos('handle')[2]

        self.obj_init_pos = self._get_state_rand_vec() if self.random_init \
            else self.init_config['obj_init_pos']
        self._target_pos = self.obj_init_pos + np.array([-0.3, -0.45, 0.])

        self.sim.model.body_pos[self.model.body_name2id('door')] = self.obj_init_pos
        self.sim.model.site_pos[self.model.site_name2id('goal')] = self._target_pos
        self._set_obj_xyz(0)
        self.maxPullDist = np.linalg.norm(self.data.get_geom_xpos('handle')[:-1] - self._target_pos[:-1])
        self.target_reward = 1000*self.maxPullDist + 1000*2

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand()

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.reachCompleted = False

    def _gripper_caging_reward(self, action, obj_position, obj_radius):
        pad_success_margin = 0.05 # obj_radius + 0.01
        grip_success_margin_low = obj_radius - 0.005
        grip_success_margin_high = obj_radius + 0.001
        x_z_success_margin = 0.01

        # scale = np.array([3. ,3. ,1.])

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

        # right_gripping = reward_utils.tolerance(delta_object_y_right_pad,
        #     bounds=(grip_success_margin_low, grip_success_margin_high),
        #     margin=right_caging_margin,
        #     sigmoid='long_tail',
        # )
        # left_gripping = reward_utils.tolerance(delta_object_y_left_pad,
        #     bounds=(grip_success_margin_low, grip_success_margin_high),
        #     margin=left_caging_margin,
        #     sigmoid='long_tail',
        # )


        assert right_caging >= 0 and right_caging <= 1
        assert left_caging >= 0 and left_caging <= 1

        y_caging = reward_utils.hamacher_product(right_caging, left_caging)
        # y_gripping = reward_utils.hamacher_product(right_gripping, left_gripping)

        assert y_caging >= 0 and y_caging <= 1

        tcp_xz = tcp + np.array([0., -tcp[1], 0.])
        obj_position_x_z = np.copy(obj_position) + np.array([0., -obj_position[1], 0.])
        tcp_obj_norm_x_z = np.linalg.norm(tcp_xz - obj_position_x_z, ord=2)
        init_obj_x_z = self.obj_init_pos + np.array([0., -self.obj_init_pos[1], 0.])
        init_tcp_x_z = self.init_tcp + np.array([0., -self.init_tcp[1], 0.])


        tcp_obj_x_z_margin = np.linalg.norm(init_obj_x_z - init_tcp_x_z) - x_z_success_margin
        x_z_caging = reward_utils.tolerance(tcp_obj_norm_x_z,
                                bounds=(0, x_z_success_margin),
                                margin=tcp_obj_x_z_margin,
                                sigmoid='long_tail',)

        assert right_caging >= 0 and right_caging <= 1
        gripper_closed = min(max(0, action[-1]), 1)
        assert gripper_closed >= 0 and gripper_closed <= 1
        caging = reward_utils.hamacher_product(y_caging, x_z_caging)
        assert caging >= 0 and caging <= 1

        # if caging > 0.95:
        #     gripping = y_gripping
        # else:
        #     gripping = 0.
        # assert gripping >= 0 and gripping <= 1
        # caging_and_gripping = reward_utils.hamacher_product(caging, gripping)
        # caging_and_gripping = (caging + gripping) / 2

        # assert caging_and_gripping >= 0 and caging_and_gripping <= 1


        return caging


    def compute_reward(self, action, obs):

        _TARGET_RADIUS = 0.05
        gripper = obs[:3]
        handle = obs[4:7]
        handle_pos_init = self.obj_init_pos

        under_handle = handle - np.array([0., 0., 0.01])
        under_handle_pos_init = handle_pos_init - np.array([0., 0., 0.01])

        gripping_reward = self._gripper_caging_reward(action, under_handle, 0.04)

        # scale = np.array([1., 3., 1.])
        # handle_error = (handle - self._target_pos) * scale
        # handle_error_init = (handle_pos_init - self._target_pos) * scale
        #
        # reward_for_opening = reward_utils.tolerance(
        #     np.linalg.norm(handle_error),
        #     bounds=(0, 0.02),
        #     margin=np.linalg.norm(handle_error_init),
        #     sigmoid='long_tail'
        # )

        # hand_to_goal_error = np.linalg.norm(gripper - self._target_pos)
        # htge_init = np.linalg.norm(self.init_tcp - self._target_pos)
        #
        # reach_to_goal = reward_utils.tolerance(
        #     hand_to_goal_error,
        #     bounds=(0, _TARGET_RADIUS),
        #     margin=htge_init,
        #     sigmoid='long_tail'
        # )


        # Emphasize XY error so that gripper is able to drop down and cage
        # handle without running into it. By doing this, we are assuming
        # that the reward in the Z direction is small enough that the agent
        # will be willing to explore raising a finger above the handle, hook it,
        # and drop back down to re-gain Z reward

        # scale = np.array([3., 3., 1.])
        # reach_error = np.linalg.norm((under_handle*scale) - (gripper*scale))
        # reach_error_init = np.linalg.norm((under_handle_pos_init*scale) - (self.init_tcp*scale))
        #
        # reach_reward = reward_utils.tolerance(
        #     reach_error,
        #     bounds=(0, 0.01),
        #     margin=reach_error_init,
        #     sigmoid='long_tail'
        # )

        lever_angle = -self.data.get_joint_qpos('doorjoint')
        lever_angle_desired = np.pi / 2.0
        lever_error = abs(lever_angle - lever_angle_desired)

        lever_engagement = reward_utils.tolerance(
            lever_error,
            bounds=(0, np.pi / 48.0),
            margin=(np.pi / 2.0),
            sigmoid='long_tail'
        )

        reward = 1.5 * gripping_reward

        if(gripping_reward > 0.95):
            gripping_reward = 1
            reward = 2 + (8 * lever_engagement)

        if np.linalg.norm(handle - self._target_pos) < _TARGET_RADIUS:
            reward = 10

        # print("REWARD: {} -- CAGING: {} -- OPENNING: {}".format(reward, gripping_reward, reward_for_opening))

        return (
            reward,
            np.linalg.norm(handle - gripper),
            obs[3],
            np.linalg.norm(handle - self._target_pos),
            gripping_reward,
            lever_engagement
        )
