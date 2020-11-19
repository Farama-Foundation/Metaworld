import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import _assert_task_is_set
from metaworld.envs.mujoco.sawyer_xyz.visual.visual_sawyer_sandbox_env import VisualSawyerSandboxEnv
from metaworld.envs.mujoco.sawyer_xyz.visual.tools import Puck

from .library import TOOLSETS


class PickPlace(VisualSawyerSandboxEnv):

    def __init__(self):
        super().__init__()
        self.init_config = {
            'hand_init_pos': self.hand_init_pos,
        }

        goal_low = (-1., -1., -1.)
        goal_high = (1., 1., 1.)

        self._target_pos = np.zeros(3)
        self._random_reset_space = Box(
            np.array([-.1, .4, -.1, .4, .1]),
            np.array([+.1, .6, +.1, .6, .3]),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self._toolset_required = TOOLSETS[type(self).__name__]
        self.randomize_extra_toolset(5)

    def _reset_required_tools(self, world, solver):
        puck = Puck()
        x = 0.0
        y = 0.0
        z = puck.resting_pos_z
        if self.random_init:
            vec = self._get_state_rand_vec()
            self._target_pos = vec[2:]

            x = world.size[0]/2.0 + vec[0]
            y = vec[1] - 0.3

        puck.specified_pos = np.array([x, y, z])
        solver.did_manual_set(puck)

        self.obj_init_pos = puck.specified_pos + np.array([
            -world.size[0]/2.0, +0.3, 0
        ])
        print(self.obj_init_pos)
        print(self._target_pos)

    @_assert_task_is_set
    def step(self, action):
        obs = super().step(action)
        obj = obs[4:7]

        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            grasp_reward,
            in_place_reward
        ) = self.compute_reward(action, obs)

        success = float(obj_to_target <= 0.02)
        near_object = float(tcp_to_obj <= 0.03)
        grasp_success = float(self.touching_main_object and (tcp_open > 0) and (
                    obj[2] - 0.02 > self.obj_init_pos[2]))

        info = {
            'success': success,
            'near_object': near_object,
            'grasp_success': grasp_success,
            'grasp_reward': grasp_reward,
            'in_place_reward': in_place_reward,
            'obj_to_target': obj_to_target,
            'unscaled_reward': reward
        }

        self.curr_path_length += 1
        return obs, reward, False, info

    def _get_pos_objects(self):
        return self.get_body_com('Puck')

    def _get_quat_objects(self):
        return self.sim.data.get_body_xquat('Puck')

    def _gripper_caging_reward(self, action, obj_position):
        pad_success_margin = 0.05
        x_z_success_margin = 0.005
        obj_radius = 0.015
        tcp = self.tcp_center
        left_pad = self.get_body_com('leftpad')
        right_pad = self.get_body_com('rightpad')
        delta_object_y_left_pad = left_pad[1] - obj_position[1]
        delta_object_y_right_pad = obj_position[1] - right_pad[1]
        right_caging_margin = abs(abs(obj_position[1] - self.init_right_pad[1])
                                  - pad_success_margin)
        left_caging_margin = abs(abs(obj_position[1] - self.init_left_pad[1])
                                 - pad_success_margin)

        right_caging = reward_utils.tolerance(delta_object_y_right_pad,
                                              bounds=(
                                              obj_radius, pad_success_margin),
                                              margin=right_caging_margin,
                                              sigmoid='long_tail', )
        left_caging = reward_utils.tolerance(delta_object_y_left_pad,
                                             bounds=(
                                             obj_radius, pad_success_margin),
                                             margin=left_caging_margin,
                                             sigmoid='long_tail', )

        y_caging = reward_utils.hamacher_product(left_caging,
                                                 right_caging)

        # compute the tcp_obj distance in the x_z plane
        tcp_xz = tcp + np.array([0., -tcp[1], 0.])
        obj_position_x_z = np.copy(obj_position) + np.array(
            [0., -obj_position[1], 0.])
        tcp_obj_norm_x_z = np.linalg.norm(tcp_xz - obj_position_x_z, ord=2)

        # used for computing the tcp to object object margin in the x_z plane
        init_obj_x_z = self.obj_init_pos + np.array(
            [0., -self.obj_init_pos[1], 0.])
        init_tcp_x_z = self.init_tcp + np.array([0., -self.init_tcp[1], 0.])
        tcp_obj_x_z_margin = np.linalg.norm(init_obj_x_z - init_tcp_x_z,
                                            ord=2) - x_z_success_margin

        x_z_caging = reward_utils.tolerance(tcp_obj_norm_x_z,
                                            bounds=(0, x_z_success_margin),
                                            margin=tcp_obj_x_z_margin,
                                            sigmoid='long_tail', )

        gripper_closed = min(max(0, action[-1]), 1)
        caging = reward_utils.hamacher_product(y_caging, x_z_caging)

        gripping = gripper_closed if caging > 0.97 else 0.
        caging_and_gripping = reward_utils.hamacher_product(caging,
                                                            gripping)
        return caging_and_gripping

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
                                          sigmoid='long_tail', )

        object_grasped = self._gripper_caging_reward(action, obj)
        in_place_and_object_grasped = reward_utils.hamacher_product(
            object_grasped,
            in_place)
        reward = in_place_and_object_grasped

        if tcp_to_obj < 0.02 and (tcp_opened > 0) and (
                obj[2] - 0.01 > self.obj_init_pos[2]):
            reward += 1. + 5. * in_place
        if obj_to_target < _TARGET_RADIUS:
            reward = 10.
        return [reward, tcp_to_obj, tcp_opened, obj_to_target, object_grasped,
                in_place]