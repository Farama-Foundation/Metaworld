import numpy as np
from gym.spaces import Box

from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_state import SawyerXYZState
from metaworld.envs.mujoco.sawyer_xyz.tools import (
    Puck, get_position_of, get_quat_of
)
from ._reward_primitives import (
    tolerance,
    hamacher_product as h_prod,
    gripper_caging_reward
)
from ._task import Task


class PickPlace(Task):

    def __init__(self):
        # The following properties are those that are necessary
        # to compute rewards, but aren't included in a SawyerXYZState
        # object.
        self._target_pos = np.zeros(3)
        self._initial_pos_obj = None
        self._initial_pos_pads_center = None

    @property
    def random_reset_space(self) -> Box:
        return Box(
            np.array([-.1, .4, -.1, .4, .1]),
            np.array([+.1, .6, +.1, .6, .3]),
        )

    def get_pos_objects(self, mjsim) -> np.ndarray:
        return get_position_of(Puck(), mjsim)

    def get_quat_objects(self, mjsim) -> np.ndarray:
        return get_quat_of(Puck(), mjsim)

    def reset_required_tools(
            self,
            world,
            solver,
            opt_rand_state_vec,  # TODO deprecate?
            opt_rand_init=True  # TODO deprecate?
    ):
        puck = Puck()
        self._target_pos = np.array([0.1, 0.8, 0.2])
        x = 0.0
        y = 0.0
        z = puck.resting_pos_z
        if opt_rand_init:
            vec = opt_rand_state_vec
            self._target_pos = vec[2:]

            x = world.size[0] / 2.0 + vec[0]
            y = vec[1] - 0.3

        puck.specified_pos = np.array([x, y, z])
        solver.did_manual_set(puck)

    def compute_reward(self, state: SawyerXYZState):
        if state.timestep == 1:
            self._initial_pos_obj = state.pos_objs[:3].copy()
            self._initial_pos_pads_center = state.pos_pads_center.copy()

        tcp = state.pos_pads_center
        obj = state.pos_objs[:3]
        tcp_opened = state.normalized_inter_pad_distance

        obj_to_target = np.linalg.norm(obj - self._target_pos)
        tcp_to_obj = np.linalg.norm(obj - tcp)
        in_place_margin = (
            np.linalg.norm(self._initial_pos_obj - self._target_pos)
        )

        in_place = tolerance(obj_to_target,
                             bounds=(0, 0.05),
                             margin=in_place_margin,
                             sigmoid='long_tail', )

        object_grasped = gripper_caging_reward(
            state,
            self._initial_pos_obj,
            self._initial_pos_pads_center,
            obj_radius=0.015,
            pad_success_thresh=0.05,
            xz_thresh=0.005
        )
        in_place_and_object_grasped = h_prod(object_grasped, in_place)
        reward = in_place_and_object_grasped

        if tcp_to_obj < 0.02 and (tcp_opened > 0) and (
                obj[2] - 0.01 > self._initial_pos_obj[2]):
            reward += 1. + 5. * in_place
        if obj_to_target < 0.05:
            reward = 10.

        return reward, {
            'success': float(obj_to_target <= 0.05),
            'near_object': float(tcp_to_obj <= 0.03),
            'grasp_success': float(
                (tcp_opened > 0) and
                (obj[2] - 0.02 > self._initial_pos_obj[2])
            ),
            'grasp_reward': object_grasped,
            'in_place_reward': in_place,
            'obj_to_target': obj_to_target,
            'unscaled_reward': reward
        }
