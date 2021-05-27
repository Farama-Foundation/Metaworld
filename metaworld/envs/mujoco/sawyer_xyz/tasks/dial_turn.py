import numpy as np
from gym.spaces import Box

from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_state import SawyerXYZState
from metaworld.envs.mujoco.sawyer_xyz.tools import (
    Dial, get_position_of, get_quat_of, get_joint_pos_of
)
from ._reward_primitives import (
    tolerance,
    hamacher_product as h_prod,
    gripper_caging_reward
)
from ._task import Task


class DialTurn(Task):

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
            np.array([-.1, .4]),
            np.array([+.1, .6]),
        )

    def get_pos_objects(self, mjsim) -> np.ndarray:
        dial_center = get_position_of(Dial(), mjsim)
        dial_angle_rad = get_joint_pos_of('dialKnob', mjsim)

        offset = np.array([
            np.sin(dial_angle_rad),
            -np.cos(dial_angle_rad),
            0
        ])
        dial_radius = 0.05

        offset *= dial_radius

        return dial_center + offset

    def get_quat_objects(self, mjsim) -> np.ndarray:
        return get_quat_of(Dial(), mjsim)

    def reset_required_tools(
            self,
            world,
            solver,
            random_reset_vec,
    ):
        dial = Dial()

        x = world.size[0] / 2.0 + random_reset_vec[0]
        y = random_reset_vec[1] - 0.3
        z = dial.resting_pos_z

        dial.specified_pos = np.array([x, y, z])
        solver.did_manual_set(dial)

    def compute_reward(self, state: SawyerXYZState):
        if state.timestep == 1:
            self._initial_pos_obj = state.pos_objs[:3].copy()
            self._initial_pos_pads_center = state.pos_pads_center.copy()
            self._target_pos = self._initial_pos_obj + np.array([-.05, .05, 0])

        obj = state.pos_objs[:3]
        tcp = state.pos_pads_center
        target = self._target_pos.copy()

        dial_push_pos = obj + np.array([0.05, 0.02, 0.09])
        dial_push_pos_init = self._initial_pos_obj + np.array([0.05, 0.02, 0.09])

        target_to_obj = np.linalg.norm(obj - target)
        target_to_obj_init = np.linalg.norm(dial_push_pos_init - target)

        in_place = tolerance(
            target_to_obj,
            bounds=(0, 0.02),
            margin=abs(target_to_obj_init - 0.02),
            sigmoid='long_tail',
        )

        tcp_to_obj = np.linalg.norm(dial_push_pos - tcp)
        tcp_to_obj_init = np.linalg.norm(
            dial_push_pos_init - self._initial_pos_pads_center
        )
        reach = tolerance(
            tcp_to_obj,
            bounds=(0, 0.005),
            margin=abs(tcp_to_obj_init - 0.005),
            sigmoid='gaussian',
        )
        gripper_closed = min(max(0, state.action[-1]), 1)

        reach = h_prod(reach, gripper_closed)
        reward = 10 * h_prod(reach, in_place)

        return reward, {
            'success': float(target_to_obj <= 0.02),
            'near_object': float(tcp_to_obj <= 0.01),
            'grasp_success': 1.,
            'grasp_reward': reach,
            'in_place_reward': in_place,
            'obj_to_target': target_to_obj,
            'unscaled_reward': reward,
        }
