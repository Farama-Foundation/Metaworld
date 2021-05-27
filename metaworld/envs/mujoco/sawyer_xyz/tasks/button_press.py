import numpy as np
from gym.spaces import Box

from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_state import SawyerXYZState
from metaworld.envs.mujoco.sawyer_xyz.tools import (
    ButtonBox, get_position_of, get_quat_of
)
from ._reward_primitives import (
    tolerance,
    hamacher_product as h_prod
)
from ._task import Task


class ButtonPress(Task):
    BUTTON_TRAVEL = 0.1

    def __init__(self):
        # The following properties are those that are necessary
        # to compute rewards, but aren't included in a SawyerXYZState
        # object.
        self._target_pos = None
        self._initial_pos_obj = None
        self._initial_pos_pads_center = None

    @property
    def random_reset_space(self) -> Box:
        return Box(
            np.zeros(2),
            np.ones(2),
        )

    def get_pos_objects(self, mjsim) -> np.ndarray:
        # TODO A bit of hack here...
        # Ideally target pos would be fully defined in reset_required_tools
        if self._target_pos is None:
            self._target_pos = mjsim.data.site_xpos[
                mjsim.model.site_name2id('ButtonBoxEnd')
            ].copy()

        return get_position_of('button', mjsim) + np.array([.0, .0, .193])

    def get_quat_objects(self, mjsim) -> np.ndarray:
        return get_quat_of('button', mjsim)

    def reset_required_tools(
            self,
            world,
            solver,
            random_reset_vec,
    ):
        button = ButtonBox()

        vec = random_reset_vec
        x = world.size[0] / 2.0 + 0.8 * (vec[0] - 0.5)
        y = world.size[1] / 8.0 + 0.3 * (vec[1] - 0.0)
        z = button.resting_pos_z

        button.specified_pos = np.array([x, y, z])
        solver.did_manual_set(button)

    def compute_reward(self, state: SawyerXYZState):
        if state.timestep == 1:
            self._initial_pos_obj = state.pos_objs[:3].copy()
            self._initial_pos_pads_center = state.pos_pads_center.copy()

        tcp = state.pos_pads_center
        obj = state.pos_objs[:3]
        tcp_closed = 1 - state.normalized_inter_pad_distance

        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(obj - self._initial_pos_pads_center)
        obj_to_target = abs(self._target_pos[2] - obj[2])

        near_button = tolerance(
            tcp_to_obj,
            bounds=(0, 0.01),
            margin=tcp_to_obj_init,
            sigmoid='long_tail',
        )
        button_pressed = tolerance(
            obj_to_target,
            bounds=(0, 0.005),
            margin=ButtonPress.BUTTON_TRAVEL,
            sigmoid='long_tail',
        )

        reward = 5 * h_prod(tcp_closed, near_button)
        if tcp_to_obj <= 0.03:
            reward += 5 * button_pressed

        return reward, {
            'success': float(obj_to_target <= 0.02),
            'near_object': float(tcp_to_obj <= 0.05),
            'grasp_success': float(tcp_closed > 0),
            'grasp_reward': near_button,
            'in_place_reward': button_pressed,
            'obj_to_target': obj_to_target,
            'unscaled_reward': reward,
        }
