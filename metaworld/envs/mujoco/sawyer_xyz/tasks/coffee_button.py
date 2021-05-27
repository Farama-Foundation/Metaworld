import numpy as np
from gym.spaces import Box

from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_state import SawyerXYZState
from metaworld.envs.mujoco.sawyer_xyz.tools import (
    CoffeeMachine, get_position_of, get_quat_of
)
from ._reward_primitives import (
    tolerance,
    hamacher_product as h_prod,
    gripper_caging_reward
)
from ._task import Task


class CoffeeButton(Task):

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
            np.array([-.2, .8]),
            np.array([+.2, .9]),
        )

    def get_pos_objects(self, mjsim) -> np.ndarray:
        return mjsim.data.site_xpos[mjsim.model.site_name2id('buttonStart')]

    def get_quat_objects(self, mjsim) -> np.ndarray:
        return np.array([1., 0., 0., 0.])

    def reset_required_tools(
            self,
            world,
            solver,
            random_reset_vec,
    ):
        machine = CoffeeMachine()

        x = world.size[0] / 2.0 + random_reset_vec[0]
        y = random_reset_vec[1] - 0.3
        z = machine.resting_pos_z

        machine.specified_pos = np.array([x, y, z])
        solver.did_manual_set(machine)

    def compute_reward(self, state: SawyerXYZState):
        if state.timestep == 1:
            self._initial_pos_obj = state.pos_objs[:3].copy()
            self._initial_pos_pads_center = state.pos_pads_center.copy()
            self._target_pos = self._initial_pos_obj + np.array([.0, 0.03, .0])

        obj = state.pos_objs[:3]
        tcp = state.pos_pads_center
        tcp_closed = max(state.normalized_inter_pad_distance, 0.0)

        obj_to_target = abs(self._target_pos[1] - obj[1])
        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(obj - self._initial_pos_pads_center)

        near_button = tolerance(
            tcp_to_obj,
            bounds=(0, 0.05),
            margin=tcp_to_obj_init,
            sigmoid='long_tail',
        )
        button_pressed = tolerance(
            obj_to_target,
            bounds=(0, 0.005),
            margin=0.03,
            sigmoid='long_tail',
        )

        reward = 2 * h_prod(tcp_closed, near_button)
        if tcp_to_obj <= 0.05:
            reward += 8 * button_pressed

        return reward, {
            'success': float(obj_to_target <= 0.02),
            'near_object': float(tcp_to_obj <= 0.05),
            'grasp_success': float(state.normalized_inter_pad_distance > 0),
            'grasp_reward': near_button,
            'in_place_reward': button_pressed,
            'obj_to_target': obj_to_target,
            'unscaled_reward': reward,
        }
