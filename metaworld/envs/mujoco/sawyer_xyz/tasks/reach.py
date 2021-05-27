import numpy as np
from gym.spaces import Box

from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_state import SawyerXYZState
from ._reward_primitives import (
    tolerance
)
from ._task import Task


class Reach(Task):
    def __init__(self):
        # The following properties are those that are necessary
        # to compute rewards, but aren't included in a SawyerXYZState
        # object.
        self._target_pos = np.zeros(3)
        self._initial_pos_hand = None

    @property
    def random_reset_space(self) -> Box:
        return Box(
            np.array([-.1, .4, .1]),
            np.array([+.1, .6, .3]),
        )

    def get_pos_objects(self, mjsim) -> np.ndarray:
        return np.array([])

    def get_quat_objects(self, mjsim) -> np.ndarray:
        return np.array([])

    def reset_required_tools(
            self,
            world,
            solver,
            random_reset_vec,
    ):
        self._target_pos = random_reset_vec.copy()

    def compute_reward(self, state: SawyerXYZState):
        if state.timestep == 1:
            self._initial_pos_hand = state.pos_hand.copy()

        tcp_to_target = np.linalg.norm(state.pos_hand - self._target_pos)
        in_place_margin = np.linalg.norm(
            self._initial_pos_hand - self._target_pos
        )

        reward = 10 * tolerance(tcp_to_target,
                                bounds=(0, 0.05),
                                margin=in_place_margin,
                                sigmoid='long_tail', )

        return reward, {
            'success': float(tcp_to_target <= 0.05),
            'near_object': 0.0,
            'grasp_success': 0.0,
            'grasp_reward': 0.0,
            'in_place_reward': reward / 10.0,
            'obj_to_target': tcp_to_target,
            'unscaled_reward': reward
        }
