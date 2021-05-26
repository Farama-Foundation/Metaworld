import abc

import numpy as np
from gym.spaces import Box

from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_state import SawyerXYZState


class Task(abc.ABC):

    # @abc.abstractmethod
    # @property
    # def id_main_object(self) -> str:
    #     pass

    @property
    @abc.abstractmethod
    def random_reset_space(self) -> Box:
        pass

    @abc.abstractmethod
    def get_pos_objects(self, mjsim) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_quat_objects(self, mjsim) -> np.ndarray:
        pass

    @abc.abstractmethod
    def reset_required_tools(
            self,
            world,
            solver,
            opt_rand_state_vec,  # TODO deprecate?
            opt_rand_init=True  # TODO deprecate?
    ):
        """
        Configures the minimal requisite set of tools for the task to be
        accomplished

        Args:
            world (VoxelSpace): The space into which tools should be placed
                (should match the private `._voxel_space` member of the solver)
            solver (Solver): The solver that will be in charge of proc gen after
                this manual placement is complete
            opt_rand_state_vec (np.ndarray): A list of positions
            opt_rand_init (boolean): Whether to use positions in opt_rand_state_vec

        """
        pass

    @abc.abstractmethod
    def evaluate_state(self, state: SawyerXYZState) -> (float, dict):
        """Evaluates an environment's current state relative to the task.
        Returns the reward as well as a dict containing the success flag and
        any other pertinent metrics

        Args:
            state: A state-based observation (NOT visual obs)

        Returns:
            (The raw reward, The debugging info dict)

        """
        pass
