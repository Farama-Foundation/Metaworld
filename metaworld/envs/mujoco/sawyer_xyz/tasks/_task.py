import abc

import numpy as np
from gym.spaces import Box

from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_state import SawyerXYZState


class Task(abc.ABC):

    @property
    @abc.abstractmethod
    def random_reset_space(self) -> Box:
        pass

    @abc.abstractmethod
    def get_pos_objects(self, mjsim) -> np.ndarray:
        """
        Returns positions of objects pertinent to the task. Since the proc gen
        env wrapper knows which tools are involved in the task, it could compute
        some standard positions on its own. But *useful* positions often involve
        rather bespoke transformations for each task, so it ends up here.

        Note: These positions will be passed through to `compute_reward`

        Args:
            mjsim (Mujoco.MjSim): The active Mujoco simulation object
        """
        pass

    @abc.abstractmethod
    def get_quat_objects(self, mjsim) -> np.ndarray:
        """
        Similar to `get_pos_objects` but for quaternions. Again, this can be
        highly task-specific.

        Note: These quaternions will be passed through to `compute_reward`

        Args:
            mjsim (Mujoco.MjSim): The active Mujoco simulation object
        """
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
    def compute_reward(self, state: SawyerXYZState) -> (float, dict):
        """Evaluates an environment's current state relative to the task.
        Returns the reward as well as a dict containing the success flag and
        any other pertinent metrics

        Args:
            state: A state-based observation (NOT visual obs)

        Returns:
            (The raw reward, The debugging info dict)

        """
        pass
