import numpy as np

from ._heuristic import Heuristic


class Ceiling(Heuristic):
    """
    Attempts to place tools such that they don't "bump their heads" on some
    imaginary ceiling. If the scene's camera is centrally located, then a
    ceiling whose height increases radially can help avoid most occlusions.
    """
    def __init__(self, func_of_i_and_j):
        self._ceil_func = func_of_i_and_j

    def apply_to(self, tool, voxel_space):
        ij = [np.arange(i) for i in voxel_space.mat.shape[:2]]
        i_grid, j_grid = np.meshgrid(*ij, indexing='ij')
        k_max = self._ceil_func(i_grid, j_grid)

        unavailable = voxel_space.mat.copy()
        unavailable[
            np.arange(unavailable.shape[2]) > k_max[:, :, np.newaxis]
        ] = True

        res = voxel_space.resolution
        tool_height = int((tool.resting_pos_z + tool.bbox[5]) * res)
        binary_topography = unavailable[:, :, :tool_height].sum(axis=2)

        return binary_topography.astype('bool'), tool.resting_pos_z * res
