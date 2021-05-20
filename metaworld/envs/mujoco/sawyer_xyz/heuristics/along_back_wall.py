import numpy as np

from .heuristic import Heuristic


class AlongBackWall(Heuristic):
    def __init__(self, y_value):
        self._y_value = y_value

    def apply_to(self, tool, voxel_space):
        res = voxel_space.resolution
        j_min = int(self._y_value * res)
        k = int(tool.resting_pos_z * res)

        binary_topography = voxel_space.mat.sum(axis=2).astype('bool')
        first_free_i_idx = binary_topography[:, j_min:].argmin(axis=0).max()
        i = first_free_i_idx - int(tool.bbox[0] * res)

        unavailable = np.ones_like(binary_topography)
        unavailable[min(i, unavailable.shape[0] - 1), j_min:] = False
        return unavailable, k
