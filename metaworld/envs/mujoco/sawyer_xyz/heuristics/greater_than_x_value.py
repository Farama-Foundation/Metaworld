import numpy as np

from .heuristic import Heuristic


class GreaterThanXValue(Heuristic):
    def __init__(self, x_value):
        self._x_value = x_value

    def apply_to(self, tool, voxel_space):
        res = voxel_space.resolution
        i_min = int(self._x_value * res)
        k = int(tool.resting_pos_z * res)

        unavailable = np.ones_like(voxel_space.mat[:, :, k])
        unavailable[i_min:, :] = False
        return unavailable, k
