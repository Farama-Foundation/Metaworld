import numpy as np

from .heuristic import Heuristic


class LessThanYValue(Heuristic):
    def __init__(self, y_value):
        self._y_value = y_value

    def apply_to(self, tool, voxel_space):
        res = voxel_space.resolution
        j_max = int(self._y_value * res)
        k = int(tool.resting_pos_z * res)

        unavailable = np.ones_like(voxel_space.mat[:, :, k])
        unavailable[:, :j_max] = False
        return unavailable, k
