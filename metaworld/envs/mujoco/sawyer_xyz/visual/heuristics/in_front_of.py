import numpy as np

from .heuristic import Heuristic


class InFrontOf(Heuristic):
    def __init__(self, base_tool):
        self._base_tool = base_tool

    def apply_to(self, tool, voxel_space):
        assert(
            self._base_tool.specified_pos is not None,
            f'Can\'t apply InFrontOf heuristic until {self._base_tool.name}\'s pos is specified'
        )
        res = voxel_space.resolution
        base_ijk = self._base_tool.specified_pos * res
        base_bbox = self._base_tool.bbox * res
        tool_bbox = tool.bbox * res

        i_min = int(base_ijk[0] + base_bbox[0] - tool_bbox[0])
        i_max = int(base_ijk[0] + base_bbox[3] - tool_bbox[3])
        if i_max <= i_min:
            i_min = int(base_ijk[0])
            i_max = i_min + 1
        k = int(tool.resting_pos_z * res)

        unavailable = voxel_space.mat[:, :, k:k + int(tool_bbox[5])].sum(axis=2)
        j = unavailable[i_min:i_max].argmax(axis=1).min() - int(tool_bbox[4])
        mask = np.ones_like(unavailable)
        mask[i_min:i_max, j] = False

        return np.any([mask, unavailable], axis=0), k
