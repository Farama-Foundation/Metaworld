import numpy as np

from .heuristic import Heuristic


class OnTopOf(Heuristic):
    def __init__(self, base_tool):
        self._base_tool = base_tool

    def apply_to(self, tool, voxel_space):
        assert(
            self._base_tool.specified_pos is not None,
            f'Can\'t apply OnTopOf heuristic until {self._base_tool.name}\'s pos is specified'
        )
        res = voxel_space.resolution
        base_ijk = self._base_tool.specified_pos * res
        base_bbox = self._base_tool.bbox * res
        base_top = base_ijk[2] + base_bbox[5]
        tool_bbox = tool.bbox * res

        i_min = int(base_ijk[0] + base_bbox[0] - tool_bbox[0])
        i_max = int(base_ijk[0] + base_bbox[3] - tool_bbox[3])
        i_max = max(i_min + 1, i_max)
        j = int(base_ijk[1])
        k = int(base_top + tool.resting_pos_z * res)

        unavailable = voxel_space.mat[:, :, int(base_top):].sum(axis=2)
        mask = np.ones_like(voxel_space.mat[:, :, k])
        mask[i_min:i_max, j] = False

        return np.any([mask, unavailable], axis=0), k
