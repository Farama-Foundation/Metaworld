import numpy as np

from .heuristic import Heuristic


class OnTopOf(Heuristic):
    def __init__(self, base_tool):
        self._base_tool = base_tool

    def apply_to(self, tool, voxel_space, rg):
        assert(
            self._base_tool.specified_pos is not None,
            f'Can\'t apply OnTopOf heuristic until {self._base_tool.name}\'s pos is specified'
        )
        res = voxel_space.resolution

        base_ij = self._base_tool.specified_pos[:2] * res  # X, Y
        base_ij = np.roll(base_ij, 1)  # Y, X
        base_ij[1] += voxel_space.mat.shape[1] // 2

        base_bbox_ij = self._base_tool.bbox.reshape(2, 3)[:, :2]  # X, Y
        base_bbox_ij = np.roll(base_bbox_ij, 1, axis=1)  # Y, X
        base_bbox_ij *= res
        base_bbox_ij += base_ij
        base_bbox_ij = base_bbox_ij.clip(
            min=0,
            max=voxel_space.mat.shape[:2]
        ).astype('uint32')

        space_above_base = voxel_space.mat[
            base_bbox_ij[0][0]:base_bbox_ij[1][0],
            base_bbox_ij[0][1]:base_bbox_ij[1][1]
        ].copy()

        pos = np.zeros(3)
        pos[0] = base_ij[1]
        pos[1] = base_ij[0]
        pos[2] = space_above_base.sum(axis=2).mean() + tool.resting_pos_z * res

        bbox = tool.bbox * res + np.hstack([pos] * 2)
        bbox = bbox.clip(
            min=0,
            max=res * np.hstack(
                [voxel_space.size[:2][::-1], voxel_space.size[2]] * 2
            )
        ).astype('uint32')

        voxel_space.mat[
            bbox[1]:bbox[4],
            bbox[0]:bbox[3],
            bbox[2]:bbox[5]
        ] = True

        # Offset x value so that 0 is at center of table
        pos[0] -= voxel_space.mat.shape[1] // 2
        tool.specified_pos = pos / res
