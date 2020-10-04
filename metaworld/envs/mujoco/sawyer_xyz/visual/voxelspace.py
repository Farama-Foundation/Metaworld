import abc

import numpy as np


class VoxelSpace(abc.ABC):
    """
    Represents a 3D space in which Tools can be placed
    """
    def __init__(self, size, resolution):
        """
        Args:
            size (tuple of float): (Length, Width, Height) of the space
            resolution (int): Number of voxels per unit of size
        """
        self.size = size
        self.resolution = resolution
        self.mat = np.zeros(
            list(map(lambda s: int(s * resolution), size)),
            dtype='int8'
        )

    @staticmethod
    def _fill(mat, ijk_bbox, value=True):
        mat[
            ijk_bbox[0]:ijk_bbox[3],
            ijk_bbox[1]:ijk_bbox[4],
            ijk_bbox[2]:ijk_bbox[5]
        ] += int(value) * 2 - 1

    def fill(self, ijk_bbox, value=True):
        VoxelSpace._fill(self.mat, ijk_bbox, value)

    def fill_tool(self, tool, value=True):
        self.fill((
            tool.bbox * self.resolution + np.hstack([tool.specified_pos] * 2)
        ).astype('uint32'), value)

    def pick_least_overlap(self, ijks, tool_bbox):
        best_ijk = None
        best_bbox = None
        highest_volume = 0

        tool_voxels = np.zeros_like(self.mat)

        for i in range(len(ijks)):
            ijk = ijks[i]
            # Scale tool's bbox to voxel space resolution (I, J, K)
            # and center it at `ijk`
            tool_bbox = tool_bbox * self.resolution + np.hstack([ijk] * 2)
            tool_bbox = tool_bbox.clip(
                min=0,
                max=self.resolution * np.hstack([self.size] * 2)
            ).astype('uint32')

            # Fill tool voxels, check volume, and erase tool voxels.
            # This allows us to reuse the tool_voxels array rather than
            # recreating it for each iteration of the for loop
            VoxelSpace._fill(tool_voxels, tool_bbox, True)
            volume = np.any([self.mat.astype('bool'), tool_voxels], axis=0).sum()
            VoxelSpace._fill(tool_voxels, tool_bbox, False)

            # If this combo happens to create to the highest volume, save it
            if volume > highest_volume:
                highest_volume = volume
                best_ijk = ijk
                best_bbox = tool_bbox

        return best_ijk, best_bbox, highest_volume
