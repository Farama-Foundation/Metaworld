import abc

import numpy as np


class VoxelSpace(abc.ABC):
    """
    Represents a 3D space in which Tools can be placed
    """
    def __init__(self, size, resolution):
        """
        Args:
            size (tuple of float): (Width, Length, Height) of the space
            resolution (int): Number of voxels per unit of size
        """
        self.size = size
        self.resolution = resolution
        # `mat` is a boolean matrix wherein `True` denotes filled space
        self.mat = np.zeros(
            list(map(lambda s: int(s * resolution), size)),
            dtype='bool'
        )

    @staticmethod
    def _fill(mat, ijk_bbox, value=True):
        """
        For a given `mat`, declares that all space within a bounding box is
        occupied (or not).
        Args:
            mat (np.ndarray): The matrix representation of a space
            ijk_bbox (tuple of int): Matrix indices defining the bounding box
                to fill. ijk_bbox[:3] is the lower-left corner; [3:] is the
                upper-right corner.
            value (bool): Whether the space is occupied (True) or not (False)
        """
        mat[
            ijk_bbox[0]:ijk_bbox[3],
            ijk_bbox[1]:ijk_bbox[4],
            ijk_bbox[2]:ijk_bbox[5]
        ] = value

    def fill(self, ijk_bbox, value=True):
        """
        Declare that all space within a bounding box is occupied (or not).
        Args:
            ijk_bbox (tuple of int): Matrix indices defining the bounding box
                to fill. ijk_bbox[:3] is the lower-left corner; [3:] is the
                upper-right corner.
            value (bool): Whether the space is occupied (True) or not (False)
        """
        VoxelSpace._fill(self.mat, ijk_bbox, value)

    def fill_tool(self, tool, value=True):
        """
        Declare whether or not a tool should be considered physically-present
        inside at its `.specified_pos` property, or if it's more of a ghost

        Args:
            tool (Tool): The tool to consider
            value (bool): Whether the tool is present (True) or not (False)
        """
        self.fill((
            tool.bbox * self.resolution + np.hstack([tool.specified_pos] * 2),
            value
        ).astype('uint32'))

    def pick_least_overlap(self, ijks, tool_bbox):
        """
        Given a list of potential positions and the volume required for a tool,
        pick the one that results in the least intersection with other objects
        in the VoxelSpace.

        ijks (list): A list of potential positions
        tool_bbox (tuple): A bounding box (6 floats) that define the
            corner-to-corner rectangular volume required by the tool
        """
        best_ijk = None
        best_bbox = None
        highest_volume = 0

        tool_voxels = np.zeros_like(self.mat)
        initial_volume = self.mat.sum()
        perfect = False

        for i in range(len(ijks)):
            ijk = ijks[i]
            # Scale tool's bbox to voxel space resolution (I, J, K)
            # and center it at `ijk`
            bbox = tool_bbox * self.resolution + np.hstack([ijk] * 2)
            bbox = bbox.clip(
                min=0,
                max=self.resolution * np.hstack([self.size] * 2)
            ).astype('uint32')

            # Fill tool voxels, check volume, and erase tool voxels.
            # This allows us to reuse the tool_voxels array rather than
            # recreating it for each iteration of the for loop
            VoxelSpace._fill(tool_voxels, bbox, True)
            volume = np.any([self.mat, tool_voxels], axis=0).sum()
            perfect = volume - initial_volume == tool_voxels.sum()
            VoxelSpace._fill(tool_voxels, bbox, False)

            # If this combo happens to create to the highest volume, save it
            if volume > highest_volume:
                highest_volume = volume
                best_ijk = ijk
                best_bbox = bbox

            # If this combo happens to have no overlaps, stop looking for more
            if perfect:
                break

        return best_ijk, best_bbox, highest_volume, perfect
