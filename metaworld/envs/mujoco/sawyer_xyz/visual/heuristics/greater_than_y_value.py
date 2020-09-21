import numpy as np

from .heuristic import Heuristic


class GreaterThanYValue(Heuristic):
    def __init__(self, y_value, tries=20):
        self._y_value = y_value
        self._tries = tries

    def apply_to(self, tool, voxel_space, rg):
        """Applies the heuristic

        Args:
            tool (Tool): The tool to which the heuristic should be applied
            voxel_space (VoxelSpace): The space into which the tool should
                be placed
            rg (Generator): Numpy random number generator
            tries (int): Sample this many points from the subset of points
                with `y > y_value`. The point that results in the least inter-
                body overlap will be selected

        """
        res = voxel_space.resolution
        y_idx = self._y_value * res
        # Set all voxels in front of y_idx to True (occupied)
        occupied = voxel_space.mat.copy()  # Y, X, Z
        occupied[:int(y_idx), :, :] = True
        # Get voxels on object's ground plane
        covered = occupied[:, :, int(tool.resting_pos_z * res)]  # Y, X

        ij = [np.arange(i) for i in covered.shape]
        coords = np.dstack(np.meshgrid(*ij, indexing='ij'))
        coords_available = coords[~covered]
        coords_chosen = rg.choice(coords_available, self._tries)

        coords_best = None
        volume_best = 0
        bbox_best = None
        for i in range(self._tries):
            padded = np.pad(np.roll(coords_chosen[i], 1), (0, 1), constant_values=tool.resting_pos_z * res)  # X, Y, Z
            bbox = tool.bbox * res + np.hstack([padded] * 2)
            bbox = bbox.clip(
                min=0,
                max=res * np.hstack(
                    [voxel_space.size[:2][::-1], voxel_space.size[2]] * 2
                )
            ).astype('uint32')

            mat = voxel_space.mat.copy()
            mat[
                bbox[1]:bbox[4],
                bbox[0]:bbox[3],
                bbox[2]:bbox[5]
            ] = True

            volume = mat.sum()
            if volume > volume_best:
                coords_best = coords_chosen[i]
                volume_best = volume
                bbox_best = bbox

        voxel_space.mat[
            bbox_best[1]:bbox_best[4],
            bbox_best[0]:bbox_best[3],
            bbox_best[2]:bbox_best[5]
        ] = True

        # Roll so that order is (X, Y)
        coords_best = np.roll(coords_best, 1)
        # Offset x value so that 0 is at center of table
        coords_best[0] -= voxel_space.mat.shape[1] // 2
        tool.specified_pos = np.hstack((coords_best / res, tool.resting_pos_z))
