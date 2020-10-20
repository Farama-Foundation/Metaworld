import abc


class Heuristic(abc.ABC):
    @abc.abstractmethod
    def apply_to(self, tool, voxel_space):
        """Applies the heuristic

        Args:
            tool (Tool): The tool to which the heuristic should be applied
            voxel_space (VoxelSpace): The space into which the tool should
                be placed

        :rtype: (np.ndarray, int)

        """
        pass
