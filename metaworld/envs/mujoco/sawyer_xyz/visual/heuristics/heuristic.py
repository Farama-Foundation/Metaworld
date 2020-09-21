import abc


class Heuristic(abc.ABC):
    @abc.abstractmethod
    def apply_to(self, tool, voxel_space, rg, tries):
        pass
