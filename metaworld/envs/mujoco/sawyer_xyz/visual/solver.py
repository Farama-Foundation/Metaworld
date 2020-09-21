import numpy as np


class Solver:
    def __init__(self, voxel_space, seed=None):
        self._voxel_space = voxel_space
        self._tools = []
        self.seed = seed
        self._rg = np.random.default_rng(seed=seed)

    @property
    def tools(self):
        return tuple(self._tools)

    def apply(self, heuristic, tools):
        for tool in tools:
            heuristic.apply_to(tool, self._voxel_space, self._rg)
            self._tools.append(tool)
