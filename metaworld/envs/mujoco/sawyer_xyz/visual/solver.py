import random

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

    def did_manual_set(self, tool):
        self._voxel_space.fill_tool(tool)
        self._tools.append(tool)

    def apply(self, heuristics, tools, tries=30, shuffle=True):
        if not isinstance(heuristics, list):
            heuristics = [heuristics]

        has_no_overlap = True

        if shuffle:
            random.shuffle(tools)
        for tool in tools:
            masks = []
            ks = []
            for heuristic in heuristics:
                m, k = heuristic.apply_to(tool, self._voxel_space)
                masks.append(m)
                ks.append(k)

            ijs_available = Solver._ijs_available(np.any(masks, axis=0))
            ijs = self._rg.choice(ijs_available, tries)
            ijks = np.hstack((ijs, np.full((len(ijs), 1), max(ks))))

            ijk, bbox, vol, perfect = self._voxel_space.pick_least_overlap(
                ijks,
                tool.bbox
            )
            has_no_overlap &= perfect
            self._voxel_space.fill(bbox)
            tool.specified_pos = ijk / self._voxel_space.resolution

            self._tools.append(tool)

        return has_no_overlap

    @staticmethod
    def _ijs_available(mask):
        ij = [np.arange(i) for i in mask.shape]
        coords = np.dstack(np.meshgrid(*ij, indexing='ij'))
        return coords[~mask]
