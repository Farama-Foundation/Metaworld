import random

import numpy as np


class Solver:
    """
    Applies heuristics to place tools into a voxelspace
    (procedural environment generation)
    """
    def __init__(self, voxel_space, seed=None):
        """
        Args:
            voxel_space (VoxelSpace): The voxelspace to solve for
            seed (int): A seed for the numpy rng that's used for proc gen
        """
        self._voxel_space = voxel_space
        self._tools = []
        self.seed = seed
        self._rg = np.random.default_rng(seed=seed)

    @property
    def tools(self):
        """A list of tools that have been placed into the voxelspace"""
        return tuple(self._tools)

    def did_manual_set(self, tool):
        """
        Declare that a given tool's position was set manually, but allow the
        solver to update other pertinent state. Allows for interoperability
        between manually-positioned objects and proc gen objects

        Args:
            tool (Tool): The tool with the manually-set position

        """
        self._voxel_space.fill_tool(tool)
        self._tools.append(tool)

    def apply(self, heuristics, tools, tries=30, shuffle=True):
        """
        Apply a list of heuristics to a list of tools, optionally with multiple
        attempts and randomized ordering. This is the big "solving" aspect of
        the Solver.

        Args:
            heuristics (Heuristic or list of Heuristic): The heuristic(s) to
                apply
            tools (list of Tool): The tools to place in the voxelspace
            tries (int): The number of potential locations to generate for each
                tool. The location that results in the least object-object
                intersection will be chosen.
            shuffle (bool): Whether to randomize the order in which tools are
                placed (True) or not (False)

        """
        if not isinstance(heuristics, list):
            heuristics = [heuristics]

        has_no_overlap = True

        if shuffle:
            random.shuffle(tools)
        for tool in tools:
            if not tool.enabled:
                continue

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
