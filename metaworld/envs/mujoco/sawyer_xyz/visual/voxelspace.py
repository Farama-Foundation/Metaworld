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
            dtype='bool'
        )
