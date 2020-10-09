import numpy as np

from .tool import Tool


class ScrewEye(Tool):
    @property
    def bbox(self):
        return np.array([-.06, -.05, -.02, .17, .05, .02])

    @property
    def resting_pos_z(self):
        return 0.05
