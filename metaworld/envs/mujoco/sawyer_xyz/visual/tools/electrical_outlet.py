import numpy as np

from .tool import Artifact


class ElectricalOutlet(Artifact):
    @property
    def bbox(self):
        return np.array([.0, -.09, .0, .24, .09, .22])
