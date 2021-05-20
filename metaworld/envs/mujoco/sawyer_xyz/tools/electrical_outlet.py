import numpy as np

from .tool import Artifact


class ElectricalOutlet(Artifact):
    @property
    def bbox(self):
        return np.array([-.02, -.09, .0, .15, .09, .22])
