import numpy as np

from .tool import Artifact


class CoffeeMachine(Artifact):
    @property
    def bbox(self):
        return np.array([-.09, -.24, .0, .09, .1, 0.37])
