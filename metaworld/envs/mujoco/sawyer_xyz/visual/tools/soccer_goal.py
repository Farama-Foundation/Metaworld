import numpy as np

from .tool import Artifact


class SoccerGoal(Artifact):
    @property
    def bbox(self):
        return np.array([-.1, -.06, .0, .1, .05, .16])
