import numpy as np

from .tool import Artifact


class Window(Artifact):
    @property
    def bbox(self):
        return np.array([-.19, -.15, .0, .19, .05, .49])
