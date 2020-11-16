import numpy as np

from .tool import Tool


class ToasterHandle(Tool):
    @property
    def bbox(self):
        return np.array([-.1, -.2, .0, .1, .1, .23])
