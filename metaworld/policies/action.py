import numpy as np


class Action:
    """
    Represents an action to be taken in an environment.

    Once initialized, fields can be assigned as if the action
    is a dictionary. Once filled, the corresponding array is
    available as an instance variable.
    """
    def __init__(self, structure):
        """
        Args:
            structure (dict): Map from field names to output array indices
        """
        self._structure = structure
        self.array = np.zeros(len(self), dtype='float')

    def __len__(self):
        return sum([1 if isinstance(idx, int) else len(idx) for idx in self._structure.items()])

    def __getitem__(self, key):
        assert key in self._structure, 'This action\'s structure does not contain %s' % key
        return self.array[self._structure[key]]

    def __setitem__(self, key, value):
        assert key in self._structure, 'This action\'s structure does not contain %s' % key
        self.array[self._structure[key]] = value
