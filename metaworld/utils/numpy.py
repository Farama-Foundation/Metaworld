import numpy as np


def randint(rng: np.random.Generator, size=None) -> int | np.ndarray:
    """Returns a random integer from [0, 2**32 - 1] using the provided RNG."""
    return rng.integers(0, 2**32, size=size, dtype=np.uint32)
