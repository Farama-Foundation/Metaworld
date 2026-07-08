import typing

import numpy as np

from metaworld.utils import reward_utils


def test_rect_prism_tolerance_type_hints_resolvable():
    # np.float_ was removed in NumPy 2.0; resolving the annotations
    # must not raise AttributeError under numpy>=2.
    hints = typing.get_type_hints(reward_utils.rect_prism_tolerance)
    assert set(hints) >= {"curr", "zero", "one"}


def test_rect_prism_tolerance_values():
    zero = np.array([0.0, 0.0, 0.0])
    one = np.array([1.0, 1.0, 1.0])
    # corner with reward 1
    assert reward_utils.rect_prism_tolerance(one, zero, one) == 1.0
    # opposite corner -> 0
    assert reward_utils.rect_prism_tolerance(zero, zero, one) == 0.0
    # outside the prism -> 1.0 (no-collision-penalty semantics)
    assert (
        reward_utils.rect_prism_tolerance(np.array([2.0, 2.0, 2.0]), zero, one) == 1.0
    )
