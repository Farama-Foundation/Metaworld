"""Regression coverage for MuJoCo versions with extended equality data."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from metaworld.sawyer_xyz_env import SawyerMocapBase


@pytest.mark.parametrize("width", [11, 12])
def test_reset_mocap_welds_preserves_additional_fields(width):
    eq_data = np.full((2, width), 2.0)
    model = SimpleNamespace(
        nmocap=1,
        eq_type=np.array([mujoco.mjtEq.mjEQ_WELD, mujoco.mjtEq.mjEQ_CONNECT]),
        eq_data=eq_data,
    )
    env = SimpleNamespace(model=model)

    SawyerMocapBase.reset_mocap_welds(env)

    np.testing.assert_array_equal(eq_data[0, :11], [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 5])
    np.testing.assert_array_equal(eq_data[0, 11:], 2)
    np.testing.assert_array_equal(eq_data[1], 2)
