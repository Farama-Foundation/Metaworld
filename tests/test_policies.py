import numpy as np
import pytest

import metaworld
from metaworld.env_dict import ENV_NAMES

import gymnasium as gym

from tests.helpers import ExpertPolicyMetaworldAgent


@pytest.mark.parametrize("env_name", ENV_NAMES)
def test_policies(env_name):
    pass
