"""Regression cases for peg grasps and alignment near the workspace boundary."""

import pickle

import numpy as np
import pytest

from metaworld.env_dict import ALL_V3_ENVIRONMENTS
from metaworld.policies import ENV_POLICY_MAP
from metaworld.types import Task


@pytest.mark.parametrize(
    "rand_vec",
    [
        [
            0.19721970269603983,
            0.6178886441208498,
            0.02,
            -0.34493546735562736,
            0.6649458959115495,
            0.0006824282910221063,
        ],
        [
            0.09267669717326395,
            0.5875304729448845,
            0.02,
            -0.3449831156501311,
            0.6645887283495605,
            0.0007041396756571139,
        ],
    ],
    ids=["grasp-offset", "leftmost-hole"],
)
def test_peg_insertion_policy_completes_boundary_tasks(rand_vec):
    """The reference policy must complete known difficult MT50 task poses."""
    name = "peg-insert-side-v3"
    env_cls = ALL_V3_ENVIRONMENTS[name]
    env = env_cls()
    try:
        env.set_task(
            Task(
                env_name=name,
                data=pickle.dumps(
                    {
                        "env_cls": env_cls,
                        "rand_vec": np.array(rand_vec, dtype=np.float64),
                        "partially_observable": False,
                    }
                ),
            )
        )
        obs, _ = env.reset(seed=42)
        policy = ENV_POLICY_MAP[name]()
        success = False
        for _ in range(300):
            obs, _, _, _, info = env.step(policy.get_action(obs))
            if info["success"]:
                success = True
                break
        assert success, f"Peg insertion failed for task pose {rand_vec}"
    finally:
        env.close()
