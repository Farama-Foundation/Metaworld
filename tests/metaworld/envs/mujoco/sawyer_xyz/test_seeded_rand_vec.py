import random

import pytest
import numpy as np

from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE


@pytest.mark.parametrize(
    'env_name',
    sorted(ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE.keys()))
def test_observations_match(env_name):
    seed = random.randrange(1000)
    env1 = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name](seed=seed)
    env1.seeded_rand_vec = True
    env2 = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name](seed=seed)
    env2.seeded_rand_vec = True

    obs1, obs2 = env1.reset(), env2.reset()
    assert (obs1 == obs2).all()

    for i in range(env1.max_path_length):
        a = np.random.uniform(low=-1, high=-1, size=4)
        obs1, r1, done1, _ = env1.step(a)
        obs2, r2, done2, _ = env2.step(a)
        assert (obs1 == obs2).all()
        assert r1 == r2
        assert not done1
        assert not done2
