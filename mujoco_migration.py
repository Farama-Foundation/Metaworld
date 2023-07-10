from metaworld import MT10
import gymnasium as gym
import pickle
import dill
import numpy as np
import copy

def data_equivalence(data_1, data_2) -> bool:
    """Assert equality between data 1 and 2, i.e observations, actions, info.

    Args:
        data_1: data structure 1
        data_2: data structure 2

    Returns:
        If observation 1 and 2 are equivalent
    """
    if type(data_1) == type(data_2):
        if isinstance(data_1, dict):
            return data_1.keys() == data_2.keys() and all(
                data_equivalence(data_1[k], data_2[k]) for k in data_1.keys()
            )
        elif isinstance(data_1, (tuple, list)):
            return len(data_1) == len(data_2) and all(
                data_equivalence(o_1, o_2) for o_1, o_2 in zip(data_1, data_2)
            )
        elif isinstance(data_1, np.ndarray):
            if data_1.shape == data_2.shape and data_1.dtype == data_2.dtype:
                if data_1.dtype == object:
                    return all(data_equivalence(a, b) for a, b in zip(data_1, data_2))
                else:
                    return np.allclose(data_1, data_2, atol=0.00001)
            else:
                return False
        else:
            return data_1 == data_2
    else:
        return False

def test_pickle_env(env: gym.Env):
    print(f'dumping {env}')
    dump = pickle.dumps(env)
    #print(f'loading {env}')
    pickled_env = pickle.loads(dump)
    r1 = env.reset()
    r2 = pickled_env.reset()
    print(data_equivalence(r1, r2))
    print(r1, r2)
    action = env.action_space.sample()
    s1 = env.step(action)
    s2 = pickled_env.step(action)
    print(data_equivalence(s1, s2))
    print(s1, s2)
    env.close()
    pickled_env.close()

mt10 = MT10()

for env in mt10.train_classes.values():
    e = env()
    #print()
    #print(e.data)
    e.reset()
    #print(e.data)
    # e.reset()
    # model = pickle.loads(pickle.dumps(e))
    test_pickle_env(e)
    #print(f'done {e}')
