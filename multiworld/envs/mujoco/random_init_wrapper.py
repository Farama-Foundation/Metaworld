import gym
import numpy as np

from multiworld.core.serializable import Serializable


INITIAL_CONFIGURATIONS_SPACE_DICT = {
    'reach': {
        'domain': 'hand_pos',
        'low': (-0.5, 0.40, 0.05),
        'high': (0.5, 1, 0.5),
    },
    'push': {
        'domain': 'obj_pos',
        'low': (-0.1, 0.6, 0.02),
        'high': (0.1, 0.7, 0.02),
    },
    'pickplace': {
        'domain': 'obj_pos',
        'low': (-0.1, 0.6, 0.02),
        'high': (0.1, 0.7, 0.02),
    },
    'door_open': {
        'domain': 'obj_pos',
        'low': (0., 0.85, 0.1),
        'high': (0.1, 0.95, 0.1),
    },
    'drawer_open': {
        'domain': 'obj_pos',
        'low': (-0.1, 0.9, 0.04),
        'high': (0.1, 0.9, 0.04),
    },
    'drawer_close': {
        'domain': 'obj_pos',
        'low': (-0.1, 0.9, 0.04),
        'high': (0.1, 0.9, 0.04),
    },
    'button_press_topdown': {
        'domain': 'obj_pos',
        'low': (-0.1, 0.8, 0.05),
        'high': (0.1, 0.9, 0.05),
    },
    'peg_insertion': {
        'domain': 'obj_pos',
        'low': (-0.1, 0.5, 0.02),
        'high': (0.1, 0.7, 0.02),
    },
    'window_open': {
        'domain': 'obj_pos',
        'low': (-0.1, 0.7, 0.16),
        'high': (0.1, 0.9, 0.16),
    },
    'window_close': {
        'domain': 'obj_pos',
        'low': (-0.1, 0.7, 0.16),
        'high': (0.1, 0.9, 0.16),
    },
}


def generate_random_init_configs(task_name, n_initial_configs=5):
    low = INITIAL_CONFIGURATIONS_SPACE_DICT[task_name]['low']
    high = INITIAL_CONFIGURATIONS_SPACE_DICT[task_name]['high']
    domain = INITIAL_CONFIGURATIONS_SPACE_DICT[task_name]['domain']

    inits = np.random.uniform(
        low=np.array(low),
        high=np.array(high),
        size=(n_initial_configs, len(low))
    ).tolist()

    return domain, inits


class RandomInitWrapper(gym.Wrapper, Serializable):

    '''
    Provide finite random initialization for environments
    
    NOTE:
        Please use this with environment argument random_init=False!!!!

    EXAMPLE USAGE:
    >>> from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place_6dof import SawyerReachPushPickPlace6DOFEnv
    >>> from multiworld.envs.mujoco.random_init_wrapper import generate_random_init_configs, RandomInitWrapper
    >>> env = RandomInitWrapper(SawyerReachPushPickPlace6DOFEnv(random_init=False), inits, domain)
    >>> env.reset()  # will set initial state first then call reset()
    '''

    def __init__(self, env, initial_configurations, domain='obj_pos'):
        assert not env.random_init,\
            "Please use this wrapper when setting environment's random_init as False!!"
        Serializable.quick_init(self, locals())
        super().__init__(env)
        self._domain= domain
        self._initial_configs = initial_configurations
        self._current_idx = 0
        self._n_initial_configs = len(self._initial_configs)

    def reset(self):
        '''Set initial configuration first'''
        self._current_idx = np.random.randint(0, self._n_initial_configs)
        if self._domain == 'hand_pos':
            self.env.hand_init_pos = np.array(self._initial_configs[self._current_idx])
        elif self._domain == 'obj_pos':
            self.env.obj_init_pos = np.array(self._initial_configs[self._current_idx])
        else:
            raise NotImplementedError
        return self.env.reset()
