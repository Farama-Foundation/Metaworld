from collections import OrderedDict
import numpy as np
from sawyer_control.envs.sawyer_env_base import SawyerEnvBase
from sawyer_control.core.serializable import Serializable

class SawyerXYPushingImgMultitaskEnv(SawyerEnvBase):
    ''' Warning: This environment will not work without being wrapped by ImageEnv'''
    def __init__(self,
                 fixed_goal=(1, 1, 1, 1),
                 pause_on_reset=True,
                 action_mode='position',
                 z=.23128,
                 **kwargs
                ):
        Serializable.quick_init(self, locals())
        SawyerEnvBase.__init__(self, action_mode=action_mode, **kwargs)
        self.goal_space = self.config.POSITION_SAFETY_BOX
        self._goal = None
        self.pause_on_reset=pause_on_reset
        self.z = z

    def step(self, action):
        self._act(action)
        observation = self._get_obs()
        reward = self.compute_reward(action, observation)
        info = self._get_info()
        done = False
        return observation, reward, done, info

    @property
    def goal_dim(self):
        return 4 #xy for object position, xy for end effector position

    def set_to_goal(self, goal):
        print('moving arm to desired object goal')
        obj_goal = np.concatenate((goal[:2], [self.z]))
        ee_goal = np.concatenate((goal[2:4], [self.z]))
        self._position_act(obj_goal-self._end_effector_pose()[:3])
        input()
        print('place object at end effector location and press enter')
        #MOVES ROBOT ARM TO GOAL POSITION:
        self._position_act(ee_goal - self._end_effector_pose()[:3])
        print('setting ee')
        self._goal = self._get_obs()

    def _reset_robot(self):
        self._act(self.reset_position - self._end_effector_pose()[:3])
        if self.pause_on_reset:
            input()
            print('move object to reset position and press enter')

    def reset(self):
        self._reset_robot()
        self._goal = self.sample_goal()
        return self._get_obs()

    def get_diagnostics(self, paths, prefix=''):
        return OrderedDict()

