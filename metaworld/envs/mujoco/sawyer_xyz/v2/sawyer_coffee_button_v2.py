import numpy as np
from gym.spaces import Box

from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerCoffeeButtonEnvV2(SawyerXYZEnv):

    def __init__(self):

        self.max_dist = 0.03

        hand_low = (-0.5, .4, 0.05)
        hand_high = (0.5, 1., 0.5)
        obj_low = (-0.1, 0.8, -.001)
        obj_high = (0.1, 0.9, +.001)
        # goal_low[3] would be .1, but objects aren't fully initialized until a
        # few steps after reset(). In that time, it could be .01
        goal_low = obj_low + np.array([-.001, -.22 + self.max_dist, .299])
        goal_high = obj_high + np.array([+.001, -.22 + self.max_dist, .301])

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([0, 0.9, 0.28]),
            'obj_init_angle': 0.3,
            'hand_init_pos': np.array([0., .4, .2]),
        }
        self.goal = np.array([0, 0.78, 0.33])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.max_path_length = 150

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.target_reward = 1000 * self.max_dist + 1000 * 2

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_coffee.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        reward, reachDist, pushDist = self.compute_reward(action, ob)
        self.curr_path_length += 1
        info = {
            'reachDist': reachDist,
            'goalDist': pushDist,
            'epRew': reward,
            'pickRew': None,
            'success': float(pushDist <= 0.02)
        }

        return ob, reward, False, info

    @property
    def _target_site_config(self):
        return [('coffee_goal', self._target_pos)]

    def _get_pos_objects(self):
        return self._get_site_pos('buttonStart')

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flatten()
        qvel = self.data.qvel.flatten()
        qpos[0:3] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()

        self.obj_init_pos = self._get_state_rand_vec() if self.random_init \
            else self.init_config['obj_init_pos']
        self.sim.model.body_pos[self.model.body_name2id(
            'coffee_machine'
        )] = self.obj_init_pos

        pos_mug = self.obj_init_pos + np.array([.0, -.22, .0])
        self._set_obj_xyz(pos_mug)

        pos_button = self.obj_init_pos + np.array([.0, -.22, .3])
        self._target_pos = pos_button + np.array([.0, self.max_dist, .0])

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand()
        self.reachCompleted = False

    def compute_reward(self, actions, obs):
        del actions

        objPos = obs[3:6]

        leftFinger = self._get_site_pos('leftEndEffector')
        fingerCOM  =  leftFinger

        pressGoal = self._target_pos[1]

        pressDist = np.abs(objPos[1] - pressGoal)
        reachDist = np.linalg.norm(objPos - fingerCOM)

        c1 = 1000
        c2 = 0.01
        c3 = 0.001
        if reachDist < 0.05:
            pressRew = 1000 * (self.max_dist - pressDist) + c1 * (np.exp(-(pressDist ** 2) / c2) + np.exp(-(pressDist ** 2) / c3))
        else:
            pressRew = 0

        pressRew = max(pressRew, 0)
        reward = -reachDist + pressRew

        return [reward, reachDist, pressDist]
