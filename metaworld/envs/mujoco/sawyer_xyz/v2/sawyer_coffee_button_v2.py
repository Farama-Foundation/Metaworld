import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
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

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_coffee.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            near_button,
            button_pressed
        ) = self.compute_reward(action, obs)

        info = {
            'success': float(obj_to_target <= 0.02),
            'near_object': float(tcp_to_obj <= 0.05),
            'grasp_success': float(tcp_open > 0),
            'grasp_reward': near_button,
            'in_place_reward': button_pressed,
            'obj_to_target': obj_to_target,
            'unscaled_reward': reward,
        }

        return reward, info

    @property
    def _target_site_config(self):
        return [('coffee_goal', self._target_pos)]

    def _get_id_main_object(self):
        return None

    def _get_pos_objects(self):
        return self._get_site_pos('buttonStart')

    def _get_quat_objects(self):
        return np.array([1., 0., 0., 0.])

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

    def compute_reward(self, action, obs):
        del action
        obj = obs[4:7]
        tcp = self.tcp_center

        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(obj - self.init_tcp)
        obj_to_target = abs(self._target_pos[1] - obj[1])

        tcp_closed = max(obs[3], 0.0)
        near_button = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, 0.05),
            margin=tcp_to_obj_init,
            sigmoid='long_tail',
        )
        button_pressed = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, 0.005),
            margin=self.max_dist,
            sigmoid='long_tail',
        )

        reward = 2 * reward_utils.hamacher_product(tcp_closed, near_button)
        if tcp_to_obj <= 0.05:
            reward += 8 * button_pressed

        return (
            reward,
            tcp_to_obj,
            obs[3],
            obj_to_target,
            near_button,
            button_pressed
        )
