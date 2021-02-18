import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerFaucetCloseEnvV2(SawyerXYZEnv):
    def __init__(self):

        hand_low = (-0.5, 0.40, -0.15)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.8, 0.0)
        obj_high = (0.1, 0.85, 0.0)
        self._handle_length = 0.175
        self._target_radius = 0.07

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([0, 0.8, 0.0]),
            'hand_init_pos': np.array([0., .4, .2])
        }
        self.hand_init_pos = self.init_config['hand_init_pos']
        self.obj_init_pos = self.init_config['obj_init_pos']

        goal_low = self.hand_low
        goal_high = self.hand_high

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_faucet.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (reward, tcp_to_obj, _, target_to_obj, object_grasped,
         in_place) = self.compute_reward(action, obs)

        info = {
            'success': float(target_to_obj <= 0.07),
            'near_object': float(tcp_to_obj <= 0.01),
            'grasp_success': 1.,
            'grasp_reward': object_grasped,
            'in_place_reward': in_place,
            'obj_to_target': target_to_obj,
            'unscaled_reward': reward,
        }

        return reward, info

    @property
    def _target_site_config(self):
        return [('goal_close', self._target_pos),
                ('goal_open', np.array([10., 10., 10.]))]

    def _get_quat_objects(self):
        return self.sim.data.get_body_xquat('faucetBase')

    def _get_pos_objects(self):
        return self._get_site_pos('handleStartClose') + np.array(
            [0., 0., -0.01])

    def reset_model(self):
        self._reset_hand()

        # Compute faucet position
        self.obj_init_pos = self._get_state_rand_vec() if self.random_init \
            else self.init_config['obj_init_pos']
        # Set mujoco body to computed position
        self.sim.model.body_pos[self.model.body_name2id(
            'faucetBase')] = self.obj_init_pos

        self._target_pos = self.obj_init_pos + np.array(
            [-self._handle_length, .0, .125])

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand()
        self.reachCompleted = False

    def compute_reward(self, action, obs):
        obj = obs[4:7]
        tcp = self.tcp_center
        target = self._target_pos.copy()

        target_to_obj = (obj - target)
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = (self.obj_init_pos - target)
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self._target_radius),
            margin=abs(target_to_obj_init - self._target_radius),
            sigmoid='long_tail',
        )

        faucet_reach_radius = 0.01
        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(self.obj_init_pos - self.init_tcp)
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, faucet_reach_radius),
            margin=abs(tcp_to_obj_init - faucet_reach_radius),
            sigmoid='gaussian',
        )

        tcp_opened = 0
        object_grasped = reach

        reward = 2 * reach + 3 * in_place
        reward *= 2
        reward = 10 if target_to_obj <= self._target_radius else reward

        return (reward, tcp_to_obj, tcp_opened, target_to_obj, object_grasped,
                in_place)
