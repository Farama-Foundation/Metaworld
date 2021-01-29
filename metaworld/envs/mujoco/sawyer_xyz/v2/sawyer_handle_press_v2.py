import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerHandlePressEnvV2(SawyerXYZEnv):
    TARGET_RADIUS=0.02

    def __init__(self):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.8, -0.001)
        obj_high = (0.1, 0.9, +0.001)
        goal_low = (-0.1, 0.55, 0.04)
        goal_high = (0.1, 0.70, 0.08)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([0, 0.9, 0.0]),
            'hand_init_pos': np.array( (0, 0.6, 0.2),),
        }
        self.goal = np.array([0, 0.8, 0.14])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_handle_press.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (reward,
        tcp_to_obj,
        _,
        target_to_obj,
        object_grasped,
        in_place) = self.compute_reward(action, obs)

        info = {
            'success': float(target_to_obj <= self.TARGET_RADIUS),
            'near_object': float(tcp_to_obj <= 0.05),
            'grasp_success': 1.,
            'grasp_reward': object_grasped,
            'in_place_reward': in_place,
            'obj_to_target': target_to_obj,
            'unscaled_reward': reward,
        }

        return reward, info

    @property
    def _target_site_config(self):
        return []

    def _get_pos_objects(self):
        return self._get_site_pos('handleStart')

    def _get_quat_objects(self):
        return np.zeros(4)

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        qvel[9] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()

        self.obj_init_pos = (self._get_state_rand_vec()
                             if self.random_init
                             else self.init_config['obj_init_pos'])

        self.sim.model.body_pos[self.model.body_name2id('box')] = self.obj_init_pos
        self._set_obj_xyz(-0.001)
        self._target_pos = self._get_site_pos('goalPress')
        self.maxDist = np.abs(self.data.site_xpos[self.model.site_name2id('handleStart')][-1] - self._target_pos[-1])
        self.target_reward = 1000*self.maxDist + 1000*2
        self._handle_init_pos = self._get_pos_objects()

        return self._get_obs()

    def compute_reward(self, actions, obs):
        del actions

        objPos = obs[4:7]
        obj = self._get_pos_objects()
        tcp = self.tcp_center
        target = self._target_pos.copy()
        
        target_to_obj = (obj[2] - target[2])
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = (self._handle_init_pos[2] - target[2])
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=abs(target_to_obj_init - self.TARGET_RADIUS),
            sigmoid='long_tail',
        )

        handle_radius = 0.02
        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(self._handle_init_pos - self.init_tcp)
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, handle_radius),
            margin=abs(tcp_to_obj_init - handle_radius),
            sigmoid='long_tail',
        )
        tcp_opened = 0
        object_grasped = reach

        reward = reward_utils.hamacher_product(reach, in_place)
        reward = 1 if target_to_obj <= self.TARGET_RADIUS else reward
        reward *= 10
        return (reward,
               tcp_to_obj,
               tcp_opened,
               target_to_obj,
               object_grasped,
               in_place)