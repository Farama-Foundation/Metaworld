import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerSweepEnvV2(SawyerXYZEnv):

    OBJ_RADIUS = 0.02

    def __init__(self):

        init_puck_z = 0.1
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1.0, 0.5)
        obj_low = (-0.1, 0.6, 0.02)
        obj_high = (0.1, 0.7, 0.02)
        goal_low = (.49, .6, 0.00)
        goal_high = (0.51, .7, 0.02)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos':np.array([0., 0.6, 0.02]),
            'obj_init_angle': 0.3,
            'hand_init_pos': np.array([0., .6, .2]),
        }
        self.goal = np.array([0.5, 0.65, 0.01])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.max_path_length = 200
        self.init_puck_z = init_puck_z

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_sweep_v2.xml')

    @_assert_task_is_set
    def step(self, action):
        obs = super().step(action)
        obj = obs[4:7]
        (
            reward,
            tcp_to_obj,
            tcp_opened,
            target_to_obj,
            object_grasped,
            in_place
        ) = self.compute_reward(action, obs)

        info = {
            'success': float(target_to_obj <= 0.05),
            'near_object': float(tcp_to_obj <= 0.03),
            'grasp_reward': object_grasped,
            'in_place_reward': in_place,
            'obj_to_target': target_to_obj,
            'unscaled_reward': reward,
        }
        self.curr_path_length += 1

        return obs, reward, False, info

    def _get_quat_objects(self):
        return self.data.get_body_xquat('obj')

    def _get_pos_objects(self):
        return self.get_body_com('obj')

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.objHeight = self.get_body_com('obj')[2]

        if self.random_init:
            obj_pos = self._get_state_rand_vec()
            self.obj_init_pos = np.concatenate((obj_pos[:2], [self.obj_init_pos[-1]]))
            self._target_pos[1] = obj_pos.copy()[1]

        self._set_obj_xyz(self.obj_init_pos)
        self.maxPushDist = np.linalg.norm(self.get_body_com('obj')[:-1] - self._target_pos[:-1])
        self.target_reward = 1000*self.maxPushDist + 1000*2

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand()

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.reachCompleted = False

    def compute_reward(self, action, obs):
        obj = obs[4:7]
        tcp_opened = obs[3]
        tcp_to_obj = np.linalg.norm(obj - self.tcp_center)
        target_to_obj = np.linalg.norm(obj - self._target_pos)
        target_to_obj_init = np.linalg.norm(obj - self.obj_init_pos)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=target_to_obj_init,
            sigmoid='long_tail',
        )

        object_grasped = self._gripper_caging_reward(action, obj, self.OBJ_RADIUS)
        reward = reward_utils.hamacher_product(object_grasped, in_place)



        if tcp_to_obj < 0.02 and tcp_opened > 0:
            reward += 1. + 5. * in_place
        if target_to_obj < self.TARGET_RADIUS:
            reward = 10.
        return (
            reward,
            tcp_to_obj,
            tcp_opened,
            target_to_obj,
            object_grasped,
            in_place
        )
