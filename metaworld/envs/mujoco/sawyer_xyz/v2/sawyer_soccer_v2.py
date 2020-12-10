import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set



class SawyerSoccerEnvV2(SawyerXYZEnv):

    OBJ_RADIUS = 0.013
    TARGET_RADIUS=0.05

    def __init__(self):

        goal_low = (-0.1, 0.8, 0.0)
        goal_high = (0.1, 0.9, 0.0)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.6, 0.03)
        obj_high = (0.1, 0.7, 0.03)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([0, 0.6, 0.03]),
            'obj_init_angle': 0.3,
            'hand_init_pos': np.array([0., .6, .2]),
        }
        self.goal = np.array([0., 0.9, 0.03])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.max_path_length = 500

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_soccer.xml')

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
        self.curr_path_length += 1

        success = float(target_to_obj <= 0.07)
        near_object = float(tcp_to_obj <= 0.03)
        grasp_success = float(self.touching_object and (tcp_opened > 0) and (obj[2] - 0.02 > self.obj_init_pos[2]))
        info = {
            'success': success,
            'near_object': near_object,
            'grasp_success': grasp_success,
            'grasp_reward': object_grasped,
            'in_place_reward': in_place,
            'obj_to_target': target_to_obj,
            'unscaled_reward': reward,
        }

        return obs, reward, False, info

    def _get_pos_objects(self):
        return self.get_body_com('soccer_ball')

    def _get_quat_objects(self):
        return Rotation.from_matrix(
            self.data.get_body_xmat('soccer_ball')
        ).as_quat()

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.objHeight = self.get_body_com('soccer_ball')[2]

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            self._target_pos = goal_pos[3:]
            while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
                goal_pos = self._get_state_rand_vec()
                self._target_pos = goal_pos[3:]
            self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))
            self.sim.model.body_pos[self.model.body_name2id('goal_whole')] = self._target_pos

        self._set_obj_xyz(self.obj_init_pos)
        self.maxPushDist = np.linalg.norm(self.obj_init_pos[:2] - np.array(self._target_pos)[:2])

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand()
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

        object_grasped = self._gripper_caging_reward(action,
                                                     obj,
                                                     object_reach_radius=0.01,
                                                     obj_radius=self.OBJ_RADIUS,
                                                     pad_success_margin=0.06,
                                                     x_z_margin=0.01,
                                                     high_density=False)

        reward = 2 * reward_utils.hamacher_product(object_grasped, in_place)

        if tcp_to_obj < 0.01 and tcp_opened > 0 and object_grasped > 0.5:
            reward = max(1.5, 2*object_grasped) + 8. * in_place
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
