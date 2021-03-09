import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerBasketballEnvV2(SawyerXYZEnv):
    PAD_SUCCESS_MARGIN = 0.06
    TARGET_RADIUS = 0.08

    def __init__(self):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.6, 0.0299)
        obj_high = (0.1, 0.7, 0.0301)
        goal_low = (-0.1, 0.85, 0.)
        goal_high = (0.1, 0.9+1e-7, 0.)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': .3,
            'obj_init_pos': np.array([0, 0.6, 0.03], dtype=np.float32),
            'hand_init_pos': np.array((0, 0.6, 0.2), dtype=np.float32),
        }
        self.goal = np.array([0, 0.9, 0])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(
            np.array(goal_low) + np.array([0, -0.083, 0.2499]),
            np.array(goal_high) + np.array([0, -0.083, 0.2501])
        )

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_basketball.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        obj = obs[4:7]

        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            grasp_reward,
            in_place_reward
        ) = self.compute_reward(action, obs)

        info = {
            'success': float(obj_to_target <= self.TARGET_RADIUS),
            'near_object': float(tcp_to_obj <= 0.05),
            'grasp_success': float(
                (tcp_open > 0) and
                (obj[2] - 0.03 > self.obj_init_pos[2])
            ),
            'grasp_reward': grasp_reward,
            'in_place_reward': in_place_reward,
            'obj_to_target': obj_to_target,
            'unscaled_reward': reward,
        }

        return reward, info

    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id('objGeom')

    def _get_pos_objects(self):
        return self.get_body_com('bsktball')

    def _get_quat_objects(self):
        return self.sim.data.get_body_xquat('bsktball')

    def reset_model(self):
        self._reset_hand()
        self.prev_obs = self._get_curr_obs_combined_no_goal()

        basket_pos = self.goal.copy()
        self.sim.model.body_pos[self.model.body_name2id('basket_goal')] = basket_pos
        self._target_pos = self.data.site_xpos[self.model.site_name2id('goal')]

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            basket_pos = goal_pos[3:]
            while np.linalg.norm(goal_pos[:2] - basket_pos[:2]) < 0.15:
                goal_pos = self._get_state_rand_vec()
                basket_pos = goal_pos[3:]
            self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))
            self.sim.model.body_pos[self.model.body_name2id('basket_goal')] = basket_pos
            self._target_pos = self.data.site_xpos[self.model.site_name2id('goal')]

        self._set_obj_xyz(self.obj_init_pos)
        return self._get_obs()

    def compute_reward(self, action, obs):
        obj = obs[4:7]
        # Force target to be slightly above basketball hoop
        target = self._target_pos.copy()
        target[2] = 0.3

        # Emphasize Z error
        scale = np.array([1., 1., 2.])
        target_to_obj = (obj - target) * scale
        target_to_obj = np.linalg.norm(target_to_obj)
        target_to_obj_init = (self.obj_init_pos - target) * scale
        target_to_obj_init = np.linalg.norm(target_to_obj_init)

        in_place = reward_utils.tolerance(
            target_to_obj,
            bounds=(0, self.TARGET_RADIUS),
            margin=target_to_obj_init,
            sigmoid='long_tail',
        )
        tcp_opened = obs[3]
        tcp_to_obj = np.linalg.norm(obj - self.tcp_center)

        object_grasped = self._gripper_caging_reward(
            action,
            obj,
            object_reach_radius=0.01,
            obj_radius=0.025,
            pad_success_thresh=0.06,
            xz_thresh=0.005,
            high_density=True
        )
        if tcp_to_obj < 0.035 and tcp_opened > 0 and \
                obj[2] - 0.01 > self.obj_init_pos[2]:
            object_grasped = 1
        reward = reward_utils.hamacher_product(object_grasped, in_place)

        if tcp_to_obj < 0.035 and tcp_opened > 0 and \
                obj[2] - 0.01 > self.obj_init_pos[2]:
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
