import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerPickOutOfHoleEnvV2(SawyerXYZEnv):
    _TARGET_RADIUS = 0.02

    def __init__(self):
        hand_low = (-0.5, 0.40, -0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (0, 0.75, 0.02)
        obj_high = (0, 0.75, 0.02)
        goal_low = (-0.1, 0.5, 0.15)
        goal_high = (0.1, 0.6, 0.3)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([0, 0.6, 0.0]),
            'obj_init_angle': 0.3,
            'hand_init_pos': np.array([0., .6, .2]),
        }
        self.goal = np.array([0., 0.6, 0.2])
        self.obj_init_pos = None
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_pick_out_of_hole.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            tcp_to_obj,
            grasp_success,
            obj_to_target,
            grasp_reward,
            in_place_reward
        ) = self.compute_reward(action, obs)

        success = float(obj_to_target <= 0.07)
        near_object = float(tcp_to_obj <= 0.03)
        grasp_success = float(grasp_success)

        info = {
            'success': success,
            'near_object': near_object,
            'grasp_success': grasp_success,
            'grasp_reward': grasp_reward,
            'in_place_reward': in_place_reward,
            'obj_to_target': obj_to_target,
            'unscaled_reward': reward
        }

        return reward, info

    @property
    def _target_site_config(self):
        l = [('goal', self.init_right_pad)]
        if self.obj_init_pos is not None:
            l[0] = ('goal', self.obj_init_pos)
        return l

    @property
    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id('objGeom')

    def _get_pos_objects(self):
        return self.get_body_com('obj')

    def _get_quat_objects(self):
        return self.sim.data.get_body_xquat('obj')

    def reset_model(self):
        self._reset_hand()

        pos_obj = self.init_config['obj_init_pos']
        pos_goal = self.goal.copy()

        if self.random_init:
            pos_obj, pos_goal = np.split(self._get_state_rand_vec(), 2)
            while np.linalg.norm(pos_obj[:2] - pos_goal[:2]) < 0.15:
                pos_obj, pos_goal = np.split(self._get_state_rand_vec(), 2)

        self.obj_init_pos = pos_obj
        self._set_obj_xyz(self.obj_init_pos)
        self._target_pos = pos_goal

        return self._get_obs()

    def compute_reward(self, action, obs):
        obj = obs[4:7]
        gripper = self.tcp_center

        obj_to_target = np.linalg.norm(obj - self._target_pos)
        tcp_to_obj = np.linalg.norm(obj - gripper)
        in_place_margin = np.linalg.norm(self.obj_init_pos - self._target_pos)

        threshold = 0.03
        # floor is a 3D funnel centered on the initial object pos
        radius = np.linalg.norm(gripper[:2] - self.obj_init_pos[:2])
        if radius <= threshold:
            floor = 0.0
        else:
            floor = 0.015 * np.log(radius - threshold) + 0.15
        # prevent the hand from running into cliff edge by staying above floor
        above_floor = 1.0 if gripper[2] >= floor else reward_utils.tolerance(
            max(floor - gripper[2], 0.0),
            bounds=(0.0, 0.01),
            margin=0.02,
            sigmoid='long_tail',
        )
        object_grasped = self._gripper_caging_reward(
            action,
            obj,
            object_reach_radius=0.01,
            obj_radius=0.015,
            pad_success_thresh=0.02,
            xz_thresh=0.03,
            desired_gripper_effort=0.1,
            high_density=True
        )
        in_place = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, 0.02),
            margin=in_place_margin,
            sigmoid='long_tail'
        )
        reward = reward_utils.hamacher_product(object_grasped, in_place)

        near_object = tcp_to_obj < 0.04
        pinched_without_obj = obs[3] < 0.33
        lifted = obj[2] - 0.02 > self.obj_init_pos[2]
        # Increase reward when properly grabbed obj
        grasp_success = near_object and lifted and not pinched_without_obj
        if grasp_success:
            reward += 1. + 5. * reward_utils.hamacher_product(
                in_place, above_floor
            )
        # Maximize reward on success
        if obj_to_target < self.TARGET_RADIUS:
            reward = 10.

        return (
            reward,
            tcp_to_obj,
            grasp_success,
            obj_to_target,
            object_grasped,
            in_place,
        )
