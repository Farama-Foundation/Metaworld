import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerNutAssemblyEnvV2(SawyerXYZEnv):
    WRENCH_HANDLE_LENGTH = 0.02

    def __init__(self):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (0, 0.6, 0.02)
        obj_high = (0, 0.6, 0.02)
        goal_low = (-0.1, 0.75, 0.1)
        goal_high = (0.1, 0.85, 0.1)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': 0.3,
            'obj_init_pos': np.array([0, 0.6, 0.02], dtype=np.float32),
            'hand_init_pos': np.array((0, 0.6, 0.2), dtype=np.float32),
        }
        self.goal = np.array([0.1, 0.8, 0.1], dtype=np.float32)
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.liftThresh = 0.1
        self.max_path_length = 500

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_assembly_peg.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)

        (
            reward,
            reward_grab,
            reward_ready,
            reward_success,
            success
        ) = self.compute_reward(action, ob)

        info = {
            'success': float(success),
            'near_object': reward_ready,
            'grasp_success': reward_grab >= 0.5,
            'grasp_reward': reward_grab,
            'in_place_reward': reward_success,
            'obj_to_target': 0,
            'unscaled_reward': reward,
        }

        self.curr_path_length += 1
        return ob, reward, False, info

    @property
    def _target_site_config(self):
        return [('pegTop', self._target_pos)]

    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id('WrenchHandle')

    def _get_pos_objects(self):
        return self.data.site_xpos[self.model.site_name2id('RoundNut-8')]

    def _get_quat_objects(self):
        return self.sim.data.get_body_xquat('RoundNut')

    def _get_obs_dict(self):
        obs_dict = super()._get_obs_dict()
        obs_dict['state_achieved_goal'] = self.get_body_com('RoundNut')
        return obs_dict

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.objHeight = self.data.site_xpos[self.model.site_name2id('RoundNut-8')][2]
        self.heightTarget = self.objHeight + self.liftThresh

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.1:
                goal_pos = self._get_state_rand_vec()
            self.obj_init_pos = goal_pos[:3]
            self._target_pos = goal_pos[-3:]

        peg_pos = self._target_pos - np.array([0., 0., 0.05])
        self._set_obj_xyz(self.obj_init_pos)
        self.sim.model.body_pos[self.model.body_name2id('peg')] = peg_pos
        self.sim.model.site_pos[self.model.site_name2id('pegTop')] = self._target_pos
        self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) - np.array(self._target_pos)) + self.heightTarget

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand()
        self.init_tcp = self.tcp_center
        self.init_left_pad = self.get_body_com('leftpad')
        self.init_right_pad = self.get_body_com('rightpad')

    @staticmethod
    def _reward_quat(obs):
        # Ideal laid-down wrench has quat [.707, 0, 0, .707]
        # Rather than deal with an angle between quaternions, just approximate:
        ideal = np.array([0.707, 0, 0, 0.707])
        error = np.linalg.norm(obs[7:11] - ideal)
        return max(1.0 - error/0.2, 0.0)

    @staticmethod
    def _reward_pos(wrench_center, target_pos):
        pos_error = target_pos - wrench_center
        scale = np.array([3., 3., 1.])
        a = 0.1  # Relative importance of just *trying* to lift the wrench
        b = 0.9  # Relative importance of placing the wrench on the peg
        lifted = wrench_center[2] > 0.02 or np.linalg.norm(pos_error[:2]) < 0.02
        in_place = a * float(lifted) + b * reward_utils.tolerance(
            np.linalg.norm(pos_error * scale),
            bounds=(0, 0.05),
            margin=0.2,
            sigmoid='long_tail',
        )
        # prevent the wrench from running into the side of the peg by creating
        # a protective torus around it. modify input to torus function by sqrt()
        # in order to stretch the torus out
        radius = np.linalg.norm(pos_error[:2]) ** 0.5
        torus_radius = target_pos[2] * 1.2  # torus is slightly taller than peg
        center_to_torus_center = (torus_radius + 0.02) ** 0.5

        floor = target_pos[2] + np.sqrt(
            torus_radius ** 2 - (center_to_torus_center - radius) ** 2
        )
        if np.isnan(floor):
            floor = 0.0
        above_floor = 1.0 if wrench_center[2] >= floor else \
            reward_utils.tolerance(
                floor - wrench_center[2],
                bounds=(0.0, 0.01),
                margin=floor / 2.0,
                sigmoid='long_tail',
            )
        return reward_utils.hamacher_product(above_floor, in_place)

    def compute_reward(self, actions, obs):
        hand = obs[:3]
        wrench = obs[4:7]
        wrench_center = self._get_site_pos('RoundNut')
        # `self._gripper_caging_reward` assumes that the target object can be
        # approximated as a sphere. This is not true for the wrench handle, so
        # to avoid re-writing the `self._gripper_caging_reward` we pass in a
        # modified wrench position.
        # This modified position's X value will perfect match the hand's X value
        # as long as it's within a certain threshold
        wrench_threshed = wrench.copy()
        threshold = SawyerNutAssemblyEnvV2.WRENCH_HANDLE_LENGTH / 2.0
        if abs(wrench[0] - hand[0]) < threshold:
            wrench_threshed[0] = hand[0]

        reward_quat = SawyerNutAssemblyEnvV2._reward_quat(obs)
        reward_grab = self._gripper_caging_reward(
            actions, wrench_threshed,
            object_reach_radius=0.01,
            obj_radius=0.015,
            pad_success_margin=0.05,
            x_z_margin=0.01,
            high_density=True,
        )
        reward_in_place = SawyerNutAssemblyEnvV2._reward_pos(
            wrench_center,
            self._target_pos
        )

        reward = 2.0 * reward_grab + 8.0 * reward_in_place * reward_quat

        # Override reward on success
        aligned = np.linalg.norm(wrench_center[:2] - self._target_pos[:2]) < .02
        hooked = obs[6] < self._target_pos[2]
        success = aligned and hooked
        if success:
            reward = 10.0

        return (
            reward,
            reward_grab,
            reward_quat,
            reward_in_place,
            success,
        )
