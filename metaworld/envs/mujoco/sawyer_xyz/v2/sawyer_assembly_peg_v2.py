import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerNutAssemblyEnvV2(SawyerXYZEnv):
    WRENCH_HANDLE_WIDTH = 0.04

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
    def _reward_grab_effort(actions):
        return (np.clip(actions[3], -1, 1) + 1.0) / 2.0

    @staticmethod
    def _reward_quat(obs):
        # Ideal laid-down wrench has quat [.707, 0, 0, .707]
        # Rather than deal with an angle between quaternions, just approximate:
        ideal = np.array([0.707, 0, 0, 0.707])
        error = np.linalg.norm(obs[7:11] - ideal)
        return max(1.0 - error/0.2, 0.0)

    @staticmethod
    def _reward_pos(obs, wrench_center, target_pos):
        hand = obs[:3]
        wrench = obs[4:7]

        # STEP 1: Preparing to lift the wrench ---------------------------------
        threshold = 0.02
        # floor is a 3D valley stretching across the wrench's handle
        radius_1 = abs(hand[1] - wrench[1])
        floor_1 = 0.0
        if radius_1 > threshold:
            floor_1 = 0.01 * np.log(radius_1 - threshold) + 0.1
        # prevent the hand from running into the handle prematurely by keeping
        # it above the "floor"
        above_floor_1 = 1.0 if hand[2] >= floor_1 else reward_utils.tolerance(
            floor_1 - hand[2],
            bounds=(0.0, 0.01),
            margin=floor_1 / 2.0,
            sigmoid='long_tail',
        )
        # grab the wrench's handle
        # grabbing anywhere along the handle is acceptable, subject to the
        # constraint that it remains level upon lifting (encouraged by the
        # reward_quat() function)
        pos_error_1 = hand - wrench
        if pos_error_1[0] <= SawyerNutAssemblyEnvV2.WRENCH_HANDLE_WIDTH / 2.0:
            pos_error_1[0] = 0.0
        in_place_1 = reward_utils.tolerance(
            np.linalg.norm(pos_error_1),
            bounds=(0, 0.02),
            margin=0.5,
            sigmoid='long_tail',
        )
        ready_to_lift = reward_utils.hamacher_product(above_floor_1, in_place_1)

        # STEP 2: Placing the wrench -------------------------------------------
        pos_error_2 = target_pos - wrench_center
        pos_error_2[2] = wrench_center[2]
        a = 0.2  # Relative importance of just *trying* to lift the wrench
        b = 0.8  # Relative importance of placing the wrench on the peg
        lifted = wrench[2] > 0.04 or np.linalg.norm(pos_error_2[:2]) < 0.02
        in_place_2 = a * float(lifted) + b * reward_utils.tolerance(
            np.linalg.norm(pos_error_2),
            bounds=(0, 0.02),
            margin=0.25,
            sigmoid='long_tail',
        )
        # prevent the wrench from running into the side of the peg by creating
        # a protective torus around it
        radius = np.linalg.norm(pos_error_2[:2])
        torus_radius = target_pos[2]  # Height of the peg
        torus_center = target_pos.copy()
        torus_center[2] = 0.0
        center_to_torus_center = torus_radius + 0.01

        floor_2 = np.sqrt(
            torus_radius ** 2 - (center_to_torus_center - radius) ** 2
        )
        if np.isnan(floor_2):
            floor_2 = 0.0
        above_floor_2 = 1.0 if wrench_center[2] >= floor_2 else \
            reward_utils.tolerance(
                floor_2 - wrench_center[2],
                bounds=(0.0, 0.01),
                margin=floor_2 / 2.0,
                sigmoid='long_tail',
            )

        on_peg = reward_utils.hamacher_product(above_floor_2, in_place_2)
        return ready_to_lift, on_peg

    def compute_reward(self, actions, obs):
        wrench_center = self._get_site_pos('RoundNut')

        reward_grab = SawyerNutAssemblyEnvV2._reward_grab_effort(actions)
        reward_quat = SawyerNutAssemblyEnvV2._reward_quat(obs)
        reward_steps = SawyerNutAssemblyEnvV2._reward_pos(
            obs,
            wrench_center,
            self._target_pos
        )

        # If first step is nearly complete...
        if reward_steps[0] > 0.9:
            # Begin incentivizing grabbing without obliterating existing reward
            # (min possible after conditional is 1.0, which was the max before)
            reward_grab = 2.0 - reward_grab
        # Rescale to [0,1]
        reward_grab /= 2.0

        reward = sum((
            2.0 * reward_utils.hamacher_product(reward_grab, reward_steps[0]),
            8.0 * reward_steps[1],
        ))

        # Override reward on success
        success = np.linalg.norm(wrench_center[:2] - self._target_pos[:2]) < 0.02 and \
            obs[6] < self._target_pos[2]
        if success:
            reward = 10.0

        # STRONG emphasis on proper wrench orientation
        reward *= reward_quat

        return (
            reward,
            reward_grab,
            *reward_steps,
            success,
        )
