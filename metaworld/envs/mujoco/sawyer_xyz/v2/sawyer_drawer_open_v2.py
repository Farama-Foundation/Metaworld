import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerDrawerOpenEnvV2(SawyerXYZEnv):
    def __init__(self):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.9, 0.0)
        obj_high = (0.1, 0.9, 0.0)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': np.array([0.3, ], dtype=np.float32),
            'obj_init_pos': np.array([0., 0.9, 0.0], dtype=np.float32),
            'hand_init_pos': np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        goal_low = self.hand_low
        goal_high = self.hand_high

        

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.maxDist = 0.2
        self.target_reward = 1000 * self.maxDist + 1000 * 2

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_drawer.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            gripper_error,
            gripped,
            handle_error,
            caging_reward,
            opening_reward
        ) = self.compute_reward(action, obs)

        info = {
            'success': float(handle_error <= 0.03),
            'near_object': float(gripper_error <= 0.03),
            'grasp_success': float(gripped > 0),
            'grasp_reward': caging_reward,
            'in_place_reward': opening_reward,
            'obj_to_target': handle_error,
            'unscaled_reward': reward,
        }

        return reward, info

    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id('objGeom')

    def _get_pos_objects(self):
        return self.get_body_com('drawer_link') + np.array([.0, -.16, .0])

    def _get_quat_objects(self):
        return self.sim.data.get_body_xquat('drawer_link')

    def reset_model(self):
        self._reset_hand()
        self.prev_obs = self._get_curr_obs_combined_no_goal()

        # Compute nightstand position
        self.obj_init_pos = self._get_state_rand_vec() if self.random_init \
            else self.init_config['obj_init_pos']
        # Set mujoco body to computed position
        self.sim.model.body_pos[self.model.body_name2id(
            'drawer'
        )] = self.obj_init_pos
        # Set _target_pos to current drawer position (closed) minus an offset
        self._target_pos = self.obj_init_pos + np.array([.0, -.16 - self.maxDist, .09])

        return self._get_obs()

    def compute_reward(self, action, obs):
        gripper = obs[:3]
        handle = obs[4:7]

        handle_error = np.linalg.norm(handle - self._target_pos)

        reward_for_opening = reward_utils.tolerance(
            handle_error,
            bounds=(0, 0.02),
            margin=self.maxDist,
            sigmoid='long_tail'
        )

        handle_pos_init = self._target_pos + np.array([.0, self.maxDist, .0])
        # Emphasize XY error so that gripper is able to drop down and cage
        # handle without running into it. By doing this, we are assuming
        # that the reward in the Z direction is small enough that the agent
        # will be willing to explore raising a finger above the handle, hook it,
        # and drop back down to re-gain Z reward
        scale = np.array([3., 3., 1.])
        gripper_error = (handle - gripper) * scale
        gripper_error_init = (handle_pos_init - self.init_tcp) * scale

        reward_for_caging = reward_utils.tolerance(
            np.linalg.norm(gripper_error),
            bounds=(0, 0.01),
            margin=np.linalg.norm(gripper_error_init),
            sigmoid='long_tail'
        )

        reward = reward_for_caging + reward_for_opening
        reward *= 5.0

        return (
            reward,
            np.linalg.norm(handle - gripper),
            obs[3],
            handle_error,
            reward_for_caging,
            reward_for_opening
        )
