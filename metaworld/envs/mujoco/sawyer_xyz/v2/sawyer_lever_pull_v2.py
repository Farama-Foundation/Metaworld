import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerLeverPullEnvV2(SawyerXYZEnv):
    """
    Motivation for V2:
        V1 was impossible to solve because the lever would have to be pulled
        through the table in order to reach the target location.
    Changelog from V1 to V2:
        - (8/12/20) Updated to Byron's XML
        - (7/7/20) Added 3 element lever position to the observation
            (for consistency with other environments)
        - (6/23/20) In `reset_model`, changed `final_pos[2] -= .17` to `+= .17`
            This ensures that the target point is above the table.
    """
    LEVER_RADIUS = 0.2

    def __init__(self):

        hand_low = (-0.5, 0.40, -0.15)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.7, 0.0)
        obj_high = (0.1, 0.8, 0.0)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([0, 0.7, 0.0]),
            'hand_init_pos': np.array([0, 0.4, 0.2], dtype=np.float32),
        }
        self.goal = np.array([.12, 0.88, .05])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']
        self._lever_pos_init = None

        goal_low = self.hand_low
        goal_high = self.hand_high

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_lever_pull.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):

        (
            reward,
            shoulder_to_lever,
            ready_to_lift,
            lever_error,
            lever_engagement
        ) = self.compute_reward(action, obs)

        info = {
            'success': float(lever_error <= np.pi / 24),
            'near_object': float(shoulder_to_lever < 0.03),
            'grasp_success': float(ready_to_lift > 0.9),
            'grasp_reward': ready_to_lift,
            'in_place_reward': lever_engagement,
            'obj_to_target': shoulder_to_lever,
            'unscaled_reward': reward,
        }

        return reward, info

    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id('objGeom')

    def _get_pos_objects(self):
        return self._get_site_pos('leverStart')

    def _get_quat_objects(self):
        return Rotation.from_matrix(self.data.get_geom_xmat('objGeom')).as_quat()

    def reset_model(self):
        self._reset_hand()
        self.obj_init_pos = self._get_state_rand_vec() if self.random_init \
            else self.init_config['obj_init_pos']
        self.sim.model.body_pos[
            self.model.body_name2id('lever')] = self.obj_init_pos

        self._lever_pos_init = self.obj_init_pos + np.array(
            [.12, -self.LEVER_RADIUS, .25]
        )
        self._target_pos = self.obj_init_pos + np.array(
            [.12, .0, .25 + self.LEVER_RADIUS]
        )
        return self._get_obs()

    def compute_reward(self, action, obs):
        gripper = obs[:3]
        lever = obs[4:7]

        # De-emphasize y error so that we get Sawyer's shoulder underneath the
        # lever prior to bumping on against
        scale = np.array([4., 1., 4.])
        # Offset so that we get the Sawyer's shoulder underneath the lever,
        # rather than its fingers
        offset = np.array([.0, .055, .07])

        shoulder_to_lever = (gripper + offset - lever) * scale
        shoulder_to_lever_init = (
            self.init_tcp + offset - self._lever_pos_init
        ) * scale

        # This `ready_to_lift` reward should be a *hint* for the agent, not an
        # end in itself. Make sure to devalue it compared to the value of
        # actually lifting the lever
        ready_to_lift = reward_utils.tolerance(
            np.linalg.norm(shoulder_to_lever),
            bounds=(0, 0.02),
            margin=np.linalg.norm(shoulder_to_lever_init),
            sigmoid='long_tail',
        )

        # The skill of the agent should be measured by its ability to get the
        # lever to point straight upward. This means we'll be measuring the
        # current angle of the lever's joint, and comparing with 90deg.
        lever_angle = -self.data.get_joint_qpos('LeverAxis')
        lever_angle_desired = np.pi / 2.0

        lever_error = abs(lever_angle - lever_angle_desired)

        # We'll set the margin to 15deg from horizontal. Angles below that will
        # receive some reward to incentivize exploration, but we don't want to
        # reward accidents too much. Past 15deg is probably intentional movement
        lever_engagement = reward_utils.tolerance(
            lever_error,
            bounds=(0, np.pi / 48.0),
            margin=(np.pi / 2.0) - (np.pi / 12.0),
            sigmoid='long_tail'
        )

        target = self._target_pos
        obj_to_target = np.linalg.norm(lever - target)
        in_place_margin = (np.linalg.norm(self._lever_pos_init - target))

        in_place = reward_utils.tolerance(obj_to_target,
                                    bounds=(0, 0.04),
                                    margin=in_place_margin,
                                    sigmoid='long_tail',)

        # reward = 2.0 * ready_to_lift + 8.0 * lever_engagement
        reward = 10.0 * reward_utils.hamacher_product(ready_to_lift, in_place)
        return (
            reward,
            np.linalg.norm(shoulder_to_lever),
            ready_to_lift,
            lever_error,
            lever_engagement
        )
