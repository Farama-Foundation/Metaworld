import numpy as np
from gymnasium.spaces import Box

from metaworld.envs.mujoco.arm_env import ArmEnv
from metaworld.envs.mujoco.mujoco_env import _assert_task_is_set


class SawyerXYZEnv(ArmEnv):
    _ACTION_DIM = 8
    _QPOS_SPACE = Box(
        np.array(
            [
                -3.05,
                -3.8,
                -3.04,
                -3.04,
                -2.98,
                -2.98,
                -4.71,
                0,
                -0.03,
            ]
        ),
        np.array(
            [
                3.05,
                -0.5,
                3.04,
                3.04,
                2.98,
                2.98,
                4.71,
                0.04,
                0,
            ]
        ),
        dtype=np.float64,
    )
    _HAND_SPACE = Box(
        np.array(
            [
                -0.525,
                0.348,
                -0.0525,
                -1,
                -1,
                -1,
                -1,
            ]
        ),
        np.array(
            [
                +0.525,
                1.025,
                0.7,
                1,
                1,
                1,
                1,
            ]
        ),
        dtype=np.float64,
    )

    def __init__(
        self,
        model_name,
        frame_skip=5,
        hand_low=...,
        hand_high=...,
        mocap_low=None,
        mocap_high=None,
        action_scale=1 / 100,
        action_rot_scale=1,
        render_mode=None,
    ):
        super().__init__(
            model_name,
            frame_skip,
            hand_low,
            hand_high,
            mocap_low,
            mocap_high,
            action_scale,
            action_rot_scale,
            render_mode,
        )

        self.hand_init_qpos = np.array(
            [1.56, -1.47, -0.0609, 2.65, -0.09, 0.387, -1.74, 0, 0]
        )
        self.init_left_pad = self.get_body_com("leftpad")
        self.init_right_pad = self.get_body_com("rightpad")

        self.action_space = Box(
            np.ones(self._ACTION_DIM) * -1,
            np.ones(self._ACTION_DIM),
            dtype=np.float64,
        )

    @property
    def tcp_center(self):
        """The COM of the gripper's 2 fingers.

        Returns:
            (np.ndarray): 3-element position
        """
        right_finger_pos = self.data.site("rightEndEffector")
        left_finger_pos = self.data.site("leftEndEffector")
        tcp_center = (right_finger_pos.xpos + left_finger_pos.xpos) / 2.0
        return tcp_center

    # @property
    # def gripper_distance_apart(self):
    #     finger_right, finger_left = (
    #         self.data.body("rightclaw"),
    #         self.data.body("leftclaw"),
    #     )
    #     # the gripper can be at maximum about ~0.1 m apart.
    #     # dividing by 0.1 normalized the gripper distance between
    #     # 0 and 1. Further, we clip because sometimes the grippers
    #     # are slightly more than 0.1m apart (~0.00045 m)
    #     # clipping removes the effects of this random extra distance
    #     # that is produced by mujoco

    #     gripper_distance_apart = np.linalg.norm(finger_right.xpos - finger_left.xpos)
    #     gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0.0, 1.0)

    #     return gripper_distance_apart

    def set_action(self, action):
        """Applies the given action to the simulation.

        Args:
            action (np.ndarray): 8-element array of actions
        """

        parsed_action = np.hstack((action, -action[-1]))
        self.do_simulation(parsed_action, n_frames=self.frame_skip)

    def gripper_effort_from_action(self, action):
        return action[-1]
