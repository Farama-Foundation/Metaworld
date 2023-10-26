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
                -3.81,
                -3.04,
                -3.04,
                -2.98,
                -2.98,
                -4.71,
                -0.0115,
                -0.0208,
            ]
        ),
        np.array(
            [
                3.05,
                2.27,
                3.04,
                3.04,
                2.98,
                2.98,
                4.71,
                0.0208,
                0.0115,
            ]
        ),
        dtype=np.float64,
    )

    def __init__(
        self,
        model_name,
        hand_low=...,
        hand_high=...,
        mocap_low=None,
        mocap_high=None,
        render_mode=None,
    ):
        super().__init__(
            model_name=model_name,
            hand_low=hand_low,
            hand_high=hand_high,
            mocap_low=mocap_low,
            mocap_high=mocap_high,
            render_mode=render_mode,
        )

        self.hand_init_qpos = np.array(
            [0, -1.18, 0, 2.18, 0, 0.57, 3.3161, 0.0208, -0.0208]
        )
        self.init_left_pad = self.get_body_com("l_finger_tip")
        self.init_right_pad = self.get_body_com("r_finger_tip")

        self.action_space = Box(
            np.array([-80, -80, -40, -40, -9, -9, -9, -0.0115]),
            np.array([80, 80, 40, 40, 9, 9, 9, 0.020833]),
            dtype=np.float64,
        )

        self.arm_col = [
            "link0_collision",
            "link1_collision",
            "link2_collision",
            "link3_collision",
            "link4_collision",
            "link5_collision",
            "link6_collision",
            "right_l1_2",
            "right_l2_2",
            "right_l4_2",
        ]

    @property
    def tcp_center(self):
        """The COM of the gripper's 2 fingers.

        Returns:
            (np.ndarray): 3-element position
        """
        right_finger_pos = self.data.body("r_finger_tip").xpos
        left_finger_pos = self.data.body("l_finger_tip").xpos
        tcp_center = (right_finger_pos + left_finger_pos) / 2.0
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

    def get_action_penalty(self, action):
        action_cost_coff = 1e-3

        action_norm = np.linalg.norm(action)
        contact = self.check_contact_table()

        penalty = action_cost_coff * action_norm
        if contact:
            penalty = 5

        return penalty
