import numpy as np
from gymnasium.spaces import Box

from metaworld.envs.mujoco.arm_env import ArmEnv
from metaworld.envs.mujoco.mujoco_env import _assert_task_is_set


class UR5eEnv(ArmEnv):
    _ACTION_DIM = 7
    _QPOS_SPACE = Box(
        np.array(
            [
                -1.57,
                -3.14,
                -3.14,
                -6.28,
                -6.28,
                -6.28,
                0,
                -0.876,
                0,
                0,
                -0.876,
                0,
            ]
        ),
        np.array(
            [
                1.57,
                3.14,
                3.14,
                6.28,
                6.28,
                6.28,
                0.8,
                0.876,
                0.876,
                0.8,
                0.876,
                0.876,
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
            [
                -0.393,
                -1.29,
                2.420,
                -3.08,
                -1.38,
                -1.95,
                0,
                -0.24,
                -0.189,
                0,
                -0.24,
                -0.189,
            ]
        )
        self.gripper_pos = 0
        # self.init_finger_1 = self.get_body_com("jaco_link_finger_1")
        # self.init_finger_2 = self.get_body_com("jaco_link_finger_2")
        # self.init_finger_3 = self.get_body_com("jaco_link_finger_3")

        self.action_space = Box(
            np.array(
                [
                    -150,
                    -150,
                    -150,
                    -28,
                    -28,
                    -28,
                    -1,
                ]
            ),
            np.array(
                [
                    150,
                    150,
                    150,
                    28,
                    28,
                    28,
                    1,
                ]
            ),
            dtype=np.float64,
        )

        self.arm_col = [
            "wrist1_col",
            "wrist2_col",
            "wrist3_col",
            "wrist3_col2",
        ]

    @property
    def tcp_center(self):
        """The COM of the gripper's 3 fingers.

        Returns:
            (np.ndarray): 3-element position
        """
        finger_1, finger_2 = (
            self.data.body("left_inner_finger"),
            self.data.body("right_inner_finger"),
        )
        tcp_center = (finger_1.xpos + finger_2.xpos) / 2.0
        return tcp_center

    def set_action(self, action):
        """Applies the given action to the simulation.

        Args:
            action (np.ndarray): 9-element array of actions
        """

        gripper_speed = 0.01
        self.gripper_pos = np.clip(
            self.gripper_pos + gripper_speed * action[-1], 0, 0.8
        )
        parsed_action = np.hstack((action[:-1], self.gripper_pos, self.gripper_pos))
        self.do_simulation(parsed_action, n_frames=self.frame_skip)

    def gripper_effort_from_action(self, action):
        return action[-1]

    def get_action_penalty(self, action):
        action_cost_coff = 1e-3

        action_norm = np.linalg.norm(action)
        contact = self.check_contact_table()

        penalty = action_cost_coff * action_norm
        if contact:
            penalty -= 5

        return penalty
