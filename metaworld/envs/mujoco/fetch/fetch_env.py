import numpy as np
from gymnasium.spaces import Box

from metaworld.envs.mujoco.arm_env import ArmEnv
from metaworld.envs.mujoco.mujoco_env import _assert_task_is_set


class FetchEnv(ArmEnv):
    _ACTION_DIM = 8
    _QPOS_SPACE = Box(
        np.array(
            [
                -1.57,
                -4.04,
                -3.14,
                -1.57,
                -3.14,
                -3.14,
                0,
                0,
                0,
            ]
        ),
        np.array(
            [
                4.71,
                0.896,
                3.17,
                4.77,
                3.21,
                3.22,
                0.716,
                0.716,
                0.709,
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

        self.hand_init_qpos = np.array([0, -1.22, 0, 2.25, 0, 0.54, 0, 0.05, 0.05])
        self.init_left_pad = self.get_body_com("l_gripper_finger_link")
        self.init_right_pad = self.get_body_com("r_gripper_finger_link")

        self.action_space = Box(
            np.ones(self._ACTION_DIM) * -1,
            np.ones(self._ACTION_DIM),
            dtype=np.float64,
        )

    @property
    def tcp_center(self):
        """The COM of the gripper's 3 fingers.

        Returns:
            (np.ndarray): 3-element position
        """
        finger_1, finger_2 = (
            self.data.site("rightEndEffector"),
            self.data.site("leftEndEffector"),
        )
        tcp_center = (finger_1.xpos + finger_2.xpos) / 2.0
        return tcp_center

    def set_action(self, action):
        """Applies the given action to the simulation.

        Args:
            action (np.ndarray): 9-element array of actions
        """

        parsed_action = np.hstack((action, action[-1]))
        self.do_simulation(parsed_action, n_frames=self.frame_skip)

    def gripper_effort_from_action(self, action):
        return action[-1]
