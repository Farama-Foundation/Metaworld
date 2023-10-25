import numpy as np
from gymnasium.spaces import Box

from metaworld.envs.mujoco.arm_env import ArmEnv
from metaworld.envs.mujoco.mujoco_env import _assert_task_is_set


class JacoEnv(ArmEnv):
    _ACTION_DIM = 9
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

        self.hand_init_qpos = np.array([1.6, 0.03, 1.57, 1.63, 0.135, -0.05, 0, 0, 0])
        self.init_finger_1 = self.get_body_com("jaco_link_finger_1")
        self.init_finger_2 = self.get_body_com("jaco_link_finger_2")
        self.init_finger_3 = self.get_body_com("jaco_link_finger_3")

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
        finger_1, finger_2, finger_3 = (
            self.data.site("finger_1_tip"),
            self.data.site("finger_2_tip"),
            self.data.site("finger_3_tip"),
        )
        tcp_center = (finger_1.xpos + finger_2.xpos + finger_3.xpos) / 3.0
        return tcp_center

    def set_action(self, action):
        """Applies the given action to the simulation.

        Args:
            action (np.ndarray): 9-element array of actions
        """

        self.do_simulation(action, n_frames=self.frame_skip)

    def gripper_effort_from_action(self, action):
        return np.mean(action[-3:])
