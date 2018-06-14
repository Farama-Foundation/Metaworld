import abc
import numpy as np
import mujoco_py

from multiworld.core.serializable import Serializable
from multiworld.envs.mujoco.mujoco_env import MujocoEnv

import copy


class SawyerMocapBase(MujocoEnv, Serializable, metaclass=abc.ABCMeta):
    """
    Provides some commonly-shared functions for Sawyer Mujoco envs that use
    mocap for XYZ control.
    """
    mocap_low = np.array([-0.2, 0.5, 0.06])
    mocap_high = np.array([0.2, 0.7, 0.6])

    def __init__(self, model_name, frame_skip=50):
        MujocoEnv.__init__(self, model_name, frame_skip=frame_skip)
        # Resets the mocap welds that we use for actuation.
        sim = self.sim
        if sim.model.nmocap > 0 and sim.model.eq_data is not None:
            for i in range(sim.model.eq_data.shape[0]):
                if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    # Define the xyz + quat of the mocap relative to the hand
                    sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.]
                    )

    def reset_mocap2body_xpos(self):
        # move mocap to weld joint
        self.data.set_mocap_pos(
            'mocap',
            np.array([self.data.get_body_xpos('hand')]),
        )
        self.data.set_mocap_quat(
            'mocap',
            np.array([self.data.get_body_quat('hand')]),
        )

    def get_endeff_pos(self):
        return self.data.get_body_xpos('hand').copy()

    def get_env_state(self):
        joint_state = self.sim.get_state()
        mocap_state = self.data.mocap_pos, self.data.mocap_quat
        state = joint_state, mocap_state
        return copy.deepcopy(state)

    def set_env_state(self, state):
        joint_state, mocap_state = state
        self.sim.set_state(joint_state)
        mocap_pos, mocap_quat = mocap_state
        self.data.set_mocap_pos('mocap', mocap_pos)
        self.data.set_mocap_quat('mocap', mocap_quat)
        self.sim.forward()


class SawyerXYZEnv(SawyerMocapBase, metaclass=abc.ABCMeta):
    def __init__(
            self,
            *args,
            hand_low=(-0.2, 0.55, 0.05),
            hand_high=(0.2, 0.75, 0.3),
            action_scale=1./100,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.action_scale = action_scale
        self.hand_low = np.array(hand_low)
        self.hand_high = np.array(hand_high)
        self.mocap_low = np.hstack(hand_low)
        self.mocap_high = np.hstack(hand_high)

    def set_xyz_action(self, action):
        action = np.clip(action, -1, 1)
        pos_delta = action * self.action_scale
        new_mocap_pos = self.data.mocap_pos + pos_delta[None]
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high,
        )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
