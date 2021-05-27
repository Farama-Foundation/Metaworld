import numpy as np
from gym.spaces import Box

from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_state import SawyerXYZState
from metaworld.envs.mujoco.sawyer_xyz.tools import (
    ScrewEye, ScrewEyePeg, get_position_of, get_quat_of
)
from ._reward_primitives import (
    tolerance,
    hamacher_product as h_prod,
    gripper_caging_reward
)
from ._task import Task


class Assemble(Task):
    WRENCH_HANDLE_LENGTH = 0.02

    def __init__(self):
        # The following properties are those that are necessary
        # to compute rewards, but aren't included in a SawyerXYZState
        # object.
        self._target_pos = np.zeros(3)
        self._initial_pos_obj = None
        self._initial_pos_pads_center = None

    @property
    def random_reset_space(self) -> Box:
        return Box(
            np.array([-.1, .2, -.1, .2]),
            np.array([+.1, .4, +.1, .4]),
        )

    def get_pos_objects(self, mjsim) -> np.ndarray:
        return np.concatenate((
            mjsim.data.site_xpos[mjsim.model.site_name2id('RoundNut-8')],
            mjsim.data.site_xpos[mjsim.model.site_name2id('RoundNut')]
        ))

    def get_quat_objects(self, mjsim) -> np.ndarray:
        return get_quat_of(ScrewEye(), mjsim)

    def reset_required_tools(
            self,
            world,
            solver,
            random_reset_vec,
    ):
        screw_eye = ScrewEye()
        peg = ScrewEyePeg()

        xyz0 = np.array([.0, .0, screw_eye.resting_pos_z])
        xyz1 = np.array([.0, .0, peg.resting_pos_z])

        vec = random_reset_vec
        displacement = vec[:2] - vec[2:]

        xyz0[:2] = vec[:2]
        xyz1[:2] = vec[:2] + 0.2 * displacement / np.linalg.norm(displacement)

        xyz0[0] += world.size[0] / 2.0
        xyz1[0] += world.size[0] / 2.0

        screw_eye.specified_pos = xyz0
        peg.specified_pos = xyz1
        solver.did_manual_set(screw_eye)
        solver.did_manual_set(peg)

        self._target_pos = xyz1

    @staticmethod
    def _reward_quat(quat):
        # Ideal laid-down wrench has quat [.707, 0, 0, .707]
        # Rather than deal with an angle between quaternions, just approximate:
        ideal = np.array([0.707, 0, 0, 0.707])
        error = np.linalg.norm(quat - ideal)
        return max(1.0 - error/0.4, 0.0)

    @staticmethod
    def _reward_pos(wrench_center, target_pos):
        pos_error = target_pos - wrench_center

        radius = np.linalg.norm(pos_error[:2])

        aligned = radius < 0.02
        hooked = pos_error[2] > 0.0
        success = aligned and hooked

        # Target height is a 3D funnel centered on the peg.
        # use the success flag to widen the bottleneck once the agent
        # learns to place the wrench on the peg -- no reason to encourage
        # tons of alignment accuracy if task is already solved
        threshold = 0.02 if success else 0.01
        target_height = 0.0
        if radius > threshold:
            target_height = 0.02 * np.log(radius - threshold) + 0.2

        pos_error[2] = target_height - wrench_center[2]

        scale = np.array([1., 1., 3.])
        a = 0.1  # Relative importance of just *trying* to lift the wrench
        b = 0.9  # Relative importance of placing the wrench on the peg
        lifted = wrench_center[2] > 0.02 or radius < threshold
        in_place = a * float(lifted) + b * tolerance(
            np.linalg.norm(pos_error * scale),
            bounds=(0, 0.02),
            margin=0.4,
            sigmoid='long_tail',
        )

        return in_place, success

    def compute_reward(self, state: SawyerXYZState):
        if state.timestep == 1:
            self._initial_pos_obj = state.pos_objs[:3].copy()
            self._initial_pos_pads_center = state.pos_pads_center.copy()

        hand = state.pos_hand
        wrench = state.pos_objs[:3]
        wrench_center = state.pos_objs[3:]
        # `self._gripper_caging_reward` assumes that the target object can be
        # approximated as a sphere. This is not true for the wrench handle, so
        # to avoid re-writing the `self._gripper_caging_reward` we pass in a
        # modified wrench position.
        # This modified position's X value will perfect match the hand's X value
        # as long as it's within a certain threshold
        wrench_threshed = wrench.copy()
        threshold = Assemble.WRENCH_HANDLE_LENGTH / 2.0
        if abs(wrench[0] - hand[0]) < threshold:
            wrench_threshed[0] = hand[0]

        reward_quat = Assemble._reward_quat(state.quat_objs[:4])
        reward_grab = gripper_caging_reward(
            state,
            self._initial_pos_obj,
            self._initial_pos_pads_center,
            obj_radius=0.015,
            pad_success_thresh=0.02,
            xz_thresh=0.01,
            include_reach_reward=True,
            reach_reward_radius=0.01
        )
        reward_in_place, success = Assemble._reward_pos(
            wrench_center,
            self._target_pos
        )

        reward = (2.0 * reward_grab + 6.0 * reward_in_place) * reward_quat
        # Override reward on success
        if success:
            reward = 10.0

        return reward, {
            'success': float(success),
            'near_object': reward_quat,
            'grasp_success': reward_grab >= 0.5,
            'grasp_reward': reward_grab,
            'in_place_reward': reward_in_place,
            'obj_to_target': 0,
            'unscaled_reward': reward,
        }
