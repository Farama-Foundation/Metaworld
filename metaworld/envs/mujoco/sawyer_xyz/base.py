import abc
import copy

from gym.spaces import Discrete
import mujoco_py
import numpy as np


from metaworld.core.serializable import Serializable
from metaworld.envs.mujoco.mujoco_env import MujocoEnv
from metaworld.envs.env_util import quat_to_zangle, zangle_to_quat, quat_create, quat_mul


OBS_TYPE = ['plain', 'with_goal_id', 'with_goal_and_id', 'with_goal', 'with_goal_init_obs']


class SawyerMocapBase(MujocoEnv, Serializable, metaclass=abc.ABCMeta):
    """
    Provides some commonly-shared functions for Sawyer Mujoco envs that use
    mocap for XYZ control.
    """
    mocap_low = np.array([-0.2, 0.5, 0.06])
    mocap_high = np.array([0.2, 0.7, 0.6])

    def __init__(self, model_name, frame_skip=20):
        MujocoEnv.__init__(self, model_name, frame_skip=frame_skip)
        self.reset_mocap_welds()

    def get_endeff_pos(self):
        return self.data.get_body_xpos('hand').copy()

    def get_gripper_pos(self):
        return np.array([self.data.qpos[7]])

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

    def __getstate__(self):
        state = super().__getstate__()
        return {**state, 'env_state': self.get_env_state()}

    def __setstate__(self, state):
        super().__setstate__(state)
        self.set_env_state(state['env_state'])

    def reset_mocap_welds(self):
        """Resets the mocap welds that we use for actuation."""
        sim = self.sim
        if sim.model.nmocap > 0 and sim.model.eq_data is not None:
            for i in range(sim.model.eq_data.shape[0]):
                if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.])
        sim.forward()


class SawyerXYZEnv(SawyerMocapBase, metaclass=abc.ABCMeta):
    def __init__(
            self,
            *args,
            hand_low=(-0.2, 0.55, 0.05),
            hand_high=(0.2, 0.75, 0.3),
            mocap_low=None,
            mocap_high=None,
            action_scale=2./100,
            action_rot_scale=1.,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.action_scale = action_scale
        self.action_rot_scale = action_rot_scale
        self.hand_low = np.array(hand_low)
        self.hand_high = np.array(hand_high)
        if mocap_low is None:
            mocap_low = hand_low
        if mocap_high is None:
            mocap_high = hand_high
        self.mocap_low = np.hstack(mocap_low)
        self.mocap_high = np.hstack(mocap_high)

        # We use continuous goal space by default and
        # can discretize the goal space by calling
        # the `discretize_goal_space` method.
        self.discrete_goal_space = None
        self.discrete_goals = []
        self.active_discrete_goal = None

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

    def set_xyz_action_rot(self, action):
        action[:3] = np.clip(action[:3], -1, 1)
        pos_delta = action[:3] * self.action_scale
        new_mocap_pos = self.data.mocap_pos + pos_delta[None]
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high,
        )
        rot_axis = action[4:] / np.linalg.norm(action[4:])
        action[3] = action[3] * self.action_rot_scale
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        # replace this with learned rotation
        quat = quat_mul(quat_create(np.array([0, 1., 0]), np.pi),
                        quat_create(np.array(rot_axis), action[3]))
        self.data.set_mocap_quat('mocap', quat)
        # self.data.set_mocap_quat('mocap', np.array([np.cos(action[3]/2), np.sin(action[3]/2)*rot_axis[0], np.sin(action[3]/2)*rot_axis[1], np.sin(action[3]/2)*rot_axis[2]]))
        # self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

    def set_xyz_action_rotz(self, action):
        action[:3] = np.clip(action[:3], -1, 1)
        pos_delta = action[:3] * self.action_scale
        new_mocap_pos = self.data.mocap_pos + pos_delta[None]
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high,
        )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        zangle_delta = action[3] * self.action_rot_scale
        new_mocap_zangle = quat_to_zangle(self.data.mocap_quat[0]) + zangle_delta

        # new_mocap_zangle = action[3]
        new_mocap_zangle = np.clip(
            new_mocap_zangle,
            -3.0,
            3.0,
        )
        if new_mocap_zangle < 0:
            new_mocap_zangle += 2 * np.pi
        self.data.set_mocap_quat('mocap', zangle_to_quat(new_mocap_zangle))

    def set_xy_action(self, xy_action, fixed_z):
        delta_z = fixed_z - self.data.mocap_pos[0, 2]
        xyz_action = np.hstack((xy_action, delta_z))
        self.set_xyz_action(xyz_action)

    def discretize_goal_space(self, goals=None):
        if goals is None:
            self.discrete_goals = [self.default_goal]
        else:
            assert len(goals) >= 1
            self.discrete_goals = goals
        # update the goal_space to a Discrete space
        self.discrete_goal_space = Discrete(len(self.discrete_goals))

    # Belows are methods for using the new wrappers.
    # `sample_goals` is implmented across the sawyer_xyz
    # as sampling from the task lists. This will be done
    # with the new `discrete_goals`. After all the algorithms
    # conform to this API (i.e. using the new wrapper), we can
    # just remove the underscore in all method signature.
    def sample_goals_(self, batch_size):
        if self.discrete_goal_space is not None:
            return [self.discrete_goal_space.sample() for _ in range(batch_size)]
        else:
            return [self.goal_space.sample() for _ in range(batch_size)]

    def set_goal_(self, goal):
        if self.discrete_goal_space is not None:
            self.active_discrete_goal = goal
            self.goal = self.discrete_goals[goal]
            self._state_goal_idx = np.zeros(len(self.discrete_goals))
            self._state_goal_idx[goal] = 1.
        else:
            self.goal = goal
    
    def set_init_config(self, config):
        assert isinstance(config, dict)
        for key, val in config.items():
            self.init_config[key] = val

    '''
    Functions that are copied and pasted everywhere and seems
    to be not used.
    '''
    def sample_goals(self, batch_size):
        '''Note: should be replaced by sample_goals_ if not used''' 
        # Required by HER-TD3
        goals = self.sample_goals_(batch_size)
        if self.discrete_goal_space is not None:
            goals = [self.discrete_goal_space[g].copy() for g in goals]
        return {
            'state_desired_goal': goals,
        }

    def sample_task(self):
        '''Note: this can be replaced by sample_goal_(batch_size=1)'''
        goal = self.sample_goals_(1)
        if self.discrete_goal_space is not None:
            return self.discrete_goals[goal]
        else:
            return goal

    def _set_obj_xyz_quat(self, pos, angle):
        quat = quat_create(np.array([0, 0, .1]), angle)
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qpos[12:16] = quat.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)
