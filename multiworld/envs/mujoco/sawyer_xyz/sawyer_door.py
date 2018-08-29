import abc
from collections import OrderedDict

import mujoco_py
import numpy as np
import sys
from multiworld.envs.mujoco.mujoco_env import MujocoEnv
from gym.spaces import Box, Dict
from multiworld.core.serializable import Serializable
from multiworld.envs.env_util import (
    get_stat_in_paths,
    create_stats_ordered_dict, get_asset_full_path,
)
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv


class SawyerDoorEnv(
    SawyerXYZEnv,
    MultitaskEnv,
    Serializable,
    metaclass=abc.ABCMeta,
):
    def __init__(
        self,
        frame_skip=50,
        goal_low=(.4, .5, .12, -.5),
        goal_high=(.6, .7, .12, 0),
        pos_action_scale=1 / 100,
        action_reward_scale=0,
        reward_type='angle_difference',
        indicator_threshold=(.02, .03),
        fix_goal=False,
        fixed_goal=(0.5, .5, .12, -.25),
        num_resets_before_door_and_hand_reset=1,
        fixed_hand_z=0.12,
        hand_low=(-0.15, -2, 0),
        hand_high=(0.15, 2, 1),
        target_pos_scale=1,
        target_angle_scale=1,
        xml_suffix='pull',
    ):
        self.quick_init(locals())
        self.xml_suffix = xml_suffix
        SawyerXYZEnv.__init__(
            self,
            self.model_name,
            frame_skip=frame_skip,
            hand_low=hand_low,
            hand_high=hand_high
        )
        MultitaskEnv.__init__(self)

        self.reward_type = reward_type
        self.indicator_threshold = indicator_threshold

        self.fix_goal = fix_goal
        self.fixed_goal = np.array(fixed_goal)
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        self._state_goal = None
        self.fixed_hand_z = fixed_hand_z

        self.action_space = Box(np.array([-1, -1]),
                                np.array([1, 1]))
        max_angle = 1.5708
        self.state_space = Box(
            np.concatenate((hand_low, [-1*max_angle])),
            np.concatenate((hand_low, [1*max_angle])),
        )
        self.angle_space = Box(
            np.array([-max_angle]),
            np.array([max_angle])
        )
        self.observation_space = Dict([
            ('observation', self.state_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.state_space),
            ('state_observation', self.state_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.state_space),
        ])
        self._pos_action_scale = pos_action_scale
        self.action_reward_scale = action_reward_scale
        self.target_pos_scale = target_pos_scale
        self.target_angle_scale = target_angle_scale
        self.reset_counter = 0
        self.num_resets_before_door_and_hand_reset = num_resets_before_door_and_hand_reset

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_door_' + self.xml_suffix + '.xml')

    def step(self, action):
        self.set_xy_action(action[:2], self.fixed_hand_z)
        u = np.zeros(7)
        self.do_simulation(u, self.frame_skip)
        info = self._get_info()
        ob = self._get_obs()
        reward = self.compute_reward(action, ob)
        done = False
        return ob, reward, done, info

    def _get_obs(self):
        pos = self.get_endeff_pos()
        angle = self.get_door_angle()
        flat_obs = np.concatenate((pos, angle))
        return dict(
            observation=flat_obs,
            desired_goal=self._state_goal,
            achieved_goal=flat_obs,
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=flat_obs,
        )

    def _get_info(self):
        angle_diff = np.abs(self.get_door_angle() - self._state_goal[-1])[0]
        hand_dist = np.linalg.norm(self.get_endeff_pos() - self._state_goal[:3])
        info = dict(
            angle_difference=angle_diff,
            angle_success=(angle_diff < self.indicator_threshold[0]).astype(
                float),
            hand_distance=hand_dist,
            hand_success=(hand_dist < self.indicator_threshold[1]).astype(
                float),
            total_distance=angle_diff + hand_dist
        )
        return info

    def get_door_angle(self):
        return np.array([self.data.get_joint_qpos('doorjoint')])

    @property
    def endeff_id(self):
        return self.model.body_names.index('leftclaw')

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        actual_angle = achieved_goals[:, -1]
        goal_angle = desired_goals[:, -1]
        pos = achieved_goals[:, :3]
        goal_pos = desired_goals[:, :3]
        angle_diff = np.abs(actual_angle - goal_angle)
        pos_dist = np.linalg.norm(pos - goal_pos, axis=1)
        if self.reward_type == 'angle_diff_and_hand_distance':
            r = - (
                angle_diff * self.target_angle_scale
                + pos_dist * self.target_pos_scale
            )
        elif self.reward_type == 'angle_difference':
            r = - angle_diff * self.target_angle_scale

        elif self.reward_type == 'hand_success':
            r = -(angle_diff < self.indicator_threshold[0] and pos_dist <
                  self.indicator_threshold[1]).astype(float)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def reset(self):
        if self.reset_counter % self.num_resets_before_door_and_hand_reset == 0:
            self._reset_hand()
            self._set_door_pos(0)
        goal = self.sample_goal()
        self.set_goal(goal)
        self.reset_counter += 1
        self.reset_mocap_welds()
        return self._get_obs()

    def _reset_hand(self):
        angles = self.data.qpos.copy()
        velocities = self.data.qvel.copy()
        angles[1:] = self.init_angles[1:]
        velocities[1:] = 0
        self.set_state(angles.flatten(), velocities.flatten())
        self.reset_mocap_welds()

    def _set_door_pos(self, pos):
        angles = self.data.qpos.copy()
        velocities = self.data.qvel.copy()
        angles[0] = pos
        velocities[0] = 0
        self.set_state(angles.flatten(), velocities.flatten())

    @property
    def init_angles(self):
        return [
            1.02866769e+00, - 6.95207647e-01, 4.22932911e-01,
            1.76670458e+00, - 5.69637604e-01, 6.24117280e-01,
            3.53404635e+00,
            0,
        ]

    ''' Multitask Functions '''

    @property
    def goal_dim(self):
        return 4

    def set_goal(self, goal):
        self._state_goal = goal['state_desired_goal']

    def sample_goals(self, batch_size):
        if self.fix_goal:
            goals = np.repeat(
                self.fixed_goal.copy()[None],
                batch_size,
                0
            )
        else:
            goals = np.random.uniform(
                self.goal_space.low,
                self.goal_space.high,
                size=(batch_size, self.goal_space.low.size),
            )
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def set_to_goal_angle(self, angle):
        self._state_goal = angle.copy()
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[0] = angle.copy()
        qvel[0] = 0
        self.set_state(qpos, qvel)

    def set_to_goal_pos(self, xyz):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', np.array(xyz))
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            u = np.zeros(7)
            self.do_simulation(u, self.frame_skip)

    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

    def set_to_goal(self, goal):
        raise NotImplementedError("Hard to do because what if the hand is in "
                                  "the door? Use presampled goals.")

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'angle_difference',
            'angle_success',
            'hand_distance',
            'hand_success',
            'total_distance',
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'Final %s%s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
            ))
        return statistics

    def get_env_state(self):
        base_state = super().get_endeff_pos()
        goal = self._state_goal.copy()
        return base_state, goal

    def set_env_state(self, state):
        base_state, goal = state
        super().set_env_state(base_state)
        self._state_goal = goal
