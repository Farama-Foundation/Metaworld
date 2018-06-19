from collections import OrderedDict
import numpy as np
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box, Dict

from railrl.core import logger

from multiworld.core.serializable import Serializable
from railrl.envs.env_utils import get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path


class SawyerReachTorqueEnv(MujocoEnv, Serializable, MultitaskEnv):
    """Implements a torque-controlled Sawyer environment"""

    def __init__(self,
                 frame_skip=30,
                 action_scale=1. / 10,
                 keep_vel_in_obs=True,
                 use_safety_box=False,
                 fix_goal=False,
                 fixed_goal=(0.15, 0.6, 0.3),
                 reward_type='euclidean',
                 indicator_threshold=.06,
                 goal_low=None,
                 goal_high=None,
                 ):
        self.quick_init(locals())
        self.action_scale = action_scale
        if goal_low is None:
            goal_low = np.array([-0.1, 0.5, 0.02])
        if goal_high is None:
            goal_high = np.array([0.1, 0.7, 0.2])
        self.safety_box = Box(
            goal_low,
            goal_high
        )
        self.keep_vel_in_obs = keep_vel_in_obs
        self.use_safety_box=use_safety_box
        self.prev_qpos = self.init_angles.copy()
        self.fix_goal = fix_goal
        self.fixed_goal = np.array(fixed_goal)
        self.goal_space = Box(goal_low, goal_high)
        self._goal_xyz = None
        self.reset()
        self.reward_type = reward_type
        self.indicator_threshold = indicator_threshold
        MultitaskEnv.__init__(self)
        MujocoEnv.__init__(self, self.model_name, frame_skip=frame_skip)


        self.action_space = Box( -1*np.ones(7), np.ones(7))
        self.observation_space = Dict([
            ('observation', self.observation_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.goal_space),
            ('state_observation', self.observation_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.goal_space),
        ])

    @property
    def model_name(self):
       return get_asset_full_path('sawyer_xyz/sawyer_reach_torque.xml')

    def reset_to_prev_qpos(self):
        angles = self.data.qpos.copy()
        velocities = self.data.qvel.copy()
        angles[:] = self.prev_qpos.copy()
        velocities[:] = 0
        self.set_state(angles.flatten(), velocities.flatten())
        self.set_goal_xyz(self._goal_xyz)

    def is_outside_box(self):
        pos = self.get_endeff_pos()
        return not self.safety_box.contains(pos)

    def set_to_qpos(self, qpos):
        angles = self.data.qpos.copy()
        velocities = self.data.qvel.copy()
        angles[:] = qpos
        velocities[:] = 0
        self.set_state(angles.flatten(), velocities.flatten())
        self.set_goal_xyz(self._goal_xyz)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 1.0

        # robot view
        # rotation_angle = 90
        # cam_dist = 1
        # cam_pos = np.array([0, 0.5, 0.2, cam_dist, -45, rotation_angle])

        # 3rd person view
        cam_dist = 0.3
        rotation_angle = 270
        cam_pos = np.array([0, 1.0, 0.5, cam_dist, -45, rotation_angle])

        # top down view
        # cam_dist = 0.2
        # rotation_angle = 0
        # cam_pos = np.array([0, 0, 1.5, cam_dist, -90, rotation_angle])

        for i in range(3):
            self.viewer.cam.lookat[i] = cam_pos[i]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid = -1

    def step(self, a):
        a = a * self.action_scale
        self.do_simulation(a, self.frame_skip)
        if self.use_safety_box:
            if self.is_outside_box():
                self.reset_to_prev_qpos()
            else:
                self.prev_qpos = self.data.qpos.copy()
        obs = self._get_obs()
        info = self._get_info()
        reward = self.compute_reward(
            obs['achieved_goal'],
            obs['desired_goal'],
            info,
        )
        done = False
        return obs, reward, done, info

    def _get_env_obs(self):
        if self.keep_vel_in_obs:
            return np.concatenate([
                self.sim.data.qpos.flat[:7],
                self.sim.data.qvel.flat,
                self.get_endeff_pos(),
            ])
        else:
            return np.concatenate([
                self.sim.data.qpos.flat[:7],
                self.get_endeff_pos(),
            ])

    def _get_obs(self):
        ee_pos = self.get_endeff_pos()
        state_obs = self._get_env_obs()
        return dict(
            observation=state_obs,
            desired_goal=self._goal_xyz,
            achieved_goal=ee_pos,

            state_observation=state_obs,
            state_desired_goal=self._goal_xyz,
            state_achieved_goal=ee_pos,
        )

    def _get_info(self):
        hand_distance = np.linalg.norm(self._goal_xyz - self.get_endeff_pos())
        return dict(
            hand_distance=hand_distance,
            hand_success=float(hand_distance < self.indicator_threshold),
        )

    def get_endeff_pos(self):
        return self.data.body_xpos[self.endeff_id].copy()

    def get_goal_pos(self):
        return self.data.body_xpos[self.goal_id].copy()

    def sample_goal_xyz(self):
        pos = np.random.uniform(
            np.array([-0.1, 0.5, 0.02]),
            np.array([0.1, 0.7, 0.2]),
        )
        return pos

    def set_goal_xyz(self, pos):
        ''' Sets the goal to a particular position '''
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[7:10] = pos.copy()
        qvel[7:10] = [0, 0, 0]
        self.set_state(qpos, qvel)

    def reset(self):
        angles = self.data.qpos.copy()
        velocities = self.data.qvel.copy()
        angles[:] = self.init_angles
        velocities[:] = 0
        self.set_state(angles.flatten(), velocities.flatten())
        goal = self.sample_goal()
        self._goal_xyz = goal['state_desired_goal']
        self.set_goal_xyz(self._goal_xyz)
        self.prev_qpos=self.init_angles
        self.sim.forward()
        return self._get_obs()



    @property
    def init_angles(self):
        return [
            1.02866769e+00, - 6.95207647e-01, 4.22932911e-01,
            1.76670458e+00, - 5.69637604e-01, 6.24117280e-01,
            3.53404635e+00,
            1.07586388e-02, 6.62018003e-01, 2.09936716e-02,
            1.00000000e+00, 3.76632959e-14, 1.36837913e-11, 1.56567415e-23
        ]

    @property
    def goal_dim(self):
        return 3

    @property
    def endeff_id(self):
        return self.model.body_names.index('leftclaw')

    @property
    def goal_id(self):
        return self.model.body_names.index('goal')

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'hand_distance',
            'hand_success',
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

    """
    Multitask functions
    """
    @property
    def goal_dim(self) -> int:
        return 3

    def sample_goal_for_rollout(self):
        return self.sample_goal_xyz()

    def get_goal(self):
        return self._goal_xyz

    def convert_obs_to_goals(self, obs):
        return obs[:, -3:]

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

    def compute_rewards(self, achieved_goals, desired_goals, info):
        hand_pos = achieved_goals
        goals = desired_goals
        distances = np.linalg.norm(hand_pos - goals, axis=1)
        if self.reward_type == 'hand_distance':
            r = -distances
        elif self.reward_type == 'hand_success':
            r = -(distances < self.indicator_threshold).astype(float)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r


if __name__ == "__main__":
    import pygame
    from pygame.locals import QUIT, KEYDOWN

    pygame.init()
    screen = pygame.display.set_mode((400, 300))
    char_to_action = {
        'w': np.array([0 , -1, 0 , 0]),
        'a': np.array([1 , 0 , 0 , 0]),
        's': np.array([0 , 1 , 0 , 0]),
        'd': np.array([-1, 0 , 0 , 0]),
        'q': np.array([1 , -1 , 0 , 0]),
        'e': np.array([-1 , -1 , 0, 0]),
        'z': np.array([1 , 1 , 0 , 0]),
        'c': np.array([-1 , 1 , 0 , 0]),
        # 'm': np.array([1 , 1 , 0 , 0]),
        'j': np.array([0 , 0 , 1 , 0]),
        'k': np.array([0 , 0 , -1 , 0]),
        'x': 'toggle',
        'r': 'reset',
    }

    # ACTION_FROM = 'controller'
    # H = 100000
    ACTION_FROM = 'random'
    H = 300
    # ACTION_FROM = 'pd'
    # H = 50

    env = SawyerReachTorqueEnv(keep_vel_in_obs=False, use_safety_box=False)
    # env = MultitaskToFlatEnv(env)
    lock_action = False
    while True:
        obs = env.reset()
        last_reward_t = 0
        returns = 0
        action = np.zeros_like(env.action_space.sample())
        for t in range(H):
            done = False
            if ACTION_FROM == 'controller':
                if not lock_action:
                    action = np.array([0,0,0,0])
                for event in pygame.event.get():
                    event_happened = True
                    if event.type == QUIT:
                        sys.exit()
                    if event.type == KEYDOWN:
                        char = event.dict['key']
                        new_action = char_to_action.get(chr(char), None)
                        if new_action == 'toggle':
                            lock_action = not lock_action
                        elif new_action == 'reset':
                            done = True
                        elif new_action is not None:
                            action = new_action
                        else:
                            action = np.array([0 , 0 , 0 , 0])
                        print("got char:", char)
                        print("action", action)
                        print("angles", env.data.qpos.copy())
            elif ACTION_FROM == 'random':
                action = env.action_space.sample()
            else:
                delta = (env.get_block_pos() - env.get_endeff_pos())[:2]
                action[:2] = delta * 100
            # if t == 0:
            #     print("goal is", env.get_goal_pos())
            obs, reward, _, info = env.step(action)

            env.render()
            if done:
                break
        print("new episode")
