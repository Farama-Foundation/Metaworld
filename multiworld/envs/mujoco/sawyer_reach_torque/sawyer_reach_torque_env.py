from collections import OrderedDict
import numpy as np
from multiworld.envs.mujoco.mujoco_env import MujocoEnv
from gym.spaces import Box, Dict
from multiworld.core.serializable import Serializable
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path


class SawyerReachTorqueEnv(MujocoEnv, Serializable, MultitaskEnv):
    """Implements a torque-controlled Sawyer environment"""

    def __init__(self,
                 frame_skip=30,
                 action_scale=10,
                 keep_vel_in_obs=True,
                 use_safety_box=False,
                 fix_goal=False,
                 fixed_goal=(0.05, 0.6, 0.15),
                 reward_type='hand_distance',
                 indicator_threshold=.05,
                 goal_low=None,
                 goal_high=None,
                 ):
        self.quick_init(locals())
        MultitaskEnv.__init__(self)
        self.action_scale = action_scale
        MujocoEnv.__init__(self, self.model_name, frame_skip=frame_skip)
        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = Box(low=low, high=high)
        if goal_low is None:
            goal_low = np.array([-0.1, 0.5, 0.02])
        else:
            goal_low = np.array(goal_low)
        if goal_high is None:
            goal_high = np.array([0.1, 0.7, 0.2])
        else:
            goal_high = np.array(goal_low)
        self.safety_box = Box(
            goal_low,
            goal_high
        )
        self.keep_vel_in_obs = keep_vel_in_obs
        self.goal_space = Box(goal_low, goal_high)
        obs_size = self._get_env_obs().shape[0]
        high = np.inf * np.ones(obs_size)
        low = -high
        self.obs_space = Box(low, high)
        self.achieved_goal_space = Box(
            -np.inf * np.ones(3),
            np.inf * np.ones(3)
        )
        self.observation_space = Dict([
            ('observation', self.obs_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.achieved_goal_space),
            ('state_observation', self.obs_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.achieved_goal_space),
        ])
        self.fix_goal = fix_goal
        self.fixed_goal = np.array(fixed_goal)
        self.use_safety_box=use_safety_box
        self.prev_qpos = self.init_angles.copy()
        self.reward_type = reward_type
        self.indicator_threshold = indicator_threshold
        self.reset()

    @property
    def model_name(self):
       return get_asset_full_path('sawyer_xyz/sawyer_reach_torque.xml')

    def reset_to_prev_qpos(self):
        angles = self.data.qpos.copy()
        velocities = self.data.qvel.copy()
        angles[:] = self.prev_qpos.copy()
        velocities[:] = 0
        self.set_state(angles.flatten(), velocities.flatten())

    def is_outside_box(self):
        pos = self.get_endeff_pos()
        return not self.safety_box.contains(pos)

    def set_to_qpos(self, qpos):
        angles = self.data.qpos.copy()
        velocities = self.data.qvel.copy()
        angles[:] = qpos
        velocities[:] = 0
        self.set_state(angles.flatten(), velocities.flatten())

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 1.0

        # 3rd person view
        cam_dist = 0.3
        rotation_angle = 270
        cam_pos = np.array([0, 1.0, 0.5, cam_dist, -45, rotation_angle])

        for i in range(3):
            self.viewer.cam.lookat[i] = cam_pos[i]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid = -1

    def step(self, action):
        action = action * self.action_scale
        self.do_simulation(action, self.frame_skip)
        if self.use_safety_box:
            if self.is_outside_box():
                self.reset_to_prev_qpos()
            else:
                self.prev_qpos = self.data.qpos.copy()
        ob = self._get_obs()
        info = self._get_info()
        reward = self.compute_reward(action, ob)
        done = False
        return ob, reward, done, info

    def _get_env_obs(self):
        if self.keep_vel_in_obs:
            return np.concatenate([
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
                self.get_endeff_pos(),
            ])
        else:
            return np.concatenate([
                self.sim.data.qpos.flat,
                self.get_endeff_pos(),
            ])

    def _get_obs(self):
        ee_pos = self.get_endeff_pos()
        state_obs = self._get_env_obs()
        return dict(
            observation=state_obs,
            desired_goal=self._state_goal,
            achieved_goal=ee_pos,

            state_observation=state_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=ee_pos,
        )

    def _get_info(self):
        hand_distance = np.linalg.norm(self._state_goal - self.get_endeff_pos())
        return dict(
            hand_distance=hand_distance,
            hand_success=float(hand_distance < self.indicator_threshold),
        )

    def get_endeff_pos(self):
        return self.data.body_xpos[self.endeff_id].copy()

    def reset_model(self):
        angles = self.data.qpos.copy()
        velocities = self.data.qvel.copy()
        angles[:] = self.init_angles
        velocities[:] = 0
        self.set_state(angles.flatten(), velocities.flatten())
        self.sim.forward()
        self.prev_qpos=self.data.qpos.copy()

    def reset(self):
        self.reset_model()
        self.set_goal(self.sample_goal())
        self.sim.forward()
        self.prev_qpos = self.data.qpos.copy()
        return self._get_obs()

    @property
    def init_angles(self):
        return [
            1.02866769e+00, - 6.95207647e-01, 4.22932911e-01,
            1.76670458e+00, - 5.69637604e-01, 6.24117280e-01,
            3.53404635e+00,
        ]

    @property
    def endeff_id(self):
        return self.model.body_names.index('leftclaw')

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

    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

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

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['achieved_goal']
        desired_goals = obs['desired_goal']
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

    def get_env_state(self):
        joint_state = self.sim.get_state()
        goal = self._state_goal.copy()
        return joint_state, goal

    def set_env_state(self, state):
        state, goal = state
        self.sim.set_state(state)
        self.sim.forward()
        self._state_goal = goal

if __name__ == "__main__":
    # import pygame
    # from pygame.locals import QUIT, KEYDOWN
    #
    # pygame.init()
    # screen = pygame.display.set_mode((400, 300))
    # char_to_action = {
    #     'w': np.array([0 , -1, 0 , 0]),
    #     'a': np.array([1 , 0 , 0 , 0]),
    #     's': np.array([0 , 1 , 0 , 0]),
    #     'd': np.array([-1, 0 , 0 , 0]),
    #     'q': np.array([1 , -1 , 0 , 0]),
    #     'e': np.array([-1 , -1 , 0, 0]),
    #     'z': np.array([1 , 1 , 0 , 0]),
    #     'c': np.array([-1 , 1 , 0 , 0]),
    #     # 'm': np.array([1 , 1 , 0 , 0]),
    #     'j': np.array([0 , 0 , 1 , 0]),
    #     'k': np.array([0 , 0 , -1 , 0]),
    #     'x': 'toggle',
    #     'r': 'reset',
    # }

    # ACTION_FROM = 'controller'
    # H = 100000
    ACTION_FROM = 'random'
    H = 300
    # ACTION_FROM = 'pd'
    # H = 50

    env = SawyerReachTorqueEnv(keep_vel_in_obs=False, use_safety_box=False)

    # env.get_goal()
    # # env = MultitaskToFlatEnv(env)
    # lock_action = False
    # while True:
    #     obs = env.reset()
    #     last_reward_t = 0
    #     returns = 0
    #     action = np.zeros_like(env.action_space.sample())
    #     for t in range(H):
    #         done = False
    #         if ACTION_FROM == 'controller':
    #             if not lock_action:
    #                 action = np.array([0,0,0,0])
    #             for event in pygame.event.get():
    #                 event_happened = True
    #                 if event.type == QUIT:
    #                     sys.exit()
    #                 if event.type == KEYDOWN:
    #                     char = event.dict['key']
    #                     new_action = char_to_action.get(chr(char), None)
    #                     if new_action == 'toggle':
    #                         lock_action = not lock_action
    #                     elif new_action == 'reset':
    #                         done = True
    #                     elif new_action is not None:
    #                         action = new_action
    #                     else:
    #                         action = np.array([0 , 0 , 0 , 0])
    #                     print("got char:", char)
    #                     print("action", action)
    #                     print("angles", env.data.qpos.copy())
    #         elif ACTION_FROM == 'random':
    #             action = env.action_space.sample()
    #         else:
    #             delta = (env.get_block_pos() - env.get_endeff_pos())[:2]
    #             action[:2] = delta * 100
    #         # if t == 0:
    #         #     print("goal is", env.get_goal_pos())
    #         obs, reward, _, info = env.step(action)
    #         env.render()
    #         if done:
    #             break
    #     print("new episode")
