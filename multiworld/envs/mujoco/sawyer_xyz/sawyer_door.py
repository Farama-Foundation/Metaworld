import abc
from collections import OrderedDict

import mujoco_py
import numpy as np
import sys
from multiworld.envs.mujoco.mujoco_env import MujocoEnv
from gym.spaces import Box, Dict
from multiworld.core.serializable import Serializable
from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from railrl.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from railrl.exploration_strategies.epsilon_greedy import EpsilonGreedy
from railrl.exploration_strategies.ou_strategy import OUStrategy
from railrl.policies.simple import ZeroPolicy


class SawyerDoorEnv(MultitaskEnv, MujocoEnv, Serializable, metaclass=abc.ABCMeta):
    def __init__(self,
                     frame_skip=50,
                     goal_low=-.5,
                     goal_high=.5,
                     pos_action_scale=1 / 100,
                     action_reward_scale=0,
                     reward_type='angle_difference',
                     indicator_threshold=0.02,
                     fix_goal=False,
                     fixed_goal=.25,
                ):
        self.quick_init(locals())
        MultitaskEnv.__init__(self)
        MujocoEnv.__init__(self, self.model_name, frame_skip=frame_skip)

        self.reward_type = reward_type
        self.indicator_threshold = indicator_threshold

        self.fix_goal = fix_goal
        self.fixed_goal = np.array([fixed_goal])
        self.goal_space = Box(np.array([goal_low]), np.array([goal_high]))
        self._state_goal = None

        self.action_space = Box(np.array([-1, -1, -1, -1]), np.array([1, 1, 1, 1]))
        max_angle = 1.5708
        self.state_space = Box(
            np.array([-1, -1, -1, -max_angle]),
            np.array([1, 1, 1, max_angle]),
        )
        self.angle_space = Box(
            np.array([-max_angle]),
            np.array([max_angle])
        )
        self.observation_space = Dict([
            ('observation', self.state_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.angle_space),
            ('state_observation', self.state_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.angle_space),
        ])
        self._pos_action_scale = pos_action_scale
        self.action_reward_scale = action_reward_scale

        self.reset()
        self.reset_mocap_welds()

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_door.xml')

    def reset_mocap_welds(self):
        """Resets the mocap welds that we use for actuation."""
        sim = self.sim
        if sim.model.nmocap > 0 and sim.model.eq_data is not None:
            for i in range(sim.model.eq_data.shape[0]):
                if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.])
        sim.forward()

    def step(self, action):
        action = np.clip(action, -1, 1)
        self.mocap_set_action(action[:3] * self._pos_action_scale)
        u = np.zeros((7))
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
            achieved_goal=angle,
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=angle,
        )

    def _get_info(self):
        angle_diff = np.abs(self.get_door_angle()-self._state_goal)
        info = dict(
            angle_difference=angle_diff,
            angle_success = (angle_diff < self.indicator_threshold).astype(float)
        )
        return info

    def mocap_set_action(self, action):
        pos_delta = action[None]
        self.reset_mocap2body_xpos()
        new_mocap_pos = self.data.mocap_pos + pos_delta
        new_mocap_pos[0, 0] = np.clip(
            new_mocap_pos[0, 0],
            -0.15,
            0.15,
        )
        new_mocap_pos[0, 1] = np.clip(
            new_mocap_pos[0, 1],
            -2,
            2,
        )
        new_mocap_pos[0, 2] = np.clip(
            0.06,
            0,
            0.5,
        )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

    def reset_mocap2body_xpos(self):
        self.data.set_mocap_pos(
            'mocap',
            np.array([self.data.body_xpos[self.endeff_id]]),
        )
        self.data.set_mocap_quat(
            'mocap',
            np.array([self.data.body_xquat[self.endeff_id]]),
        )

    def get_endeff_pos(self):
        return self.data.body_xpos[self.endeff_id].copy()

    def get_door_angle(self):
        return np.array([self.data.get_joint_qpos('doorjoint')])

    @property
    def endeff_id(self):
        return self.model.body_names.index('leftclaw')

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        reward = np.linalg.norm(achieved_goals-desired_goals, axis=1)
        if self.reward_type == 'angle_difference':
            reward =  -reward
        elif self.reward_type == 'angle_success':
            reward = -(reward < self.indicator_threshold).astype(float)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return reward

    def reset(self):
        angles = self.data.qpos.copy()
        velocities = self.data.qvel.copy()
        angles[:] = self.init_angles
        velocities[:] = 0
        self.set_state(angles.flatten(), velocities.flatten())
        goal = self.sample_goal()
        self._state_goal = goal['state_desired_goal']
        return self._get_obs()

    @property
    def init_angles(self):
        return [
            0,
            1.02866769e+00, - 6.95207647e-01, 4.22932911e-01,
            1.76670458e+00, - 5.69637604e-01, 6.24117280e-01,
            3.53404635e+00,
        ]

    ''' Multitask Functions '''

    @property
    def goal_dim(self):
        return 1

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
        state_goal = goal['state_desired_goal']
        self.set_to_goal_angle(state_goal)

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'angle_difference',
            'angle_success',
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
        joint_state = self.sim.get_state()
        mocap_state = self.data.mocap_pos, self.data.mocap_quat
        base_state = joint_state, mocap_state
        goal = self._state_goal.copy()
        return base_state, goal

    def set_env_state(self, state):
        state, goal = state
        joint_state, mocap_state = state
        self.sim.set_state(joint_state)
        mocap_pos, mocap_quat = mocap_state
        self.data.set_mocap_pos('mocap', mocap_pos)
        self.data.set_mocap_quat('mocap', mocap_quat)
        self.sim.forward()
        self._state_goal = goal

class SawyerDoorPushOpenEnv(SawyerDoorEnv):
    def __init__(self,
                 goal_low=0,
                 goal_high=.5,
                 max_x_pos=.1,
                 max_y_pos=.7,
                 **kwargs
                ):
        self.quick_init(locals())
        self.max_x_pos = max_x_pos
        self.max_y_pos = max_y_pos
        self.min_y_pos = .5
        super().__init__(goal_low=goal_low, goal_high=goal_high, **kwargs)

    def mocap_set_action(self, action):
        pos_delta = action[None]
        self.reset_mocap2body_xpos()
        new_mocap_pos = self.data.mocap_pos + pos_delta
        new_mocap_pos[0, 0] = np.clip(
            new_mocap_pos[0, 0],
            -self.max_x_pos,
            self.max_x_pos,
        )
        new_mocap_pos[0, 1] = np.clip(
            new_mocap_pos[0, 1],
            self.min_y_pos,
            self.max_y_pos,
        )
        new_mocap_pos[0, 2] = np.clip(
            0.06,
            0,
            0.5,
        )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

    def set_to_goal(self, goal):
        ee_pos = np.random.uniform(np.array([-self.max_x_pos, self.min_y_pos, .06]),
                                   np.array([self.max_x_pos, .6, .06]))
        self.set_to_goal_pos(ee_pos)
        self.set_to_goal_angle(goal['state_desired_goal'])


class SawyerDoorPushOpenAndReachEnv(SawyerDoorPushOpenEnv):
    def __init__(self,
                 frame_skip=30,
                 goal_low=(-.1, .5, .06, 0),
                 goal_high=(.1, .6, .06,.5),
                 action_reward_scale=0,
                 pos_action_scale=1 / 100,
                 reward_type='angle_difference',
                 indicator_threshold=(.02, .03),
                 fix_goal=False,
                 fixed_goal=(0.15, 0.6, 0.3, 0),
                 target_pos_scale=0.25,
                 target_angle_scale=1,
                 max_x_pos=.1,
                 max_y_pos=.7,
                 ):

        self.quick_init(locals())
        MultitaskEnv.__init__(self)
        MujocoEnv.__init__(self, self.model_name, frame_skip=frame_skip)

        self.reward_type = reward_type
        self.indicator_threshold = indicator_threshold

        self.fix_goal = fix_goal
        self.fixed_goal = np.array(fixed_goal)
        self.goal_space = Box(
            np.array(goal_low),
            np.array(goal_high),
        )
        self._state_goal = None

        self.action_space = Box(np.array([-1, -1, -1, -1]), np.array([1, 1, 1, 1]))
        max_angle = 1.5708
        self.state_space = Box(
            np.array([-1, -1, -1, -max_angle]),
            np.array([1, 1, 1, max_angle]),
        )
        self.observation_space = Dict([
            ('observation', self.state_space),
            ('desired_goal', self.state_space),
            ('achieved_goal', self.state_space),
            ('state_observation', self.state_space),
            ('state_desired_goal', self.state_space),
            ('state_achieved_goal', self.state_space),
        ])
        self._pos_action_scale = pos_action_scale
        self.target_pos_scale = target_pos_scale
        self.action_reward_scale = action_reward_scale
        self.target_angle_scale = target_angle_scale

        self.max_x_pos = max_x_pos
        self.max_y_pos = max_y_pos
        self.min_y_pos = .5
        
        self.reset()
        self.reset_mocap_welds()

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
        angle_diff = np.abs(self.get_door_angle()-self._state_goal[-1])[0]
        hand_dist = np.linalg.norm(self.get_endeff_pos()-self._state_goal[:3])
        info = dict(
            angle_difference=angle_diff,
            angle_success = (angle_diff < self.indicator_threshold[0]).astype(float),
            hand_distance=hand_dist,
            hand_success=(hand_dist<self.indicator_threshold[1]).astype(float),
            total_distance=angle_diff+hand_dist
        )
        return info

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        actual_angle = achieved_goals[:, -1]
        goal_angle = desired_goals[:, -1]
        pos = achieved_goals[:, :3]
        goal_pos = desired_goals[:, :3]
        angle_diff = np.abs(actual_angle - goal_angle)
        pos_dist = np.linalg.norm(pos - goal_pos, axis=1)
        if self.reward_type == 'angle_difference':
            r = - (angle_diff*self.target_angle_scale+pos_dist*self.target_pos_scale)
        elif self.reward_type == 'hand_success':
            r = -(angle_diff < self.indicator_threshold[0] and pos_dist < self.indicator_threshold[1]).astype(float)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r
        
    ''' Multitask Functions '''

    @property
    def goal_dim(self):
        return 4

    def set_to_goal_angle(self, angle):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[0] = angle.copy()
        qvel[0] = 0
        self.set_state(qpos, qvel)

    def set_to_goal(self, goal):
        goal = goal['state_desired_goal']
        self.set_to_goal_pos(goal[:3])
        self.set_to_goal_angle(goal[-1])

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

    def mocap_set_action(self, action):
        pos_delta = action[None]
        self.reset_mocap2body_xpos()
        new_mocap_pos = self.data.mocap_pos + pos_delta
        new_mocap_pos[0, 0] = np.clip(
            new_mocap_pos[0, 0],
            -self.max_x_pos,
            self.max_x_pos,
        )
        new_mocap_pos[0, 1] = np.clip(
            new_mocap_pos[0, 1],
            self.min_y_pos,
            self.max_y_pos,
        )
        new_mocap_pos[0, 2] = np.clip(
            0.06,
            0,
            0.5,
        )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

class SawyerDoorPullOpenEnv(SawyerDoorEnv):
    def __init__(self,
                 goal_low=-.5,
                 goal_high=0,
                 max_x_pos=.1,
                 min_y_pos=.4,
                 max_y_pos=.6,
                 use_line=False,
                 **kwargs
                ):
        self.quick_init(locals())
        self.max_x_pos = max_x_pos
        self.min_y_pos = min_y_pos
        self.max_y_pos = max_y_pos
        self.use_line=use_line
        super().__init__(goal_low=goal_low, goal_high=goal_high, **kwargs)

    def mocap_set_action(self, action):
        pos_delta = action[None]
        self.reset_mocap2body_xpos()
        new_mocap_pos = self.data.mocap_pos + pos_delta
        new_mocap_pos[0, 0] = np.clip(
            new_mocap_pos[0, 0],
            -self.max_x_pos,
            self.max_x_pos,
        )
        new_mocap_pos[0, 1] = np.clip(
            new_mocap_pos[0, 1],
            self.min_y_pos,
            self.max_y_pos,
        )
        new_mocap_pos[0, 2] = np.clip(
            0.06,
            0,
            0.5,
        )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

    def set_to_goal(self, goal):
        goal = goal['state_desired_goal']
        if self.use_line:
            ymax = self.min_y_pos
            x_max = (ymax -.6)/np.tan(goal)[0] - .15
            if np.abs(x_max) > self.max_x_pos:
                xmax = self.max_x_pos
            else:
                xmax = x_max
            x_pos = np.random.uniform(-self.max_x_pos, xmax)
            y_min = self.min_y_pos
            y_max = .6 + np.tan(goal)[0]*(x_pos+.15)
            y_pos = np.random.uniform(y_min, y_max)
            ee_pos = np.array([x_pos, y_pos, .06])
        else:
            ee_pos = np.random.uniform(np.array([-self.max_x_pos, self.min_y_pos, .06]),
                                   np.array([self.max_x_pos, .5, .06]))
        self.set_to_goal_pos(ee_pos)
        self.set_to_goal_angle(goal)

class SawyerPushAndPullDoorEnv(SawyerDoorEnv):
    def __init__(self,
                     goal_low=-.5,
                     goal_high=.5,
                     max_x_pos=.1,
                     min_y_pos=.5,
                     max_y_pos=.7,
                     use_line=True,
                     **kwargs
                ):
        self.quick_init(locals())
        self.max_x_pos = max_x_pos
        self.min_y_pos = min_y_pos
        self.max_y_pos = max_y_pos
        self.use_line = use_line
        super().__init__(goal_low=goal_low, goal_high=goal_high, **kwargs)

    def mocap_set_action(self, action):
        pos_delta = action[None]
        self.reset_mocap2body_xpos()
        new_mocap_pos = self.data.mocap_pos + pos_delta
        new_mocap_pos[0, 0] = np.clip(
            new_mocap_pos[0, 0],
            -self.max_x_pos,
            self.max_x_pos,
        )
        new_mocap_pos[0, 1] = np.clip(
            new_mocap_pos[0, 1],
            self.min_y_pos,
            self.max_y_pos,
        )
        new_mocap_pos[0, 2] = np.clip(
            0.06,
            0,
            0.5,
        )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

    def set_to_goal(self, goal):
        state_goal = goal['state_desired_goal']
        if state_goal > 0:
            ee_pos = np.random.uniform(np.array([-self.max_x_pos, .5, .06]),
                                       np.array([self.max_x_pos, .6, .06]))
            self.set_to_goal_pos(ee_pos)
        else:
            if self.use_line:
                ymax = self.min_y_pos
                x_max = (ymax - .6) / np.tan(state_goal)[0] - .15
                if np.abs(x_max) > self.max_x_pos:
                    xmax = self.max_x_pos
                else:
                    xmax = x_max
                x_pos = np.random.uniform(-self.max_x_pos, xmax)
                y_min = self.min_y_pos
                y_max = .6 + np.tan(state_goal)[0] * (x_pos + .15)
                y_pos = np.random.uniform(y_min, y_max)
                ee_pos = np.array([x_pos, y_pos, .06])
            else:
                ee_pos = np.random.uniform(np.array([-self.max_x_pos, self.min_y_pos, .06]),
                                           np.array([self.max_x_pos, .5, .06]))
            self.set_to_goal_pos(ee_pos)
        self.set_to_goal_angle(state_goal)
if __name__ == "__main__":
    import pygame
    from pygame.locals import QUIT, KEYDOWN

    pygame.init()

    screen = pygame.display.set_mode((400, 300))

    char_to_action = {
        'w': np.array([0, 1, 0, 0]),
        'a': np.array([-1, 0, 0, 0]),
        's': np.array([0, -1, 0, 0]),
        'd': np.array([1, 0, 0, 0]),
        'q': np.array([1, -1, 0, 0]),
        'e': np.array([-1, -1, 0, 0]),
        'z': np.array([1, 1, 0, 0]),
        'c': np.array([-1, 1, 0, 0]),
        'x': 'toggle',
        'r': 'reset',
    }
    # np.random.seed(100)
    env = SawyerDoorPullOpenEnv(fix_goal=True, min_y_pos=.3)
    policy = ZeroPolicy(env.action_space.low.size)
    es = OUStrategy(
        env.action_space,
        theta=1
    )
    es = EpsilonGreedy(
        action_space=env.action_space,
        prob_random_action=0.1,
    )
    policy = exploration_policy = PolicyWrappedWithExplorationStrategy(
        exploration_strategy=es,
        policy=policy,
    )

    env.reset()
    ACTION_FROM = 'hardcoded'
    # ACTION_FROM = 'pd'
    # ACTION_FROM = 'random'
    H = 10000
    # H = 300
    # H = 50
    goal = .25

    while True:
        lock_action = False
        obs = env.reset()
        last_reward_t = 0
        returns = 0
        action, _ = policy.get_action(None)
        goal=env._state_goal
        print(goal)
        for t in range(H):
            done = False
            if ACTION_FROM == 'controller':
                if not lock_action:
                    action = np.array([0, 0, 0, 0])
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
                            action = np.array([0, 0, 0, 0])
                        print("got char:", char)
                        print("action", action)
                        print("angles", env.data.qpos.copy())
                        print("position", env.get_endeff_pos())
            elif ACTION_FROM=='hardcoded':
                action=np.array([1, -2, 0, 0])
            else:
                action = env.action_space.sample()
            if np.abs(env.data.qpos[0]+.4) < .001:
                print(env.get_endeff_pos())
                break
            # obs, reward, _, info = env.step(action)
            # print(obs['state_observation'][-1])
            # print(env.get_goal()['desired_goal'])
            env.set_to_goal_pos(np.array([-.15, .6, -.35]))
            env.render()
            print(t)
            # if done:
            #     break
            # print("new episode")
        break