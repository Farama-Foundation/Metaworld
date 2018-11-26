from collections import OrderedDict
import numpy as np
from gym.spaces import Dict, Box
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.core.serializable import Serializable
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict, get_asset_full_path
from multiworld.envs.mujoco.mujoco_env import MujocoEnv

class HalfCheetahEnv(MujocoEnv, MultitaskEnv, Serializable):
    def __init__(self, action_scale=1, frame_skip=5, reward_type='vel_distance', indicator_threshold=.1, fixed_goal=5, fix_goal=False, max_speed=6):
        self.quick_init(locals())
        MultitaskEnv.__init__(self)
        self.action_scale = action_scale
        MujocoEnv.__init__(self, self.model_name, frame_skip=frame_skip)
        bounds = self.model.actuator_ctrlrange.copy()
        low = bounds[:, 0]
        high = bounds[:, 1]
        self.action_space = Box(low=low, high=high)
        self.reward_type = reward_type
        self.indicator_threshold=indicator_threshold
        self.fixed_goal = fixed_goal
        self.fix_goal = fix_goal
        self._state_goal = None
        self.goal_space = Box(np.array(-1*max_speed), np.array(max_speed))
        obs_size = self._get_env_obs().shape[0]
        high = np.inf * np.ones(obs_size)
        low = -high
        self.obs_space = Box(low, high)
        self.achieved_goal_space = Box(self.obs_space.low[8], self.obs_space.high[8])
        self.observation_space = Dict([
            ('observation', self.obs_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.achieved_goal_space),
            ('state_observation', self.obs_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.achieved_goal_space),
        ])
        self.reset()

    @property
    def model_name(self):
        return get_asset_full_path('classic_mujoco/half_cheetah.xml')

    def step(self, action):
        action = action * self.action_scale
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        info = self._get_info()
        reward = self.compute_reward(action, ob)
        done = False
        return ob, reward, done, info

    def _get_env_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])
    def _get_obs(self):
        state_obs = self._get_env_obs()
        achieved_goal = state_obs[8]
        return dict(
            observation=state_obs,
            desired_goal=self._state_goal,
            achieved_goal=achieved_goal,
            state_observation=state_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=achieved_goal,
        )

    def _get_info(self, ):
        state_obs = self._get_env_obs()
        xvel = state_obs[8]
        desired_xvel = self._state_goal
        xvel_error = np.linalg.norm(xvel - desired_xvel)
        info = dict()
        info['vel_distance'] = xvel_error
        info['vel_difference'] =np.abs(xvel - desired_xvel)
        info['vel_success'] = (xvel_error < self.indicator_threshold).astype(float)
        return info

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['achieved_goal']
        desired_goals = obs['desired_goal']
        distances = np.linalg.norm(achieved_goals - desired_goals, axis=1)
        if self.reward_type == 'vel_distance':
            r = -distances
        elif self.reward_type == 'vel_success':
            r = -(distances > self.indicator_threshold).astype(float)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def reset(self):
        self.reset_model()
        goal = self.sample_goal()
        self._state_goal = goal['state_desired_goal']
        return self._get_obs()

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'vel_distance',
            'vel_success',
            'vel_difference',
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
        return 1

    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

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

    def set_to_goal(self, goal):
        pass

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
    env = HalfCheetah()
    env.get_goal()
    env.step(np.array(1))
    env.reset()