from gym.spaces import Dict
import sawyer_control.envs.sawyer_pushing as sawyer_pushing
from multiworld.core.serializable import Serializable
from multiworld.core.multitask_env import MultitaskEnv

class SawyerPushXYEnv(sawyer_pushing.SawyerPushXYEnv, MultitaskEnv):
    ''' Must Wrap with Image Env to use!'''
    def __init__(self,
                 **kwargs
                ):
        Serializable.quick_init(self, locals())
        sawyer_pushing.SawyerPushXYEnv.__init__(self, **kwargs)
        self.observation_space = Dict([
            ('observation', self.observation_space),
            ('desired_goal', self.goal_space),
            ('achieved_goal', self.goal_space),
            ('state_observation', self.observation_space),
            ('state_desired_goal', self.goal_space),
            ('state_achieved_goal', self.goal_space),
        ])

    def step(self, action):
        self._act(action)
        observation = self._get_obs()
        reward = None
        info = self._get_info()
        done = False
        return observation, reward, done, info

    def compute_rewards(self, actions, obs):
        raise NotImplementedError('Use Image based reward')

    def _get_obs(self):
        achieved_goal = None
        state_obs = super()._get_obs()
        return dict(
            observation=state_obs,
            desired_goal=self._state_goal,
            achieved_goal=achieved_goal,

            state_observation=state_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=achieved_goal,
        )

    def reset(self):
        if self.action_mode == "position":
            self._position_act(self.reset_pos - self._get_endeffector_pose(), in_reset=True)
        else:
            self._reset_robot()
        goal = self.sample_goal()
        self._state_goal = goal['state_desired_goal']
        return self._get_obs()

    """
    Multitask functions
    """

    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

    def sample_goal(self):
        return MultitaskEnv.sample_goal(self)

    def sample_goals(self, batch_size):
        goals = super().sample_goals(batch_size)
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def set_to_goal(self, goal):
        goal = goal['state_desired_goal']
        super().set_to_goal(goal)

if __name__=="__main__":
    env = SawyerPushXYEnv()
    env.reset()
    env.set_to_goal({'state_desired_goal':[0, .1]})
