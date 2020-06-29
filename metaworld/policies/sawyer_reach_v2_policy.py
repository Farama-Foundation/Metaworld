import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerReachV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_xyz': obs[:3],
            'puck_xyz': obs[3:-3],
            'goal_vec': obs[-3:]
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_pow': 3
        })

        action['delta_pos'] = move(o_d['hand_xyz'], to_xyz=self._desired_xyz(o_d), p=5.)
        action['grab_pow'] = 0.

        return action.array

    @staticmethod
    def _desired_xyz(o_d):
        return o_d['hand_xyz'] + o_d['goal_vec']
