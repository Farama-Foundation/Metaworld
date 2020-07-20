import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerDialTurnV1Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'dial_pos': obs[3:6],
            'goal_pos': obs[6:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_pow': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_xyz(o_d), p=5.)
        action['grab_pow'] = 0.

        return action.array

    @staticmethod
    def _desired_xyz(o_d):
        hand_pos = o_d['hand_pos']
        dial_pos = o_d['dial_pos'] + np.array([0.0, -0.028, 0.0])
        if abs(hand_pos[2] - dial_pos[2]) > 0.02:
            return np.array([hand_pos[0], hand_pos[1], dial_pos[2]])
        elif abs(hand_pos[1] - dial_pos[1]) > 0.02:
            return np.array([dial_pos[0]+0.20, dial_pos[1], dial_pos[2]])
        return np.array([dial_pos[0]-0.10, dial_pos[1], dial_pos[2]])
