import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerPlateSlideSideV1Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'puck_pos': obs[3:6],
            'unused_info': obs[6:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=25.)
        action['grab_effort'] = 1.

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_puck = o_d['puck_pos'] + np.array([.07, .0, -.005])

        if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.04:
            return pos_puck + np.array([.0, .0, .1])
        elif abs(pos_curr[2] - pos_puck[2]) > 0.04:
            return pos_puck
        elif pos_curr[0] > -.2:
            return np.array([pos_curr[0] - .1, .63, pos_curr[2]])
        else:
            return pos_puck + np.array([-.1, .0, .0])
