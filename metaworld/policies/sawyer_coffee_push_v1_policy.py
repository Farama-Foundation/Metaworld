import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerCoffeePushV1Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'mug_pos': obs[3:-3],
            'extra_info': obs[-3:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=10.)
        action['grab_effort'] = self._grab_effort(o_d)

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_mug = o_d['mug_pos'] + np.array([.0, .0, .01])

        if np.linalg.norm(pos_curr[:2] - pos_mug[:2]) > 0.06:
            return pos_mug + np.array([.0, .0, .3])
        elif abs(pos_curr[2] - pos_mug[2]) > 0.02:
            return pos_mug
        elif abs(pos_curr[0] - -.1) > 0.03:
            return np.array([-.1, pos_curr[1], .1])
        else:
            return np.array([-.1, .8, .1])

    @staticmethod
    def _grab_effort(o_d):
        pos_curr = o_d['hand_pos']
        pos_mug = o_d['mug_pos']

        if np.linalg.norm(pos_curr[:2] - pos_mug[:2]) > 0.06 or \
                abs(pos_curr[2] - pos_mug[2]) > 0.15:
            return -1.
        else:
            return .5
