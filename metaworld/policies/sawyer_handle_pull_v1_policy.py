import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerHandlePullV1Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'handle_pos': obs[3:6],
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
        pos_button = o_d['handle_pos'] + np.array([.0, -.02, .0])

        if abs(pos_curr[0] - pos_button[0]) > 0.04:
            return pos_button + np.array([0., 0., 0.2])
        elif abs(pos_curr[2] - pos_button[2]) > 0.03:
            return pos_button + np.array([.0, -.1, -.01])
        elif abs(pos_curr[1] - pos_button[1]) > .01:
            return np.array([pos_button[0], pos_button[1] + .04, pos_curr[2]])
        else:
            return pos_button + np.array([.0, .04, .1])
