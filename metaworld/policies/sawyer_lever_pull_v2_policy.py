import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerLeverPullV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_xyz': obs[:3],
            'lever_xyz': obs[3:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_pow': 3
        })

        action['delta_pos'] = move(o_d['hand_xyz'], to_xyz=self._desired_xyz(o_d), p=25.)
        action['grab_pow'] = 1.

        return action.array

    @staticmethod
    def _desired_xyz(o_d):
        pos_curr = o_d['hand_xyz']
        pos_lever = o_d['lever_xyz'] + np.array([.0, -.04, .0])

        if np.linalg.norm(pos_curr[:2] - pos_lever[:2]) > 0.04:
            return pos_lever + np.array([0., 0., 0.3])
        elif abs(pos_curr[2] - pos_lever[2]) > 0.02:
            return pos_lever
        else:
            return pos_lever + np.array([.0, .1, .01])
