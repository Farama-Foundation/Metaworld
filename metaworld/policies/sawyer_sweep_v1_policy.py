import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerSweepV1Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_xyz': obs[:3],
            'cube_xyz': obs[3:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_pow': 3
        })

        action['delta_pos'] = move(o_d['hand_xyz'], to_xyz=self._desired_xyz(o_d), p=25.)
        action['grab_pow'] = self._grab_pow(o_d)

        return action.array

    @staticmethod
    def _desired_xyz(o_d):
        pos_curr = o_d['hand_xyz']
        pos_cube = o_d['cube_xyz'] + np.array([.0, .0, .015])

        if pos_curr[0] < .2:
            if np.linalg.norm(pos_curr[:2] - pos_cube[:2]) > 0.04:
                return pos_cube + np.array([0., 0., 0.3])
            elif abs(pos_curr[2] - pos_cube[2]) > 0.02:
                return pos_cube

        return np.array([.5, pos_cube[1], .1])

    @staticmethod
    def _grab_pow(o_d):
        pos_curr = o_d['hand_xyz']
        pos_cube = o_d['cube_xyz']

        if np.linalg.norm(pos_curr[:2] - pos_cube[:2]) > 0.04 \
            or abs(pos_curr[2] - pos_cube[2]) > 0.15:
            return -1.
        elif pos_cube[0] < .35:
            return .7
        else:
            return -1.
