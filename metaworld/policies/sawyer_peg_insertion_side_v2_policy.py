import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerPegInsertionSideV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_xyz': obs[:3],
            'peg_xyz': obs[3:-1],
            'hole_vec': obs[-1]
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
        pos_peg = o_d['peg_xyz'] + np.array([.0, .0, .01])
        # lowest X is -.35, doesn't matter if we overshoot
        # Y is given by hole_vec
        # Z is constant at .16
        pos_hole = np.array([-.35, pos_curr[1] + o_d['hole_vec'], .16])

        if np.linalg.norm(pos_curr[:2] - pos_peg[:2]) > .04:
            return pos_peg + np.array([.0, .0, .3])
        elif abs(pos_curr[2] - pos_peg[2]) > .025:
            return pos_peg
        elif np.linalg.norm(pos_peg[1:] - pos_hole[1:]) > 0.04:
            return pos_hole + np.array([.3, .0, .0])
        else:
            return pos_hole

    @staticmethod
    def _grab_pow(o_d):
        pos_curr = o_d['hand_xyz']
        pos_peg = o_d['peg_xyz']

        if np.linalg.norm(pos_curr[:2] - pos_peg[:2]) > 0.04 \
            or abs(pos_curr[2] - pos_peg[2]) > 0.15:
            return -1.
        else:
            return .6
