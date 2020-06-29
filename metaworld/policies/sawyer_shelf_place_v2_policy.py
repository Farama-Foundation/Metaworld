import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerShelfPlaceV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_xyz': obs[:3],
            'block_xyz': obs[3:-1],
            'x_vec': obs[-1:]
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
        pos_block = o_d['block_xyz'] + np.array([.005, .0, .015])
        x_vec = o_d['x_vec']
        if np.linalg.norm(pos_curr[:2] - pos_block[:2]) > 0.04:
            # positioning over block
            return pos_block + np.array([0., 0., 0.3])
        elif abs(pos_curr[2] - pos_block[2]) > 0.02:
            # grabbing block 
            return pos_block
        elif np.abs(x_vec) > 0.02:
            # centering with goal pos
            pos_new = pos_curr + np.array([x_vec, 0., 0.])
            return pos_new
        elif pos_curr[2] < 0.25:
            # move up to correct height
            pos_new = pos_curr + np.array([0., 0., 0.25])
            return pos_new
        else:
            # move forward to goal
            pos_new = pos_curr + np.array([0., 0.05, 0.])
            return pos_new

    @staticmethod
    def _grab_pow(o_d):
        pos_curr = o_d['hand_xyz']
        pos_block = o_d['block_xyz']

        if np.linalg.norm(pos_curr[:2] - pos_block[:2]) > 0.04 \
            or abs(pos_curr[2] - pos_block[2]) > 0.15:
            return -1.
        else:
            return .7
