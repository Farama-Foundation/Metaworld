import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerWindowOpenV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def parse_obs(obs):
        return {
            'hand_xyz': obs[:3],
            'wndw_xyz': obs[3:],
        }

    def get_action(self, obs):
        o_d = self.parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_pow': 3
        })

        action['delta_pos'] = move(o_d['hand_xyz'], to_xyz=self.desired_xyz(o_d), p=25.)
        action['grab_pow'] = 1.

        return action.array

    @staticmethod
    def desired_xyz(o_d):
        pos_curr = o_d['hand_xyz']
        pos_wndw = o_d['wndw_xyz']
        pos_wndw += np.array([-0.03, -0.03, -0.1])

        if np.linalg.norm(pos_curr[:2] - pos_wndw[:2]) > 0.04:
            return pos_wndw + np.array([0., 0., 0.3])
        elif abs(pos_curr[2] - pos_wndw[2]) > 0.02:
            return pos_wndw
        else:
            return pos_wndw + np.array([0.1, 0., 0.])
