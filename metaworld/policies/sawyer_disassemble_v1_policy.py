import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerDisassembleV1Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'wrench_pos': obs[3:6],
            'peg_pos': obs[9:],
            'unused_info': obs[6:9],
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
        pos_wrench = o_d['wrench_pos'] + np.array([.01, -.01, .01])
        pos_peg = o_d['peg_pos'] + np.array([.07, .0, .15])

        # If XY error is greater than 0.02, place end effector above the wrench
        if np.linalg.norm(pos_curr[:2] - pos_wrench[:2]) > 0.02:
            return pos_wrench + np.array([0., 0., 0.12])
        # Once XY error is low enough, drop end effector down on top of wrench
        elif abs(pos_curr[2] - pos_wrench[2]) > 0.03:
            return pos_wrench
        # If still hooked on peg, move upwards
        elif pos_wrench[2] < 0.12:
            return pos_peg + np.array([.0, .0, .1])
        # Move away from peg
        else:
            return pos_curr + np.array([.0, -.1, .0])

    @staticmethod
    def _grab_effort(o_d):
        pos_curr = o_d['hand_pos']
        pos_wrench = o_d['wrench_pos'] + np.array([.01, .0, .0])

        if np.linalg.norm(pos_curr[:2] - pos_wrench[:2]) > 0.02 or abs(pos_curr[2] - pos_wrench[2]) > 0.08:
            return 0.
        else:
            return 0.8
