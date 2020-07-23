import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerAssemblyV1Policy(Policy):

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
        pos_wrench = o_d['wrench_pos'] + np.array([.01, .0, .0])
        pos_peg = o_d['peg_pos'] + np.array([.07, .0, .15])

        # If XY error is greater than 0.02, place end effector above the wrench
        if np.linalg.norm(pos_curr[:2] - pos_wrench[:2]) > 0.02:
            return pos_wrench + np.array([0., 0., 0.1])
        # (For later) if lined up with peg, drop down on top of it
        elif np.linalg.norm(pos_curr[:2] - pos_peg[:2]) <= 0.02:
            return pos_peg + np.array([.0, .0, -.07])
        # Once XY error is low enough, drop end effector down on top of wrench
        elif abs(pos_curr[2] - pos_wrench[2]) > 0.05:
            return pos_wrench + np.array([0., 0., 0.03])
        # If not at the same Z height as the goal, move up to that plane
        elif abs(pos_curr[2] - pos_peg[2]) > 0.04:
            return np.array([pos_curr[0], pos_curr[1], pos_peg[2]])
        # If XY error is greater than 0.02, place end effector above the peg
        else:
            return pos_peg

    @staticmethod
    def _grab_effort(o_d):
        pos_curr = o_d['hand_pos']
        pos_wrench = o_d['wrench_pos'] + np.array([.01, .0, .0])
        pos_peg = o_d['peg_pos'] + np.array([.07, .0, .15])

        if np.linalg.norm(pos_curr[:2] - pos_wrench[:2]) > 0.02 or abs(pos_curr[2] - pos_wrench[2]) > 0.1:
            return 0.
        # Until hovering over peg, keep hold of wrench
        elif np.linalg.norm(pos_curr[:2] - pos_peg[:2]) > 0.01:
            return 0.6
        else:
            return -1.
