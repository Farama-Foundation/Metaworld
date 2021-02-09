import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerPegInsertionSideV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'gripper_distance_apart': obs[3],
            'peg_pos': obs[4:7],
            'peg_rot': obs[7:11],
            'goal_pos': obs[-3:],
            'unused_info_curr_obs': obs[11:18],
            '_prev_obs': obs[18:36]
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=25.)
        action['grab_effort'] = self._grab_effort(o_d)

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_peg = o_d['peg_pos']
        # lowest X is -.35, doesn't matter if we overshoot
        # Y is given by hole_vec
        # Z is constant at .16
        pos_hole = np.array([-.35, o_d['goal_pos'][1], .16])

        if np.linalg.norm(pos_curr[:2] - pos_peg[:2]) > .04:
            return pos_peg + np.array([.0, .0, .3])
        elif abs(pos_curr[2] - pos_peg[2]) > .025:
            return pos_peg
        elif np.linalg.norm(pos_peg[1:] - pos_hole[1:]) > 0.03:
            return pos_hole + np.array([.4, .0, .0])
        else:
            return pos_hole

    @staticmethod
    def _grab_effort(o_d):
        pos_curr = o_d['hand_pos']
        pos_peg = o_d['peg_pos']

        if np.linalg.norm(pos_curr[:2] - pos_peg[:2]) > 0.04 \
            or abs(pos_curr[2] - pos_peg[2]) > 0.15:
            return -1.
        else:
            return .6
