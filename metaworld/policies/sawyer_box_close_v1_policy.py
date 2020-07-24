import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerBoxCloseV1Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'lid_pos': obs[3:6],
            'box_pos': obs[9:11],
            'extra_info': obs[[6, 7, 8, 11]],
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
        pos_lid = o_d['lid_pos'] + np.array([-.04, .0, -.06])
        pos_box = np.array([*o_d['box_pos'], 0.15]) + np.array([-.04, .0, .0])

        # If error in the XY plane is greater than 0.02, place end effector above the puck
        if np.linalg.norm(pos_curr[:2] - pos_lid[:2]) > 0.01:
            return pos_lid + np.array([0., 0., 0.1])
        # Once XY error is low enough, drop end effector down on top of puck
        elif abs(pos_curr[2] - pos_lid[2]) > 0.05:
            return pos_lid
        # If not at the same Z height as the goal, move up to that plane
        elif abs(pos_curr[2] - pos_box[2]) > 0.04:
            return np.array([pos_curr[0], pos_curr[1], pos_box[2]])
        # Move to the goal
        else:
            return pos_box

    @staticmethod
    def _grab_effort(o_d):
        pos_curr = o_d['hand_pos']
        pos_puck = o_d['lid_pos'] + np.array([-.04, .0, -.06])

        if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.01 or abs(pos_curr[2] - pos_puck[2]) > 0.13:
            return 0.
        # While end effector is moving down toward the puck, begin closing the grabber
        else:
            return .8
