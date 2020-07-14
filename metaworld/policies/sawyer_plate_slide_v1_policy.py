import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerPlateSlideV1Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'puck_pos': obs[3:6],
            'shelf_x': obs[-3],
            'unused_info': obs[[6, 7, 8, 10, 11]],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=10.)
        action['grab_effort'] = -1.

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_puck = o_d['puck_pos'] + np.array([.0, -.055, .03])

        aligned_with_puck = np.linalg.norm(pos_curr[:2] - pos_puck[:2]) <= 0.03

        if not aligned_with_puck:
            return pos_puck + np.array([.0, .0, .1])
        elif abs(pos_curr[2] - pos_puck[2]) > 0.04:
            return pos_puck
        else:
            return np.array([o_d['shelf_x'], .9, pos_puck[2]])
