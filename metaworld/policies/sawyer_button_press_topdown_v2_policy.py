import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerButtonPressTopdownV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'hand_closed': obs[3],
            'button_pos': obs[4:7],
            'unused_info': obs[7:],
        }

    def get_action(self, state, p_scale=1.0):
        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(state.pos_hand, to_xyz=self._desired_pos(state), p=25. * p_scale)
        action['grab_effort'] = 1.

        return action.array

    @staticmethod
    def _desired_pos(state):
        pos_curr = state.pos_hand
        pos_button = state.pos_objs[:3]

        if np.linalg.norm(pos_curr[:2] - pos_button[:2]) > 0.04:
            return pos_button + np.array([0., 0., 0.1])
        else:
            return pos_button
