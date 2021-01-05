import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerDoorUnlockV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'gripper': obs[3],
            'lock_pos': obs[4:7],
            'unused_info': obs[7:],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=25.)
        action['grab_effort'] = 1.

        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_lock = o_d['lock_pos'] + np.array([-.04, -.02, -.03])

        if np.linalg.norm(pos_curr[:2] - pos_lock[:2]) > 0.02:
            if pos_curr[2] > .15:
                return pos_curr + np.array([.0, -.1, -.1])
            return pos_lock
        else:
            return pos_lock + np.array([.1, .0, .01])
