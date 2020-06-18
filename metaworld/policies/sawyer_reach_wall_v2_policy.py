import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, move


class SawyerReachWallV2Policy(Policy):

    @staticmethod
    def parse_obs(obs):
        return {
            'hand_xyz': obs[:3],
            'obj_xyz': obs[3:-3],
            'goal_vec': obs[-3:]
        }

    def get_action(self, obs):
        o_d = self.parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_pow': 3
        })

        action['delta_pos'] = move(
            o_d['hand_xyz'],
            to_xyz=self.desired_xyz(o_d), p=5.
        )
        action['grab_pow'] = self.grab_pow(o_d)

        return action.array

    @staticmethod
    def desired_xyz(o_d):
        pos_hand = o_d['hand_xyz']
        if(-0.1 <= pos_hand[0] <= 0.3 and
                0.60 <= pos_hand[1] <= 0.80 and
                pos_hand[2] < 0.2):
            return o_d['hand_xyz'] + \
                [o_d['goal_vec'][0], o_d['goal_vec'][1], 1]
        return o_d['hand_xyz'] + o_d['goal_vec']

    @staticmethod
    def grab_pow(o_d):
        return 0.
