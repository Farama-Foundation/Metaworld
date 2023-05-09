import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerBasketballV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'gripper': obs[3],
            'ball_pos': obs[4:7],
            'hoop_x': obs[-3],
            'hoop_yz': obs[-2:],
            'unused_info': obs[7:-3],
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=self._desired_pos(o_d), p=50.)
        action['grab_effort'] = self._grab_effort(o_d)
        print(action.array)
        return action.array

    @staticmethod
    def _desired_pos(o_d):
        pos_curr = o_d['hand_pos']
        pos_ball = o_d['ball_pos'] + np.array([.0, .0, .01])
        # X is given by hoop_pos
        # Y varies between .85 and .9, so we take avg
        # Z is constant at .35
        pos_hoop = np.array([o_d['hoop_x'], .875, .35])
        print(f'curr {pos_curr}')
        print(f'ball {pos_ball}')
        print(f'hoop {pos_hoop}')
        if np.linalg.norm(pos_curr[:2] - pos_ball[:2]) > .04:
            print('not over ball')
            return pos_ball + np.array([.0, .0, .3])
        elif abs(pos_curr[2] - pos_ball[2]) > 0.035:
            print('not at same level as ball')
            print(pos_ball)
            print(abs(pos_curr[2] - pos_ball[2]))
            return pos_ball
        elif abs(pos_ball[2] - pos_hoop[2]) > 0.05:
            print('not above hoop')
            return np.array([pos_curr[0], pos_curr[1], pos_hoop[2]])
        else:
            print('move to hoop')
            return pos_hoop

    @staticmethod
    def _grab_effort(o_d):
        pos_curr = o_d['hand_pos']
        pos_ball = o_d['ball_pos']

        if np.linalg.norm(pos_curr[:2] - pos_ball[:2]) > 0.04 \
            or abs(pos_curr[2] - pos_ball[2]) > 0.07:
            return -1.
        else:
            return .6
