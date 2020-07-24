import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move


class SawyerSoccerV1Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_pos': obs[:3],
            'ball_pos': obs[3:6],
            'goal_pos': obs[9:],
            'unused_info': obs[6:9],
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
        pos_ball = o_d['ball_pos'] + np.array([.0, .0, .03])
        pos_goal = o_d['goal_pos']

        curr_to_ball = pos_ball - pos_curr
        curr_to_ball /= np.linalg.norm(curr_to_ball)

        ball_to_goal = pos_goal - pos_ball
        ball_to_goal /= np.linalg.norm(ball_to_goal)

        scaling = .1
        if np.dot(curr_to_ball[:2], ball_to_goal[:2]) < .7:
            scaling *= -1

        return pos_ball + scaling * ball_to_goal
