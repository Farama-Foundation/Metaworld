import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, move, assert_fully_parsed


class SawyerPickPlaceWallV2Policy(Policy):

    @staticmethod
    @assert_fully_parsed
    def _parse_obs(obs):
        return {
            'hand_xyz': obs[:3],
            'puck_xyz': obs[3:-3],
            'goal_vec': obs[-3:]
        }

    def get_action(self, obs):
        o_d = self._parse_obs(obs)

        action = Action({
            'delta_pos': np.arange(3),
            'grab_pow': 3
        })

        action['delta_pos'] = move(o_d['hand_xyz'], to_xyz=self.desired_xyz(o_d), p=10.)
        action['grab_pow'] = self.grab_pow(o_d)

        return action.array

    @staticmethod
    def desired_xyz(o_d):
        pos_curr = o_d['hand_xyz']
        pos_puck = o_d['puck_xyz']
        goal_vec = o_d['goal_vec']

        # If error in the XY plane is greater than 0.02, place end effector above the puck
        if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.02:
            return pos_puck + np.array([0., 0., 0.1])
        # Once XY error is low enough, drop end effector down on top of puck
        elif abs(pos_curr[2] - pos_puck[2]) > 0.05 and pos_puck[-1] < 0.03:
            return pos_puck + np.array([0., 0., 0.03])
        # Move to the goal
        else:
            # if wall is in the way of arm, straight up above the wall
            if(-0.15 <= pos_curr[0] <= 0.35 and
                    0.60 <= pos_curr[1] <= 0.80 and
                    pos_curr[2] < 0.25):
                    return pos_curr + [0, 0, 1]
            #move towards the goal while staying above the wall
            elif(-0.15 <= pos_curr[0] <= 0.35 and
                    0.60 <= pos_curr[1] <= 0.80 and
                    pos_curr[2] < 0.35):
                return pos_curr + [goal_vec[0], goal_vec[1], 0]
            # If not at the same Z height as the goal, move up to that plane
            elif abs(goal_vec[-1]) > 0.04:
                return pos_curr + np.array([0., 0., goal_vec[-1]])
            return pos_curr + goal_vec

    @staticmethod
    def grab_pow(o_d):
        pos_curr = o_d['hand_xyz']
        pos_puck = o_d['puck_xyz']

        if np.linalg.norm(pos_curr[:2] - pos_puck[:2]) > 0.02 or abs(pos_curr[2] - pos_puck[2]) > 0.1:
            return 0.
        # While end effector is moving down toward the puck, begin closing the grabber
        else:
            return 0.6
