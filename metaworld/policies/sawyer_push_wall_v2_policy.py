import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, move


class SawyerPushWallV2Policy(Policy):

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

        action['delta_pos'] = move(o_d['hand_xyz'], to_xyz=self.desired_xyz(o_d), p=10.)
        action['grab_pow'] = self.grab_pow(o_d)

        return action.array

    @staticmethod
    def desired_xyz(o_d):
        pos_curr = o_d['hand_xyz']
        pos_obj = o_d['obj_xyz']

        # If error in the XY plane is greater than 0.02, place end effector above the puck
        if np.linalg.norm(pos_curr[:2] - pos_obj[:2]) > 0.02:
            return pos_obj + np.array([0., 0., 0.2])
        # Once XY error is low enough, drop end effector down on top of obj
        elif abs(pos_curr[2] - pos_obj[2]) > 0.04:
            return pos_obj + np.array([0., 0., 0.03])
        # Move to the goal
        else:
            #if the wall is between the puck and the goal, go around the wall
            if(-0.1 <= pos_obj[0] <= 0.3 and 0.65 <= pos_obj[1] <= 0.75):
                return pos_curr + np.array([-1, 0, 0])
            elif ((-0.15 < pos_obj[0] < 0.05 or 0.15 < pos_obj[0] < 0.35)
                    and 0.695 <= pos_obj[1] <= 0.755):
                return pos_curr + np.array([0, 1, 0])
            return pos_curr + o_d['goal_vec']

    @staticmethod
    def grab_pow(o_d):
        pos_curr = o_d['hand_xyz']
        pos_obj = o_d['obj_xyz']

        if np.linalg.norm(pos_curr[:2] - pos_obj[:2]) > 0.02 or \
                          abs(pos_curr[2] - pos_obj[2]) > 0.1:
            return 0.0
        # While end effector is moving down toward the obj, begin closing the grabber
        else:
            return 0.6
