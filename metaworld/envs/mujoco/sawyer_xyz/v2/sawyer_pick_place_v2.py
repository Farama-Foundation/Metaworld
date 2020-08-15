import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerPickPlaceEnvV2(SawyerXYZEnv):
    """
    Motivation for V2:
        V1 was very difficult to solve because the observation didn't say where
        to move after picking up the puck.
    Changelog from V1 to V2:
        - (7/7/20) Removed 3 element vector. Replaced with 3 element position
            of the goal (for consistency with other environments)
        - (6/15/20) Added a 3 element vector to the observation. This vector
            points from the end effector to the goal coordinate.
            i.e. (self._target_pos - pos_hand)
        - (6/15/20) Separated reach-push-pick-place into 3 separate envs.
    """
    def __init__(self):
        liftThresh = 0.04

        goal_low = (-0.1, 0.8, 0.05)
        goal_high = (0.1, 0.9, 0.3)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.6, 0.02)
        obj_high = (0.1, 0.7, 0.02)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': .3,
            'obj_init_pos': np.array([0, 0.6, 0.02]),
            'hand_init_pos': np.array([0, .6, .2]),
        }

        self.goal = np.array([0.1, 0.8, 0.2])

        self.obj_init_angle = self.init_config['obj_init_angle']
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.liftThresh = liftThresh
        self.max_path_length = 500

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.num_resets = 0
        self.obj_init_pos = None

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_pick_place_v2.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)

        obs = self._get_obs()

        reward, tcp_to_obj, tcp_open, obj_to_target, grasp_reward, in_place_reward = self.compute_reward(action, obs)
        success = float(obj_to_target <= 0.07)
        near_object = float(tcp_to_obj <= 0.03)
        grasp_success = float(tcp_open <= 0.73 and near_object)
        info = {
            'success': success,
            'near_object': near_object,
            'grasp_success': grasp_success,
            'grasp_reward': grasp_reward,
            'in_place_reward': in_place_reward
        }

        self.curr_path_length += 1
        return obs, reward, False, info

    def _get_pos_objects(self):
        return self.get_body_com('obj')

    def fix_extreme_obj_pos(self, orig_init_pos):
        # This is to account for meshes for the geom and object are not
        # aligned. If this is not done, the object could be initialized in an
        # extreme position
        diff = self.get_body_com('obj')[:2] - \
               self.get_body_com('obj')[:2]
        adjusted_pos = orig_init_pos[:2] + diff
        # The convention we follow is that body_com[2] is always 0,
        # and geom_pos[2] is the object height
        return [
            adjusted_pos[0],
            adjusted_pos[1],
            self.get_body_com('obj')[-1]
        ]

    def reset_model(self):
        self._reset_hand()
        self._target_pos = self.goal.copy()
        self.obj_init_pos = self.fix_extreme_obj_pos(self.init_config['obj_init_pos'])
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.objHeight = self.get_body_com('obj')[2]
        self.heightTarget = self.objHeight + self.liftThresh

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            self._target_pos = goal_pos[3:]
            while np.linalg.norm(goal_pos[:2] - self._target_pos[:2]) < 0.15:
                goal_pos = self._get_state_rand_vec()
                self._target_pos = goal_pos[3:]
            self._target_pos = goal_pos[-3:]
            self.obj_init_pos = goal_pos[:3]
            finger_right, finger_left = (
                self.get_site_pos('rightEndEffector'),
                self.get_site_pos('leftEndEffector')
            )
            self.init_tcp = (finger_right + finger_left) / 2

        self._set_obj_xyz(self.obj_init_pos)
        self.maxPlacingDist = np.linalg.norm(
            np.array([self.obj_init_pos[0],
                      self.obj_init_pos[1],
                      self.heightTarget]) -
            np.array(self._target_pos)) + self.heightTarget
        self.target_reward = 1000*self.maxPlacingDist + 1000*2
        self.num_resets += 1

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand()

        finger_right, finger_left = (
            self._get_site_pos('rightEndEffector'),
            self._get_site_pos('leftEndEffector')
        )
        self.init_finger_center = (finger_right + finger_left) / 2
        self.pick_completed = False

    def compute_reward(self, action, obs):
        finger_right, finger_left = (
            self._get_site_pos('rightEndEffector'),
            self._get_site_pos('leftEndEffector')
        )
        tcp = (finger_right + finger_left) / 2
        obj = obs[4:7]
        tcp_opened = obs[3]
        target = self._state_goal
        _TARGET_RADIUS_GRASP = 0.03
        _TARGET_RADIUS = 0.07
        tcp_to_obj = np.linalg.norm(obj - tcp)

        tcp_obj_margin = (np.linalg.norm(self.obj_init_pos - self.init_tcp)
            - _TARGET_RADIUS_GRASP)
        tcp_obj = reward_utils.tolerance(tcp_to_obj,
                                bounds=(0, _TARGET_RADIUS_GRASP),
                                margin=tcp_obj_margin,
                                sigmoid='long_tail',)
        
        # rewards for closing the gripper
        tcp_opened_margin = 0.73  # computed using scripted policy manually
        tcp_close = reward_utils.tolerance(tcp_opened,
                                bounds=(0, tcp_opened_margin),
                                margin=1 - tcp_opened_margin,
                                sigmoid='long_tail',)

        # based on Hamacher Product T-Norm
        hammacher_prod_tcp_obj_tcp_close = (tcp_obj * tcp_close) / (tcp_obj + tcp_close - (tcp_obj * tcp_close))
        grasp = (tcp_obj + hammacher_prod_tcp_obj_tcp_close) / 2
        if tcp_opened <= tcp_opened_margin and tcp_to_obj <= _TARGET_RADIUS_GRASP:
            assert grasp >= 1.
        else:
            if not grasp < 1.:
                import ipdb; ipdb.set_trace()
            
        obj_to_target = np.linalg.norm(obj - target)

        in_place_margin = (np.linalg.norm(self.obj_init_pos - target))
        in_place = reward_utils.tolerance(obj_to_target,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=in_place_margin,
                                    sigmoid='long_tail',)
                                    # value_at_margin=0.05)
        # based on Hamacher Product T-Norm
        in_place_and_grasp = (in_place * grasp) / (in_place + grasp - (in_place * grasp))
        assert in_place_and_grasp <= 1.
        # here's a simple fix for most "hoving is equivalent to finishing" 
        # issues: add a small control cost to to the reward function
        c = 0.05
        reward = in_place_and_grasp - (c * np.linalg.norm(action[:3]))
        return [10 * reward, tcp_to_obj, tcp_opened, obj_to_target, grasp, in_place]
