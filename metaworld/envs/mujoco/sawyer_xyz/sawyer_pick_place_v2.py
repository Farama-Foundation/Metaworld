import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.env_util import get_asset_full_path
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv, _assert_task_is_set


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
            i.e. (self._state_goal - pos_hand)
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
        return get_asset_full_path('sawyer_xyz/sawyer_pick_place_v2.xml')

    @_assert_task_is_set
    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)

        obs = self._get_obs()
        obs_dict = self._get_obs_dict()

        reward, tcp_to_obj, _, obj_to_target = self.compute_reward(action, obs_dict)
        success = float(obj_to_target <= 0.07)
        # success = float(tcp_to_obj <= 0.03)

        info = {
            'success': success,
        }

        self.curr_path_length += 1
        return obs, reward, False, info

    def _get_pos_objects(self):
        return self.get_body_com('obj')

    def _set_goal_marker(self, goal):
        self.data.site_xpos[self.model.site_name2id('goal')] = goal[:3]

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

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
        self._state_goal = self.goal.copy()
        self.obj_init_pos = self.fix_extreme_obj_pos(self.init_config['obj_init_pos'])
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.objHeight = self.get_body_com('obj')[2]
        self.heightTarget = self.objHeight + self.liftThresh

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            self._state_goal = goal_pos[3:]
            while np.linalg.norm(goal_pos[:2] - self._state_goal[:2]) < 0.15:
                goal_pos = self._get_state_rand_vec()
                self._state_goal = goal_pos[3:]
            self._state_goal = goal_pos[-3:]
            self.obj_init_pos = goal_pos[:3]
            finger_right, finger_left = (
                self.get_site_pos('rightEndEffector'),
                self.get_site_pos('leftEndEffector')
            )
            self.init_tcp = (finger_right + finger_left) / 2

        self._set_goal_marker(self._state_goal)
        self._set_obj_xyz(self.obj_init_pos)
        self.maxPlacingDist = np.linalg.norm(
            np.array([self.obj_init_pos[0],
                      self.obj_init_pos[1],
                      self.heightTarget]) -
            np.array(self._state_goal)) + self.heightTarget
        self.target_reward = 1000*self.maxPlacingDist + 1000*2
        self.num_resets += 1

        return self._get_obs()

    def _reset_hand(self):
        for _ in range(50):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1, 1], self.frame_skip)

        finger_right, finger_left = (
            self.get_site_pos('rightEndEffector'),
            self.get_site_pos('leftEndEffector')
        )
        self.init_finger_center = (finger_right + finger_left) / 2
        self.pick_completed = False

    def compute_reward(self, actions, obs):
        finger_right, finger_left = (
            self.get_site_pos('rightEndEffector'),
            self.get_site_pos('leftEndEffector')
        )
        obs = obs['state_observation']
        tcp = (finger_right + finger_left) / 2
        obj = obs[3:6]
        target = self._state_goal
        _TARGET_RADIUS_GRASP = 0.03
        _TARGET_RADIUS = 0.07
        tcp_to_obj = np.linalg.norm(obj - tcp)

        grasp_margin = (np.linalg.norm(self.obj_init_pos - self.init_tcp)
            - _TARGET_RADIUS_GRASP)
        grasp = reward_utils.tolerance(tcp_to_obj,
                                bounds=(0, _TARGET_RADIUS_GRASP),
                                margin=grasp_margin,
                                sigmoid='long_tail',)

        obj_to_target = np.linalg.norm(obj - target)

        in_place_margin = (np.linalg.norm(self.obj_init_pos - target)
            - _TARGET_RADIUS)
        in_place = reward_utils.tolerance(obj_to_target,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=in_place_margin,
                                    sigmoid='long_tail',)
                                    # value_at_margin=0.05)
        in_place_weight = 5.
        # based on Hamacher Product T-Norm
        in_place_and_grasp = (in_place * grasp) / (in_place + grasp - (in_place * grasp))
        assert in_place_and_grasp <= 1.
        reward = (grasp + in_place_weight * in_place_and_grasp) / (1 + in_place_weight)
        if obj_to_target <= _TARGET_RADIUS and tcp_to_obj <= _TARGET_RADIUS_GRASP:
            assert reward >= 1.
        else:
            assert reward < 1.
        return [reward, tcp_to_obj, 0, obj_to_target]
