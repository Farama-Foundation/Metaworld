import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.env_util import get_asset_full_path
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv, _assert_task_is_set


class SawyerLeverPullEnvV2(SawyerXYZEnv):
    """
    Motivation for V2:
        V1 was impossible to solve because the lever would have to be pulled
        through the table in order to reach the target location.
    Changelog from V1 to V2:
        - (7/7/20) Added 3 element lever position to the observation
            (for consistency with other environments)
        - (6/23/20) In `reset_model`, changed `final_pos[2] -= .17` to `+= .17`
            This ensures that the target point is above the table.
    """
    def __init__(self):

        hand_low = (-0.5, 0.40, -0.15)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.7, 0.05)
        obj_high = (0.1, 0.8, 0.05)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([0, 0.7, 0.05]),
            'hand_init_pos': np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0, 0.75, -0.12])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        goal_low = self.hand_low
        goal_high = self.hand_high

        self.max_path_length = 150

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_lever_pull.xml')

    @_assert_task_is_set
    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        reward, reachDist, pullDist = self.compute_reward(action, ob)
        self.curr_path_length += 1

        info = {'reachDist': reachDist, 'goalDist': pullDist, 'epRew' : reward, 'pickRew':None, 'success': float(pullDist <= 0.05)}
        info['goal'] = self.goal

        return ob, reward, False, info

    def _get_pos_objects(self):
        return self.get_site_pos('leverStart')

    def _set_goal_marker(self, goal):
        self.data.site_xpos[self.model.site_name2id('goal')] = (
            goal[:3]
        )

    def reset_model(self):
        self._reset_hand()
        self._state_goal = self.goal.copy()
        self.obj_init_pos = self.init_config['obj_init_pos']

        if self.random_init:
            goal_pos = np.random.uniform(
                self._random_reset_space.low,
                self._random_reset_space.high,
                size=(self._random_reset_space.low.size),
            )
            self.obj_init_pos = goal_pos[:3]
            final_pos = goal_pos.copy()
            final_pos[1] += 0.05
            final_pos[2] += 0.17
            self._state_goal = final_pos

        self.sim.model.body_pos[self.model.body_name2id('lever')] = self.obj_init_pos
        self._set_goal_marker(self._state_goal)
        self.maxPullDist = np.linalg.norm(self._state_goal - self.obj_init_pos)

        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.reachCompleted = False

    def compute_reward(self, actions, obs):
        del actions

        obj = obs[3:6]
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        tcp = (rightFinger + leftFinger)/2
        target = self._state_goal
        _TARGET_RADIUS = 0.05
        tcp_to_obj = np.linalg.norm(obj - tcp)
        obj_to_target = np.linalg.norm(obj - target)
        grasp = reward_utils.tolerance(tcp_to_obj,
                                       bounds=(0, _TARGET_RADIUS),
                                       margin=_TARGET_RADIUS,
                                       sigmoid='long_tail')
        in_place = reward_utils.tolerance(obj_to_target,
                                          bounds=(0, _TARGET_RADIUS),
                                          margin=_TARGET_RADIUS,
                                          sigmoid='long_tail')
        in_place_weight = 10.
        reward = (grasp + in_place_weight * in_place) / (1 + in_place_weight)

        return [reward, tcp_to_obj, obj_to_target]
