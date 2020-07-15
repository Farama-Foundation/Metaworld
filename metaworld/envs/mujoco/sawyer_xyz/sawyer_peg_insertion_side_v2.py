import numpy as np
from gym.spaces import Box

from metaworld.envs.env_util import get_asset_full_path
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv, _assert_task_is_set


class SawyerPegInsertionSideEnvV2(SawyerXYZEnv):
    """
    Motivation for V2:
        V1 was difficult to solve because the observation didn't say where
        to insert the peg (the hole's location). Furthermore, the hole object
        could be initialized in such a way that it severely restrained the
        sawyer's movement.
    Changelog from V1 to V2:
        - (7/7/20) Removed 1 element vector. Replaced with 3 element position
            of the hole (for consistency with other environments)
        - (6/16/20) Added a 1 element vector to the observation. This vector
            points from the end effector to the hole in the Y direction.
            i.e. (self._state_goal - pos_hand)[1]
        - (6/16/20) Used existing goal_low and goal_high values to constrain
            the hole's position, as opposed to hand_low and hand_high
    """
    def __init__(self):
        liftThresh = 0.11
        hand_init_pos = (0, 0.6, 0.2)

        goal_low = (-0.35, 0.5, 0.05)
        goal_high = (-0.25, 0.8, 0.05)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.5, 0.02)
        obj_high = (0.1, 0.7, 0.02)


        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([0, 0.6, 0.02]),
            'hand_init_pos': np.array([0, .6, .2]),
        }

        self.goal = np.array([-0.3, 0.6, 0.05])

        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.liftThresh = liftThresh
        self.max_path_length = 150

        self.hand_init_pos = np.array(hand_init_pos)

        self.obj_and_goal_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.observation_space = Box(
            np.hstack((self.hand_low, obj_low, obj_low, goal_low)),
            np.hstack((self.hand_high, obj_high, obj_high, goal_high)),
        )

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_peg_insertion_side.xml')

    @_assert_task_is_set
    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation([action[-1], -action[-1]])

        ob = self._get_obs()
        obs_dict = self._get_obs_dict()

        rew, reach_dist, pick_rew, placing_dist = self.compute_reward(action, obs_dict)
        success = float(placing_dist <= 0.07)

        info = {
            'reachDist': reach_dist,
            'pickRew': pick_rew,
            'epRew': rew,
            'goalDist': placing_dist,
            'success': success,
            'goal': self.goal
        }

        self.curr_path_length += 1
        return ob, rew, False, info

    def _get_pos_objects(self):
        return self.get_body_com('peg')

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()

        self.sim.model.body_pos[self.model.body_name2id('box')] = np.array([-0.3, 0.6, 0.05])
        self._state_goal = self.sim.model.site_pos[self.model.site_name2id('hole')] + self.sim.model.body_pos[self.model.body_name2id('box')]
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.objHeight = self.get_body_com('peg').copy()[2]
        self.heightTarget = self.objHeight + self.liftThresh

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.1:
                goal_pos = self._get_state_rand_vec()
            self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))
            self.sim.model.body_pos[self.model.body_name2id('box')] = goal_pos[-3:]
            self._state_goal = self.sim.model.site_pos[self.model.site_name2id('hole')] + self.sim.model.body_pos[self.model.body_name2id('box')]

        self._set_obj_xyz(self.obj_init_pos)
        self.obj_init_pos = self.get_body_com('peg')
        self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) - np.array(self._state_goal)) + self.heightTarget
        self.target_reward = 1000*self.maxPlacingDist + 1000*2
        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
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
        obs = obs['state_observation']
        pos_obj = obs[3:6]
        pos_peg_head = self.get_site_pos('pegHead')

        finger_right, finger_left = (
            self.get_site_pos('rightEndEffector'),
            self.get_site_pos('leftEndEffector')
        )
        finger_center = (finger_right + finger_left) / 2
        heightTarget = self.heightTarget

        tolerance = 0.01
        self.pick_completed = pos_obj[2] >= (heightTarget - tolerance)


        placingGoal = self._state_goal

        reach_dist = np.linalg.norm(pos_obj - finger_center)

        placingDistHead = np.linalg.norm(pos_peg_head - placingGoal)
        placing_dist = np.linalg.norm(pos_obj - placingGoal)

        def obj_dropped():
            # Object on the ground, far away from the goal, and from the gripper
            # Can tweak the margin limits
            return (pos_obj[2] < (self.objHeight + 0.005))\
                   and (placing_dist > 0.02)\
                   and (reach_dist > 0.02)

        def reach_reward():
            reach_xy = np.linalg.norm(pos_obj[:-1] - finger_center[:-1])
            z_rew = np.linalg.norm(
                finger_center[-1] - self.init_finger_center[-1])

            reach_rew = -reach_dist if reach_xy < 0.05 else -reach_xy - z_rew
            # Incentive to close fingers when reachDist is small
            if reach_dist < 0.05:
                reach_rew = -reach_dist + max(actions[-1], 0)/50

            return reach_rew, reach_dist

        def pick_reward():
            h_scale = 100
            if self.pick_completed and not obj_dropped():
                return h_scale * heightTarget
            elif (reach_dist < 0.1) and (pos_obj[2] > (self.objHeight + 0.005)):
                return h_scale * min(heightTarget, pos_obj[2])
            else:
                return 0

        def place_reward():
            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            if self.pick_completed and reach_dist < 0.1 and not obj_dropped():
                if placingDistHead <= 0.05:
                    place_rew = c1 * (self.maxPlacingDist - placing_dist) + \
                                c1 * (np.exp(-(placing_dist ** 2) / c2) +
                                      np.exp(-(placing_dist ** 2) / c3))
                else:
                    place_rew = c1 * (self.maxPlacingDist - placingDistHead) + \
                                c1 * (np.exp(-(placingDistHead ** 2) / c2) +
                                      np.exp(-(placingDistHead ** 2) / c3))
                place_rew = max(place_rew, 0)
                return [place_rew, placing_dist]
            else:
                return [0, placing_dist]

        reach_rew, reach_dist = reach_reward()
        pick_rew = pick_reward()
        place_rew, placing_dist = place_reward()
        assert ((place_rew >= 0) and (pick_rew >= 0))

        reward = reach_rew + pick_rew + place_rew
        return [reward, reach_dist, pick_rew, placing_dist]
