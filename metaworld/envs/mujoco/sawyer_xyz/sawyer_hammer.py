import numpy as np
from gym.spaces import Box

from metaworld.envs.env_util import get_asset_full_path
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv, _assert_task_is_set


class SawyerHammerEnv(SawyerXYZEnv):
    def __init__(self):

        liftThresh = 0.09
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.5, 0.02)
        obj_high = (0.1, 0.6, 0.02)
        goal_low = (0., 0.85, 0.05)
        goal_high = (0.3, 0.9, 0.05)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'hammer_init_pos': np.array([0, 0.6, 0.02]),
            'hand_init_pos': np.array([0, 0.6, 0.2]),
        }
        self.goal = self.init_config['hammer_init_pos']
        self.hammer_init_pos = self.init_config['hammer_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.liftThresh = liftThresh
        self.max_path_length = 200

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
        return get_asset_full_path('sawyer_xyz/sawyer_hammer.xml')

    @_assert_task_is_set
    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation
        ob = self._get_obs()
        obs_dict = self._get_obs_dict()
        reward, _, reachDist, pickRew, _, _, screwDist = self.compute_reward(action, obs_dict)
        self.curr_path_length += 1

        info = {'reachDist': reachDist, 'pickRew':pickRew, 'epRew' : reward, 'goalDist': screwDist, 'success': float(screwDist <= 0.05)}
        info['goal'] = self.goal

        return ob, reward, False, info

    def _get_pos_objects(self):
        return self.get_body_com('hammer').copy()

    def _get_obs_dict(self):
        obs_dict = super()._get_obs_dict()
        obs_dict['state_observation'] = np.concatenate((
            self.get_endeff_pos(),
            self.get_body_com('hammer').copy(),
            self.data.get_geom_xpos('hammerHead').copy(),
            self.data.site_xpos[self.model.site_name2id('screwHead')]
        ))
        obs_dict['state_achieved_goal'] = self.data.site_xpos[self.model.site_name2id('screwHead')]
        return obs_dict

    def _set_hammer_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()
        self.sim.model.body_pos[self.model.body_name2id('box')] = np.array([0.24, 0.85, 0.05])
        self.sim.model.body_pos[self.model.body_name2id('screw')] = np.array([0.24, 0.71, 0.11])
        self._state_goal = self.sim.model.site_pos[self.model.site_name2id('goal')] + self.sim.model.body_pos[self.model.body_name2id('box')]
        self.obj_init_pos = np.array([0.24, 0.71, 0.11])
        self.hammer_init_pos = self.init_config['hammer_init_pos']
        self.hammerHeight = self.get_body_com('hammer').copy()[2]
        self.heightTarget = self.hammerHeight + self.liftThresh

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.1:
                goal_pos = self._get_state_rand_vec()
            self.hammer_init_pos = np.concatenate((goal_pos[:2], [self.hammer_init_pos[-1]]))

        self._set_hammer_xyz(self.hammer_init_pos)
        self.obj_init_pos = self.sim.model.site_pos[self.model.site_name2id('screwHead')] + self.sim.model.body_pos[self.model.body_name2id('screw')]
        self.maxHammerDist = np.linalg.norm(np.array([self.hammer_init_pos[0], self.hammer_init_pos[1], self.heightTarget]) - np.array(self.obj_init_pos)) + \
                                self.heightTarget + np.abs(self.obj_init_pos[1] - self._state_goal[1])

        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False

    def compute_reward(self, actions, obs):
        obs = obs['state_observation']

        hammerPos = obs[3:6]
        hammerHeadPos = self.data.get_geom_xpos('hammerHead').copy()
        objPos = self.data.site_xpos[self.model.site_name2id('screwHead')]

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        heightTarget = self.heightTarget

        hammerDist = np.linalg.norm(objPos - hammerHeadPos)
        screwDist = np.abs(objPos[1] - self._state_goal[1])
        reachDist = np.linalg.norm(hammerPos - fingerCOM)

        def reachReward():
            reachRew = -reachDist
            # incentive to close fingers when reachDist is small
            if reachDist < 0.05:
                reachRew = -reachDist + max(actions[-1],0)/50
            return reachRew , reachDist

        def pickCompletionCriteria():
            tolerance = 0.01
            if hammerPos[2] >= (heightTarget- tolerance):
                return True
            else:
                return False

        if pickCompletionCriteria():
            self.pickCompleted = True


        def objDropped():
            return (hammerPos[2] < (self.hammerHeight + 0.005)) and (hammerDist >0.02) and (reachDist > 0.02)
            # Object on the ground, far away from the goal, and from the gripper
            # Can tweak the margin limits

        def orig_pickReward():
            hScale = 100

            if self.pickCompleted and not(objDropped()):
                return hScale*heightTarget
            elif (reachDist < 0.1) and (hammerPos[2]> (self.hammerHeight + 0.005)) :
                return hScale* min(heightTarget, hammerPos[2])
            else:
                return 0

        def hammerReward():
            c1 = 1000
            c2 = 0.01
            c3 = 0.001

            cond = self.pickCompleted and (reachDist < 0.1) and not(objDropped())
            if cond:
                hammerRew = 1000*(self.maxHammerDist - hammerDist - screwDist) + c1*(np.exp(-((hammerDist+screwDist)**2)/c2) + np.exp(-((hammerDist+screwDist)**2)/c3))
                hammerRew = max(hammerRew,0)
                return [hammerRew , hammerDist, screwDist]
            else:
                return [0 , hammerDist, screwDist]

        reachRew, reachDist = reachReward()
        pickRew = orig_pickReward()
        hammerRew , hammerDist, screwDist = hammerReward()
        assert ((hammerRew >=0) and (pickRew>=0))
        reward = reachRew + pickRew + hammerRew

        return [reward, reachRew, reachDist, pickRew, hammerRew, hammerDist, screwDist]
