import numpy as np
from gym.spaces import Box

from metaworld.envs.asset_path_utils import full_v1_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerHammerEnv(SawyerXYZEnv):
    def __init__(self):

        liftThresh = 0.09
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.5, 0.02)
        obj_high = (0.1, 0.6, 0.02)
        goal_low = (0.2399, .7399, 0.109)
        goal_high = (0.2401, .7401, 0.111)

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

        self._random_reset_space = Box(np.array(obj_low), np.array(obj_high))
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v1_path_for('sawyer_xyz/sawyer_hammer.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        reward, _, reachDist, pickRew, _, _, screwDist = self.compute_reward(action, ob)

        info = {
            'reachDist': reachDist,
            'pickRew': pickRew,
            'epRew': reward,
            'goalDist': screwDist,
            'success': float(screwDist <= 0.05)
        }

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

        # Set position of box & nail (these are not randomized)
        self.sim.model.body_pos[self.model.body_name2id(
            'box'
        )] = np.array([0.24, 0.85, 0.05])
        self.sim.model.body_pos[self.model.body_name2id(
            'screw'
        )] = np.array([0.24, 0.71, 0.11])
        # Update _target_pos
        self._target_pos = self._get_site_pos('goal')

        # Update heights (for use in reward function)
        self.hammerHeight = self.get_body_com('hammer').copy()[2]
        self.heightTarget = self.hammerHeight + self.liftThresh

        # Randomize hammer position
        self.hammer_init_pos = self._get_state_rand_vec() if self.random_init \
            else self.init_config['hammer_init_pos']
        self._set_hammer_xyz(self.hammer_init_pos)

        # Update distances (for use in reward function)
        self.obj_init_pos = self.sim.model.site_pos[self.model.site_name2id('screwHead')] + self.sim.model.body_pos[self.model.body_name2id('screw')]
        self.maxHammerDist = np.linalg.norm(np.array([self.hammer_init_pos[0], self.hammer_init_pos[1], self.heightTarget]) - np.array(self.obj_init_pos)) + \
                                self.heightTarget + np.abs(self.obj_init_pos[1] - self._target_pos[1])

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand(10)

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False

    def compute_reward(self, actions, obs):

        hammerPos = obs[3:6]
        hammerHeadPos = self.data.get_geom_xpos('hammerHead').copy()
        objPos = self.data.site_xpos[self.model.site_name2id('screwHead')]

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        heightTarget = self.heightTarget

        hammerDist = np.linalg.norm(objPos - hammerHeadPos)
        screwDist = np.abs(objPos[1] - self._target_pos[1])
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
