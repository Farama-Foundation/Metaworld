import numpy as np
from gym.spaces import  Box

from metaworld.envs.asset_path_utils import full_v1_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerPegUnplugSideEnv(SawyerXYZEnv):

    def __init__(self):

        liftThresh = 0.04
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.25, 0.6, 0.05)
        obj_high = (-0.15, 0.8, 0.05)
        goal_low = (-0.05, 0.6, 0.019)
        goal_high = (0.2, 0.8, 0.021)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([-0.225, 0.6, 0.05]),
            'hand_init_pos': np.array(((0, 0.6, 0.2))),
        }
        self.goal = np.array([-0.225, 0.6, 0.05])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.liftThresh = liftThresh

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v1_path_for('sawyer_xyz/sawyer_peg_unplug_side.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        reward, _, reachDist, pickRew, _, placingDist = self.compute_reward(action, ob)

        info = {
            'reachDist': reachDist,
            'pickRew': pickRew,
            'epRew': reward,
            'goalDist': placingDist,
            'success': float(placingDist <= 0.07)
        }

        return ob, reward, False, info

    def _get_pos_objects(self):
        return self._get_site_pos('pegEnd')

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        qvel[9] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()
        self.sim.model.body_pos[self.model.body_name2id('box')] = self.goal.copy()
        hole_pos = self.sim.model.site_pos[self.model.site_name2id('hole')] + self.sim.model.body_pos[self.model.body_name2id('box')]
        self.obj_init_pos = hole_pos
        self._target_pos = np.concatenate(([hole_pos[0] + 0.2], hole_pos[1:]))

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            self.sim.model.body_pos[self.model.body_name2id('box')] = goal_pos
            hole_pos = self.sim.model.site_pos[self.model.site_name2id('hole')] + self.sim.model.body_pos[self.model.body_name2id('box')]
            self.obj_init_pos = hole_pos
            self._target_pos = np.concatenate(([hole_pos[0] + 0.2], hole_pos[1:]))

        self.sim.model.body_pos[self.model.body_name2id('peg')] = self.obj_init_pos
        self._set_obj_xyz(0)
        self.objHeight = self.get_body_com('peg').copy()[0]
        self.heightTarget = self.objHeight + self.liftThresh
        self.obj_init_pos = self.get_body_com('peg')
        self.maxPlacingDist = np.linalg.norm(self._target_pos - self.obj_init_pos)
        self.target_reward = 1000*self.maxPlacingDist + 1000*2

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand(10)

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.reachCompleted = False

    def compute_reward(self, actions, obs):
        objPos = obs[3:6]

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        placingGoal = self._target_pos

        reachDist = np.linalg.norm(objPos - fingerCOM)

        placingDist = np.linalg.norm(objPos[:-1] - placingGoal[:-1])


        def reachReward():
            reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
            zRew = np.linalg.norm(fingerCOM[-1] - self.hand_init_pos[-1])

            if reachDistxy < 0.05:
                reachRew = -reachDist
            else:
                reachRew =  -reachDistxy - 2*zRew

            # incentive to close fingers when reachDist is small
            if reachDist < 0.05:
                reachRew = -reachDist + max(actions[-1],0)/50
            return reachRew, reachDist

        self.reachCompleted = reachDist < 0.05

        def placeReward():
            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            if self.reachCompleted:
                placeRew = 1000*(self.maxPlacingDist - placingDist) + c1*(np.exp(-(placingDist**2)/c2) + np.exp(-(placingDist**2)/c3))
                placeRew = max(placeRew,0)
                return [placeRew , placingDist]
            else:
                return [0 , placingDist]

        reachRew, reachDist = reachReward()
        placeRew, placingDist = placeReward()
        assert placeRew >=0
        reward = reachRew + placeRew

        return [reward, reachRew, reachDist, None, placeRew, placingDist]
