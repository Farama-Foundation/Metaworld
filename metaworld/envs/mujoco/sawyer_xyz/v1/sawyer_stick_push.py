import numpy as np
from gym.spaces import Box

from metaworld.envs.asset_path_utils import full_v1_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerStickPushEnv(SawyerXYZEnv):
    def __init__(self):

        liftThresh = 0.04
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.08, 0.58, 0.02)
        obj_high = (-0.03, 0.62, 0.02)
        goal_low = (0.399, 0.55, 0.0199)
        goal_high = (0.401, 0.6, 0.0201)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'stick_init_pos': np.array([-0.1, 0.6, 0.02]),
            'hand_init_pos': np.array([0, 0.6, 0.2]),
        }
        self.goal = self.init_config['stick_init_pos']
        self.stick_init_pos = self.init_config['stick_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.liftThresh = liftThresh# For now, fix the object initial position.
        self.obj_init_pos = np.array([0.2, 0.6, 0.04])
        self.obj_init_qpos = np.array([0.0, 0.0])
        self.obj_space = Box(np.array(obj_low), np.array(obj_high))
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )

    @property
    def model_name(self):
        return full_v1_path_for('sawyer_xyz/sawyer_stick_obj.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        reward, _, reachDist, pickRew, _, pushDist = self.compute_reward(action, ob)

        info = {
            'reachDist': reachDist,
            'pickRew': pickRew,
            'epRew': reward,
            'goalDist': pushDist,
            'success': float(pushDist <= 0.1 and reachDist <= 0.05)
        }

        return ob, reward, False, info

    @property
    def _target_site_config(self):
        return [('goal', np.hstack(
            (*self._target_pos[:2], self.stick_init_pos[2])
        ))]

    def _get_pos_objects(self):
        return np.hstack((
            self.get_body_com('stick').copy(),
            self.get_body_com('object').copy(),
        ))

    def _get_quat_objects(self):
        del self
        return np.array([0.] * 8)

    def _get_obs_dict(self):
        obs_dict = super()._get_obs_dict()
        obs_dict['state_achieved_goal'] = self.get_body_com('object').copy()
        return obs_dict

    def _set_stick_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[16:18] = pos.copy()
        qvel[16:18] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()
        self.stick_init_pos = self.init_config['stick_init_pos']
        self._target_pos = np.array([0.4, 0.6, self.stick_init_pos[-1]])
        self.stickHeight = self.get_body_com('stick').copy()[2]
        self.heightTarget = self.stickHeight + self.liftThresh

        if self.random_init:
            goal_pos = self._get_state_rand_vec()
            while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.1:
                goal_pos = self._get_state_rand_vec()
            self.stick_init_pos = np.concatenate((goal_pos[:2], [self.stick_init_pos[-1]]))
            self._target_pos = np.concatenate((goal_pos[-3:-1], [self.stick_init_pos[-1]]))

        self._set_stick_xyz(self.stick_init_pos)
        self._set_obj_xyz(self.obj_init_qpos)
        self.obj_init_pos = self.get_body_com('object').copy()
        self.maxPlaceDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) - np.array(self.stick_init_pos)) + self.heightTarget
        self.maxPushDist = np.linalg.norm(self.obj_init_pos[:2] - self._target_pos[:2])

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand(10)

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False

    def compute_reward(self, actions, obs):

        stickPos = obs[3:6]
        objPos = obs[6:9]

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        heightTarget = self.heightTarget
        pushGoal = self._target_pos

        pushDist = np.linalg.norm(objPos[:2] - pushGoal[:2])
        placeDist = np.linalg.norm(objPos - stickPos)
        reachDist = np.linalg.norm(stickPos - fingerCOM)

        def reachReward():
            reachRew = -reachDist
            # incentive to close fingers when reachDist is small
            if reachDist < 0.05:
                reachRew = -reachDist + max(actions[-1],0)/50

            return reachRew , reachDist

        def pickCompletionCriteria():
            tolerance = 0.01
            return stickPos[2] >= (heightTarget- tolerance)

        self.pickCompleted = pickCompletionCriteria()


        def objDropped():
            return (stickPos[2] < (self.stickHeight + 0.005)) and (pushDist >0.02) and (reachDist > 0.02)
            # Object on the ground, far away from the goal, and from the gripper
            # Can tweak the margin limits

        def orig_pickReward():
            hScale = 100
            if self.pickCompleted and not(objDropped()):
                return hScale*heightTarget
            elif (reachDist < 0.1) and (stickPos[2]> (self.stickHeight + 0.005)):
                return hScale* min(heightTarget, stickPos[2])
            else:
                return 0

        def pushReward():
            c1 = 1000
            c2 = 0.01
            c3 = 0.001
            cond = self.pickCompleted and (reachDist < 0.1) and not(objDropped())
            if cond:
                pushRew = 1000*(self.maxPlaceDist - placeDist) + c1*(np.exp(-(placeDist**2)/c2) + np.exp(-(placeDist**2)/c3))
                if placeDist < 0.05:
                    c4 = 2000
                    c5 = 0.001
                    c6 = 0.0001
                    pushRew += 1000*(self.maxPushDist - pushDist) + c4*(np.exp(-(pushDist**2)/c5) + np.exp(-(pushDist**2)/c6))
                pushRew = max(pushRew,0)

                return [pushRew , pushDist]
            else:
                return [0 , pushDist]

        reachRew, reachDist = reachReward()
        pickRew = orig_pickReward()
        pushRew , pushDist = pushReward()
        assert ((pushRew >=0) and (pickRew>=0))
        reward = reachRew + pickRew + pushRew

        return [reward, reachRew, reachDist, pickRew, pushRew, pushDist]
