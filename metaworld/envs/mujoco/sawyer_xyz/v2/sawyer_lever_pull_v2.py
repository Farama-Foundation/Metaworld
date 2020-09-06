import numpy as np
from gym.spaces import Box

from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerLeverPullEnvV2(SawyerXYZEnv):
    """
    Motivation for V2:
        V1 was impossible to solve because the lever would have to be pulled
        through the table in order to reach the target location.
    Changelog from V1 to V2:
        - (8/12/20) Updated to Byron's XML
        - (7/7/20) Added 3 element lever position to the observation
            (for consistency with other environments)
        - (6/23/20) In `reset_model`, changed `final_pos[2] -= .17` to `+= .17`
            This ensures that the target point is above the table.
    """
    def __init__(self):

        hand_low = (-0.5, 0.40, -0.15)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.7, 0.0)
        obj_high = (0.1, 0.8, 0.0)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([0, 0.7, 0.0]),
            'hand_init_pos': np.array([0, 0.4, 0.2], dtype=np.float32),
        }
        self.goal = np.array([.12, 0.88, .05])
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
        return full_v2_path_for('sawyer_xyz/sawyer_lever_pull.xml')

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        reward, reachDist, pullDist = self.compute_reward(action, ob)
        self.curr_path_length += 1

        info = {
            'reachDist': reachDist,
            'goalDist': pullDist,
            'epRew': reward,
            'pickRew': None,
            'success': float(pullDist <= 0.05)
        }

        return ob, reward, False, info

    def _get_pos_objects(self):
        return self._get_site_pos('leverStart')

    def reset_model(self):
        self._reset_hand()
        self.obj_init_pos = self._get_state_rand_vec() if self.random_init \
            else self.init_config['obj_init_pos']
        self.sim.model.body_pos[
            self.model.body_name2id('lever')] = self.obj_init_pos

        self._target_pos = self.obj_init_pos + np.array([.12, .0, .45])

        self.maxPullDist = np.linalg.norm(self._target_pos - self.obj_init_pos)

        return self._get_obs()

    def _reset_hand(self):
        super()._reset_hand()
        self.reachCompleted = False

    def compute_reward(self, actions, obs):
        del actions

        objPos = obs[3:6]

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        pullGoal = self._target_pos

        pullDist = np.linalg.norm(objPos - pullGoal)
        reachDist = np.linalg.norm(objPos - fingerCOM)
        reachRew = -reachDist

        self.reachCompleted = reachDist < 0.05

        def pullReward():
            c1 = 1000
            c2 = 0.01
            c3 = 0.001

            if self.reachCompleted:
                pullRew = 1000*(self.maxPullDist - pullDist) + c1*(np.exp(-(pullDist**2)/c2) + np.exp(-(pullDist**2)/c3))
                pullRew = max(pullRew,0)
                return pullRew
            else:
                return 0

        pullRew = pullReward()
        reward = reachRew + pullRew

        return [reward, reachDist, pullDist]
