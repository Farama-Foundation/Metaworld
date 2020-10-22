import numpy as np
from gym.spaces import Box

from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv


class SawyerHandlePressSideEnvV2(SawyerXYZEnv):
    """
    Motivation for V2:
        V1 was very difficult to solve because the end effector's wrist has a
        nub that got caught on the box before pushing the handle all the way
        down. There are a number of ways to fix this, e.g. moving box to right
        sie of table, extending handle's length, decreasing handle's damping,
        or moving the goal position slightly upward. I just the last one.
    Changelog from V1 to V2:
        - (8/05/20) Updated to new XML
        - (6/30/20) Increased goal's Z coordinate by 0.01 in XML
    """
    def __init__(self):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.35, 0.65, -0.001)
        obj_high = (-0.25, 0.75, +0.001)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_pos': np.array([-0.3, 0.7, 0.0]),
            'hand_init_pos': np.array((0, 0.6, 0.2),),
        }
        self.goal = np.array([-0.2, 0.7, 0.14])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.hand_init_pos = self.init_config['hand_init_pos']

        goal_low = self.hand_low
        goal_high = self.hand_high

        self.max_path_length = 150

        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([1, 1, 1, 1]),
        )

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_handle_press_sideways.xml')

    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation
        ob = self._get_obs()
        reward, reachDist, pressDist = self.compute_reward(action, ob)
        self.curr_path_length += 1

        info = {
            'reachDist': reachDist,
            'goalDist': pressDist,
            'epRew': reward,
            'pickRew': None,
            'success': float(pressDist <= 0.04)
        }

        return ob, reward, False, info

    def _get_pos_objects(self):
        return self._get_site_pos('handleStart')

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        qvel[9] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()

        self.obj_init_pos = (self._get_state_rand_vec()
                             if self.random_init
                             else self.init_config['obj_init_pos'])

        self.sim.model.body_pos[self.model.body_name2id('box')] = self.obj_init_pos
        self._set_obj_xyz(0)
        self._target_pos = self._get_site_pos('goalPress')
        self.maxDist = np.abs(self.data.site_xpos[self.model.site_name2id('handleStart')][-1] - self._target_pos[-1])
        self.target_reward = 1000*self.maxDist + 1000*2

        return self._get_obs()

    def compute_reward(self, actions, obs):
        del actions

        objPos = obs[3:6]

        leftFinger = self._get_site_pos('leftEndEffector')
        fingerCOM  =  leftFinger

        pressGoal = self._target_pos[-1]

        pressDist = np.abs(objPos[-1] - pressGoal)
        reachDist = np.linalg.norm(objPos - fingerCOM)

        c1 = 1000
        c2 = 0.01
        c3 = 0.001
        if reachDist < 0.05:
            pressRew = 1000*(self.maxDist - pressDist) + c1*(np.exp(-(pressDist**2)/c2) + np.exp(-(pressDist**2)/c3))
        else:
            pressRew = 0

        pressRew = max(pressRew, 0)
        reward = -reachDist + pressRew

        return [reward, reachDist, pressDist]
