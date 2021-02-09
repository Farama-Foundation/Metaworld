import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerDoorEnvV2(SawyerXYZEnv):
    def __init__(self):

        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (0., 0.85, 0.15)
        obj_high = (0.1, 0.95, 0.15)
        goal_low = (-.3, 0.4, 0.1499)
        goal_high = (-.2, 0.5, 0.1501)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': np.array([0.3, ]),
            'obj_init_pos': np.array([0.1, 0.95, 0.15]),
            'hand_init_pos': np.array([0, 0.6, 0.2]),
        }

        self.goal = np.array([-0.2, 0.7, 0.15])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self.door_angle_idx = self.model.get_joint_qpos_addr('doorjoint')
        

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_door_pull.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            reward_grab,
            reward_ready,
            reward_success,
        ) = self.compute_reward(action, obs)

        success = float(abs(obs[4] - self._target_pos[0]) <= 0.08)

        info = {
            'success': success,
            'near_object': reward_ready,
            'grasp_success': reward_grab >= 0.5,
            'grasp_reward': reward_grab,
            'in_place_reward': reward_success,
            'obj_to_target': 0,
            'unscaled_reward': reward,
        }

        return reward, info

    @property
    def _target_site_config(self):
        return []

    def _get_pos_objects(self):
        return self.data.get_geom_xpos('handle').copy()

    def _get_quat_objects(self):
        return Rotation.from_matrix(self.data.get_geom_xmat('handle')).as_quat()

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        qpos[self.door_angle_idx] = pos
        qvel[self.door_angle_idx] = 0
        self.set_state(qpos.flatten(), qvel.flatten())

    def reset_model(self):
        self._reset_hand()

        self.objHeight = self.data.get_geom_xpos('handle')[2]

        self.obj_init_pos = self._get_state_rand_vec() if self.random_init \
            else self.init_config['obj_init_pos']
        self._target_pos = self.obj_init_pos + np.array([-0.3, -0.45, 0.])

        self.sim.model.body_pos[self.model.body_name2id('door')] = self.obj_init_pos
        self.sim.model.site_pos[self.model.site_name2id('goal')] = self._target_pos
        self._set_obj_xyz(0)
        self.maxPullDist = np.linalg.norm(self.data.get_geom_xpos('handle')[:-1] - self._target_pos[:-1])
        self.target_reward = 1000*self.maxPullDist + 1000*2

        return self._get_obs()

    @staticmethod
    def _reward_grab_effort(actions):
        return (np.clip(actions[3], -1, 1) + 1.0) / 2.0

    @staticmethod
    def _reward_pos(obs, theta):
        hand = obs[:3]
        door = obs[4:7] + np.array([-0.05, 0, 0])

        threshold = 0.12
        # floor is a 3D funnel centered on the door handle
        radius = np.linalg.norm(hand[:2] - door[:2])
        if radius <= threshold:
            floor = 0.0
        else:
            floor = 0.04 * np.log(radius - threshold) + 0.4
        # prevent the hand from running into the handle prematurely by keeping
        # it above the "floor"
        above_floor = 1.0 if hand[2] >= floor else reward_utils.tolerance(
            floor - hand[2],
            bounds=(0.0, 0.01),
            margin=floor / 2.0,
            sigmoid='long_tail',
        )
        # move the hand to a position between the handle and the main door body
        in_place = reward_utils.tolerance(
            np.linalg.norm(hand - door - np.array([0.05, 0.03, -0.01])),
            bounds=(0, threshold / 2.0),
            margin=0.5,
            sigmoid='long_tail',
        )
        ready_to_open = reward_utils.hamacher_product(above_floor, in_place)

        # now actually open the door
        door_angle = -theta
        a = 0.2  # Relative importance of just *trying* to open the door at all
        b = 0.8  # Relative importance of fully opening the door
        opened = a * float(theta < -np.pi/90.) + b * reward_utils.tolerance(
            np.pi/2. + np.pi/6 - door_angle,
            bounds=(0, 0.5),
            margin=np.pi/3.,
            sigmoid='long_tail',
        )

        return ready_to_open, opened

    def compute_reward(self, actions, obs):
        theta = self.data.get_joint_qpos('doorjoint')

        reward_grab = SawyerDoorEnvV2._reward_grab_effort(actions)
        reward_steps = SawyerDoorEnvV2._reward_pos(obs, theta)

        reward = sum((
            2.0 * reward_utils.hamacher_product(reward_steps[0], reward_grab),
            8.0 * reward_steps[1],
        ))

        # Override reward on success flag
        if abs(obs[4] - self._target_pos[0]) <= 0.08:
            reward = 10.0

        return (
            reward,
            reward_grab,
            *reward_steps,
        )
