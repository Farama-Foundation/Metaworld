import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils
from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set



class SawyerPlateSlideEnvV2(SawyerXYZEnv):

    OBJ_RADIUS = 0.04

    def __init__(self):

        goal_low = (-0.1, 0.85, 0.)
        goal_high = (0.1, 0.9, 0.)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (0., 0.6, 0.)
        obj_high = (0., 0.6, 0.)

        super().__init__(
            self.model_name,
            hand_low=hand_low,
            hand_high=hand_high,
        )

        self.init_config = {
            'obj_init_angle': 0.3,
            'obj_init_pos': np.array([0., 0.6, 0.], dtype=np.float32),
            'hand_init_pos': np.array((0, 0.6, 0.2), dtype=np.float32),
        }
        self.goal = np.array([0., 0.85, 0.02])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_plate_slide.xml')

    @_assert_task_is_set
    def evaluate_state(self, obs, action):
        (
            reward,
            tcp_to_obj,
            tcp_opened,
            obj_to_target,
            object_grasped,
            in_place
        ) = self.compute_reward(action, obs)

        success = float(obj_to_target <= 0.07)
        near_object = float(tcp_to_obj <= 0.03)

        info = {
            'success': success,
            'near_object': near_object,
            'grasp_reward': object_grasped,
            'grasp_success': 0.0,
            'in_place_reward': in_place,
            'obj_to_target': obj_to_target,
            'unscaled_reward': reward
        }
        return reward, info

    def _get_pos_objects(self):
        return self.data.get_geom_xpos('puck')

    def _get_quat_objects(self):
        return Rotation.from_matrix(self.data.get_geom_xmat('puck')).as_quat()

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:11] = pos
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()

        self.obj_init_pos = self.init_config['obj_init_pos']
        self._target_pos = self.goal.copy()

        if self.random_init:
            rand_vec = self._get_state_rand_vec()
            self.init_tcp = self.tcp_center
            self.obj_init_pos = rand_vec[:3]
            self._target_pos = rand_vec[3:]

        self.sim.model.body_pos[
            self.model.body_name2id('puck_goal')] = self._target_pos
        self._set_obj_xyz(np.zeros(2))

        return self._get_obs()

    def compute_reward(self, action, obs):
        _TARGET_RADIUS = 0.05
        tcp = self.tcp_center
        obj = obs[4:7]
        tcp_opened = obs[3]
        target = self._target_pos

        obj_to_target = np.linalg.norm(obj - target)
        in_place_margin = np.linalg.norm(self.obj_init_pos - target)

        in_place = reward_utils.tolerance(obj_to_target,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=in_place_margin,
                                    sigmoid='long_tail',)

        tcp_to_obj = np.linalg.norm(tcp - obj)
        obj_grasped_margin = np.linalg.norm(self.init_tcp - self.obj_init_pos)

        object_grasped = reward_utils.tolerance(tcp_to_obj,
                                    bounds=(0, _TARGET_RADIUS),
                                    margin=obj_grasped_margin,
                                    sigmoid='long_tail',)

        in_place_and_object_grasped = reward_utils.hamacher_product(object_grasped,
                                                                    in_place)
        reward = 8 * in_place_and_object_grasped

        if obj_to_target < _TARGET_RADIUS:
            reward = 10.
        return [
            reward,
            tcp_to_obj,
            tcp_opened,
            obj_to_target,
            object_grasped,
            in_place
        ]
