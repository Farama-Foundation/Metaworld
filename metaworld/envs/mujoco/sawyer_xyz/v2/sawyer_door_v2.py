import numpy as np
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from metaworld.envs import reward_utils


from metaworld.envs.asset_path_utils import full_v2_path_for
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import SawyerXYZEnv, _assert_task_is_set


class SawyerDoorEnvV2(SawyerXYZEnv):

    OBJ_RADIUS = 0.03

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

        self.max_path_length = 150

        self._random_reset_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self.door_angle_idx = self.model.get_joint_qpos_addr('doorjoint')

    @property
    def model_name(self):
        return full_v2_path_for('sawyer_xyz/sawyer_door_pull.xml')

    @_assert_task_is_set
    def step(self, action):
        obs = super().step(action)
        # (
        #     reward,
        #     tcp_to_obj,
        #     tcp_opened,
        #     obj_to_target,
        #     object_grasped,
        #     in_place
        # ) = self.compute_reward(action, obs)

        (
            reward,
            gripper_error,
            gripped,
            handle_error,
            caging_reward,
            opening_reward
        ) = self.compute_reward(action, obs)

        goal_dist = np.linalg.norm(obs[4:7] - self._target_pos)

        info = {
            'success': float(goal_dist <= 0.03),
            'near_object': float(gripper_error <= 0.03),
            'grasp_success': float(gripped > 0),
            'grasp_reward': caging_reward,
            'in_place_reward': opening_reward,
            'obj_to_target': handle_error,
            'unscaled_reward': reward,
        }


        self.curr_path_length += 1
        # info = {
        #     'reward': reward,
        #     'tcp_to_obj': tcp_to_obj,
        #     'obj_to_target': obj_to_target,
        #     'in_place_reward': in_place,
        #     'success': float(obj_to_target <= 0.08)
        # }

        return obs, reward, False, info

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

    def _reset_hand(self):
        super()._reset_hand()

        rightFinger, leftFinger = self._get_site_pos('rightEndEffector'), self._get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.reachCompleted = False

    def compute_reward(self, action, obs):
        # _TARGET_RADIUS = 0.05
        # tcp = self.tcp_center
        # obj = obs[4:7]
        # tcp_opened = obs[3]
        # target = self._target_pos
        #
        # obj_to_target = np.linalg.norm(obj - target)
        # tcp_to_obj = np.linalg.norm(obj - tcp)
        # in_place_margin = (np.linalg.norm(self.obj_init_pos - target)) + 0.2
        #
        # in_place = reward_utils.tolerance(obj_to_target,
        #                             bounds=(0, _TARGET_RADIUS),
        #                             margin=in_place_margin,
        #                             sigmoid='long_tail',)
        #
        # object_grasped = self._gripper_caging_reward(action, obj, self.OBJ_RADIUS)
        # in_place_and_object_grasped = reward_utils.hamacher_product(object_grasped,
        #                                                             in_place)
        # reward = in_place_and_object_grasped
        #
        # if tcp_to_obj < 0.1:
        #     reward += 1. + 5. * in_place
        # if obj_to_target < _TARGET_RADIUS:
        #     reward = 10.
        # print("REWARD: {} -- TCP_TO_OBJ: {}".format(reward, tcp_to_obj))
        # return [reward, tcp_to_obj, tcp_opened, obj_to_target, object_grasped, in_place]

        gripper = obs[:3]
        handle = obs[4:7]
        handle_pos_init = self.obj_init_pos

        scale = np.array([1., 3., 1.])
        handle_error = (handle - self._target_pos) * scale
        handle_error_init = (handle_pos_init - self._target_pos) * scale

        reward_for_opening = reward_utils.tolerance(
            np.linalg.norm(handle_error),
            bounds=(0, 0.02),
            margin=np.linalg.norm(handle_error_init),
            sigmoid='long_tail'
        )


        # Emphasize XY error so that gripper is able to drop down and cage
        # handle without running into it. By doing this, we are assuming
        # that the reward in the Z direction is small enough that the agent
        # will be willing to explore raising a finger above the handle, hook it,
        # and drop back down to re-gain Z reward
        scale = np.array([3., 3., 1.])
        gripper_error = (handle - gripper) * scale
        gripper_error_init = (handle_pos_init - self.init_tcp) * scale

        print(np.linalg.norm(gripper_error), np.linalg.norm(gripper_error_init))


        reward_for_caging = reward_utils.tolerance(
            np.linalg.norm(gripper_error),
            bounds=(0, 0.04),
            margin=np.linalg.norm(gripper_error_init),
            sigmoid='long_tail'
        )

        reward = reward_for_caging + reward_for_opening
        reward *= 5.0

        print("REWARD: {} -- CAGING: {} -- OPENNING: {}".format(reward, reward_for_caging, reward_for_opening))

        return (
            reward,
            np.linalg.norm(handle - gripper),
            obs[3],
            np.linalg.norm(handle_error),
            reward_for_caging,
            reward_for_opening
        )
