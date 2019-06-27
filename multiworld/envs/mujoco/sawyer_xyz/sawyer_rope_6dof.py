from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict

from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv
from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera, sawyer_pick_and_place_camera_slanted_angle
from multiworld.envs.mujoco.dynamic_mjc.rope import rope

def get_beads_xy(qpos, num_beads):
    init_joint_offset = 9
    num_free_joints = 7

    xy_list = []
    for j in range(num_beads):
        offset = init_joint_offset + j*num_free_joints
        xy_list.append(qpos[offset:offset+2])

    return np.asarray(xy_list)

def get_com(xy_list):
    return np.mean(xy_list, axis=0)

def calculate_distance(qpos1, qpos2, num_beads):

    xy1 = get_beads_xy(qpos1, num_beads)
    xy2 = get_beads_xy(qpos2, num_beads)

    com1 = get_com(xy1)
    com2 = get_com(xy2)

    xy1_translate = xy1 + (com2 - com1)
    xy1_translate_flip = np.flip(xy1, axis=0) + (com2 - com1)

    distance = np.linalg.norm(xy1_translate - xy2)
    distance_flip = np.linalg.norm(xy1_translate_flip - xy2)

    return distance, distance_flip


class SawyerRope6DOFEnv(MultitaskEnv, SawyerXYZEnv):
    def __init__(
            self,
            obj_low=(-0.1, 0.35, 0.05),
            obj_high=(0.1, 0.55, 0.05),

            reward_type='dense',
            indicator_threshold=0.06,

            random_init=True,

            # hand_low=(-0.5, 0.40, 0.05),
            # hand_high=(0.5, 1, 0.5),
            hand_low=(-0.7, 0.30, 0.05),
            hand_high=(0.7, 1, 0.5),
            goal_low=None,
            goal_high=None,
            reset_free=False,

            num_beads=7,
            init_pos=(0., 0.4, 0.02),
            substeps=50, 
            log_video=False, 
            video_substeps=5, 
            video_h=500, 
            video_w=500,
            camera_name='leftcam',
            sparse=False,
            action_penalty_const=1e-2,
            rotMode='fixed',#'fixed',

            **kwargs
    ):
        self.quick_init(locals())
        MultitaskEnv.__init__(self)
        model = rope(num_beads=num_beads, init_pos=init_pos, texture=True)
        import os
        model.save(os.path.expanduser('~/sawyer_rope.xml'))
        with model.asfile() as f:
            SawyerXYZEnv.__init__(
                self,
                model_name=f.name,
                frame_skip=5,
                **kwargs
            )
        if obj_low is None:
            obj_low = self.hand_low
        if obj_high is None:
            obj_high = self.hand_high
        self.obj_low = obj_low
        self.obj_high = obj_high
        self.reset_free = reset_free

        #sim params
        self.num_beads = num_beads
        self.init_pos = init_pos
        self.substeps = substeps # number of intermediate positions to generate

        #reward params
        self.sparse = sparse
        self.action_penalty_const = action_penalty_const

        #video params
        self.log_video = log_video
        self.video_substeps = video_substeps
        self.video_h = video_h
        self.video_w = video_w
        self.camera_name = camera_name

        self._state_goal = np.asarray([1.0, 0.0])
        self.rotMode = rotMode
        self.random_init = random_init
        if rotMode == 'fixed':
            self.action_space = Box(
                np.array([-1, -1, -1, -1]),
                np.array([1, 1, 1, 1]),
            )
        elif rotMode == 'rotz':
            self.action_rot_scale = 1./10
            self.action_space = Box(
                np.array([-1, -1, -1, -1, -1]),
                np.array([1, 1, 1, 1, 1]),
            )
        elif rotMode == 'quat':
            self.action_space = Box(
                np.array([-1, -1, -1, 0, -1, -1, -1, -1]),
                np.array([1, 1, 1, 2*np.pi, 1, 1, 1, 1]),
            )
        else:
            self.action_space = Box(
                np.array([-1, -1, -1, -np.pi/2, -np.pi/2, 0, -1]),
                np.array([1, 1, 1, np.pi/2, np.pi/2, np.pi*2, 1]),
            )

        self.obj_and_goal_space = Box(
            np.array(obj_low),
            np.array(obj_high),
            dtype=np.float32
        )
        self.hand_space = Box(
            self.hand_low,
            self.hand_high,
            dtype=np.float32
        )

        self.observation_space = Box(
            np.hstack((self.hand_low, list(obj_low)*self.num_beads)),
            np.hstack((self.hand_high, list(obj_high)*self.num_beads)),
            dtype=np.float32
        )

        self.hand_reset_pos = np.array([0, .6, .2])
        self.reset()

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_rope.xml')

    def mode(self, name):
        if 'train' not in name:
            self.oracle_reset_prob = 0.0

    def viewer_setup(self):
        # sawyer_pick_and_place_camera(self.viewer.cam)
        # sawyer_pick_and_place_camera_slanted_angle(self.viewer.cam)
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.distance = 1.0

        # robot view
        # rotation_angle = 90
        # cam_dist = 1
        # cam_pos = np.array([0, 0.5, 0.2, cam_dist, -45, rotation_angle])

        # 3rd person view
        cam_dist = 0.3
        rotation_angle = 270
        cam_pos = np.array([0, 1.0, 0.5, cam_dist, -45, rotation_angle])

        # top down view
        # cam_dist = 0.2
        # rotation_angle = 0
        # cam_pos = np.array([0, 0, 1.5, cam_dist, -90, rotation_angle])

        for i in range(3):
            self.viewer.cam.lookat[i] = cam_pos[i]
        self.viewer.cam.distance = cam_pos[3]
        self.viewer.cam.elevation = cam_pos[4]
        self.viewer.cam.azimuth = cam_pos[5]
        self.viewer.cam.trackbodyid = -1

    def step(self, action):
        if self.rotMode == 'euler':
            action_ = np.zeros(7)
            action_[:3] = action[:3]
            action_[3:] = euler2quat(action[3:6])
            self.set_xyz_action_rot(action_)
        elif self.rotMode == 'fixed':
            self.set_xyz_action(action[:3])
        elif self.rotMode == 'rotz':
            self.set_xyz_action_rotz(action[:4])
        else:
            self.set_xyz_action_rot(action[:7])
        self.do_simulation([action[-1], -action[-1]])
        # new_obj_pos = self.get_obj_pos()
        # for i in range(self.num_beads):
        #     new_obj_pos[i][0:2] = np.clip(
        #         new_obj_pos[i][0:2],
        #         self.obj_low[0:2],
        #         self.obj_high[0:2]
        #     )
        # self._set_obj_xyz(new_obj_pos)
        # self.last_obj_pos = new_obj_pos.copy()
        ob = self._get_obs()
        ob_dict = self._get_obs_dict()
        reward, abs_cos, is_success = self.compute_reward(action, ob_dict)
        done = False
        return ob, reward, done, {'reward': reward, 'abs_cos': abs_cos, 'success': is_success}

    def _get_obs(self):
        hand = self.get_endeff_pos()
        objPos =  np.array(self.get_obj_pos()).flatten()
        flat_obs = np.concatenate((hand, objPos))
        return flat_obs


    def _get_obs_dict(self):
        hand = self.get_endeff_pos()
        objPos =  np.array(self.get_obj_pos()).flatten()
        flat_obs = np.concatenate((hand, objPos))

        return dict(
            observation=flat_obs,
            desired_goal=self._state_goal,
            achieved_goal=flat_obs,
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=flat_obs,
        )

    def get_obj_pos(self):
        return [self.get_body_com('bead_%d' % i) for i in range(self.num_beads)]

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        for i in range(self.num_beads):
            qpos[9+7*i:12+7*i] = pos[i].copy()
            qvel[9+6*i:15+6*i] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()
        if self.reset_free:
            return self._get_obs()
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        if self.random_init:
            obj_pos = np.random.uniform(
                self.obj_and_goal_space.low,
                self.obj_and_goal_space.high,
                size=(self.obj_and_goal_space.low.size),
            )
            obj_poses = [obj_pos + i*np.array([0, 0.05, 0]) for i in range(self.num_beads)]
            self._set_obj_xyz(obj_poses)
        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.hand_reset_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation(None, self.frame_skip)

    """
    Multitask functions
    """
    def compute_rewards(self, actions, obs):
        return [self.compute_reward(actions[i], obs[i])[0] for i in range(obs['observation'].shape[0])]

    def compute_reward(self, action, obs):
        first_bead = self.get_body_com("bead_0")[:2]
        last_bead = self.get_body_com("bead_{}".format(self.num_beads-1))[:2]
        vec_1 = first_bead - last_bead
        vec_2 = self._state_goal #horizontal line
        cosine = np.dot(vec_1, vec_2)/(np.linalg.norm(vec_1) + 1e-10)
        abs_cos = np.abs(cosine)

        #compute action penalty
        # act_dim = self.action_space.shape[0]
        action_penalty = np.linalg.norm(action[:3])
        length_penalty = np.abs(np.linalg.norm(vec_1) - self.num_beads*0.05)

        if self.sparse:
            if abs_cos > 0.9:
                main_rew = 1.0
            else:
                main_rew = 0.0
        else:
            main_rew = abs_cos 
        is_success = (abs_cos > 0.9)

        reward = main_rew + self.action_penalty_const*action_penalty + length_penalty
        return [reward, abs_cos, is_success]

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'is_success',
            'action_penalty',
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s%s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'Final %s%s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
            ))
        return statistics

    def get_goal(self):
        """
        Returns a dictionary
        """
        return dict(
            desired_goal=self._state_goal,
            state_desired_goal=self._state_goal,
        )

    """
    Implement the batch-version of these functions.
    """
    def sample_goals(self, batch_size):
        """
        :param batch_size:
        :return: Returns a dictionary mapping desired goal keys to arrays of
        size BATCH_SIZE x Z, where Z depends on the key.
        """
        return dict(
            desired_goal=np.array([self._state_goal for _ in batch_size]),
            state_desired_goal=np.array([self._state_goal for _ in batch_size]),
        )

    def get_env_state(self):
        base_state = super().get_env_state()
        goal = self._state_goal.copy()
        return base_state, goal

    def set_env_state(self, state):
        base_state, goal = state
        super().set_env_state(base_state)
        self._state_goal = goal
