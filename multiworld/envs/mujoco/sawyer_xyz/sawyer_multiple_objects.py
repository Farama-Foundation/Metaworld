"""This environment allows a Sawyer to interact with multiple custom objects.
It is largely copied over from Frederik Ebert and Sudeep Dasari:
https://github.com/febert/visual_mpc/ but adapted to the multiworld interface.
"""

import multiworld
import numpy as np
import cv2
import mujoco_py
from pyquaternion import Quaternion
from multiworld.envs.mujoco.util.create_xml import create_object_xml, create_root_xml, clean_xml
from collections import OrderedDict
from multiworld.envs.mujoco.util.interpolation import TwoPointCSpline

from mujoco_py.builder import MujocoException
import copy
from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv

from gym.spaces import Box, Dict
import time

from multiworld.core.image_env import ImageEnv
from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v2
#[0.33802861 0.72943006 0.15373863]
#[-0.3326288   0.70221613  0.14004561]
#[0.32792462 0.23910806 0.10462233]
#[-0.32757104  0.24410117  0.10482]
#fix_z = 0.087

low = [-0.31, 0.24, 0.0845]
high = [0.32, 0.7, 0.14]


def quat_to_zangle(quat):
    angle = -(Quaternion(axis = [0,1,0], angle = np.pi).inverse * Quaternion(quat)).angle
    if angle < 0:
        return angle + 2 * np.pi
    return angle

def zangle_to_quat(zangle):
    """
    :param zangle in rad
    :return: quaternion
    """
    return (Quaternion(axis=[0,1,0], angle=np.pi) * Quaternion(axis=[0, 0, -1], angle= zangle)).elements

BASE_DIR = '/'.join(str.split(multiworld.__file__, '/')[:-2])
asset_base_path = BASE_DIR + '/multiworld/envs/assets/multi_object_sawyer_xyz/'

low_bound = np.array([-0.27, 0.52, 0.05, 0, -1])
high_bound = np.array([0.27, 0.95, 0.1, 2 * np.pi - 0.001, 1])
NEUTRAL_JOINTS = np.array([1.65474475, - 0.53312487, - 0.65980174, 1.1841825, 0.62772584, 1.11682223, 1.31015104, -0.05, 0.05])

class MultiSawyerEnv(MultitaskEnv, SawyerXYZEnv):
    def __init__(self,
        filename='sawyer_multiobj.xml',
        mode_rel=np.array([True, True, True, True, False]),
        num_objects = 3, object_mass = 1, friction=1.0, finger_sensors=True,
        maxlen=0.12, minlen=0.01, preload_obj_dict=None,
        object_meshes=['Bowl', 'GlassBowl', 'LotusBowl01', 'ElephantBowl', 'RuggedBowl'],
        obj_classname = 'freejoint', block_height=0.02, block_width = 0.02,
        cylinder_radius = 0.05,
        viewer_image_height = 84, viewer_image_width = 84, skip_first=100,
        substeps=100, init_hand_xyz=([-0.00031741, 0.24092109, 0.08816302]),
        randomize_initial_pos = False, state_goal = None,
        randomize_goal_at_reset = True,
        hand_z_position=0.089,
        fix_z = True, fix_gripper = True, fix_rotation = True,
        fix_reset_pos = True, do_render = False,
        match_orientation = False,
        workspace_low = np.array([-0.31, 0.24, 0]),
        workspace_high = np.array([0.32, 0.7, 0.14]),
        hand_low = np.array([-0.31, 0.24, 0.0]),
        hand_high = np.array([0.32, 0.7, 0.14]),
        env_seed = 0,
    ):
        self.quick_init(locals())

        self.seed(env_seed)

        base_filename = asset_base_path + filename

        friction_params = (friction, 0.1, 0.02)
        self.obj_stat_prop = create_object_xml(base_filename, num_objects, object_mass,
                                               friction_params, object_meshes, finger_sensors,
                                               maxlen, minlen, preload_obj_dict, obj_classname,
                                               block_height, block_width, cylinder_radius)


        gen_xml = create_root_xml(base_filename)
        SawyerXYZEnv.__init__(self,
            model_name=gen_xml,
            mocap_low=hand_low,
            mocap_high=hand_high,
        )
        clean_xml(gen_xml)

        if self.sim.model.nmocap > 0 and self.sim.model.eq_data is not None:
            for i in range(self.sim.model.eq_data.shape[0]):
                if self.sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    # Define the xyz + quat of the mocap relative to the hand
                    self.sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.]
                    )

        self._base_sdim, self._base_adim, self.mode_rel = 5, 5, mode_rel
        self.hand_low = np.array(hand_low)
        self.hand_high = np.array(hand_high)
        self.workspace_low = np.array(workspace_low)
        self.workspace_high = np.array(workspace_high)
        self.object_low = self.workspace_low.copy()
        self.object_high = self.workspace_high.copy()
        self.object_low[0], self.object_low[1] = self.object_low[0] + cylinder_radius, \
                                                 self.object_low[1] + cylinder_radius
        self.object_high[0], self.object_high[1] = self.object_high[0] - cylinder_radius, \
                                                 self.object_high[1] - cylinder_radius
        self.action_scale = 5/100
        self.num_objects, self.skip_first, self.substeps = num_objects, skip_first, substeps
        self.randomize_initial_pos = randomize_initial_pos
        self.finger_sensors, self._maxlen = finger_sensors, maxlen
        self._threshold = 0
        self._previous_target_qpos, self._n_joints = None, 7
        self._object_pos = np.zeros((num_objects, 7))
        self._reset_xyz = np.zeros((num_objects, 3))
        self._reset_quat = np.zeros((num_objects, 4))
        self._reset_pos = np.copy(self._object_pos)
        self._object_names = ['obj_' + str(i) for i in range(num_objects)]
        self._initialized = False
        self._state_goal = state_goal
        self.hand_z_position = hand_z_position
        self.fix_z = fix_z
        self.fix_rotation = fix_rotation
        self.fix_gripper = fix_gripper
        self._randomize_goal_at_reset = randomize_goal_at_reset
        self.do_render = do_render
        self.init_hand_xyz = np.array(init_hand_xyz)
        self.match_orientation = match_orientation
        self._object_reset_pos = self.sample_object_positions()
        self.goal_dim_per_object = 7 if match_orientation else 3
        self.cylinder_radius = cylinder_radius
        # self.finger_sensors = False # turning this off for now

        action_bounds = [1, 1, ]
        if not fix_z:
            action_bounds.append(1)
        if not fix_gripper:
            action_bounds.append(1)
        if not fix_rotation:
            action_bounds.append(1)
        action_bounds = np.array(action_bounds)
        self.action_space = Box(-action_bounds, action_bounds, dtype=np.float32)
        # action space is 5-dim: x, y, z, ? ?

        self.hand_space = Box(self.hand_low, self.hand_high, dtype=np.float32)
        obs_dim = 3 * num_objects + 3
        obs_space = Box(np.array([-1] * obs_dim), np.array([1] * obs_dim), dtype=np.float32)
        goal_space = Box(
            np.array([-1] * self.goal_dim_per_object * num_objects),
            np.array([1] * self.goal_dim_per_object * num_objects),
            dtype=np.float32,
        )
        self.observation_space = Dict([
            ('observation', obs_space),
            ('desired_goal', goal_space),
            ('achieved_goal', goal_space),
            ('state_observation', obs_space),
            ('state_desired_goal', goal_space),
            ('state_achieved_goal', goal_space),
        ])

        self._state_goal = self.sample_goal()
        self.fix_reset_pos = fix_reset_pos
        o = self.reset()
        self.reset()

    def _clip_gripper(self):
        self.sim.data.qpos[7:9] = np.clip(self.sim.data.qpos[7:9], [-0.055, 0.0027], [-0.0027, 0.055])

    def samp_xyz_rot(self):
        low = self.object_low[:3]
        high = self.object_high[:3]
        rand_xyz = np.random.uniform(low, high)
        rand_xyz[-1] = 0.02
        return rand_xyz, np.random.uniform(-np.pi / 2, np.pi / 2)

    def snapshot_noarm(self):
        raise NotImplementedError

    @property
    def adim(self):
        return self._adim

    @property
    def sdim(self):
        return self._sdim

    def sample_object_positions(self):
        return [(np.array([0, 0.4, 0.02]), 0)]

        last_rands = [1]
        object_poses = []
        for i in range(self.num_objects):
            obji_xyz, rot = self.samp_xyz_rot()
            #rejection sampling to ensure objects don't crowd each other
            while len(last_rands) > 0 and min([np.linalg.norm(obji_xyz - obj_j)
                                               for obj_j in last_rands]) < self._maxlen:
                obji_xyz, rot = self.samp_xyz_rot()
            object_poses.append((obji_xyz, rot))
        return object_poses

    def reset(self):
        self._reset_hand()
        gripper_pos = self.init_hand_xyz.copy()
        last_rands = [gripper_pos]
        for i in range(self.num_objects):
            if self.randomize_initial_pos:
                obji_xyz, rot = self.samp_xyz_rot()
                #rejection sampling to ensure objects don't crowd each other

                while len(last_rands) > 0 and min([np.linalg.norm(obji_xyz - obj_j)
                                                   for obj_j in last_rands]) < self._maxlen:
                    obji_xyz, rot = self.samp_xyz_rot()

            else:
                obji_xyz, rot = self._object_reset_pos[i]

            last_rands.append(obji_xyz)

            rand_quat = Quaternion(axis=[0, 0, -1], angle= rot).elements
            self._reset_xyz[i] = obji_xyz
            self._reset_quat[i] = rand_quat
            self.sim.data.qpos[self._n_joints + i * 7: self._n_joints + 3 + i * 7] = obji_xyz
            self.sim.data.qpos[self._n_joints + 3 + i * 7: self._n_joints + 7 + i * 7] = rand_quat
            self.sim.data.qvel[self._n_joints + i * 6: self._n_joints + 3 + i * 6] = np.zeros((3))
            self.sim.data.qvel[self._n_joints + 3 + i * 6: self._n_joints + 6 + i * 6] = np.zeros((3))
            self._object_pos[i] = np.concatenate((obji_xyz, rand_quat))
        self._initialized = True

        # self.sim.forward()

        finger_force = np.zeros(2)
        for _ in range(self.skip_first):
            for _ in range(20):
                # self._clip_gripper()
                try:
                    self.sim.step()
                except MujocoException:
                    #if randomly generated start causes 'bad' contacts Mujoco will error. Have to reset again
                    print('except')
                    return self.reset()

            if self.finger_sensors:
                finger_force += self.sim.data.sensordata[:2]
        finger_force /= 10 * self.skip_first

        self._previous_target_qpos = np.zeros(self._base_sdim)
        self._previous_target_qpos[:3] = self.sim.data.get_body_xpos('hand')
        self._previous_target_qpos[3] = quat_to_zangle(self.sim.data.get_body_xquat('hand'))
        self._previous_target_qpos[-1] = low_bound[-1]

        self._init_dynamics()
        if self._randomize_goal_at_reset:
            self._state_goal = self.sample_goal()

        obs = self._get_obs(finger_force)
        return obs

    def _reset_hand(self):
        velocities = self.data.qvel.copy()
        angles = self.data.qpos.copy()
        angles[:7] = self.init_angles[:7]
        self.set_state(angles.flatten(), velocities.flatten())
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.init_hand_xyz.copy())
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

    @property
    def init_angles(self):
        return [2.07332628e+00, -4.47460459e-01, -4.55678050e-01,
                2.35962299e+00, 2.17558228e+00, 4.98524319e-01, 1.37302554e+00,
                4.08324093e-02,
                5.05442647e-04, 6.00496057e-01, 3.06443862e-02,
                1, 0, 0, 0]

    # @property
    # def init_angles(self):
    #     # return [1.83632216, -0.49958089, -0.21426779, 1.22169484,
    #     #      -2.89597359, -0.87371492, 0.17658116]
    #     return [1.89985363, -0.4995636, -0.52156845, 2.10777374, -1.62297273, -0.45320344,
    #      - 1.27655332]

    def _get_obs(self, finger_sensors=None):
        obs, touch_offset = {}, 0
        # report finger sensors as needed
        if self.finger_sensors:
            obs['finger_sensors'] = np.array([np.max(finger_sensors)]).reshape(-1)
            touch_offset = 2

        # joint poisitions and velocities
        obs['qpos'] = copy.deepcopy(self.sim.data.qpos[:self._n_joints].squeeze())
        obs['qvel'] = copy.deepcopy(self.sim.data.qvel[:self._n_joints].squeeze())

        # control state
        # obs['state'] = np.zeros(self._base_sdim)
        # obs['state'][:3] = self.sim.data.get_body_xpos('hand')[:3]
        # obs['state'][3] = quat_to_zangle(self.sim.data.get_body_xquat('hand'))
        # obs['state'][-1] = self._previous_target_qpos[-1]

        obs['object_poses_full'] = np.zeros((self.num_objects, 7))
        obs['object_poses'] = np.zeros((self.num_objects, 3))

        # import pdb; pdb.set_trace()

        for i in range(self.num_objects):
            name = "object" + str(i)
            xyz = self.data.get_body_xpos(name).copy()
            q = self.data.get_body_xquat(name).copy()

            fullpose = np.concatenate((xyz, q))
            obs['object_poses_full'][i, :] = fullpose
            obs['object_poses'][i, :3] = fullpose[:3]
            # obs['object_poses'][i, 2] = quat_to_zangle(fullpose[3:])

        flat_obs = self.get_endeff_pos()
        object_poses_full = obs['object_poses_full'].copy().flatten()
        object_poses_xyz = obs['object_poses'].copy().flatten()
        state_obs = np.concatenate((flat_obs, object_poses_xyz))
        obs['observation']= state_obs
        obs['desired_goal'] = self._state_goal.flatten()
        obs['achieved_goal'] = object_poses_xyz
        obs['state_observation'] = state_obs
        obs['state_desired_goal'] = self._state_goal.flatten()
        obs['state_achieved_goal'] = object_poses_xyz
        # report object poses
        # copy non-image data for environment's use (if needed)
        self._last_obs = copy.deepcopy(obs)
        # get images
        if self.do_render:
            self.render()
        return obs


    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [obj_name + '_distance' for obj_name in self._object_names]:
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


    def _sim_integrity(self):
        xyztheta = np.zeros(4)
        xyztheta[:3] = self.sim.data.get_body_xpos('hand')
        xyztheta[3] = quat_to_zangle(self.sim.data.get_body_xquat('hand'))
        if not all(np.logical_and(xyztheta <= high_bound[:4] + 0.05, xyztheta >= low_bound[:4] - 0.05)):
            print('robot', xyztheta)
            return False

        for i in range(self.num_objects):
            obj_xy = self._last_obs['object_poses_full'][i][:2]
            z = self._last_obs['object_poses_full'][i][2]
            if not all(np.logical_and(obj_xy <= high_bound[:2] + 0.05, obj_xy >= low_bound[:2] - 0.05)):
                return False
            if z >= 0.5 or z <= -0.1:
                return False

        return True

    def step(self, action):
        finger_force = np.zeros(2)
        if self.fix_z:
            self.set_xy_action(action[:2], self.hand_z_position)
        else:
            self.set_xyz_action(action[:3])
        u = np.zeros(8)

        try:
            self.do_simulation(u)
        except MujocoException:
            print("MujocoException, skipping")

        for i in range(self.num_objects):
            x, y = self._n_joints + i * 7, self._n_joints + 3 + i * 7
            pos = self.sim.data.qpos[x:y]
            clipped_pos = np.clip(pos, self.workspace_low, self.workspace_high)
            self.sim.data.qpos[x:y] = clipped_pos

        finger_force /= self.substeps * 10
        if np.sum(finger_force) > 0:
            print("finger force", finger_force)

        ob = self._get_obs(finger_force)
        reward = -np.linalg.norm(ob['state_achieved_goal'] - ob['state_desired_goal'])

        done = False
        info = self._get_info()

        return ob, reward, done, info

    def _get_info(self):
        infos = dict()
        for i in range(self.num_objects):
            x = i * self.goal_dim_per_object
            y = x + self.goal_dim_per_object
            obj_pos = self._last_obs['object_poses'][i]
            obj_goal = self._state_goal[x:y]
            name = self._object_names[i] + '_distance'
            infos[name] = np.linalg.norm(obj_pos - obj_goal)
        return infos

    def _post_step(self):
        """
        Add custom behavior in sub classes for post-step checks
        (eg if goal has been reached)
            -Occurs after _get_obs so last_obs is available...
        :return: None
        """
        return

    def valid_rollout(self):
        object_zs = self._last_obs['object_poses_full'][:, 2]
        return not any(object_zs < -2e-2) and self._sim_integrity()

    def _init_dynamics(self):
        self._goal_reached = False
        self._gripper_closed = False
        self._prev_touch = False

    def _next_qpos(self, action):

        target = self._previous_target_qpos * self.mode_rel + action
        if action[-1] <= self._threshold:                     #if policy outputs an "open" action then override auto-grasp
            self._gripper_closed = False
            target[-1] = -1

        return target

    def _post_step(self):
        finger_sensors_thresh = np.max(self._last_obs['finger_sensors']) > 0
        z_thresholds = np.amax(self._last_obs['object_poses_full'][:, 2]) > 0.15 and self._last_obs['state'][2] > 0.23
        if z_thresholds and finger_sensors_thresh:
            self._goal_reached = True

    def has_goal(self):
        return True

    def goal_reached(self):
        return self._goal_reached

    def compute_rewards(self, action, obs):
        r = -np.linalg.norm(obs['state_achieved_goal'] - obs['state_desired_goal'], axis=1)
        return r

    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }


    def set_goal(self, goal):
        self._state_goal = goal['state_desired_goal'][0]



    def set_to_goal(self, goal):
        self._reset_hand()

        last_rands = []
        for i in range(self.num_objects):
            x, y = i * 3, i * 3 + 3
            # print(x, y, goal)
            obji_xyz, rot = goal["state_desired_goal"][x:y], Quaternion(axis=[0, 0, -1], angle=0).elements
            self.sim.data.qpos[self._n_joints + i * 7: self._n_joints + 3 + i * 7] = obji_xyz
            self.sim.data.qpos[self._n_joints + 3 + i * 7: self._n_joints + 7 + i * 7] = rot

        u = np.zeros(8)
        self.do_simulation(u)

        # self.sim.data.set_mocap_pos('mocap', np.array([0, 0, 2]))
        # self.sim.data.set_mocap_quat('mocap', zangle_to_quat(np.random.uniform(low_bound[3], high_bound[3])))

        # placing objects then resetting to neutral risks bad contacts
        # try:
        #     for _ in range(5):
        #         self.sim.step()
        #     self.sim.data.qpos[:9] = NEUTRAL_JOINTS
        #     for _ in range(5):
        #         self.sim.step()
        # except MujocoException:
        #     return self.reset()
        # if self.randomize_initial_pos:
        #     xyz = np.random.uniform(low_bound[:3], high_bound[:3])
        #     self.sim.data.set_mocap_pos('mocap', xyz)
        #     self.sim.data.set_mocap_quat('mocap', zangle_to_quat(np.random.uniform(low_bound[3], high_bound[3])))
        # else:
        #     self.sim.data.set_mocap_pos('mocap', np.array([0, 0.5, 0]))
        #     self.sim.data.set_mocap_quat('mocap', zangle_to_quat(np.pi))
        # # reset gripper
        # self.sim.data.qpos[7:9] = NEUTRAL_JOINTS[7:9]
        # self.sim.data.ctrl[:] = [-1, 1]

        # finger_force = np.zeros(2)
        # for _ in range(self.skip_first):
        #     for _ in range(20):
        #         # self._clip_gripper()
        #         try:
        #             self.sim.step()
        #         except MujocoException:
        #             # if randomly generated start causes 'bad' contacts Mujoco will error. Have to reset again
        #             print('except')
        #             return self.reset()

        #     if self.finger_sensors:
        #         finger_force += self.sim.data.sensordata[:2]
        # finger_force /= 10 * self.skip_first
        # self._init_dynamics()

        obs = self._get_obs()
        # print(obs["state_observation"][3:])
        return obs

    def sample_goals(self, batch_size):
        goals = []
        for i in range(batch_size):
            goals.append(self.sample_goal())

        goals = np.array(goals)

        return  {'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def sample_goal(self):
        if self.match_orientation:
            goal = np.zeros((self.num_objects, 7))
        else:
            goal = np.zeros((self.num_objects, 3))
        last_rands = []
        for i in range(self.num_objects):
            obji_xyz, rot = self.samp_xyz_rot()
            #rejection sampling to ensure objects don't crowd each other
            while len(last_rands) > 0 and min([np.linalg.norm(obji_xyz - obj_j) for obj_j in last_rands]) < self._maxlen:
                obji_xyz, rot = self.samp_xyz_rot()
            last_rands.append(obji_xyz)
            rand_quat = Quaternion(axis=[0, 0, -1], angle= rot).elements
            if self.match_orientation:
                goal[i] = np.concatenate((obji_xyz, rand_quat))
            else:
                goal[i] = obji_xyz
        return goal.flatten()


if __name__ == '__main__':
    # env = MultiSawyerEnv(
    #     do_render=False,
    #     finger_sensors=False,
    #     num_objects=3,
    #     object_meshes=None,
    #     randomize_initial_pos=False,
    #     fix_z=True,
    #     fix_gripper=True,
    #     fix_rotation=True,
    # )
    # env = ImageEnv(env,
    #     non_presampled_goal_img_is_garbage=True,
    #     recompute_reward=False,
    #     init_camera=sawyer_pusher_camera_upright_v2,
    # )
    # for i in range(10000):
    #     a = np.random.uniform(-1, 1, 5)
    #     o, _, _, _ = env.step(a)
    #     if i % 10 == 0:
    #         env.reset()

    #     img = o["image_observation"].transpose().reshape(84, 84, 3)
    #     cv2.imshow('window', img)
    #     cv2.waitKey(100)





    size = 0.1
    low = np.array([-size, 0.4 - size, 0])
    high = np.array([size, 0.4 + size, 0.1])
    env = MultiSawyerEnv(
        do_render=False,
        finger_sensors=False,
        num_objects=1,
        object_meshes=None,
        # randomize_initial_pos=True,
        fix_z=True,
        fix_gripper=True,
        fix_rotation=True,
        cylinder_radius = 0.03,
        maxlen = 0.03,
        workspace_low = low,
        workspace_high = high,
        hand_low = low,
        hand_high = high,
        init_hand_xyz=(0, 0.4-size, 0.089),
    )
    for i in range(10000):
        a = np.random.uniform(-1, 1, 5)
        o, r, _, _ = env.step(a)
        if i % 100 == 0:
            o = env.reset()
        # print(i, r)
        # print(o["state_observation"])
        # print(o["state_desired_goal"])
        env.render()





    # from robosuite.devices import SpaceMouse

    # device = SpaceMouse()
    # size = 0.1
    # low = np.array([-size, 0.4 - size, 0])
    # high = np.array([size, 0.4 + size, 0.1])
    # env = MultiSawyerEnv(
    #     do_render=False,
    #     finger_sensors=False,
    #     num_objects=1,
    #     object_meshes=None,
    #     workspace_low = low,
    #     workspace_high = high,
    #     hand_low = low,
    #     hand_high = high,
    #     fix_z=True,
    #     fix_gripper=True,
    #     fix_rotation=True,
    #     cylinder_radius=0.03,
    #     maxlen=0.03,
    #     init_hand_xyz=(0, 0.4-size, 0.089),
    # )
    # for i in range(10000):
    #     state = device.get_controller_state()
    #     dpos, rotation, grasp, reset = (
    #         state["dpos"],
    #         state["rotation"],
    #         state["grasp"],
    #         state["reset"],
    #     )

    #     # convert into a suitable end effector action for the environment
    #     # current = env._right_hand_orn
    #     # drotation = current.T.dot(rotation)  # relative rotation of desired from current
    #     # dquat = T.mat2quat(drotation)
    #     # grasp = grasp - 1.  # map 0 to -1 (open) and 1 to 0 (closed halfway)
    #     # action = np.concatenate([dpos, dquat, [grasp]])

    #     a = dpos * 10 # 200

    #     # a[:3] = np.array((0, 0.7, 0.1)) - env.get_endeff_pos()
    #     # a = np.array([np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05), 0.1, 0 ,  1])
    #     o, _, _, _ = env.step(a)
    #     if i % 100 == 0:
    #         env.reset()
    #     # print(env.sim.data.qpos[:7])
    #     env.render()
