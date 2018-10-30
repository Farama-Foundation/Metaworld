from multiworld.envs.mujoco.base_mujoco_env import BaseMujocoEnv
import multiworld
import numpy as np
import cv2
import mujoco_py
from pyquaternion import Quaternion
import ipdb
debug = ipdb.set_trace
from multiworld.envs.mujoco.util.create_xml import create_object_xml, create_root_xml, clean_xml
from collections import OrderedDict
from multiworld.envs.mujoco.util.interpolation import TwoPointCSpline

from mujoco_py.builder import MujocoException
import copy
from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv



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

low_bound = np.array([-0.27, 0.52, 0.1, 0, -1])
high_bound = np.array([0.27, 0.95, 0.3, 2 * np.pi - 0.001, 1])
NEUTRAL_JOINTS = np.array([1.65474475, - 0.53312487, - 0.65980174, 1.1841825, 0.62772584, 1.11682223, 1.31015104, -0.05, 0.05])

class MultiSawyerEnv(BaseMujocoEnv, MultitaskEnv, SawyerXYZEnv):
    def __init__(self, filename='sawyer_grasp.xml', mode_rel=np.array([True, True, True, True, False]), num_objects = 3, object_mass = 1, friction=1.0, finger_sensors=True,
                 maxlen=0.12, minlen=0.01, preload_obj_dict=None, object_meshes=['Bowl', 'GlassBowl', 'LotusBowl01', 'ElephantBowl', 'RuggedBowl'], obj_classname = 'freejoint',
                 block_height=0.02, block_width = 0.02, viewer_image_height = 84, viewer_image_width = 84,
                 skip_first=100, substeps=100, randomize_initial_pos = False, state_goal = None, randomize_goal_at_reset = False,):

        base_filename = asset_base_path + filename

        friction_params = (friction, 0.1, 0.02)
        self.obj_stat_prop = create_object_xml(base_filename, num_objects, object_mass,
                                               friction_params, object_meshes, finger_sensors,
                                               maxlen, minlen, preload_obj_dict, obj_classname,
                                               block_height, block_width)


        gen_xml = create_root_xml(base_filename)
        BaseMujocoEnv.__init__(self, gen_xml, viewer_image_height, viewer_image_width)
        clean_xml(gen_xml)

        if self.sim.model.nmocap > 0 and self.sim.model.eq_data is not None:
            for i in range(self.sim.model.eq_data.shape[0]):
                if self.sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    # Define the xyz + quat of the mocap relative to the hand
                    self.sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.]
                    )

        self._base_sdim, self._base_adim, self.mode_rel = 5, 5, mode_rel
        self.num_objects, self.skip_first, self.substeps = num_objects, skip_first, substeps
        self.randomize_initial_pos = randomize_initial_pos
        self.finger_sensors, self._maxlen = finger_sensors, maxlen
        self._threshold = 0
        self._previous_target_qpos, self._n_joints = None, 9
        self._object_pos = np.zeros((num_objects, 7))
        self._reset_xyz = np.zeros((num_objects, 3))
        self._reset_quat = np.zeros((num_objects, 4))
        self._reset_pos = np.copy(self._object_pos)
        self._object_names = ['obj_' + str(i) for i in range(num_objects)]
        self._initialized = False
        self._state_goal = state_goal
        self._randomize_goal_at_reset = randomize_goal_at_reset
        self.reset()

    def _clip_gripper(self):
        self.sim.data.qpos[7:9] = np.clip(self.sim.data.qpos[7:9], [-0.055, 0.0027], [-0.0027, 0.055])

    def samp_xyz_rot(self):
        rand_xyz = np.random.uniform(low_bound[:3] + self._maxlen / 2 + 0.02, high_bound[:3] - self._maxlen / 2 + 0.02)
        rand_xyz[-1] = 0.05
        return rand_xyz, np.random.uniform(-np.pi / 2, np.pi / 2)



    def reset(self):
        last_rands = []
        if not self._initialized:
            for i in range(self.num_objects):
                obji_xyz, rot = self.samp_xyz_rot()
                #rejection sampling to ensure objects don't crowd each other
                while len(last_rands) > 0 and min([np.linalg.norm(obji_xyz - obj_j) for obj_j in last_rands]) < self._maxlen:
                    obji_xyz, rot = self.samp_xyz_rot()
                last_rands.append(obji_xyz)

                rand_quat = Quaternion(axis=[0, 0, -1], angle= rot).elements
                self._reset_xyz[i] = obji_xyz
                self._reset_quat[i] = rand_quat
                self.sim.data.qpos[self._n_joints + i * 7: self._n_joints + 3 + i * 7] = obji_xyz
                self.sim.data.qpos[self._n_joints + 3 + i * 7: self._n_joints + 7 + i * 7] = rand_quat
                self._object_pos[i] = np.concatenate((obji_xyz, rand_quat))

        else:
            for i in range(self.num_objects):
                obji_xyz, rand_quat = self._reset_xyz[i], self._reset_quat[i]
                self.sim.data.qpos[self._n_joints + i * 7: self._n_joints + 3 + i * 7] = obji_xyz
                self.sim.data.qpos[self._n_joints + 3 + i * 7: self._n_joints + 7 + i * 7] = rand_quat
                self._object_pos[i] = np.concatenate((obji_xyz, rand_quat))


        self.sim.data.set_mocap_pos('mocap', np.array([0,0,2]))
        self.sim.data.set_mocap_quat('mocap', zangle_to_quat(np.random.uniform(low_bound[3], high_bound[3])))

        #placing objects then resetting to neutral risks bad contacts
        try:
            for _ in range(5):
                self.sim.step()
            self.sim.data.qpos[:9] = NEUTRAL_JOINTS
            for _ in range(5):
                self.sim.step()
        except MujocoException:
            return self.reset()
        if self.randomize_initial_pos:
            xyz = np.random.uniform(low_bound[:3], high_bound[:3])
            self.sim.data.set_mocap_pos('mocap', xyz)
            self.sim.data.set_mocap_quat('mocap', zangle_to_quat(np.random.uniform(low_bound[3], high_bound[3])))
        else:
            self.sim.data.set_mocap_pos('mocap', np.array([0, 0.5, 0.17]))
            self.sim.data.set_mocap_quat('mocap', zangle_to_quat(np.pi))
        #reset gripper
        self.sim.data.qpos[7:9] = NEUTRAL_JOINTS[7:9]
        self.sim.data.ctrl[:] = [-1, 1]

        finger_force = np.zeros(2)
        for _ in range(self.skip_first):
            for _ in range(20):
                self._clip_gripper()
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
            self._state_goal = self.sample_goals(1)[0]
        return self._get_obs(finger_force)

    def _get_obs(self, finger_sensors):
        obs, touch_offset = {}, 0
        # report finger sensors as needed
        if self.finger_sensors:
            obs['finger_sensors'] = np.array([np.max(finger_sensors)]).reshape(-1)
            touch_offset = 2

        # joint poisitions and velocities
        obs['qpos'] = copy.deepcopy(self.sim.data.qpos[:self._n_joints].squeeze())
        obs['qvel'] = copy.deepcopy(self.sim.data.qvel[:self._n_joints].squeeze())

        # control state
        obs['state'] = np.zeros(self._base_sdim)
        obs['state'][:3] = self.sim.data.get_body_xpos('hand')[:3]
        obs['state'][3] = quat_to_zangle(self.sim.data.get_body_xquat('hand'))
        obs['state'][-1] = self._previous_target_qpos[-1]






        obs['object_poses_full'] = np.zeros((self.num_objects, 7))
        obs['object_poses'] = np.zeros((self.num_objects, 3))
        for i in range(self.num_objects):
            fullpose = self.sim.data.qpos[i * 7 + self._n_joints:(i + 1) * 7 + self._n_joints].squeeze().copy()
            fullpose[:3] = self.sim.data.sensordata[touch_offset + i * 3:touch_offset + (i + 1) * 3]

            obs['object_poses_full'][i] = fullpose
            obs['object_poses'][i, :2] = fullpose[:2]
            obs['object_poses'][i, 2] = quat_to_zangle(fullpose[3:])
        obs['observation']= obs['object_poses_full'].copy()
        obs['desired_goal'] = self._state_goal
        obs['achieved_goal'] = obs['object_poses_full'].copy()
        obs['state_observation'] = obs['object_poses_full'].copy()
        obs['state_desired_goal'] = self._state_goal
        obs['state_achieved_goal'] = obs['object_poses_full'].copy()
        # report object poses
        # copy non-image data for environment's use (if needed)
        self._last_obs = copy.deepcopy(obs)
        # get images
        obs['images'] = self.render()

        obj_image_locations = np.zeros((2, self.num_objects + 1, 2))
        for i, cam in enumerate(['maincam', 'leftcam']):
            obj_image_locations[i, 0] = self.project_point(self.sim.data.get_body_xpos('hand')[:3], cam)
            for j in range(self.num_objects):
                obj_image_locations[i, j + 1] = self.project_point(obs['object_poses_full'][j, :3], cam)
        obs['obj_image_locations'] = obj_image_locations
        obs['img'] = obs['images'][0]
        return obs


    def get_image(self):
        return self.render()[0]


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
        # if not self._sim_integrity():
        #     print('Sim reset (integrity)')
        #     raise ValueError

        target_qpos = np.clip(self._next_qpos(action), low_bound, high_bound)
        assert target_qpos.shape[0] == self._base_sdim
        finger_force = np.zeros(2)

        xyz_interp = TwoPointCSpline(self.sim.data.get_body_xpos('hand').copy(), target_qpos[:3])
        self.sim.data.set_mocap_quat('mocap', zangle_to_quat(target_qpos[3]))
        for st in range(self.substeps):
            alpha = 1.
            if not self.substeps == 1:
                alpha = st / (self.substeps - 1)

            self.sim.data.set_mocap_pos('mocap', xyz_interp.get(alpha)[0])

            if st < 3 * self.substeps // 4:
                self.sim.data.ctrl[0] = self._previous_target_qpos[-1]
                self.sim.data.ctrl[1] = -self._previous_target_qpos[-1]
            else:
                self.sim.data.ctrl[0] = target_qpos[-1]
                self.sim.data.ctrl[1] = -target_qpos[-1]

            for _ in range(20):
                self._clip_gripper()
                if self.finger_sensors:
                    finger_force += copy.deepcopy(self.sim.data.sensordata[:2].squeeze())
                try:
                    self.sim.step()
                except MujocoException:
                    print('Sim reset (bad contact)')
                    raise ValueError

        finger_force /= self.substeps * 10
        if np.sum(finger_force) > 0:
            print(finger_force)
        self._previous_target_qpos = target_qpos

        ob = self._get_obs(finger_force)
        self._post_step()
        reward = self.compute_rewards(action, ob)
        done = False
        info = self._get_info()
        return ob, reward, done, info

    def _get_info(self):
        infos = dict()
        for i in range(self.num_objects):
            infos[self._object_names[i] + '_distance'] = np.linalg.norm(self._last_obs['object_poses'][i]
                                                                        - self._state_goal[i][:3])
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
        assert action.shape[0] == 5
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
        return np.sum(np.linalg.norm(obs['state_achieved_goal'] - obs['state_desired_goal'], axis=1))

    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }


    def set_goal(self, goal):
        self._state_goal = goal['state_desired_goal'][0]



    def set_to_goal(self, goal):
        last_rands = []
        for i in range(self.num_objects):
            obji_xyz, rot = goal[i][:3], goal[i][3:]
            self.sim.data.qpos[self._n_joints + i * 7: self._n_joints + 3 + i * 7] = obji_xyz
            self.sim.data.qpos[self._n_joints + 3 + i * 7: self._n_joints + 7 + i * 7] = rot

        self.sim.data.set_mocap_pos('mocap', np.array([0, 0, 2]))
        self.sim.data.set_mocap_quat('mocap', zangle_to_quat(np.random.uniform(low_bound[3], high_bound[3])))

        # placing objects then resetting to neutral risks bad contacts
        try:
            for _ in range(5):
                self.sim.step()
            self.sim.data.qpos[:9] = NEUTRAL_JOINTS
            for _ in range(5):
                self.sim.step()
        except MujocoException:
            return self.reset()
        if self.randomize_initial_pos:
            xyz = np.random.uniform(low_bound[:3], high_bound[:3])
            self.sim.data.set_mocap_pos('mocap', xyz)
            self.sim.data.set_mocap_quat('mocap', zangle_to_quat(np.random.uniform(low_bound[3], high_bound[3])))
        else:
            self.sim.data.set_mocap_pos('mocap', np.array([0, 0.5, 0]))
            self.sim.data.set_mocap_quat('mocap', zangle_to_quat(np.pi))
        # reset gripper
        self.sim.data.qpos[7:9] = NEUTRAL_JOINTS[7:9]
        self.sim.data.ctrl[:] = [-1, 1]

        finger_force = np.zeros(2)
        for _ in range(self.skip_first):
            for _ in range(20):
                self._clip_gripper()
                try:
                    self.sim.step()
                except MujocoException:
                    # if randomly generated start causes 'bad' contacts Mujoco will error. Have to reset again
                    print('except')
                    return self.reset()

            if self.finger_sensors:
                finger_force += self.sim.data.sensordata[:2]
        finger_force /= 10 * self.skip_first
        self._init_dynamics()

        return self._get_obs(finger_force)

    def sample_goals(self, batch_size):
        goals = []
        for i in range(batch_size):
            goals.append(self.sample_goal())

        goals = np.array(goals)

        return  {'desired_goal': goals,
            'state_desired_goal': goals,
        }
    def sample_goal(self):
        goal = np.zeros((self.num_objects, 7))
        last_rands = []
        for i in range(self.num_objects):
            obji_xyz, rot = self.samp_xyz_rot()
            #rejection sampling to ensure objects don't crowd each other
            while len(last_rands) > 0 and min([np.linalg.norm(obji_xyz - obj_j) for obj_j in last_rands]) < self._maxlen:
                obji_xyz, rot = self.samp_xyz_rot()
            last_rands.append(obji_xyz)
            rand_quat = Quaternion(axis=[0, 0, -1], angle= rot).elements
            goal[i] = np.concatenate((obji_xyz, rand_quat))
        return goal


if __name__ == '__main__':
    env = MultiSawyerEnv()
    env.set_goal(env.sample_goals(1))
    env.step(np.zeros(5))
    env._clip_gripper()
    # g = env.sample_goal()
    img = env.render()[0]


    cv2.imshow('window', img)
    cv2.waitKey(10000)
