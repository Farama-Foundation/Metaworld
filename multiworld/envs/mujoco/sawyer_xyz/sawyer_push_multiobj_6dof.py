from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict
import mujoco_py

from multiworld.core.serializable import Serializable
from multiworld.envs.env_util import (
    get_stat_in_paths,
    create_stats_ordered_dict,
    get_asset_full_path,
)

from multiworld.envs.mujoco.mujoco_env import MujocoEnv
import copy

from multiworld.core.multitask_env import MultitaskEnv

from multiworld.envs.mujoco.utils.create_xml import create_object_xml, create_root_xml, clean_xml
import multiworld

BASE_DIR = '/'.join(str.split(multiworld.__file__, '/')[:-2])
asset_base_path = BASE_DIR + '/multiworld/envs/assets/multi_object_sawyer_xyz/'

class SawyerMultiobject6DOFEnv(MujocoEnv, Serializable, MultitaskEnv):
    INIT_HAND_POS = np.array([0, 0.4, 0.02])

    def __init__(
            self,
            reward_info=None,
            frame_skip=2,
            pos_action_scale=2. / 100,
            randomize_goals=False,
            puck_goal_low=(-0.1, 0.5),
            puck_goal_high=(0.1, 0.7),
            hand_goal_low=(-0.1, 0.5),
            hand_goal_high=(0.1, 0.7),
            mocap_low=(-0.1, 0.5, 0.02), #(-0.1, 0.5, 0)
            mocap_high=(0.1, 0.7, 0.5),
            # unused
            init_block_low=(-0.05, 0.55),
            init_block_high=(0.05, 0.65),
            fixed_puck_goal=(0., 0.7),
            fixed_hand_goal=(-0.05, 0.6),
            # multi-object
            num_objects=1,
            target_object_idx=0,
            filename='sawyer_multiobj.xml',
            object_mass=1,
            # object_meshes=['Bowl', 'GlassBowl', 'LotusBowl01', 'ElephantBowl', 'RuggedBowl'],
            object_meshes=None,
            obj_classname = None,
            block_height=0.02,
            block_width = 0.02,
            cylinder_radius = 0.05,
            finger_sensors=False,
            maxlen=0.06,
            minlen=0.01,
            preload_obj_dict=None,

            reset_to_initial_position=True,
            object_low=(-0.1, 0.5, 0.0),
            object_high=(0.1, 0.7, 0.5),
            action_repeat=1,
            fixed_start=True,
            objjoint_xy=True,
    ):
        self.quick_init(locals())
        self.reward_info = reward_info
        self.randomize_goals = randomize_goals
        self._pos_action_scale = pos_action_scale
        self.reset_to_initial_position = reset_to_initial_position
        self.objjoint_xy = objjoint_xy

        self.init_block_low = np.array(init_block_low)
        self.init_block_high = np.array(init_block_high)
        self.puck_goal_low = np.array(puck_goal_low)
        self.puck_goal_high = np.array(puck_goal_high)
        self.hand_goal_low = np.array(hand_goal_low)
        self.hand_goal_high = np.array(hand_goal_high)
        self.fixed_puck_goal = np.array(fixed_puck_goal)
        self.fixed_hand_goal = np.array(fixed_hand_goal)
        self.mocap_low = np.array(mocap_low)
        self.mocap_high = np.array(mocap_high)
        self.object_low = np.array(object_low)
        self.object_high = np.array(object_high)
        self.action_repeat = action_repeat

        self.num_objects = num_objects
        self.target_object_idx = target_object_idx
        self.fixed_start = fixed_start
        self.maxlen = maxlen

        # Generate XML
        base_filename = asset_base_path + filename
        friction_params = (0.1, 0.1, 0.02)
        self.obj_stat_prop = create_object_xml(base_filename, num_objects, object_mass,
                                               friction_params, object_meshes, finger_sensors,
                                               maxlen, minlen, preload_obj_dict, obj_classname,
                                               block_height, block_width, cylinder_radius,
                                               objjoint_xy=objjoint_xy)
        gen_xml = create_root_xml(base_filename)
        MujocoEnv.__init__(self, gen_xml, frame_skip=frame_skip)
        clean_xml(gen_xml)

        self.state_goal = self.sample_goal_xyxy()
        # MultitaskEnv.__init__(self, distance_metric_order=2)
        # MujocoEnv.__init__(self, gen_xml, frame_skip=frame_skip)

        # self.action_space = Box(
        #     np.array([-1, -1]),
        #     np.array([1, 1]),
        # )

        # Add quaternion
        self.action_space = Box(
            np.array([-1, -1, 0, -1, -1, -1]),
            np.array([1, 1, 2*np.pi, 1, 1, 1]),
        )

        self.num_objects = num_objects
        low = (self.num_objects + 1) * [-0.2, 0.5]
        high = (self.num_objects + 1) * [0.2, 0.7]
        self.observation_space = Box(
            np.array(low),
            np.array(high),
        )
        goal_low = np.array([-0.2, 0.5]) # np.concatenate((self.hand_goal_low, self.puck_goal_low))
        goal_high = np.array([0.2, 0.7]) # np.concatenate((self.hand_goal_high, self.puck_goal_high))
        self.goal_box = Box(
            goal_low,
            goal_high,
        )
        # hack for state-based experiments for other envs
        # self.observation_space = Box(
        #     np.array([-0.2, 0.5, -0.2, 0.5, -0.2, 0.5]),
        #     np.array([0.2, 0.7, 0.2, 0.7, 0.2, 0.7]),
        # )
        # self.goal_space = Box(
        #     np.array([-0.2, 0.5, -0.2, 0.5, -0.2, 0.5]),
        #     np.array([0.2, 0.7, 0.2, 0.7, 0.2, 0.7]),
        # )

        self.set_initial_object_positions()

        self.reset()
        self.reset_mocap_welds()

    @property
    def model_name(self):
        return get_asset_full_path(
            'sawyer_xyz/sawyer_push_and_reach_mocap_goal_hidden.xml'
        )

    def viewer_setup(self):
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

    def step(self, a):
        a[:2] = np.clip(a[:2], -1, 1)
        mocap_delta_z = 0.06 - self.data.mocap_pos[0, 2]
        new_mocap_action = np.hstack((
            a[:2],
            np.array([mocap_delta_z]),
            a[2:]
        ))
        new_mocap_action[:3] = new_mocap_action[:3]*self._pos_action_scale
        u = np.zeros(7)
        for _ in range(self.action_repeat):
            self.mocap_set_action(new_mocap_action)
            self.do_simulation(u, self.frame_skip)

        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        for i in range(self.num_objects):
            if not self.objjoint_xy:
                x = 7 + i * 7
                y = 10 + i * 7
                qpos[x:y] = np.clip(qpos[x:y], self.object_low, self.object_high)
            else:
                x = 7 + i * 2
                y = 9 + i * 2
                qpos[x:y] = np.clip(qpos[x:y], self.object_low[:2][::-1], self.object_high[:2][::-1])
        self.set_state(qpos, qvel)
        # print('qpos is', qpos)
        # print('object position is', self.data.body_xpos[self.model.body_names.index('object' + str(self.target_object_idx))].copy())

        obs = self._get_obs()
        # reward = self.compute_reward(obs, u, obs, self._goal_xyxy)
        reward = self.compute_reward(a, obs)
        done = False

        endeff_pos = self.get_endeff_pos()
        object_distances = {}
        touch_distances = {}
        object_name = "object_distance"
        object_distance = np.linalg.norm(
            self.state_goal - self.get_object_pos(self.target_object_idx)
        )
        object_distances[object_name] = object_distance
        touch_name = "touch_distance"
        touch_distance = np.linalg.norm(
            endeff_pos - self.get_object_pos(self.target_object_idx)
        )
        touch_distances[touch_name] = touch_distance
        info = dict(
            success=float(sum(object_distances.values()) < 0.06),
            **object_distances,
            **touch_distances,
        )
        return obs, reward, done, info

    def mocap_set_action(self, action):
        pos_delta = np.squeeze(action)[:3]
        new_mocap_pos = self.data.mocap_pos + pos_delta
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high
        )
        # new_mocap_pos[0, 0] = np.clip(
        #     new_mocap_pos[0, 0],
        #     -0.1,
        #     0.1,
        # )
        # new_mocap_pos[0, 1] = np.clip(
        #     new_mocap_pos[0, 1],
        #     -0.1 + 0.6,
        #     0.1 + 0.6,
        #     )
        # new_mocap_pos[0, 2] = np.clip(
        #     new_mocap_pos[0, 2],
        #     0,
        #     0.5,
        # )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        # self.data.set_mocap_quat('mocap', np.array([np.cos(action[3]/2), pos_delta[0]*np.sin(action[3]/2), pos_delta[1]*np.sin(action[3]/2), pos_delta[2]*np.sin(action[3]/2)]))
        # self.data.set_mocap_quat('mocap', np.array([np.cos(action[3]/2), pos_delta[0]*np.abs(np.sin(action[3]/2))/np.linalg.norm(pos_delta), pos_delta[1]*np.abs(np.sin(action[3]/2))/np.linalg.norm(pos_delta), pos_delta[2]*np.abs(np.sin(action[3]/2))/np.linalg.norm(pos_delta)]))
        rot_axis = action[4:] / np.linalg.norm(action[4:])
        self.data.set_mocap_quat('mocap', np.array([np.cos(action[3]/2), rot_axis[0]*np.sin(action[3]/2), rot_axis[1]*np.sin(action[3]/2), rot_axis[2]*np.sin(action[3]/2)]))
        # self.data.set_mocap_quat('mocap', np.array([np.cos(action[3]/2), 0., 1., 0.]))

    # def _set_goal_marker(self, goal):
    #     """
    #     This should be use ONLY for visualization. Use self.state_goal for
    #     logging, learning, etc.
    #     """
    #     self.data.site_xpos[self.model.site_name2id('puck-goal-marker')] = (
    #         np.concatenate((goal, [0.02]))
    #     )
    #     if self.hide_goal_markers:
    #         self.data.site_xpos[self.model.site_name2id('puck-goal-marker'), 2] = (
    #             -1000
    #         )

    def _get_obs(self):
        e = self.get_endeff_pos()[:2]
        bs = []
        for i in range(self.num_objects):
            b = self.get_object_pos(i)[:2]
            bs.append(b)
        b = np.concatenate(bs)
        x = np.concatenate((e, b))
        g = self.state_goal

        new_obs = x

        return new_obs

    def get_object_pos(self, id):
        mujoco_id = self.model.body_names.index('object' + str(id))
        return self.data.body_xpos[mujoco_id].copy()[:2]

    def get_endeff_pos(self):
        return self.data.body_xpos[self.endeff_id].copy()[:2]

    @property
    def endeff_id(self):
        return self.model.body_names.index('leftclaw')

    def sample_goal_xyxy(self):
        if self.randomize_goals:
            puck = np.array([np.random.uniform(self.puck_goal_low, self.puck_goal_high)])
        else:
            puck = self.fixed_puck_goal.copy()
        return np.hstack((hand, puck))

    def sample_puck_xy(self):
        return np.array([0, 0.6])
    # def sample_puck_xy(self):
    #     raise NotImplementedError("Shouldn't you use "
    #                               "SawyerPushAndReachXYEasyEnv? Ask Vitchyr")
    #     pos = np.random.uniform(self.init_block_low, self.init_block_high)
    #     while np.linalg.norm(self.get_endeff_pos()[:2] - pos) < 0.035:
    #         pos = np.random.uniform(self.init_block_low, self.init_block_high)
    #     return pos

    def set_object_xy(self, i, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        if not self.objjoint_xy:
            x = 7 + i * 7
            y = 10 + i * 7
            z = 14 + i * 7
            qpos[x:y] = np.hstack((pos.copy(), np.array([0.02])))
            qpos[y:z] = np.array([1, 0, 0, 0])
            x = 7 + i * 6
            y = 13 + i * 6
            qvel[x:y] = 0
        else:
            x = 7 + i * 2
            y = 9 + i * 2
            qpos[x:y] = pos.copy()[::-1]
            qvel[x:y] = 0
        self.set_state(qpos, qvel)

    def reset_mocap_welds(self):
        """Resets the mocap welds that we use for actuation."""
        sim = self.sim
        if sim.model.nmocap > 0 and sim.model.eq_data is not None:
            for i in range(sim.model.eq_data.shape[0]):
                if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.])
        sim.forward()

    def reset_mocap2body_xpos(self):
        # move mocap to weld joint
        self.data.set_mocap_pos(
            'mocap',
            np.array([self.data.body_xpos[self.endeff_id]]),
        )
        self.data.set_mocap_quat(
            'mocap',
            np.array([self.data.body_xquat[self.endeff_id]]),
        )

    def set_initial_object_positions(self):
        if self.fixed_start: # set object to middle of workspace always
            for i in range(self.num_objects):
                self.set_object_xy(i, np.array([0, 0.6]))
        else: # set object to middle of workspace always
            while True:
                pos = [self.INIT_HAND_POS[:2], ]
                for i in range(self.num_objects):
                    r = np.random.uniform(self.puck_goal_low, self.puck_goal_high)
                    pos.append(r)
                touching = []
                for i in range(self.num_objects + 1):
                    for j in range(i):
                        t = np.linalg.norm(pos[i] - pos[j]) <= self.maxlen
                        touching.append(t)
                if not any(touching):
                    break
            pos.reverse()
            for i in range(self.num_objects):
                self.set_object_xy(i, pos[i])

    def reset(self):
        velocities = self.data.qvel.copy()
        angles = self.data.qpos.copy()
        angles[:7] = np.array(self.init_angles[:7]) # just change robot joints
        self.set_state(angles.flatten(), velocities.flatten())
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.INIT_HAND_POS)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        if self.reset_to_initial_position:
            self.set_initial_object_positions()
        # set_state resets the goal xy, so we need to explicit set it again
        self.state_goal = self.sample_goal_for_rollout()
        self.reset_mocap_welds()
        return self._get_obs()

    def compute_rewards(self, action, obs, info=None):
        r = -np.linalg.norm(obs[:, 2+self.target_object_idx*2:2+(self.target_object_idx+1)*2] - self.state_goal, axis=1)
        return r

    def compute_reward(self, action, obs, info=None):
        r = -np.linalg.norm(obs[2+self.target_object_idx*2:2+(self.target_object_idx+1)*2] - self.state_goal)
        r += -np.linalg.norm(obs[2+self.target_object_idx*2:2+(self.target_object_idx+1)*2] - obs[:2])
        return r

    # REPLACING REWARD FN
    # def compute_reward(self, ob, action, next_ob, goal, env_info=None):
    #     hand_xy = next_ob[:2]
    #     puck_xy = next_ob[-2:]
    #     hand_goal_xy = goal[:2]
    #     puck_goal_xy = goal[-2:]
    #     hand_dist = np.linalg.norm(hand_xy - hand_goal_xy)
    #     puck_dist = np.linalg.norm(puck_xy - puck_goal_xy)
    #     if not self.reward_info or self.reward_info["type"] == "euclidean":
    #         r = - hand_dist - puck_dist
    #     elif self.reward_info["type"] == "state_distance":
    #         r = -np.linalg.norm(next_ob - goal)
    #     elif self.reward_info["type"] == "hand_only":
    #         r = - hand_dist
    #     elif self.reward_info["type"] == "puck_only":
    #         r = - puck_dist
    #     elif self.reward_info["type"] == "sparse":
    #         t = self.reward_info["threshold"]
    #         r = float(
    #             hand_dist + puck_dist < t
    #         ) - 1
    #     else:
    #         raise NotImplementedError("Invalid/no reward type.")
    #     return r

    def compute_her_reward_np(self, ob, action, next_ob, goal, env_info=None):
        return self.compute_reward(ob, action, next_ob, goal, env_info=env_info)

    # @property
    # def init_angles(self):
    #     return [
    #         1.06139477e+00, -6.93988797e-01, 3.76729934e-01, 1.78410587e+00,
    #         - 5.36763074e-01, 5.88122189e-01, 3.51531533e+00,
    #         0.05, 0.55, 0.02,
    #         1, 0, 0, 0,
    #         0, 0.6, 0.02,
    #         1, 0, 1, 0,
    #         0, 0.6, 0.02,
    #         1, 0, 1, 0,
    #     ]
    @property
    def init_angles(self):
        return [1.78026069e+00, - 6.84415781e-01, - 1.54549231e-01,
                2.30672090e+00, 1.93111471e+00,  1.27854012e-01,
                1.49353907e+00, 1.80196716e-03, 7.40415706e-01,
                2.09895360e-02,  9.99999990e-01,  3.05766105e-05,
                - 3.78462492e-06, 1.38684523e-04, - 3.62518873e-02,
                6.13435141e-01, 2.09686080e-02,  7.07106781e-01,
                1.48979724e-14, 7.07106781e-01, - 1.48999170e-14,
                        0, 0.6, 0.02,
                        1, 0, 1, 0,
                ]

    def log_diagnostics(self, paths, logger=None, prefix=""):
        if logger is None:
            return

        statistics = OrderedDict()
        for stat_name in [
            'hand_distance',
            'success',
        ] + [
            "object%d_distance" % i for i in range(self.num_objects)
        ] + [
            "touch%d_distance" % i for i in range(self.num_objects)
        ]:
            stat_name = stat_name
            stat = get_stat_in_paths(paths, 'env_infos', stat_name)
            statistics.update(create_stats_ordered_dict(
                '%s %s' % (prefix, stat_name),
                stat,
                always_show_all_stats=True,
            ))
            statistics.update(create_stats_ordered_dict(
                'Final %s %s' % (prefix, stat_name),
                [s[-1] for s in stat],
                always_show_all_stats=True,
            ))

        for key, value in statistics.items():
            logger.record_tabular(key, value)

    """
    Multitask functions
    """

    @property
    def goal_dim(self) -> int:
        return 4

    def sample_goals(self, batch_size):
        # goals = np.zeros((batch_size, self.goal_box.low.size))
        # for b in range(batch_size):
        #     goals[b, :] = self.sample_goal_xyxy()
        goals = np.random.uniform(
            self.goal_box.low,
            self.goal_box.high,
            size=(batch_size, self.goal_box.low.size),
        )
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def sample_goal_for_rollout(self):
        g = self.sample_goal_xyxy()
        return g

    # OLD SET GOAL
    # def set_goal(self, goal):
    #     MultitaskEnv.set_goal(self, goal)
    #     self.set_goal_xyxy(goal)
    #     # hack for VAE
    #     self.set_to_goal(goal)

    def get_goal(self):
        return {
            'desired_goal': self.state_goal,
            'state_desired_goal': self.state_goal,
        }

    def set_goal(self, goal):
        self.state_goal = goal['state_desired_goal']

    def set_to_goal(self, goal):
        state_goal = goal['state_desired_goal']
        self.set_hand_xy(state_goal[:2])
        for i in range(self.num_objects):
            x = 2 + 2 * i
            y = 4 + 2 * i
            self.set_object_xy(i, state_goal[x:y])

    def convert_obs_to_goals(self, obs):
        return obs

    def set_hand_xy(self, xy):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', np.array([xy[0], xy[1], 0.02]))
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            u = np.zeros(7)
            self.do_simulation(u, self.frame_skip)

    def get_env_state(self):
        joint_state = self.sim.get_state()
        mocap_state = self.data.mocap_pos, self.data.mocap_quat
        state = joint_state, mocap_state
        return copy.deepcopy(state)

    def set_env_state(self, state):
        joint_state, mocap_state = state
        self.sim.set_state(joint_state)
        mocap_pos, mocap_quat = mocap_state
        self.data.set_mocap_pos('mocap', mocap_pos)
        self.data.set_mocap_quat('mocap', mocap_quat)
        self.sim.forward()


class SawyerTwoObject6DOFEnv(SawyerMultiobject6DOFEnv):
    """
    This environment matches exactly the 2-object pushing environment in the RIG paper
    """
    PUCK1_GOAL_LOW = np.array([0.0, 0.5])
    PUCK1_GOAL_HIGH = np.array([0.2, 0.7])
    PUCK2_GOAL_LOW = np.array([-0.2, 0.5])
    PUCK2_GOAL_HIGH = np.array([0.0, 0.7])
    HAND_GOAL_LOW = np.array([-0.05, 0.55])
    HAND_GOAL_HIGH = np.array([0.05, 0.65])

    low = np.hstack((HAND_GOAL_LOW, PUCK1_GOAL_LOW, PUCK2_GOAL_LOW))
    high = np.hstack((HAND_GOAL_HIGH, PUCK1_GOAL_HIGH, PUCK2_GOAL_HIGH))

    def __init__(
            self,
            **kwargs
    ):
        self.quick_init(locals())
        x = 0.2
        y1 = 0.5
        y2 = 0.7
        SawyerMultiobject6DOFEnv.__init__(
            self,
            hand_goal_low = (-x, y1),
            hand_goal_high = (x, y2),
            puck_goal_low = (-x, y1),
            puck_goal_high = (x, y2),
            mocap_low=(-0.1, y1, 0.0),
            mocap_high=(0.1, y2, 0.5),

            num_objects=2,
            preload_obj_dict=[
                dict(color2=(0.1, 0.1, 0.9)),
                dict(color2=(0.1, 0.9, 0.1))
            ],
            **kwargs
        )

    def sample_goal_xyxy(self):
        if self.randomize_goals:
            touching = [True]
            while any(touching):
                g = np.random.uniform(self.low, self.high)
                e = self.get_endeff_pos()[:2]
                b1 = self.get_object_pos(0)[:2]
                b2 = self.get_object_pos(1)[:2]
                diffs = [g-e, g-b1, g-b2]
                touching = [np.linalg.norm(d) <= 0.08 for d in diffs]
        else:
            g = self.fixed_puck_goal.copy()
        return g

    def sample_goals(self, batch_size):
        # goals = np.zeros((batch_size, self.goal_box.low.size))
        # for b in range(batch_size):
        #     goals[b, :] = self.sample_goal_xyxy()
        goals = np.random.uniform(
            self.low,
            self.high,
            size=(batch_size, self.low.size),
        )
        return {
            'desired_goal': goals,
            'state_desired_goal': goals,
        }

    def reset(self):
        velocities = self.data.qvel.copy()
        angles = self.data.qpos.copy()
        angles[:7] = np.array(self.init_angles[:7]) # just change robot joints
        self.set_state(angles.flatten(), velocities.flatten())
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.INIT_HAND_POS)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

        # explicitly set starting location of two blocks
        self.set_object_xy(0, np.array([0.1, 0.55]))
        self.set_object_xy(1, np.array([-0.1, 0.55]))

        # set_state resets the goal xy, so we need to explicit set it again
        self.state_goal = self.sample_goal_for_rollout()

        self.reset_mocap_welds()
        return self._get_obs()
