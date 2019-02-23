from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict

from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv
from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera, sawyer_pick_and_place_camera_slanted_angle


class SawyerStack6DOFEnv(MultitaskEnv, SawyerXYZEnv):
    def __init__(
            self,
            obj_low=None,
            obj_high=None,

            reward_type='dense',
            indicator_threshold=0.06,

            obj_init_positions=((0, 0.6, 0.02),),
            random_init=False,

            fix_goal=False,
            fixed_goal=(0.2, 0.9, 0.02),
            hand_low=(-0.5, 0.40, 0.05),
            hand_high=(0.5, 1, 0.5),
            goal_low=None,
            goal_high=None,
            reset_free=False,

            hide_goal_markers=False,
            oracle_reset_prob=0.0,
            presampled_goals=None,
            num_goals_presampled=1000,
            p_obj_in_hand=.75,

            **kwargs
    ):
        self.quick_init(locals())
        MultitaskEnv.__init__(self)
        SawyerXYZEnv.__init__(
            self,
            model_name=self.model_name,
            frame_skip=5,
            **kwargs
        )
        if obj_low is None:
            obj_low = self.hand_low
        if obj_high is None:
            obj_high = self.hand_high
        self.obj_low = obj_low
        self.obj_high = obj_high
        if goal_low is None:
            goal_low = np.hstack((self.hand_low, obj_low))
        if goal_high is None:
            goal_high = np.hstack((self.hand_high, obj_high))

        self.reward_type = reward_type
        self.random_init = random_init
        self.p_obj_in_hand = p_obj_in_hand
        self.indicator_threshold = indicator_threshold

        self.obj_init_z = obj_init_positions[0][2]
        self.obj_init_positions = np.array(obj_init_positions)
        self.last_obj_pos = self.obj_init_positions[0]

        self.fix_goal = fix_goal
        self.fixed_goal = np.array(fixed_goal)
        self._state_goal = None
        self.reset_free = reset_free
        self.oracle_reset_prob = oracle_reset_prob

        self.hide_goal_markers = hide_goal_markers

        self.action_space = Box(
            np.array([-1, -1, -1, 0, -1, -1, -1, -1]),
            np.array([1, 1, 1, 2*np.pi, 1, 1, 1, 1]),
            # np.array([-1, -1, -1, -np.pi/4, -1]),
            # np.array([1, 1, 1, np.pi/4, 1]),
            dtype=np.float32
        )
        self.hand_and_obj_space = Box(
            np.hstack((self.hand_low, obj_low)),
            np.hstack((self.hand_high, obj_high)),
            dtype=np.float32
        )
        self.hand_space = Box(
            self.hand_low,
            self.hand_high,
            dtype=np.float32
        )
        self.gripper_and_hand_and_obj_space = Box(
            np.hstack(([0.0], self.hand_low, obj_low)),
            np.hstack(([0.04], self.hand_high, obj_high)),
            dtype=np.float32
        )

        self.observation_space = Box(
            np.hstack(([0.0], self.hand_low, obj_low, obj_low)),
            np.hstack(([0.04], self.hand_high, obj_high, obj_high)),
            dtype=np.float32
        )
        # self.observation_space = Dict([
        #     ('observation', self.gripper_and_hand_and_obj_space),
        #     ('desired_goal', self.hand_and_obj_space),
        #     ('achieved_goal', self.hand_and_obj_space),
        #     ('state_observation', self.gripper_and_hand_and_obj_space),
        #     ('state_desired_goal', self.hand_and_obj_space),
        #     ('state_achieved_goal', self.hand_and_obj_space),
        #     ('proprio_observation', self.hand_space),
        #     ('proprio_desired_goal', self.hand_space),
        #     ('proprio_achieved_goal', self.hand_space),
        # ])
        self.hand_reset_pos = np.array([0, .6, .2])

        if presampled_goals is not None:
            self._presampled_goals = presampled_goals
            self.num_goals_presampled = len(list(self._presampled_goals.values)[0])
        else:
            # presampled_goals will be created when sample_goal is first called
            self._presampled_goals = None
            self.num_goals_presampled = num_goals_presampled
        self._get_reference()
        self.reset()

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_stack.xml')

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

    def set_xyz_action(self, action):
        action[:3] = np.clip(action[:3], -1, 1)
        pos_delta = action[:3] * self.action_scale
        new_mocap_pos = self.data.mocap_pos + pos_delta[None]
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high,
        )
        rot_axis = action[4:] / np.linalg.norm(action[4:])
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        # replace this with learned rotation
        self.data.set_mocap_quat('mocap', np.array([np.cos(action[3]/2), np.sin(action[3]/2)*rot_axis[0], np.sin(action[3]/2)*rot_axis[1], np.sin(action[3]/2)*rot_axis[2]]))
        # self.data.set_mocap_quat('mocap', np.array([np.cos(action[3]/2), pos_delta[0]*np.abs(np.sin(action[3]/2))/np.linalg.norm(pos_delta), pos_delta[1]*np.abs(np.sin(action[3]/2))/np.linalg.norm(pos_delta), pos_delta[2]*np.abs(np.sin(action[3]/2))/np.linalg.norm(pos_delta)]))
        # self.data.set_mocap_quat('mocap', np.array([np.cos(action[3]/2), rot_axis[0]*np.abs(np.sin(action[3]/2)), rot_axis[1]*np.abs(np.sin(action[3]/2)), rot_axis[2]*np.abs(np.sin(action[3]/2))]))

    def step(self, action):
        self.render()
        self.set_xyz_action(action[:7])
        self.do_simulation(action[7:])
        new_obj_pos = self.get_obj_pos()
        if new_obj_pos[2] < .05:
            new_obj_pos[0:2] = np.clip(
                new_obj_pos[0:2],
                self.obj_low[0:2],
                self.obj_high[0:2]
            )
        self._set_obj_xyz(new_obj_pos)
        self.last_obj_pos = new_obj_pos.copy()
        ob = self._get_obs()
        ob_dict = self._get_obs_dict()
        reward = self.compute_reward(action, ob_dict)
        info = self._get_info()
        done = False
        return ob, reward, done, info

    def _get_obs(self):
        e = self.get_endeff_pos()
        b = self.get_obj_pos()
        gripper = self.get_gripper_pos()
        flat_obs = np.concatenate((e, b))
        flat_obs_with_gripper = np.concatenate((gripper, e, b))

        return np.concatenate([flat_obs_with_gripper,
                                self._state_goal])

        # return dict(
        #     observation=flat_obs_with_gripper,
        #     desired_goal=self._state_goal,
        #     achieved_goal=flat_obs,
        #     state_observation=flat_obs_with_gripper,
        #     state_desired_goal=self._state_goal,
        #     state_achieved_goal=flat_obs,
        #     proprio_observation=e,
        #     proprio_achieved_goal=e,
        #     proprio_desired_goal=hand_goal,
        # )

    def _get_obs_dict(self):
        e = self.get_endeff_pos()
        b = self.get_obj_pos()
        gripper = self.get_gripper_pos()
        flat_obs = np.concatenate((e, b))
        flat_obs_with_gripper = np.concatenate((gripper, e, b))

        return dict(
            observation=flat_obs_with_gripper,
            desired_goal=self._state_goal,
            achieved_goal=flat_obs,
            state_observation=flat_obs_with_gripper,
            state_desired_goal=self._state_goal,
            state_achieved_goal=flat_obs,
        )

    def _get_info(self):
        obj_goal = self._state_goal
        obj_distance = np.linalg.norm(obj_goal - self.get_obj_pos())
        touch_distance = np.linalg.norm(
            self.get_endeff_pos() - self.get_obj_pos()
        )
        return dict(
            obj_distance=obj_distance,
            touch_distance=touch_distance,
            obj_success=float(obj_distance < self.indicator_threshold),
            touch_success=float(touch_distance < self.indicator_threshold),
        )

    def get_obj_pos(self):
        return self.data.get_body_xpos('obj').copy()

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[8:11] = pos.copy()
        qvel[8:15] = 0
        self.set_state(qpos, qvel)

    def reset_model(self):
        self._reset_hand()
        if self.reset_free:
            self._set_obj_xyz(self.last_obj_pos)
            self.set_goal(self.sample_goal())
            return self._get_obs()

        if self.random_init:
            goal = np.random.uniform(
                self.hand_and_obj_space.low[3:],
                self.hand_and_obj_space.high[3:],
                size=(1, self.hand_and_obj_space.low.size - 3),
            )
            goal[:, 2] = self.obj_init_z
            self._set_obj_xyz(goal)
        else:
            obj_idx = np.random.choice(len(self.obj_init_positions))
            self._set_obj_xyz(self.obj_init_positions[obj_idx])

        if self.oracle_reset_prob > np.random.random():
            self.set_to_goal(self.sample_goal())

        self.set_goal(self.sample_goal())
        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.hand_reset_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation(None, self.frame_skip)

    def set_to_goal(self, goal):
        """
        This function can fail due to mocap imprecision or impossible object
        positions.
        """
        state_goal = goal['state_desired_goal']
        error = self.data.get_site_xpos('endeffector') - hand_goal
        corrected_obj_pos = state_goal + error
        corrected_obj_pos[2] = max(corrected_obj_pos[2], self.obj_init_z)
        self._set_obj_xyz(corrected_obj_pos)
        if corrected_obj_pos[2] > .03:
            action = np.array(1)
        else:
            action = np.array(1 - 2 * np.random.choice(2))

        for _ in range(10):
            self.do_simulation(action)
        self.sim.forward()

    """
    Multitask functions
    """
    def get_goal(self):
        return {
            'desired_goal': self._state_goal,
            'state_desired_goal': self._state_goal,
        }

    def set_goal(self, goal):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[15:18] = goal['desired_goal'].copy()
        qvel[15:22] = 0
        self.set_state(qpos, qvel)
        self._state_goal = self.data.get_body_xpos('goal').copy()

    def sample_goals(self, batch_size):
        if not self.fix_goal:
            if self._presampled_goals is None:
                self._presampled_goals = {}

                self._presampled_goals['desired_goal'] = np.random.uniform(
                        self.hand_and_obj_space.low[3:],
                        self.hand_and_obj_space.high[3:],
                        size=(self.num_goals_presampled, self.hand_and_obj_space.low.size - 3),
                    )
                self._presampled_goals['desired_goal'][:, 2] = self.obj_init_z
                self._presampled_goals['state_desired_goal'] = self._presampled_goals['desired_goal']
            idx = np.random.randint(0, self.num_goals_presampled, batch_size)
            sampled_goals = {
                k: v[idx] for k, v in self._presampled_goals.items()
            }
        else:
            sampled_goals = {}
            sampled_goals = {'desired_goal': np.tile(np.expand_dims(np.array(self.fixed_goal), axis=0), [batch_size, 1])}
            sampled_goals['state_desired_goal'] = sampled_goals['desired_goal']
        return sampled_goals

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        self.obj_body_id = self.sim.model.body_name2id("obj")
        self.goal_body_id = self.sim.model.body_name2id("goal")
        self.l_finger_geom_ids = self.sim.model.geom_name2id("leftclaw_it")
        self.r_finger_geom_ids = self.sim.model.geom_name2id("rightclaw_it")
        self.obj_geom_id = self.sim.model.geom_name2id("objbox")
        self.goal_geom_id = self.sim.model.geom_name2id("goalbox")

    def compute_rewards(self, actions, obs):
        r_reach, r_lift, r_stack = self.staged_rewards()
        if self.reward_type == 'dense':
            reward = max(r_reach, r_lift, r_stack)
        else:
            reward = 1.0 if r_stack > 0 else 0.0

        return [reward for _ in range(obs['observation'].shape[0])]

    def staged_rewards(self):
        """
        Helper function to return staged rewards based on current physical states.
        Returns:
            r_reach (float): reward for reaching and grasping
            r_lift (float): reward for lifting and aligning
            r_stack (float): reward for stacking
        """
        # reaching is successful when the gripper site is close to
        # the center of the cube
        table_height = 0.
        obj_pos = self.sim.data.body_xpos[self.obj_body_id]
        goal_pos = self.sim.data.body_xpos[self.goal_body_id]
        gripper_site_pos = self.sim.data.site_xpos[self.sim.model.site_name2id("endeffector")]
        dist = np.linalg.norm(gripper_site_pos - obj_pos)
        r_reach = (1 - np.tanh(10.0 * dist)) * 0.25

        # collision checking
        touch_left_finger = False
        touch_right_finger = False
        touch_obj_goal = False

        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            if c.geom1 == self.l_finger_geom_ids and c.geom2 == self.obj_geom_id:
                touch_left_finger = True
            if c.geom1 == self.obj_geom_id and c.geom2 == self.l_finger_geom_ids:
                touch_left_finger = True
            if c.geom1 == self.r_finger_geom_ids and c.geom2 == self.obj_geom_id:
                touch_right_finger = True
            if c.geom1 == self.obj_geom_id and c.geom2 == self.r_finger_geom_ids:
                touch_right_finger = True
            if c.geom1 == self.obj_geom_id and c.geom2 == self.goal_geom_id:
                touch_obj_goal = True
            if c.geom1 == self.goal_geom_id and c.geom2 == self.obj_geom_id:
                touch_obj_goal = True

        # additional grasping reward
        if touch_left_finger and touch_right_finger:
            r_reach += 0.25

        # lifting is successful when the cube is above the table top
        # by a margin
        obj_height = obj_pos[2]
        obj_lifted = obj_height > table_height + 0.04
        r_lift = 1.0 if obj_lifted else 0.0

        # Aligning is successful when obj is right above cubeB
        if obj_lifted:
            horiz_dist = np.linalg.norm(
                np.array(obj_pos[:2]) - np.array(goal_pos[:2])
            )
            r_lift += 0.5 * (1 - np.tanh(horiz_dist))

        # stacking is successful when the block is lifted and
        # the gripper is not holding the object
        r_stack = 0
        not_touching = not touch_left_finger and not touch_right_finger
        if not_touching and r_lift > 0 and touch_obj_goal:
            r_stack = 2.0

        return (r_reach, r_lift, r_stack)

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'touch_distance',
            'obj_success',
            'touch_success',
            'obj_distance',
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

    def get_env_state(self):
        base_state = super().get_env_state()
        goal = self._state_goal.copy()
        return base_state, goal

    def set_env_state(self, state):
        base_state, goal = state
        super().set_env_state(base_state)
        self._state_goal = goal


if __name__ == '__main__':
    import time
    env = SawyerStack6DOFEnv()
    for _ in range(1000):
        env.reset()
        for _ in range(50):
            env.render()
            env.step(env.action_space.sample())
            # env.step(np.array([np.random.uniform(low=-1., high=1.), np.random.uniform(low=-1., high=1.), 0.]))
            time.sleep(0.05)

