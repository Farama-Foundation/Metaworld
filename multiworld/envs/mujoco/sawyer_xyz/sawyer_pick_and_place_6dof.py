from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict

from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv
from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera, sawyer_pick_and_place_camera_slanted_angle


class SawyerPickAndPlace6DOFEnv(MultitaskEnv, SawyerXYZEnv):
    def __init__(
            self,
            obj_low=None,
            obj_high=None,

            reward_type='touch_and_obj_distance',
            indicator_threshold=0.06,

            obj_init_positions=((0, 0.6, 0.02),),
            random_init=False,

            fix_goal=False,
            fixed_goal=(0.15, 0.6, 0.055, -0.15, 0.6),
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
        self.reset()

    @property
    def model_name(self):
        return get_asset_full_path('sawyer_xyz/sawyer_pick_and_place.xml')

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
        # self.render()
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
        # The marker seems to get reset every time you do a simulation
        self._set_goal_marker(self._state_goal)
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
        hand_goal = self._state_goal[:3]

        return np.concatenate([flat_obs_with_gripper,
                                self._state_goal[3:]])

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
        hand_goal = self._state_goal[:3]

        return dict(
            observation=flat_obs_with_gripper,
            desired_goal=self._state_goal,
            achieved_goal=flat_obs,
            state_observation=flat_obs_with_gripper,
            state_desired_goal=self._state_goal,
            state_achieved_goal=flat_obs,
            proprio_observation=e,
            proprio_achieved_goal=e,
            proprio_desired_goal=hand_goal,
        )

    def _get_info(self):
        hand_goal = self._state_goal[:3]
        obj_goal = self._state_goal[3:]
        hand_distance = np.linalg.norm(hand_goal - self.get_endeff_pos())
        obj_distance = np.linalg.norm(obj_goal - self.get_obj_pos())
        touch_distance = np.linalg.norm(
            self.get_endeff_pos() - self.get_obj_pos()
        )
        return dict(
            hand_distance=hand_distance,
            obj_distance=obj_distance,
            hand_and_obj_distance=hand_distance+obj_distance,
            touch_distance=touch_distance,
            hand_success=float(hand_distance < self.indicator_threshold),
            obj_success=float(obj_distance < self.indicator_threshold),
            hand_and_obj_success=float(
                hand_distance+obj_distance < self.indicator_threshold
            ),
            touch_success=float(touch_distance < self.indicator_threshold),
        )

    def get_obj_pos(self):
        return self.data.get_body_xpos('obj').copy()

    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('hand-goal-site')] = (
            goal[:3]
        )
        self.data.site_xpos[self.model.site_name2id('obj-goal-site')] = (
            goal[3:]
        )
        if self.hide_goal_markers:
            self.data.site_xpos[self.model.site_name2id('hand-goal-site'), 2] = (
                -1000
            )
            self.data.site_xpos[self.model.site_name2id('obj-goal-site'), 2] = (
                -1000
            )

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
            self._set_goal_marker(self._state_goal)
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
        self._set_goal_marker(self._state_goal)
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
        hand_goal = state_goal[:3]
        for _ in range(30):
            self.data.set_mocap_pos('mocap', hand_goal)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation(np.array([-1]))
        error = self.data.get_site_xpos('endeffector') - hand_goal
        corrected_obj_pos = state_goal[3:] + error
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
        self._state_goal = goal['state_desired_goal']
        self._set_goal_marker(self._state_goal)

    def sample_goals(self, batch_size):
        if self._presampled_goals is None:
            self._presampled_goals = \
                    corrected_state_goals(
                        self,
                        self.generate_uncorrected_env_goals(
                            self.num_goals_presampled
                        )
                    )
        idx = np.random.randint(0, self.num_goals_presampled, batch_size)
        sampled_goals = {
            k: v[idx] for k, v in self._presampled_goals.items()
        }
        return sampled_goals


    def compute_rewards(self, actions, obs):
        achieved_goals = obs['state_achieved_goal']
        desired_goals = obs['state_desired_goal']
        hand_pos = achieved_goals[:, :3]
        obj_pos = achieved_goals[:, 3:]
        hand_goals = desired_goals[:, :3]
        obj_goals = desired_goals[:, 3:]

        hand_distances = np.linalg.norm(hand_goals - hand_pos, axis=1)
        obj_distances = np.linalg.norm(obj_goals - obj_pos, axis=1)
        hand_and_obj_distances = hand_distances + obj_distances
        touch_distances = np.linalg.norm(hand_pos - obj_pos, axis=1)
        touch_and_obj_distances = touch_distances + obj_distances

        if self.reward_type == 'hand_distance':
            r = -hand_distances
        elif self.reward_type == 'hand_success':
            r = -(hand_distances > self.indicator_threshold).astype(float)
        elif self.reward_type == 'obj_distance':
            r = -obj_distances
        elif self.reward_type == 'obj_success':
            r = -(obj_distances > self.indicator_threshold).astype(float)
        elif self.reward_type == 'hand_and_obj_distance':
            r = -hand_and_obj_distances
        elif self.reward_type == 'touch_and_obj_distance':
            r = -touch_and_obj_distances
            if touch_distances.mean() <= 0.05:
                r += max(actions[:, -1], 0) / 10
        elif self.reward_type == 'hand_and_obj_success':
            r = -(
                hand_and_obj_distances < self.indicator_threshold
            ).astype(float)
        elif self.reward_type == 'touch_distance':
            r = -touch_distances
        elif self.reward_type == 'touch_success':
            r = -(touch_distances > self.indicator_threshold).astype(float)
        else:
            raise NotImplementedError("Invalid/no reward type.")
        return r

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        for stat_name in [
            'touch_distance',
            'hand_success',
            'obj_success',
            'hand_and_obj_success',
            'touch_success',
            'hand_distance',
            'obj_distance',
            'hand_and_obj_distance',
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
        self._set_goal_marker(goal)

    def generate_uncorrected_env_goals(self, num_goals):
        """
        Due to small errors in mocap, moving to a specified hand position may be
        slightly off. This is an issue when the object must be placed into a given
        hand goal since high precision is needed. The solution used is to try and
        set to the goal manually and then take whatever goal the hand and object
        end up in as the "corrected" goal. The downside to this is that it's not
        possible to call set_to_goal with the corrected goal as input as mocap
        errors make it impossible to rereate the exact same hand position.

        The return of this function should be passed into
        corrected_image_env_goals or corrected_state_env_goals
        """
        if self.fix_goal:
            goals = np.repeat(self.fixed_goal.copy()[None], num_goals, 0)
        else:
            goals = np.random.uniform(
                self.hand_and_obj_space.low,
                self.hand_and_obj_space.high,
                size=(num_goals, self.hand_and_obj_space.low.size),
            )
            num_objs_in_hand = int(num_goals * self.p_obj_in_hand)
            if num_goals == 1:
                num_objs_in_hand = int(np.random.random() < self.p_obj_in_hand)

            # Put object in hand
            goals[:num_objs_in_hand, 3:] = goals[:num_objs_in_hand, :3].copy()
            goals[:num_objs_in_hand, 4] -= 0.01
            goals[:num_objs_in_hand, 5] += 0.01

            # Put object one the table (not floating)
            goals[num_objs_in_hand:, 5] = self.obj_init_z
            return {
                'desired_goal': goals,
                'state_desired_goal': goals,
                'proprio_desired_goal': goals[:, :3]
            }

class SawyerPickAndPlace6DOFEnvYZ(SawyerPickAndPlace6DOFEnv):

    def __init__(
        self,
        x_axis=0.0,
        *args,
        **kwargs
    ):
        self.quick_init(locals())
        super().__init__(*args, **kwargs)
        self.x_axis = x_axis
        pos_arrays = [
            self.hand_and_obj_space.low[:3],
            self.hand_and_obj_space.low[3:],
            self.hand_and_obj_space.high[:3],
            self.hand_and_obj_space.high[3:],

            self.gripper_and_hand_and_obj_space.low[1:4],
            self.gripper_and_hand_and_obj_space.low[4:],
            self.gripper_and_hand_and_obj_space.high[1:4],
            self.gripper_and_hand_and_obj_space.high[4:],

            self.hand_space.high[:3],
            self.hand_space.low[:3],
        ]
        for pos in pos_arrays:
            pos[0] = x_axis

        self.action_space = Box(
            np.array([-1, -1, -1]),
            np.array([1, 1, 1]),
            dtype=np.float32
        )
        self.hand_reset_pos = np.array([x_axis, .6, .2])

    def convert_2d_action(self, action):
        cur_x_pos = self.get_endeff_pos()[0]
        adjust_x = self.x_axis - cur_x_pos
        return np.r_[adjust_x, action]

    def step(self, action):
        new_obj_pos = self.data.get_site_xpos('obj')
        new_obj_pos[0] = self.x_axis
        self._set_obj_xyz(new_obj_pos)
        action = self.convert_2d_action(action)
        return super().step(action)

    def set_to_goal(self, goal):
        super().set_to_goal(goal)
        obj_pos = self.get_obj_pos()
        obj_pos[0] = self.x_axis
        self._set_obj_xyz(obj_pos)


def corrected_state_goals(pickup_env, pickup_env_goals):
    pickup_env._state_goal = np.zeros(6)
    goals = pickup_env_goals.copy()
    num_goals = len(list(goals.values())[0])
    for idx in range(num_goals):
        pickup_env.set_to_goal(
            {'state_desired_goal': goals['state_desired_goal'][idx]}
        )
        corrected_state_goal = pickup_env._get_obs_dict()['achieved_goal']
        corrected_proprio_goal = pickup_env._get_obs_dict()['proprio_achieved_goal']

        goals['desired_goal'][idx] = corrected_state_goal
        goals['proprio_desired_goal'][idx] = corrected_proprio_goal
        goals['state_desired_goal'][idx] = corrected_state_goal
    return goals

def corrected_image_env_goals(image_env, pickup_env_goals):
    """
    This isn't as easy as setting to the corrected since mocap will fail to
    move to the exact position, and the object will fail to stay in the hand.
    """

    image_env.wrapped_env._state_goal = np.zeros(6)
    goals = pickup_env_goals.copy()

    num_goals = len(list(goals.values())[0])
    goals = dict(
        image_desired_goal=np.zeros((num_goals, image_env.image_length)),
        desired_goal=np.zeros((num_goals, image_env.image_length)),
        state_desired_goal=np.zeros((num_goals, 6)),
        proprio_desired_goal=np.zeros((num_goals, 3))
    )
    for idx in range(num_goals):
        if idx % 100 == 0:
            print(idx)
        image_env.set_to_goal(
            {'state_desired_goal': pickup_env_goals['state_desired_goal'][idx]}
        )
        corrected_state_goal = image_env._get_obs_dict()['state_achieved_goal']
        corrected_proprio_goal = image_env._get_obs_dict()['proprio_achieved_goal']
        corrected_image_goal = image_env._get_obs_dict()['image_achieved_goal']

        goals['image_desired_goal'][idx] = corrected_image_goal
        goals['desired_goal'][idx] = corrected_image_goal
        goals['state_desired_goal'][idx] = corrected_state_goal
        goals['proprio_desired_goal'][idx] = corrected_proprio_goal
    return goals

def get_image_presampled_goals(image_env, num_presampled_goals):
    image_env.reset()
    pickup_env = image_env.wrapped_env
    image_env_goals = corrected_image_env_goals(
        image_env,
        pickup_env.generate_uncorrected_env_goals(num_presampled_goals)
    )
    return image_env_goals

if __name__ == '__main__':
    import time
    env = SawyerPickAndPlaceEnv()
    for _ in range(1000):
        env.reset()
        for _ in range(50):
            env.render()
            env.step(env.action_space.sample())
            # env.step(np.array([np.random.uniform(low=-1., high=1.), np.random.uniform(low=-1., high=1.), 0.]))
            time.sleep(0.05)

