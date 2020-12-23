import abc
import copy
import pickle

from gym.spaces import Box
from gym.spaces import Discrete
import mujoco_py
import numpy as np

from metaworld.envs import reward_utils
from metaworld.envs.mujoco.mujoco_env import MujocoEnv, _assert_task_is_set


class SawyerMocapBase(MujocoEnv, metaclass=abc.ABCMeta):
    """
    Provides some commonly-shared functions for Sawyer Mujoco envs that use
    mocap for XYZ control.
    """
    mocap_low = np.array([-0.2, 0.5, 0.06])
    mocap_high = np.array([0.2, 0.7, 0.6])

    def __init__(self, model_name, frame_skip=5):
        MujocoEnv.__init__(self, model_name, frame_skip=frame_skip)
        self.reset_mocap_welds()

    def get_endeff_pos(self):
        return self.data.get_body_xpos('hand').copy()

    @property
    def tcp_center(self):
        """The COM of the gripper's 2 fingers

        Returns:
            (np.ndarray): 3-element position
        """
        right_finger_pos = self._get_site_pos('rightEndEffector')
        left_finger_pos = self._get_site_pos('leftEndEffector')
        tcp_center = (right_finger_pos + left_finger_pos) / 2.0
        return tcp_center

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

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['model']
        del state['sim']
        del state['data']
        mjb = self.model.get_mjb()
        return {'state': state, 'mjb': mjb, 'env_state': self.get_env_state()}

    def __setstate__(self, state):
        self.__dict__ = state['state']
        self.model = mujoco_py.load_model_from_mjb(state['mjb'])
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.set_env_state(state['env_state'])

    def reset_mocap_welds(self):
        """Resets the mocap welds that we use for actuation."""
        sim = self.sim
        if sim.model.nmocap > 0 and sim.model.eq_data is not None:
            for i in range(sim.model.eq_data.shape[0]):
                if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.])
        sim.forward()


class SawyerXYZEnv(SawyerMocapBase, metaclass=abc.ABCMeta):
    _HAND_SPACE = Box(
        np.array([-0.525, .348, -.0525]),
        np.array([+0.525, 1.025, .525])
    )
    max_path_length = 500

    TARGET_RADIUS = 0.05

    def __init__(
            self,
            model_name,
            frame_skip=5,
            hand_low=(-0.2, 0.55, 0.05),
            hand_high=(0.2, 0.75, 0.3),
            mocap_low=None,
            mocap_high=None,
            action_scale=1./100,
            action_rot_scale=1.,
    ):
        super().__init__(model_name, frame_skip=frame_skip)
        self.random_init = True
        self.action_scale = action_scale
        self.action_rot_scale = action_rot_scale
        self.hand_low = np.array(hand_low)
        self.hand_high = np.array(hand_high)
        if mocap_low is None:
            mocap_low = hand_low
        if mocap_high is None:
            mocap_high = hand_high
        self.mocap_low = np.hstack(mocap_low)
        self.mocap_high = np.hstack(mocap_high)
        self.curr_path_length = 0
        self._freeze_rand_vec = True
        self._last_rand_vec = None

        # We use continuous goal space by default and
        # can discretize the goal space by calling
        # the `discretize_goal_space` method.
        self.discrete_goal_space = None
        self.discrete_goals = []
        self.active_discrete_goal = None

        self.init_left_pad = self.get_body_com('leftpad')
        self.init_right_pad = self.get_body_com('rightpad')

        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([+1, +1, +1, +1]),
        )

        self._obs_obj_max_len = 14
        self._obs_obj_possible_lens = (7, 14)

        self._set_task_called = False
        self._partially_observable = True

        self.hand_init_pos = None  # OVERRIDE ME
        self._target_pos = None  # OVERRIDE ME
        self._random_reset_space = None  # OVERRIDE ME

        # Note: It is unlikely that the positions and orientations stored
        # in this initiation of prev_obs are correct. That being said, it
        # doesn't seem to matter (it will only effect frame-stacking for the
        # very first observation)
        self.prev_obs = self._get_curr_obs_combined_no_goal()

    def _set_task_inner(self):
        # Doesn't absorb "extra" kwargs, to ensure nothing's missed.
        pass

    def set_task(self, task):
        self._set_task_called = True
        data = pickle.loads(task.data)
        assert isinstance(self, data['env_cls'])
        del data['env_cls']
        self._last_rand_vec = data['rand_vec']
        self._freeze_rand_vec = True
        self._last_rand_vec = data['rand_vec']
        del data['rand_vec']
        self._partially_observable = data['partially_observable']
        del data['partially_observable']
        self._set_task_inner(**data)

    def set_xyz_action(self, action):
        action = np.clip(action, -1, 1)
        pos_delta = action * self.action_scale
        new_mocap_pos = self.data.mocap_pos + pos_delta[None]

        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high,
        )
        self.data.set_mocap_pos('mocap', new_mocap_pos)
        self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))

    def discretize_goal_space(self, goals):
        assert False
        assert len(goals) >= 1
        self.discrete_goals = goals
        # update the goal_space to a Discrete space
        self.discrete_goal_space = Discrete(len(self.discrete_goals))

    # Belows are methods for using the new wrappers.
    # `sample_goals` is implmented across the sawyer_xyz
    # as sampling from the task lists. This will be done
    # with the new `discrete_goals`. After all the algorithms
    # conform to this API (i.e. using the new wrapper), we can
    # just remove the underscore in all method signature.
    def sample_goals_(self, batch_size):
        assert False
        if self.discrete_goal_space is not None:
            return [self.discrete_goal_space.sample() for _ in range(batch_size)]
        else:
            return [self.goal_space.sample() for _ in range(batch_size)]

    def set_goal_(self, goal):
        assert False
        if self.discrete_goal_space is not None:
            self.active_discrete_goal = goal
            self.goal = self.discrete_goals[goal]
            self._target_pos_idx = np.zeros(len(self.discrete_goals))
            self._target_pos_idx[goal] = 1.
        else:
            self.goal = goal

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def _get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def _set_pos_site(self, name, pos):
        """Sets the position of the site corresponding to `name`

        Args:
            name (str): The site's name
            pos (np.ndarray): Flat, 3 element array indicating site's location
        """
        assert isinstance(pos, np.ndarray)
        assert pos.ndim == 1

        self.data.site_xpos[self.model.site_name2id(name)] = pos[:3]

    @property
    def _target_site_config(self):
        """Retrieves site name(s) and position(s) corresponding to env targets

        :rtype: list of (str, np.ndarray)
        """
        return [('goal', self._target_pos)]

    @property
    def touching_main_object(self):
        """Calls `touching_object` for the ID of the env's main object

        Returns:
            (bool) whether the gripper is touching the object

        """
        return self.touching_object(self._get_id_main_object)

    def touching_object(self, object_geom_id):
        """Determines whether the gripper is touching the object with given id

        Args:
            object_geom_id (int): the ID of the object in question

        Returns:
            (bool): whether the gripper is touching the object

        """
        leftpad_geom_id = self.unwrapped.model.geom_name2id('leftpad_geom')
        rightpad_geom_id = self.unwrapped.model.geom_name2id('rightpad_geom')

        leftpad_object_contacts = [
            x for x in self.unwrapped.data.contact
            if (leftpad_geom_id in (x.geom1, x.geom2)
                and object_geom_id in (x.geom1, x.geom2))
        ]

        rightpad_object_contacts = [
            x for x in self.unwrapped.data.contact
            if (rightpad_geom_id in (x.geom1, x.geom2)
                and object_geom_id in (x.geom1, x.geom2))
        ]

        leftpad_object_contact_force = sum(
            self.unwrapped.data.efc_force[x.efc_address]
            for x in leftpad_object_contacts)

        rightpad_object_contact_force = sum(
            self.unwrapped.data.efc_force[x.efc_address]
            for x in rightpad_object_contacts)

        return 0 < leftpad_object_contact_force and \
               0 < rightpad_object_contact_force

    @property
    def _get_id_main_object(self):
        return self.unwrapped.model.geom_name2id('objGeom')

    def _get_pos_objects(self):
        """Retrieves object position(s) from mujoco properties or instance vars

        Returns:
            np.ndarray: Flat array (usually 3 elements) representing the
                object(s)' position(s)
        """
        # Throw error rather than making this an @abc.abstractmethod so that
        # V1 environments don't have to implement it
        raise NotImplementedError

    def _get_quat_objects(self):
        """Retrieves object quaternion(s) from mujoco properties

        Returns:
            np.ndarray: Flat array (usually 4 elements) representing the
                object(s)' quaternion(s)

        """
        # Throw error rather than making this an @abc.abstractmethod so that
        # V1 environments don't have to implement it
        raise NotImplementedError

    def _get_pos_goal(self):
        """Retrieves goal position from mujoco properties or instance vars

        Returns:
            np.ndarray: Flat array (3 elements) representing the goal position
        """
        assert isinstance(self._target_pos, np.ndarray)
        assert self._target_pos.ndim == 1
        return self._target_pos

    def _get_curr_obs_combined_no_goal(self):
        """Combines the end effector's {pos, closed amount} and the object(s)'
            {pos, quat} into a single flat observation. The goal's position is
            *not* included in this.

        Returns:
            np.ndarray: The flat observation array (18 elements)

        """
        pos_hand = self.get_endeff_pos()

        finger_right, finger_left = (
            self._get_site_pos('rightEndEffector'),
            self._get_site_pos('leftEndEffector')
        )

        # the gripper can be at maximum about ~0.1 m apart.
        # dividing by 0.1 normalized the gripper distance between
        # 0 and 1. Further, we clip because sometimes the grippers
        # are slightly more than 0.1m apart (~0.00045 m)
        # clipping removes the effects of this random extra distance
        # that is produced by mujoco
        gripper_distance_apart = np.linalg.norm(finger_right - finger_left)
        gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0., 1.)

        obs_obj_padded = np.zeros(self._obs_obj_max_len)

        obj_pos = self._get_pos_objects()
        obj_quat = self._get_quat_objects()
        assert len(obj_pos) % 3 == 0
        assert len(obj_quat) % 4 == 0
        obj_pos_split = np.split(obj_pos, len(obj_pos) // 3)
        obj_quat_split = np.split(obj_quat, len(obj_quat) // 4)

        obs_obj_padded[:len(obj_pos) + len(obj_quat)] = np.hstack([
            np.hstack((pos, quat))
            for pos, quat in zip(obj_pos_split, obj_quat_split)
        ])
        assert(len(obs_obj_padded) in self._obs_obj_possible_lens)

        return np.hstack((pos_hand, gripper_distance_apart, obs_obj_padded))

    def _get_obs(self):
        """Frame stacks `_get_curr_obs_combined_no_goal()` and concatenates the
            goal position to form a single flat observation.

        Returns:
            np.ndarray: The flat observation array (39 elements)
        """
        # do frame stacking
        pos_goal = self._get_pos_goal()
        if self._partially_observable:
            pos_goal = np.zeros_like(pos_goal)
        curr_obs = self._get_curr_obs_combined_no_goal()
        # do frame stacking
        obs = np.hstack((curr_obs, self.prev_obs, pos_goal))
        self.prev_obs = curr_obs
        return obs

    def _get_obs_dict(self):
        obs = self._get_obs()
        return dict(
            state_observation=obs,
            state_desired_goal=self._get_pos_goal(),
            state_achieved_goal=obs[3:-3],
        )

    @property
    def observation_space(self):
        obj_low = np.full(self._obs_obj_max_len, -np.inf)
        obj_high = np.full(self._obs_obj_max_len, +np.inf)
        goal_low = np.zeros(3) if self._partially_observable \
            else self.goal_space.low
        goal_high = np.zeros(3) if self._partially_observable \
            else self.goal_space.high
        gripper_low = -1.
        gripper_high = +1.
        return Box(
            np.hstack((self._HAND_SPACE.low, gripper_low, obj_low, self._HAND_SPACE.low, gripper_low, obj_low, goal_low)),
            np.hstack((self._HAND_SPACE.high, gripper_high, obj_high, self._HAND_SPACE.high, gripper_high, obj_high, goal_high))
        )

    @_assert_task_is_set
    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation([action[-1], -action[-1]])

        for site in self._target_site_config:
            self._set_pos_site(*site)

        return self._get_obs()

    def reset(self):
        self.curr_path_length = 0
        return super().reset()

    def _reset_hand(self, steps=50):
        self.init_tcp = self.tcp_center
        for _ in range(steps):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1, 1], self.frame_skip)

    def _get_state_rand_vec(self):
        if self._freeze_rand_vec:
            assert self._last_rand_vec is not None
            return self._last_rand_vec
        else:
            rand_vec = np.random.uniform(
                self._random_reset_space.low,
                self._random_reset_space.high,
                size=self._random_reset_space.low.size)
            self._last_rand_vec = rand_vec
            return rand_vec

    def _gripper_caging_reward(self,
                               action,
                               obj_pos,
                               obj_radius,
                               pad_success_margin,
                               object_reach_radius,
                               x_z_margin,
                               high_density=False,
                               medium_density=False):
        """Reward for agent grasping obj
            Args:
                action(np.ndarray): (4,) array representing the action
                    delta(x), delta(y), delta(z), gripper_effort
                obj_pos(np.ndarray): (3,) array representing the obj x,y,z
                obj_radius(float):radius of object's bounding sphere
                pad_success_margin(float): successful distance of gripper_pad
                    to object
                object_reach_radius(float): successful distance of gripper center
                    to the object.
                x_z_margin(float): successful distance of gripper in x_z axis to the
                    object. Y axis not included since the caging function handles
                        successful grasping in the Y axis.
        """
        if high_density and medium_density:
            raise ValueError("Can only be either high_density or medium_density")
        # MARK: Left-right gripper information for caging reward----------------
        left_pad = self.get_body_com('leftpad')
        right_pad = self.get_body_com('rightpad')

        tcp = self.tcp_center
        tcp_to_obj = np.linalg.norm(obj_pos - tcp)
        tcp_to_obj_init = np.linalg.norm(self.obj_init_pos - self.init_tcp)
        reach = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, object_reach_radius),
            margin=abs(tcp_to_obj_init-object_reach_radius),
            sigmoid='long_tail',
        )

        pad_y_lr = np.hstack((left_pad[1], right_pad[1]))
        pad_y_lr_init = np.hstack((self.init_left_pad[1], self.init_right_pad[1]))

        obj_to_pad_lr = np.abs(pad_y_lr - obj_pos[1])
        obj_to_pad_lr_init = np.abs(pad_y_lr_init - self.obj_init_pos[1])

        caging_margin_lr = np.abs(obj_to_pad_lr_init - pad_success_margin)
        caging_lr = [reward_utils.tolerance(
            obj_to_pad_lr[i],
            bounds=(obj_radius, pad_success_margin),
            margin=caging_margin_lr[i],
            sigmoid='long_tail',
        ) for i in range(2)]
        caging_y = reward_utils.hamacher_product(*caging_lr)

        # MARK: X-Z gripper information for caging reward-----------------------
        tcp = self.tcp_center
        xz = [0, 2]
        xz_margin = np.linalg.norm(self.obj_init_pos[xz] - self.init_tcp[xz])
        xz_margin -= x_z_margin

        caging_xz = reward_utils.tolerance(
            np.linalg.norm(tcp[xz] - obj_pos[xz]),
            bounds=(0, x_z_margin),
            margin=xz_margin,
            sigmoid='long_tail',
        )

        # MARK: Closed-extent gripper information for caging reward-------------
        gripper_closed = min(max(0, action[-1]), 1)

        # MARK: Combine components----------------------------------------------
        caging = reward_utils.hamacher_product(caging_y, caging_xz)
        gripping = gripper_closed if caging > 0.97 else 0.
        caging_and_gripping = reward_utils.hamacher_product(caging, gripping)

        if high_density:
            caging_and_gripping = (caging_and_gripping + caging) / 2
        if medium_density:
            caging_and_gripping = (caging_and_gripping + reach) / 2

        return caging_and_gripping
