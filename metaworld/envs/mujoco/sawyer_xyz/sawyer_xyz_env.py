import abc
import copy
import pickle

from gymnasium.spaces import Box
from gymnasium.spaces import Discrete
from gymnasium.utils.ezpickle import EzPickle

import mujoco
import numpy as np
import sys

from metaworld.envs import reward_utils
from metaworld.envs.mujoco.mujoco_env import _assert_task_is_set
from gymnasium.utils import seeding
from gymnasium.envs.mujoco import MujocoEnv as mjenv_gym

class SawyerMocapBase(mjenv_gym):
    """Provides some commonly-shared functions for Sawyer Mujoco envs that use mocap for XYZ control."""

    mocap_low = np.array([-0.2, 0.5, 0.06])
    mocap_high = np.array([0.2, 0.7, 0.6])
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 80,
    }

    def __init__(self, model_name, frame_skip=5):
        mjenv_gym.__init__(self, model_name, frame_skip=frame_skip, observation_space=self.observation_space)
        self.reset_mocap_welds()
        self.frame_skip = frame_skip

    def get_endeff_pos(self):
        return self.data.body('hand').xpos

    @property
    def tcp_center(self):
        """The COM of the gripper's 2 fingers.

        Returns:
            (np.ndarray): 3-element position
        """
        right_finger_pos = self.data.site('rightEndEffector')
        left_finger_pos = self.data.site('leftEndEffector')
        tcp_center = (right_finger_pos.xpos + left_finger_pos.xpos) / 2.0
        return tcp_center

    def get_env_state(self):
        qpos = np.copy(self.data.qpos)
        qvel = np.copy(self.data.qvel)
        return copy.deepcopy((qpos, qvel))

    def set_env_state(self, state):
        mocap_pos, mocap_quat = state
        self.set_state(mocap_pos, mocap_quat)

    def __getstate__(self):
        state = self.__dict__.copy()
        # del state['model']
        # del state['data']
        return {'state': state, 'mjb': self.model_name, 'mocap':self.get_env_state()}

    def __setstate__(self, state):
        self.__dict__ = state['state']
        mjenv_gym.__init__(self, state['mjb'], frame_skip=self.frame_skip, observation_space=self.observation_space)
        self.set_env_state(state['mocap'])

    def reset_mocap_welds(self):
        """Resets the mocap welds that we use for actuation."""
        if self.model.nmocap > 0 and self.model.eq_data is not None:
            for i in range(self.model.eq_data.shape[0]):
                if self.model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                    self.model.eq_data[i] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])


class SawyerXYZEnv(SawyerMocapBase, EzPickle):
    _HAND_SPACE = Box(
        np.array([-0.525, .348, -.0525]),
        np.array([+0.525, 1.025, .7]),
        dtype=np.float64
    )
    max_path_length = 500

    TARGET_RADIUS = 0.05

    current_task = 0
    classes = None
    classes_kwargs = None
    tasks = None
    def __init__(
        self,
        model_name,
        frame_skip=5,
        hand_low=(-0.2, 0.55, 0.05),
        hand_high=(0.2, 0.75, 0.3),
        mocap_low=None,
        mocap_high=None,
        action_scale=1.0 / 100,
        action_rot_scale=1.0,
    ):


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
        self.seeded_rand_vec = False
        self._freeze_rand_vec = True
        self._last_rand_vec = None
        self.num_resets = 0
        self.current_seed = None

        # We use continuous goal space by default and
        # can discretize the goal space by calling
        # the `discretize_goal_space` method.
        self.discrete_goal_space = None
        self.discrete_goals = []
        self.active_discrete_goal = None

        self._partially_observable = True

        super().__init__(model_name, frame_skip=frame_skip)

        mujoco.mj_forward(self.model, self.data)  # *** DO NOT REMOVE: EZPICKLE WON'T WORK *** #

        self._did_see_sim_exception = False
        self.init_left_pad = self.get_body_com("leftpad")
        self.init_right_pad = self.get_body_com("rightpad")

        self.action_space = Box(
            np.array([-1, -1, -1, -1]),
            np.array([+1, +1, +1, +1]),
            dtype=np.float64,
        )

        # Technically these observation lengths are different between v1 and v2,
        # but we handle that elsewhere and just stick with v2 numbers here
        self._obs_obj_max_len = 14

        self._set_task_called = False

        self.hand_init_pos = None  # OVERRIDE ME
        self._target_pos = None  # OVERRIDE ME
        self._random_reset_space = None  # OVERRIDE ME

        self._last_stable_obs = None
        # Note: It is unlikely that the positions and orientations stored
        # in this initiation of _prev_obs are correct. That being said, it
        # doesn't seem to matter (it will only effect frame-stacking for the
        # very first observation)
        self.seeded_rand_vec = False

        self._prev_obs = self._get_curr_obs_combined_no_goal()
        self.current_task = 0



        EzPickle.__init__(
            self,
            model_name,
            frame_skip,
            hand_low,
            hand_high,
            mocap_low,
            mocap_high,
            action_scale,
            action_rot_scale,
        )

    @staticmethod
    def _set_task_inner():
        # Doesn't absorb "extra" kwargs, to ensure nothing's missed.
        pass

    def set_task(self):
        data = self.tasks[self.current_task]
        data = pickle.loads(data.data)
        self.current_task = (self.current_task + 1) % len(self.tasks)
        self._set_task_called = True
        if type(self) == data['env_cls']:
            pass
        elif type(self) == type(data['env_cls']):
            pass
        else:
            raise ValueError("self is not same type as data['env_cls']")
        del data['env_cls']
        self._freeze_rand_vec = True
        self._last_rand_vec = copy.copy(data['rand_vec'])
        del data['rand_vec']
        self._partially_observable = data['partially_observable']
        del data['partially_observable']
        self._set_task_inner(**data)
        
    @staticmethod
    def _make_tasks(classes, args_kwargs, kwargs_override, seed=None):
        st0 = None
        if seed is not None:
            st0 = np.random.get_state()
            np.random.seed(seed)
        tasks = []
        env = classes()
        env._freeze_rand_vec = False
        env._set_task_called = True
        rand_vecs = []
        kwargs = self.classes_kwargs.copy()
        del kwargs['task_id']
        env._set_task_inner(**kwargs)
        for _ in range(_N_GOALS):
            env.reset()
            rand_vecs.append(env._last_rand_vec)
        unique_task_rand_vecs = np.unique(np.array(rand_vecs), axis=0)
        assert unique_task_rand_vecs.shape[0] == _N_GOALS, unique_task_rand_vecs.shape[0]
        env.close()
        for rand_vec in rand_vecs:
            kwargs = args['kwargs'].copy()
            del kwargs['task_id']
            kwargs.update(dict(rand_vec=rand_vec, env_cls=env_cls))
            kwargs.update(kwargs_override)
            tasks.append(_encode_task(env_name, kwargs))
        if seed is not None:
            np.random.set_state(st0)
        self.current_task = 0
        return tasks

    def set_xyz_action(self, action):
        action = np.clip(action, -1, 1)
        pos_delta = action * self.action_scale
        new_mocap_pos = self.data.mocap_pos + pos_delta[None]
        new_mocap_pos[0, :] = np.clip(
            new_mocap_pos[0, :],
            self.mocap_low,
            self.mocap_high,
        )
        self.data.mocap_pos = new_mocap_pos
        self.data.mocap_quat = np.array([1, 0, 1, 0])

    def discretize_goal_space(self, goals):
        assert False
        assert len(goals) >= 1
        self.discrete_goals = goals
        # update the goal_space to a Discrete space
        self.discrete_goal_space = Discrete(len(self.discrete_goals))

    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def _get_site_pos(self, siteName):
        _id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, siteName)
        return self.data.site_xpos[_id].copy()

    def _set_pos_site(self, name, pos):
        """Sets the position of the site corresponding to `name`.

        Args:
            name (str): The site's name
            pos (np.ndarray): Flat, 3 element array indicating site's location
        """
        assert isinstance(pos, np.ndarray)
        assert pos.ndim == 1

        _id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        self.data.site_xpos[_id] = pos[:3]

    @property
    def _target_site_config(self):
        """Retrieves site name(s) and position(s) corresponding to env targets.

        :rtype: list of (str, np.ndarray)
        """
        return [("goal", self._target_pos)]

    @property
    def touching_main_object(self):
        """Calls `touching_object` for the ID of the env's main object.

        Returns:
            (bool) whether the gripper is touching the object

        """
        return self.touching_object(self._get_id_main_object)

    def touching_object(self, object_geom_id):
        """Determines whether the gripper is touching the object with given id.

        Args:
            object_geom_id (int): the ID of the object in question

        Returns:
            (bool): whether the gripper is touching the object

        """

        leftpad_geom_id = self.data.geom('leftpad_geom').id
        rightpad_geom_id = self.data.geom('rightpad_geom').id


        leftpad_object_contacts = [
            x
            for x in self.unwrapped.data.contact
            if (
                leftpad_geom_id in (x.geom1, x.geom2)
                and object_geom_id in (x.geom1, x.geom2)
            )
        ]

        rightpad_object_contacts = [
            x
            for x in self.unwrapped.data.contact
            if (
                rightpad_geom_id in (x.geom1, x.geom2)
                and object_geom_id in (x.geom1, x.geom2)
            )
        ]

        leftpad_object_contact_force = sum(
            self.unwrapped.data.efc_force[x.efc_address]
            for x in leftpad_object_contacts
        )

        rightpad_object_contact_force = sum(
            self.unwrapped.data.efc_force[x.efc_address]
            for x in rightpad_object_contacts
        )

        return 0 < leftpad_object_contact_force and 0 < rightpad_object_contact_force

    @property
    def _get_id_main_object(self):
        return self.data.geom('objGeom').id  #[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'objGeom')]


    def _get_pos_objects(self):
        """Retrieves object position(s) from mujoco properties or instance vars.

        Returns:
            np.ndarray: Flat array (usually 3 elements) representing the
                object(s)' position(s)
        """
        # Throw error rather than making this an @abc.abstractmethod so that
        # V1 environments don't have to implement it
        raise NotImplementedError

    def _get_quat_objects(self):
        """Retrieves object quaternion(s) from mujoco properties.

        Returns:
            np.ndarray: Flat array (usually 4 elements) representing the
                object(s)' quaternion(s)

        """
        # Throw error rather than making this an @abc.abstractmethod so that
        # V1 environments don't have to implement it
        raise NotImplementedError


    def _get_pos_goal(self):
        """Retrieves goal position from mujoco properties or instance vars.

        Returns:
            np.ndarray: Flat array (3 elements) representing the goal position
        """
        assert isinstance(self._target_pos, np.ndarray)
        assert self._target_pos.ndim == 1
        return self._target_pos

    def _get_curr_obs_combined_no_goal(self):
        """Combines the end effector's {pos, closed amount} and the object(s)' {pos, quat} into a single flat observation.

        Note: The goal's position is *not* included in this.

        Returns:
            np.ndarray: The flat observation array (18 elements)

        """

        pos_hand = self.get_endeff_pos()

        finger_right, finger_left = (

            self.data.body('rightclaw'),
            self.data.body('leftclaw')
        )
        # the gripper can be at maximum about ~0.1 m apart.
        # dividing by 0.1 normalized the gripper distance between
        # 0 and 1. Further, we clip because sometimes the grippers
        # are slightly more than 0.1m apart (~0.00045 m)
        # clipping removes the effects of this random extra distance
        # that is produced by mujoco

        gripper_distance_apart = np.linalg.norm(finger_right.xpos - finger_left.xpos)
        gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0., 1.)

        obs_obj_padded = np.zeros(self._obs_obj_max_len)
        obj_pos = self._get_pos_objects()
        assert len(obj_pos) % 3 == 0
        obj_pos_split = np.split(obj_pos, len(obj_pos) // 3)

        obj_quat = self._get_quat_objects()
        assert len(obj_quat) % 4 == 0
        obj_quat_split = np.split(obj_quat, len(obj_quat) // 4)
        obs_obj_padded[:len(obj_pos) + len(obj_quat)] = np.hstack([
            np.hstack((pos, quat))
            for pos, quat in zip(obj_pos_split, obj_quat_split)
        ])
        return np.hstack((pos_hand, gripper_distance_apart, obs_obj_padded))

    def _get_obs(self):
        """Frame stacks `_get_curr_obs_combined_no_goal()` and concatenates the goal position to form a single flat observation.

        Returns:
            np.ndarray: The flat observation array (39 elements)
        """
        # do frame stacking
        pos_goal = self._get_pos_goal()
        if self._partially_observable:
            pos_goal = np.zeros_like(pos_goal)
        curr_obs = self._get_curr_obs_combined_no_goal()
        # do frame stacking
        obs = np.hstack((curr_obs, self._prev_obs, pos_goal))
        self._prev_obs = curr_obs
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
        obs_obj_max_len = 14
        obj_low = np.full(obs_obj_max_len, -np.inf, dtype=np.float64)
        obj_high = np.full(obs_obj_max_len, +np.inf, dtype=np.float64)
        goal_low = np.zeros(3) if self._partially_observable \
            else self.goal_space.low
        goal_high = np.zeros(3) if self._partially_observable \
            else self.goal_space.high
        gripper_low = -1.
        gripper_high = +1.
        return Box(
            np.hstack((self._HAND_SPACE.low, gripper_low, obj_low, self._HAND_SPACE.low, gripper_low, obj_low, goal_low)),
            np.hstack((self._HAND_SPACE.high, gripper_high, obj_high, self._HAND_SPACE.high, gripper_high, obj_high, goal_high)),
            dtype=np.float64
        )

    @_assert_task_is_set
    def step(self, action):
        self.set_xyz_action(action[:3])
        self.do_simulation([action[-1], -action[-1]], n_frames=self.frame_skip)
        self.curr_path_length += 1

        # Running the simulator can sometimes mess up site positions, so
        # re-position them here to make sure they're accurate
        for site in self._target_site_config:
            self._set_pos_site(*site)

        if self._did_see_sim_exception:
            return (
                self._last_stable_obs,  # observation just before going unstable
                0.0,  # reward (penalize for causing instability)
                False,  # termination flag always False
                {  # info
                    "success": False,
                    "near_object": 0.0,
                    "grasp_success": False,
                    "grasp_reward": 0.0,
                    "in_place_reward": 0.0,
                    "obj_to_target": 0.0,
                    "unscaled_reward": 0.0,
                },
            )

        self._last_stable_obs = self._get_obs()

        self._last_stable_obs = np.clip(self._last_stable_obs,
                                        a_max=self.observation_space.high,
                                        a_min=self.observation_space.low,
                                        dtype=np.float64)
        reward, info = self.evaluate_state(self._last_stable_obs, action)
        done = True if int(info['success']) == 1 else False
        return np.array(self._last_stable_obs, dtype=np.float64), reward, done, False, info

    def evaluate_state(self, obs, action):
        """Does the heavy-lifting for `step()` -- namely, calculating reward and populating the `info` dict with training metrics.

        Returns:
            float: Reward between 0 and 10
            dict: Dictionary which contains useful metrics (success,
                near_object, grasp_success, grasp_reward, in_place_reward,
                obj_to_target, unscaled_reward)

        """
        # Throw error rather than making this an @abc.abstractmethod so that
        # V1 environments don't have to implement it
        raise NotImplementedError

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.tasks = self._make_tasks(self.classes, self.cls_kwargs, self._partially_observable, seed=seed)
            self.set_task()
        elif bool(self.tasks) and not self.seeded_rand_vec:
            self.set_task()
        self.curr_path_length = 0
        obs, info = super().reset()
        self._prev_obs = obs[:18].copy()
        obs[18:36] = self._prev_obs
        obs = np.float64(obs)
        return obs, info

    def _reset_hand(self, steps=50):
        mocap_id = self.model.body_mocapid[self.data.body("mocap").id]
        for _ in range(steps):
            self.data.mocap_pos[mocap_id][:] = self.hand_init_pos
            self.data.mocap_quat[mocap_id][:] = np.array([1, 0, 1, 0])
            self.do_simulation([-1, 1], self.frame_skip)
        self.init_tcp = self.tcp_center

        self.init_tcp = self.tcp_center

    def _get_state_rand_vec(self):
        if self._freeze_rand_vec:
            assert self._last_rand_vec is not None
            return self._last_rand_vec
        else: 
            rand_vec = np.random.uniform(
                self._random_reset_space.low,
                self._random_reset_space.high,
                size=self._random_reset_space.low.size).astype(np.float64)
            self._last_rand_vec = rand_vec
            return rand_vec

    def _gripper_caging_reward(
        self,
        action,
        obj_pos,
        obj_radius,
        pad_success_thresh,
        object_reach_radius,
        xz_thresh,
        desired_gripper_effort=1.0,
        high_density=False,
        medium_density=False,
    ):
        """Reward for agent grasping obj.

        Args:
            action(np.ndarray): (4,) array representing the action
                delta(x), delta(y), delta(z), gripper_effort
            obj_pos(np.ndarray): (3,) array representing the obj x,y,z
            obj_radius(float):radius of object's bounding sphere
            pad_success_thresh(float): successful distance of gripper_pad
                to object
            object_reach_radius(float): successful distance of gripper center
                to the object.
            xz_thresh(float): successful distance of gripper in x_z axis to the
                object. Y axis not included since the caging function handles
                    successful grasping in the Y axis.
            desired_gripper_effort(float): desired gripper effort, defaults to 1.0.
            high_density(bool): flag for high-density. Cannot be used with medium-density.
            medium_density(bool): flag for medium-density. Cannot be used with high-density.
        """
        if high_density and medium_density:
            raise ValueError("Can only be either high_density or medium_density")
        # MARK: Left-right gripper information for caging reward----------------
        left_pad = self.get_body_com("leftpad")
        right_pad = self.get_body_com("rightpad")

        # get current positions of left and right pads (Y axis)
        pad_y_lr = np.hstack((left_pad[1], right_pad[1]))
        # compare *current* pad positions with *current* obj position (Y axis)
        pad_to_obj_lr = np.abs(pad_y_lr - obj_pos[1])
        # compare *current* pad positions with *initial* obj position (Y axis)
        pad_to_objinit_lr = np.abs(pad_y_lr - self.obj_init_pos[1])

        # Compute the left/right caging rewards. This is crucial for success,
        # yet counterintuitive mathematically because we invented it
        # accidentally.
        #
        # Before touching the object, `pad_to_obj_lr` ("x") is always separated
        # from `caging_lr_margin` ("the margin") by some small number,
        # `pad_success_thresh`.
        #
        # When far away from the object:
        #       x = margin + pad_success_thresh
        #       --> Thus x is outside the margin, yielding very small reward.
        #           Here, any variation in the reward is due to the fact that
        #           the margin itself is shifting.
        # When near the object (within pad_success_thresh):
        #       x = pad_success_thresh - margin
        #       --> Thus x is well within the margin. As long as x > obj_radius,
        #           it will also be within the bounds, yielding maximum reward.
        #           Here, any variation in the reward is due to the gripper
        #           moving *too close* to the object (i.e, blowing past the
        #           obj_radius bound).
        #
        # Therefore, before touching the object, this is very nearly a binary
        # reward -- if the gripper is between obj_radius and pad_success_thresh,
        # it gets maximum reward. Otherwise, the reward very quickly falls off.
        #
        # After grasping the object and moving it away from initial position,
        # x remains (mostly) constant while the margin grows considerably. This
        # penalizes the agent if it moves *back* toward `obj_init_pos`, but
        # offers no encouragement for leaving that position in the first place.
        # That part is left to the reward functions of individual environments.
        caging_lr_margin = np.abs(pad_to_objinit_lr - pad_success_thresh)
        caging_lr = [
            reward_utils.tolerance(
                pad_to_obj_lr[i],  # "x" in the description above
                bounds=(obj_radius, pad_success_thresh),
                margin=caging_lr_margin[i],  # "margin" in the description above
                sigmoid="long_tail",
            )
            for i in range(2)
        ]
        caging_y = reward_utils.hamacher_product(*caging_lr)

        # MARK: X-Z gripper information for caging reward-----------------------
        tcp = self.tcp_center
        xz = [0, 2]

        # Compared to the caging_y reward, caging_xz is simple. The margin is
        # constant (something in the 0.3 to 0.5 range) and x shrinks as the
        # gripper moves towards the object. After picking up the object, the
        # reward is maximized and changes very little
        caging_xz_margin = np.linalg.norm(self.obj_init_pos[xz] - self.init_tcp[xz])
        caging_xz_margin -= xz_thresh
        caging_xz = reward_utils.tolerance(
            np.linalg.norm(tcp[xz] - obj_pos[xz]),  # "x" in the description above
            bounds=(0, xz_thresh),
            margin=caging_xz_margin,  # "margin" in the description above
            sigmoid="long_tail",
        )

        # MARK: Closed-extent gripper information for caging reward-------------
        gripper_closed = (
            min(max(0, action[-1]), desired_gripper_effort) / desired_gripper_effort
        )

        # MARK: Combine components----------------------------------------------
        caging = reward_utils.hamacher_product(caging_y, caging_xz)
        gripping = gripper_closed if caging > 0.97 else 0.0
        caging_and_gripping = reward_utils.hamacher_product(caging, gripping)

        if high_density:
            caging_and_gripping = (caging_and_gripping + caging) / 2
        if medium_density:
            tcp = self.tcp_center
            tcp_to_obj = np.linalg.norm(obj_pos - tcp)
            tcp_to_obj_init = np.linalg.norm(self.obj_init_pos - self.init_tcp)
            # Compute reach reward
            # - We subtract `object_reach_radius` from the margin so that the
            #   reward always starts with a value of 0.1
            reach_margin = abs(tcp_to_obj_init - object_reach_radius)
            reach = reward_utils.tolerance(
                tcp_to_obj,
                bounds=(0, object_reach_radius),
                margin=reach_margin,
                sigmoid="long_tail",
            )
            caging_and_gripping = (caging_and_gripping + reach) / 2

        return caging_and_gripping
