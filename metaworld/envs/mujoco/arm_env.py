import copy
import pickle

import mujoco
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv as mjenv_gym
from gymnasium.spaces import Box, Discrete
from gymnasium.utils import seeding
from gymnasium.utils.ezpickle import EzPickle

from metaworld.envs import reward_utils
from metaworld.envs.mujoco.mujoco_env import _assert_task_is_set


class MocapBase(mjenv_gym):
    """Provides some commonly-shared functions for Arm Mujoco envs that use mocap for XYZ control."""

    mocap_low = np.array([-0.2, 0.5, 0.06])
    mocap_high = np.array([0.2, 0.7, 0.6])
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 40,
    }

    def __init__(self, model_name, frame_skip=5, render_mode=None):
        mjenv_gym.__init__(
            self,
            model_name,
            frame_skip=frame_skip,
            observation_space=self.arm_observation_space,
            render_mode=render_mode,
        )
        self.reset_mocap_welds()
        self.frame_skip = frame_skip

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
        return {"state": state, "mjb": self.model_name, "mocap": self.get_env_state()}

    def __setstate__(self, state):
        self.__dict__ = state["state"]
        mjenv_gym.__init__(
            self,
            state["mjb"],
            frame_skip=self.frame_skip,
            observation_space=self.sawyer_observation_space,
        )
        self.set_env_state(state["mocap"])

    def reset_mocap_welds(self):
        """Resets the mocap welds that we use for actuation."""
        if self.model.nmocap > 0 and self.model.eq_data is not None:
            for i in range(self.model.eq_data.shape[0]):
                if self.model.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
                    self.model.eq_data[i] = np.array(
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                    )


class ArmEnv(MocapBase, EzPickle):
    _QPOS_SPACE = None
    _HAND_SPACE = None

    max_path_length = 250

    TARGET_RADIUS = 0.05

    current_task = 0
    classes = None
    classes_kwargs = None
    tasks = None

    def __init__(
        self,
        model_name,
        frame_skip=10,
        hand_low=(-0.2, 0.55, 0.05),
        hand_high=(0.2, 0.75, 0.3),
        mocap_low=None,
        mocap_high=None,
        render_mode=None,
    ):
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

        super().__init__(model_name, frame_skip=frame_skip, render_mode=render_mode)

        mujoco.mj_forward(
            self.model, self.data
        )  # *** DO NOT REMOVE: EZPICKLE WON'T WORK *** #

        self._did_see_sim_exception = False

        # Technically these observation lengths are different between v1 and v2,
        # but we handle that elsewhere and just stick with v2 numbers here
        self._obs_obj_max_len = 14

        self._set_task_called = False

        self.action_cost_coff = 0.01
        self.init_left_pad = None  # OVERRIDE ME
        self.init_right_pad = None  # OVERRIDE ME
        self.hand_init_qpos = None  # OVERRIDE ME
        self._target_pos = None  # OVERRIDE ME
        self._random_reset_space = None  # OVERRIDE ME
        self.action_space = None  # OVERRIDE ME
        self.arm_col = None  # OVERRIDE ME

        self.env_col = [
            "table_col",
            "wall_col_1",
            "wall_col_2",
            "wall_col_3",
            "wall_col_4",
        ]

        self._last_stable_obs = None

        EzPickle.__init__(
            self,
            model_name,
            frame_skip,
            hand_low,
            hand_high,
            mocap_low,
            mocap_high,
        )

    def seed(self, seed):
        assert seed is not None
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.goal_space.seed(seed)
        return [seed]

    @staticmethod
    def _set_task_inner():
        # Doesn't absorb "extra" kwargs, to ensure nothing's missed.
        pass

    def set_task(self, task):
        self._set_task_called = True
        data = pickle.loads(task.data)
        assert isinstance(self, data["env_cls"])
        del data["env_cls"]
        self._last_rand_vec = data["rand_vec"]
        self._freeze_rand_vec = True
        self._last_rand_vec = data["rand_vec"]
        del data["rand_vec"]
        self._partially_observable = data["partially_observable"]
        del data["partially_observable"]
        self._set_task_inner(**data)

    def discretize_goal_space(self, goals):
        assert False
        assert len(goals) >= 1
        self.discrete_goals = goals
        # update the goal_space to a Discrete space
        self.discrete_goal_space = Discrete(len(self.discrete_goals))

    def _set_obj_xyz(self, pos):
        arm_nqpos = self._QPOS_SPACE.low.size
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        # freejoint qpos: x, y, z qvel: vx, vy, vz, ax, ay, az
        qpos[arm_nqpos : arm_nqpos + 3] = pos.copy()
        qvel[arm_nqpos : arm_nqpos + 6] = 0
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

        raise NotImplementedError

    @property
    def _get_id_main_object(self):
        return self.data.geom(
            "objGeom"
        ).id  # [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'objGeom')]

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

    @property
    def tcp_center(self):
        """The COM of the gripper.

        Returns:
            (np.ndarray): 3-element position
        """
        raise NotImplementedError

    @property
    def gripper_distance_apart(self):
        raise NotImplementedError

    def _get_curr_obs_combined_no_goal(self):
        """Combines the end effector's {pos, closed amount} and the object(s)' {pos, quat} into a single flat observation.

        Note: The goal's position is *not* included in this.

        Returns:
            np.ndarray: The flat observation array (18 elements)

        """

        # pos_hand = self.get_endeff_pos()
        qpos = self.data.qpos.flat.copy()[: self._QPOS_SPACE.low.size]

        # gripper_distance_apart = self.gripper_distance_apart

        obs_obj_padded = np.zeros(self._obs_obj_max_len)
        obj_pos = self._get_pos_objects()
        assert len(obj_pos) % 3 == 0
        obj_pos_split = np.split(obj_pos, len(obj_pos) // 3)

        obj_quat = self._get_quat_objects()
        assert len(obj_quat) % 4 == 0
        obj_quat_split = np.split(obj_quat, len(obj_quat) // 4)
        obs_obj_padded[: len(obj_pos) + len(obj_quat)] = np.hstack(
            [np.hstack((pos, quat)) for pos, quat in zip(obj_pos_split, obj_quat_split)]
        )
        return np.hstack((qpos, obs_obj_padded))
        # return np.hstack((pos_hand, qpos, gripper_distance_apart, obs_obj_padded))
        # 7 + _QPOS_SPACE.low.size + 1 + _obs_obj_max_len

    @property
    def joint_pos(self):
        return np.array(
            [self.data.qpos[x] for x in range(self._QPOS_SPACE.low.size)],
        )

    @property
    def joint_vel(self):
        return np.array([self.data.qvel[x] for x in range(self._QPOS_SPACE.low.size)])

    @property
    def endeff_pos(self):
        return self.data.body("hand").xpos

    @property
    def endeff_quat(self):
        return self.data.body("hand").xquat

    @property
    def endeff_lin_vel(self):
        return self.data.body("hand").subtree_linvel

    @property
    def endeff_ang_mom(self):
        return self.data.body("hand").subtree_angmom

    @property
    def obs_obj_padded(self):
        obs_obj_padded = np.zeros(self._obs_obj_max_len)
        obj_pos = self._get_pos_objects()
        assert len(obj_pos) % 3 == 0
        obj_pos_split = np.split(obj_pos, len(obj_pos) // 3)

        obj_quat = self._get_quat_objects()
        assert len(obj_quat) % 4 == 0
        obj_quat_split = np.split(obj_quat, len(obj_quat) // 4)
        obs_obj_padded[: len(obj_pos) + len(obj_quat)] = np.hstack(
            [np.hstack((pos, quat)) for pos, quat in zip(obj_pos_split, obj_quat_split)]
        )
        return obs_obj_padded

    def _get_obs(self):
        qpos = self.joint_pos
        qpos_cos = np.cos(qpos)
        qpos_sin = np.sin(qpos)
        qvel = self.joint_vel

        endeff_pos = self.endeff_pos
        endeff_quat = self.endeff_quat

        obs_obj_padded = self.obs_obj_padded

        pos_goal = self._get_pos_goal()
        if self._partially_observable:
            pos_goal = np.zeros_like(pos_goal)

        return np.hstack(
            (
                qpos_cos,  # nq
                qpos_sin,  # nq
                qvel,  # nq
                endeff_pos,  # 3
                endeff_quat,  # 4
                obs_obj_padded,  # 14
                pos_goal,  # 3
            )  # 3 * nq + 24
        ), np.hstack(
            (endeff_pos, 0, obs_obj_padded)
        )  # for metaworld

    def _get_obs_dict(self):
        obs = self._get_obs()
        return dict(
            state_observation=obs,
            state_desired_goal=self._get_pos_goal(),
            state_achieved_goal=None,
        )

    @property
    def arm_observation_space(self):
        """Returns the observation space for the arm.

        Returns:
            gym.spaces.Box: The observation space for the arm
        """
        obs_obj_max_len = 14
        obj_low = np.full(obs_obj_max_len, -np.inf, dtype=np.float64)
        obj_high = np.full(obs_obj_max_len, +np.inf, dtype=np.float64)
        goal_low = np.zeros(3) if self._partially_observable else self.goal_space.low
        goal_high = np.zeros(3) if self._partially_observable else self.goal_space.high
        sin_low = np.full(self._QPOS_SPACE.low.size, -1.0, dtype=np.float64)
        sin_high = np.full(self._QPOS_SPACE.high.size, +1.0, dtype=np.float64)
        qvel_low = np.full(self._QPOS_SPACE.low.size, -np.inf, dtype=np.float64)
        qvel_high = np.full(self._QPOS_SPACE.high.size, +np.inf, dtype=np.float64)
        pos_low = np.full(3, -np.inf, dtype=np.float64)
        pos_high = np.full(3, +np.inf, dtype=np.float64)
        quat_low = np.full(4, -np.inf, dtype=np.float64)
        quat_high = np.full(4, +np.inf, dtype=np.float64)
        return Box(
            np.hstack(
                (
                    sin_low,
                    sin_low,
                    qvel_low,
                    pos_low,
                    quat_low,
                    obj_low,
                    goal_low,
                )
            ),
            np.hstack(
                (
                    sin_high,
                    sin_high,
                    qvel_high,
                    pos_high,
                    quat_high,
                    obj_high,
                    goal_high,
                )
            ),
            dtype=np.float64,
        )

    def set_action(self, action):
        """Applies the given action to the simulation.

        Args:
            action (np.ndarray): The action to apply
        """
        raise NotImplementedError

    def gripper_effort_from_action(self, action):
        raise NotImplementedError

    # def parse_obs(self, obs: np.ndarray):
    #     """Parses the observation into a format of Metaworld.

    #     Args:
    #         obs (np.ndarray): The observation to parse

    #     Returns:
    #         np.ndarray: The parsed observation
    #     """

    #     pos_hand = self.endeff_pos[:3]
    #     obs_obj_padded = self.obs_obj_padded
    #     ic(pos_hand, obs_obj_padded)

    #     return np.hstack((pos_hand, obs_obj_padded))

    def parse_action(self, action: np.ndarray):
        """Parses the action into a format of Metaworld.

        Args:
            action (np.ndarray): The action to parse

        Returns:
            np.ndarray: The parsed action
        """

        return np.hstack((np.zeros(3), self.gripper_effort_from_action(action)))

    def check_contact_table(self):
        assert self.arm_col is not None

        ncon = self.data.ncon
        contact = False
        for coni in range(ncon):
            con = self.data.contact[coni]
            geom1 = self.model.geom(con.geom1).name
            geom2 = self.model.geom(con.geom2).name

            if (geom1 in self.env_col and geom2 in self.arm_col) or (
                geom2 in self.env_col and geom1 in self.arm_col
            ):
                contact = True
                break
        return contact

    def get_action_penalty(self, action):
        raise NotImplementedError

    @_assert_task_is_set
    def step(self, action):
        action_dim = self.action_space.low.size
        assert (
            len(action) == action_dim
        ), f"Actions should be size {action_dim}, got {len(action)}"
        if self.curr_path_length >= self.max_path_length:
            raise ValueError("You must reset the env manually once truncate==True")

        action = np.clip(
            action,
            a_max=self.action_space.high,
            a_min=self.action_space.low,
            dtype=np.float64,
        )
        self.set_action(action)

        self.curr_path_length += 1

        # Running the simulator can sometimes mess up site positions, so
        # re-position them here to make sure they're accurate
        for site in self._target_site_config:
            self._set_pos_site(*site)

        if self._did_see_sim_exception:
            return (
                self._last_stable_obs,  # observation just before going unstable
                0.0,  # reward (penalize for causing instability)
                False,
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

        self._last_stable_obs, parsed_obs = self._get_obs()

        self._last_stable_obs = np.clip(
            self._last_stable_obs,
            a_max=self.arm_observation_space.high,
            a_min=self.arm_observation_space.low,
            dtype=np.float64,
        )

        # reward, info = self.evaluate_state(self._last_stable_obs, action)
        reward, info = self.evaluate_state(parsed_obs, self.parse_action(action))
        action_penalty = self.get_action_penalty(action)
        info["action_penalty"] = action_penalty
        reward -= action_penalty

        # step will never return a terminate==True if there is a success
        # but we can return truncate=True if the current path length == max path length
        truncate = False
        if self.curr_path_length == self.max_path_length:
            truncate = True
        return (
            np.array(self._last_stable_obs, dtype=np.float64),
            reward,
            False,
            truncate,
            info,
        )

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
        np.random.seed(seed)
        self.curr_path_length = 0
        _, info = super().reset()
        obs, _ = self._get_obs()
        obs = np.float64(obs)
        return obs, info

    def _reset_hand(self):
        assert self.hand_init_qpos is not None

        for i, qpos in enumerate(self.hand_init_qpos):
            self.data.qpos[i] = qpos
        mujoco.mj_forward(self.model, self.data)

        self.init_tcp = self.tcp_center

    def _get_state_rand_vec(self):
        if self._freeze_rand_vec:
            assert self._last_rand_vec is not None
            return self._last_rand_vec
        else:
            rand_vec = np.random.uniform(
                self._random_reset_space.low,
                self._random_reset_space.high,
                size=self._random_reset_space.low.size,
            ).astype(np.float64)
            self._last_rand_vec = rand_vec
            return rand_vec

    def _gripper_caging_reward(  # TODO
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
        raise NotImplementedError

    def render(self, mode=None):
        """Returns rendering as uint8 in range [0...255]"""

        if mode is None:
            mode = self.render_mode

        # Cam Params
        cam_config = {
            "azimuth": 40,
            "elevation": -35,
            "distance": 2,
            "lookat": [0, 0.6, 0],  # Tuple and list are both legal
        }

        if self.mujoco_renderer._viewers == {}:
            self.mujoco_renderer.default_cam_config = cam_config

        return self.mujoco_renderer.render(mode)
