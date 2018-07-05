import cv2
import mujoco_py
import numpy as np
import warnings
from PIL import Image
from gym.spaces import Box, Dict

from multiworld.core.wrapper_env import ProxyEnv
from multiworld.envs.mujoco.sawyer_reach_torque.generate_goal_data_set import generate_goal_data_set


class ImageEnv(ProxyEnv):
    def __init__(
            self,
            wrapped_env,
            imsize=84,
            init_camera=None,
            transpose=False,
            grayscale=False,
            normalize=False,
            use_goal_caching=False,
            cached_goal_generation_function=generate_goal_data_set,
            num_cached_goals=100,
            cached_goal_keys=None,
            goal_sizes=None,
            obs_to_goal_fctns=None,
            observation_keys=None,
            use_cached_dataset=False,
            reward_type='image_distance',
            threshold=10,
    ):
        self.quick_init(locals())
        super().__init__(wrapped_env)
        self.wrapped_env.hide_goal_markers = True
        self.imsize = imsize
        self.init_camera = init_camera
        self.transpose = transpose
        self.grayscale = grayscale
        self.normalize = normalize

        if grayscale:
            self.image_length = self.imsize * self.imsize
        else:
            self.image_length = 3 * self.imsize * self.imsize
        # This is torch format rather than PIL image
        self.image_shape = (self.imsize, self.imsize)
        # Flattened past image queue
        # init camera
        if init_camera is not None:
            sim = self._wrapped_env.initialize_camera(init_camera)
            # viewer = mujoco_py.MjRenderContextOffscreen(sim, device_id=-1)
            # init_camera(viewer.cam)
            # sim.add_render_context(viewer)
        self._render_local = False
        self._img_goal = None

        img_space = Box(0, 1, (self.image_length,))
        spaces = self.wrapped_env.observation_space.spaces
        spaces['observation'] = img_space
        spaces['desired_goal'] = img_space
        spaces['achieved_goal'] = img_space
        spaces['image_observation'] = img_space
        spaces['image_desired_goal'] = img_space
        spaces['image_achieved_goal'] = img_space
        self.observation_space = Dict(spaces)
        self.reward_type=reward_type
        self.threshold = threshold
        self.use_goal_caching = use_goal_caching
        if self.use_goal_caching:
            self._img_goal = np.random.uniform(0, 1, self.image_length)
            # hardcoded for torque control for now
            cached_goal_keys = ['image_desired_goal', 'state_desired_goal', 'joint_desired_goal']
            goal_sizes = [(self.imsize ** 2) * 3, 3, 7]
            obs_to_goal_fctns = [lambda x: x, lambda x: x[-3:], lambda x: x[:7]]
            observation_keys = ['image_observation', 'state_observation', 'state_observation']
            goal_generation_dict = dict()
            for goal_key, goal_size, obs_to_goal_fctn, obs_key in zip(cached_goal_keys, goal_sizes, obs_to_goal_fctns,
                                                                      observation_keys):
                goal_generation_dict[goal_key] = [goal_size, obs_to_goal_fctn, obs_key]
            self.goals = cached_goal_generation_function(self, goal_generation_dict=goal_generation_dict, num_goals=num_cached_goals, use_cached_dataset=use_cached_dataset)
            self.goals['desired_goal'] = self.goals['image_desired_goal']
            self._wrapped_env.goals = self.goals
            self._wrapped_env.use_goal_caching = True

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = self._update_obs(obs)
        reward = self.compute_reward(action, new_obs)
        return new_obs, reward, done, info

    def reset(self, _resample_on_reset=True):
        obs = self.wrapped_env.reset(_resample_on_reset=_resample_on_reset)
        if _resample_on_reset:
            if self.use_goal_caching:
                idx = np.random.randint(0, self.num_cached_goals)
                self._img_goal = self.goals['image_desired_goal'][idx]
                self._wrapped_env._state_goal = self.goals['state_desired_goal'][idx]
                self._wrapped_env._goal_angles = self.goals['joint_desired_goal'][idx]
                for key in self.goals.keys():
                    obs[key] = self.goals[key][idx]
            else:
                env_state = self.wrapped_env.get_env_state()
                self.wrapped_env.set_to_goal(self.wrapped_env.get_goal())
                self._img_goal = self._get_flat_img()
                self.wrapped_env.set_env_state(env_state)
            return self._update_obs(obs)

    def _update_obs(self, obs):
        img_obs = self._get_flat_img()
        obs['image_observation'] = img_obs
        obs['image_desired_goal'] = self._img_goal
        obs['image_achieved_goal'] = img_obs
        obs['observation'] = img_obs
        obs['desired_goal'] = self._img_goal
        obs['achieved_goal'] = img_obs
        return obs

    def _get_flat_img(self):
        # returns the image as a torch format np array
        image_obs = self._wrapped_env.get_image()
        if self._render_local:
            cv2.imshow('env', image_obs)
            cv2.waitKey(1)
        if self.grayscale:
            image_obs = Image.fromarray(image_obs).convert('L')
            image_obs = np.array(image_obs)
        if self.normalize:
            image_obs = image_obs / 255.0
        if self.transpose:
            image_obs = image_obs.transpose()
        return image_obs.flatten()

    def enable_render(self):
        self._render_local = True

    """
    Multitask functions
    """
    def get_goal(self):
        goal = self.wrapped_env.get_goal()
        goal['desired_goal'] = self._img_goal
        goal['image_desired_goal'] = self._img_goal
        return goal

    def sample_goals(self, batch_size):
        if self.use_goal_caching:
            idxs = np.random.randint(0, self.num_cached_goals, batch_size)
            goals = dict()
            for key in self.goals.keys():
                goals[key] = self.goals[key][idxs]
            return goals
        if batch_size > 1:
            warnings.warn("Sampling goal images is slow")
        img_goals = np.zeros((batch_size, self.image_length))
        goals = self.wrapped_env.sample_goals(batch_size)
        for i in range(batch_size):
            goal = self.unbatchify_dict(goals, i)
            self.wrapped_env.set_to_goal(goal)
            img_goals[i, :] = self._get_flat_img()
        goals['desired_goal'] = img_goals
        goals['image_desired_goal'] = img_goals
        return goals

    def compute_rewards(self, actions, obs):
        achieved_goals = obs['achieved_goal']
        desired_goals = obs['desired_goal']
        dist = np.linalg.norm(achieved_goals - desired_goals, axis=1)
        if self.reward_type=='image_distance':
            return -dist
        elif self.reward_type=='image_sparse':
            return -(dist<self.threshold).astype(float)
        else:
            raise NotImplementedError()

def normalize_image(image):
    assert image.dtype == np.uint8
    return np.float64(image) / 255.0

def unormalize_image(image):
    assert image.dtype != np.uint8
    return np.uint8(image * 255.0)
