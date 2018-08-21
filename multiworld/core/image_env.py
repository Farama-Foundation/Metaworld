import cv2
import mujoco_py
import numpy as np
import warnings
from PIL import Image
from gym.spaces import Box, Dict

from multiworld.core.wrapper_env import ProxyEnv
from multiworld.envs.env_util import concatenate_box_spaces


class ImageEnv(ProxyEnv):
    def __init__(
            self,
            wrapped_env,
            imsize=84,
            init_camera=None,
            transpose=False,
            grayscale=False,
            normalize=False,
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
        self.channels = 1 if grayscale else 3
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

        self.return_image_proprio = False
        if 'proprio_observation' in spaces.keys():
            self.return_image_proprio = True
            spaces['image_proprio_observation'] = concatenate_box_spaces(
                spaces['image_observation'],
                spaces['proprio_observation']
            )
            spaces['image_proprio_desired_goal'] = concatenate_box_spaces(
                spaces['image_desired_goal'],
                spaces['proprio_desired_goal']
            )
            spaces['image_proprio_achieved_goal'] = concatenate_box_spaces(
                spaces['image_achieved_goal'],
                spaces['proprio_achieved_goal']
            )

        self.observation_space = Dict(spaces)

    def step(self, action, **kwargs):
        obs, reward, done, info = self.wrapped_env.step(action, **kwargs)
        new_obs = self._update_obs(obs)
        return new_obs, reward, done, info

    def reset(self):
        obs = self.wrapped_env.reset()
        env_state = self.wrapped_env.get_env_state()
        self.wrapped_env.set_to_goal(self.wrapped_env.get_goal(), set_goal=True)
        goal = self.wrapped_env._state_goal.copy()
        self._img_goal = self._get_flat_img()
        self.wrapped_env.set_env_state(env_state)
        self.wrapped_env._state_goal = goal
        return self._update_obs(obs)

    def _update_obs(self, obs):
        img_obs = self._get_flat_img()
        obs['image_observation'] = img_obs
        obs['image_desired_goal'] = self._img_goal
        obs['image_achieved_goal'] = img_obs
        obs['observation'] = img_obs
        obs['desired_goal'] = self._img_goal
        obs['achieved_goal'] = img_obs

        if self.return_image_proprio:
            obs['image_proprio_observation'] = np.concatenate(
                (obs['image_observation'], obs['proprio_observation'])
            )
            obs['image_proprio_desired_goal'] = np.concatenate(
                (obs['image_desired_goal'], obs['proprio_desired_goal'])
            )
            obs['image_proprio_achieved_goal'] = np.concatenate(
                (obs['image_achieved_goal'], obs['proprio_achieved_goal'])
            )

        return obs

    def _get_flat_img(self):
        # returns the image as a torch format np array
        image_obs = self._wrapped_env.get_image(width=self.imsize, height=self.imsize)
        # image_obs = self._wrapped_env.get_image(width=200, height=200)
        # start_x = 60
        # start_y = 45
        # image_obs = image_obs[start_x:start_x + self.imsize, start_y:start_y + self.imsize, :]
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

    def compute_rewards(self, achieved_goals, desired_goals, info):
        return - np.linalg.norm(achieved_goals - desired_goals, axis=1)

def normalize_image(image):
    assert image.dtype == np.uint8
    return np.float64(image) / 255.0

def unormalize_image(image):
    assert image.dtype != np.uint8
    return np.uint8(image * 255.0)
