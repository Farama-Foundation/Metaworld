import random

import cv2
import mujoco_py
import numpy as np
import warnings
from PIL import Image
from gym.spaces import Box, Dict

from multiworld.core.multitask_env import MultitaskEnv
from multiworld.core.wrapper_env import ProxyEnv
from multiworld.envs.env_util import concatenate_box_spaces
from multiworld.envs.env_util import get_stat_in_paths, create_stats_ordered_dict

from gym.spaces import Box, Dict

class GymToMultiEnv(ProxyEnv): # MultitaskEnv):
    def __init__(
            self,
            wrapped_env,
    ):
        """Minimal env to convert a gym env to one with dict observations"""
        self.quick_init(locals())
        super().__init__(wrapped_env)

        obs_box = wrapped_env.observation_space
        self.observation_space = Dict([
            ('observation', obs_box),
            ('state_observation', obs_box),
        ])

    def step(self, action):
        obs, reward, done, info = self.wrapped_env.step(action)
        new_obs = dict(
            observation=obs,
            state_observation=obs,
        )
        return new_obs, reward, done, info

    def reset(self):
        obs = self.wrapped_env.reset()
        new_obs = dict(
            observation=obs,
            state_observation=obs,
        )
        return new_obs

    def _get_obs(self):
        raise NotImplementedError()

class MujocoGymToMultiEnv(GymToMultiEnv):
    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel,
                                         old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames=None):
        if n_frames is None:
            n_frames = self.frame_skip
        if self.sim.data.ctrl is not None and ctrl is not None:
            self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def render(self, mode='human'):
        if mode == 'rgb_array':
            self._get_viewer().render()
            # window size used for old mujoco-py:
            width, height = 500, 500
            width, height = 4000, 4000
            data = self._get_viewer().read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().render()

    def close(self):
        if self.viewer is not None:
            self.viewer.finish()
            self.viewer = None

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer_setup()
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])

    def get_image(self, width=84, height=84, camera_name=None):
        return self.sim.render(
            width=width,
            height=height,
            camera_name=camera_name,
        )

    def initialize_camera(self, init_fctn):
        sim = self.sim
        viewer = mujoco_py.MjRenderContextOffscreen(sim, device_id=self.device_id)
        init_fctn(viewer.cam)
        sim.add_render_context(viewer)

    def get_diagnostics(self, paths, **kwargs):
        return {}
