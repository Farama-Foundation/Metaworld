import os

import glfw
from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym

try:
	import mujoco_py
except ImportError as e:
	raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 500

class MujocoEnv(gym.Env):
	"""
	This is a simplified version of the gym MujocoEnv class.

	Some differences are:
	 - Do not automatically set the observation/action space.
	"""

	max_path_length = 150

	def __init__(self, model_path, frame_skip, device_id=-1, automatically_set_spaces=False):
		if model_path.startswith("/"):
			fullpath = model_path
		else:
			fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
		if not path.exists(fullpath):
			raise IOError("File %s does not exist" % fullpath)
		self.frame_skip = frame_skip
		self.model = mujoco_py.load_model_from_path(fullpath)
		self.sim = mujoco_py.MjSim(self.model)
		self.data = self.sim.data
		self.viewer = None
		self._viewers = {}

		self.metadata = {
			'render.modes': ['human', 'rgb_array'],
			'video.frames_per_second': int(np.round(1.0 / self.dt))
		}
		if device_id == -1 and 'gpu_id' in os.environ:
			device_id =int(os.environ['gpu_id'])
		self.device_id = device_id
		self.init_qpos = self.sim.data.qpos.ravel().copy()
		self.init_qvel = self.sim.data.qvel.ravel().copy()
		if automatically_set_spaces:
			observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
			assert not done
			self.obs_dim = observation.size

			bounds = self.model.actuator_ctrlrange.copy()
			low = bounds[:, 0]
			high = bounds[:, 1]
			self.action_space = spaces.Box(low=low, high=high)

			high = np.inf*np.ones(self.obs_dim)
			low = -high
			self.observation_space = spaces.Box(low, high)

		self.seed()

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	# methods to override:
	# ----------------------------

	def reset_model(self):
		"""
		Reset the robot degrees of freedom (qpos and qvel).
		Implement this in each subclass.
		"""
		raise NotImplementedError

	def viewer_setup(self):
		"""
		This method is called when the viewer is initialized and after every reset
		Optionally implement this method, if you need to tinker with camera position
		and so forth.
		"""
		pass

	# -----------------------------

	def reset(self):
		self.sim.reset()
		ob = self.reset_model()
		if self.viewer is not None:
			self.viewer_setup()
		return ob

	def reset_to_idx(self, idx):
		self.sim.reset()
		ob = self.reset_model_to_idx(idx)
		if self.viewer is not None:
			self.viewer_setup()
		return ob

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
		if getattr(self, 'curr_path_length', 0) > self.max_path_length:
			raise ValueError('Maximum path length allowed by the benchmark has been exceeded')
		if n_frames is None:
			n_frames = self.frame_skip
		if self.sim.data.ctrl is not None and ctrl is not None:
			self.sim.data.ctrl[:] = ctrl
		for _ in range(n_frames):
			self.sim.step()

	def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE, depth=False):
		if 'rgb_array' in mode:
			self._get_viewer(mode).render(width, height)
			# window size used for old mujoco-py:
			data = self._get_viewer(mode).read_pixels(width, height, depth=depth)
			# original image is upside-down, so flip it
			if not depth:
				return data[::-1, :, :]
			else:
				return data[0][::-1, :, :], data[1][::-1, :]
		elif mode == 'human':
			self._get_viewer(mode).render()

	def close(self):
		if self.viewer is not None:
			glfw.destroy_window(self.viewer.window)
			self.viewer = None

	def _get_viewer(self, mode):
		self.viewer = self._viewers.get(mode)
		if self.viewer is None:
			if mode == 'human':
				self.viewer = mujoco_py.MjViewer(self.sim)
			elif 'rgb_array' in mode:
				self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, 0)
			self.viewer_setup()
			self._viewers[mode] = self.viewer
		# if mode == 'rgb_array_y':
		#     self.viewer_setup(view_angle='y')
		# else:
		#     self.viewer_setup(view_angle='x')
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
		# viewer = mujoco_py.MjViewer(sim)
		init_fctn(viewer.cam)
		sim.add_render_context(viewer)
