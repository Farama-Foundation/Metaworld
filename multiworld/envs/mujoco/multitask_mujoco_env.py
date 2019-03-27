import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
from gym.spaces import Box
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj_6dof import SawyerTwoObject6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place_6dof import SawyerPickAndPlace6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door_6dof import SawyerDoor6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach_6dof import SawyerReach6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_stack_6dof import SawyerStack6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place_wsg_6dof import SawyerPickAndPlaceWsg6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_rope_6dof import SawyerRope6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_assembly_peg_6dof import SawyerNutAssembly6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_bin_picking_6dof import SawyerBinPicking6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_drawer_open_6dof import SawyerDrawerOpen6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_drawer_close_6dof import SawyerDrawerClose6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_laptop_close_6dof import SawyerLaptopClose6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_box_open_6dof import SawyerBoxOpen6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_box_close_6dof import SawyerBoxClose6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_stick_push_6dof import SawyerStickPush6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_stick_pull_6dof import SawyerStickPull6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_hammer_6dof import SawyerHammer6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_button_press_6dof import SawyerButtonPress6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_button_press_topdown_6dof import SawyerButtonPressTopdown6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_peg_insertion_side_6dof import SawyerPegInsertionSide6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_shelf_place_6dof import SawyerShelfPlace6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place_6dof import SawyerReachPushPickPlace6DOFEnv

try:
	import mujoco_py
except ImportError as e:
	raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

DEFAULT_SIZE = 500
ENV_LIST = [SawyerReachPushPickPlace6DOFEnv, SawyerShelfPlace6DOFEnv, SawyerDrawerOpen6DOFEnv, SawyerDrawerClose6DOFEnv, SawyerButtonPress6DOFEnv,
			SawyerButtonPressTopdown6DOFEnv, SawyerPegInsertionSide6DOFEnv]

class MultiTaskMujocoEnv(gym.Env):
	"""
	An multitask mujoco environment that contains a list of mujoco environments.
	"""
	def __init__(self):
		self.mujoco_envs = []
		for i, env in enumerate(ENV_LIST):
			self.mujoco_envs.append(env(multitask=True, multitask_num=len(ENV_LIST)+2))
			# set the one-hot task representation
			if i != 0:
				self.mujoco_envs[i]._state_goal_idx = np.zeros((len(ENV_LIST)+2))
				self.mujoco_envs[i]._state_goal_idx[i+2] = 1.
		# TODO: make sure all observation spaces across tasks are the same / use self-attention
		self.num_resets = 0
		self.task_idx = max((self.num_resets % (len(ENV_LIST)+2)) - 2, 0)
		self.action_space = self.mujoco_envs[self.task_idx].action_space
		self.observation_space = self.mujoco_envs[self.task_idx].observation_space
		self.goal_space = Box(np.zeros(len(ENV_LIST)+2), np.ones(len(ENV_LIST)+2))
		self.reset()

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):
		return self.mujoco_envs[self.task_idx].step(action)

	def reset(self):
		# self.task_idx = np.random.choice([0, 0] + list(range(len(self.mujoco_envs))))
		self.task_idx = max((self.num_resets % (len(ENV_LIST)+2)) - 2, 0)
		self.num_resets += 1
		return self.mujoco_envs[self.task_idx].reset()

	@property
	def dt(self):
		return self.mujoco_envs[self.task_idx].model.opt.timestep * self.mujoco_envs[self.task_idx].frame_skip

	def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE, depth=False):
		return self.mujoco_envs[self.task_idx].render(mode=mode, width=width, height=height, depth=depth)

	def close(self):
		self.mujoco_envs[self.task_idx].close()
