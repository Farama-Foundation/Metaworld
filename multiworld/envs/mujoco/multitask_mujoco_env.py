import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
from collections import deque
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
# ENV_LIST = [SawyerReachPushPickPlace6DOFEnv, SawyerShelfPlace6DOFEnv, SawyerDrawerOpen6DOFEnv, SawyerDrawerClose6DOFEnv, SawyerButtonPress6DOFEnv,
# 			SawyerButtonPressTopdown6DOFEnv, SawyerPegInsertionSide6DOFEnv]
# ENV_LIST = [SawyerReachPushPickPlace6DOFEnv, SawyerReachPushPickPlace6DOFEnv, SawyerReachPushPickPlace6DOFEnv, SawyerShelfPlace6DOFEnv, SawyerDrawerOpen6DOFEnv,
# 			SawyerDrawerClose6DOFEnv, SawyerButtonPress6DOFEnv, SawyerButtonPressTopdown6DOFEnv, SawyerPegInsertionSide6DOFEnv]
# ENV_LIST = [SawyerReachPushPickPlace6DOFEnv, SawyerReachPushPickPlace6DOFEnv, SawyerDrawerOpen6DOFEnv, SawyerDrawerClose6DOFEnv, SawyerButtonPress6DOFEnv, SawyerButtonPressTopdown6DOFEnv]
# ENV_LIST = [SawyerReachPushPickPlace6DOFEnv, SawyerReachPushPickPlace6DOFEnv, SawyerDrawerClose6DOFEnv, SawyerButtonPress6DOFEnv, SawyerButtonPressTopdown6DOFEnv]
ENV_LIST = [SawyerReachPushPickPlace6DOFEnv, SawyerReachPushPickPlace6DOFEnv, SawyerReachPushPickPlace6DOFEnv]
# ENV_LIST = [SawyerReachPushPickPlace6DOFEnv, SawyerReachPushPickPlace6DOFEnv, SawyerReachPushPickPlace6DOFEnv, SawyerShelfPlace6DOFEnv,
# 			SawyerDrawerClose6DOFEnv, SawyerButtonPress6DOFEnv, SawyerButtonPressTopdown6DOFEnv, SawyerPegInsertionSide6DOFEnv]

class MultiTaskMujocoEnv(gym.Env):
	"""
	An multitask mujoco environment that contains a list of mujoco environments.
	"""
	def __init__(self,
				if_render=True,
				adaptive_sampling=False):
		self.mujoco_envs = []
		self.adaptive_sampling = adaptive_sampling
		if adaptive_sampling:
			self.scores = {i:deque(maxlen=10) for i in range(len(ENV_LIST))}
			self.current_score = 0.
			self.sample_probs = np.array([1./(len(ENV_LIST)) for _ in range(len(ENV_LIST))])
			self.target_scores = []
			self.sample_tau = 0.05
		for i, env in enumerate(ENV_LIST):
			if i < 3:
				self.mujoco_envs.append(env(multitask=True, multitask_num=len(ENV_LIST), random_init=False, if_render=if_render, fix_task=True, task_idx=i))
				# self.mujoco_envs.append(env(multitask=True, multitask_num=len(ENV_LIST), random_init=False, if_render=if_render, fix_task=True, task_idx=2))
			else:
				self.mujoco_envs.append(env(multitask=True, multitask_num=len(ENV_LIST), random_init=False, if_render=if_render))
			# if i < 2:#3:
			# 	self.mujoco_envs.append(env(multitask=True, multitask_num=len(ENV_LIST), random_init=False, if_render=if_render, fix_task=True, task_idx=i+1))
			# else:
			# 	self.mujoco_envs.append(env(multitask=True, multitask_num=len(ENV_LIST), random_init=False, if_render=if_render))
			# set the one-hot task representation
			self.mujoco_envs[i]._state_goal_idx = np.zeros((len(ENV_LIST)))
			self.mujoco_envs[i]._state_goal_idx[i] = 1.
			if adaptive_sampling:
				self.target_scores.append(self.mujoco_envs[i].target_reward)
		# TODO: make sure all observation spaces across tasks are the same / use self-attention
		self.num_resets = 0
		self.task_idx = self.num_resets % len(ENV_LIST)
		# only sample from pushing task
		# self.task_idx = 2
		self.action_space = self.mujoco_envs[self.task_idx].action_space
		self.observation_space = self.mujoco_envs[self.task_idx].observation_space
		self.goal_space = Box(np.zeros(len(ENV_LIST)), np.ones(len(ENV_LIST)))
		self.reset()
		self.num_resets -= 1

	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]

	def step(self, action):
		ob, reward, done, info = self.mujoco_envs[self.task_idx].step(action)
		# TODO: need to figure out how to calculate target scores in this case!
		if self.adaptive_sampling:
			self.current_score += reward
			if done:
				self.scores[self.task_idx].append(reward)
				# self.scores[self.task_idx].append(self.current_score)
		assert ob[-len(ENV_LIST):].argmax() == self.task_idx
		return ob, reward, done, info

	def reset(self):
		# self.task_idx = max((self.num_resets % (len(ENV_LIST)+2)) - 2, 0)
		if self.num_resets < len(ENV_LIST)*2 or not self.adaptive_sampling:
			self.task_idx = self.num_resets % len(ENV_LIST)
		else:
			if self.num_resets % (8*len(ENV_LIST)) == 0:
				avg_scores = [np.mean(self.scores[i]) for i in range(len(ENV_LIST))]
				self.sample_probs = np.exp((np.array(self.target_scores) - np.array(avg_scores))/np.array(self.target_scores)/self.sample_tau)
				self.sample_probs = self.sample_probs / self.sample_probs.sum()
				print('Sampling prob is', self.sample_probs)
			self.task_idx = np.random.choice(range(len(ENV_LIST)), p=self.sample_probs)
		# only sample from pushing tasks
		# self.task_idx = 2
		self.num_resets += 1
		self.current_score = 0.
		return self.mujoco_envs[self.task_idx].reset()

	@property
	def dt(self):
		return self.mujoco_envs[self.task_idx].model.opt.timestep * self.mujoco_envs[self.task_idx].frame_skip

	def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE, depth=False):
		return self.mujoco_envs[self.task_idx].render(mode=mode, width=width, height=height, depth=depth)

	def close(self):
		self.mujoco_envs[self.task_idx].close()
