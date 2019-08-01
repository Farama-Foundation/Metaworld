import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
from collections import deque
import gym
from gym.spaces import Box
from metaworld.envs.mujoco.sawyer_xyz.sawyer_assembly_peg_6dof import SawyerNutAssembly6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_bin_picking_6dof import SawyerBinPicking6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_drawer_open_6dof import SawyerDrawerOpen6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_drawer_close_6dof import SawyerDrawerClose6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_box_close_6dof import SawyerBoxClose6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_stick_push_6dof import SawyerStickPush6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_stick_pull_6dof import SawyerStickPull6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_hammer_6dof import SawyerHammer6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_button_press_6dof import SawyerButtonPress6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_button_press_topdown_6dof import SawyerButtonPressTopdown6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_peg_insertion_side_6dof import SawyerPegInsertionSide6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_shelf_place_6dof import SawyerShelfPlace6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place_6dof import SawyerReachPushPickPlace6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_door_6dof import SawyerDoor6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_door_close import SawyerDoorClose6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_window_open_6dof import SawyerWindowOpen6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_window_close_6dof import SawyerWindowClose6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_sweep import SawyerSweep6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_sweep_into_goal import SawyerSweepIntoGoal6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_hand_insert import SawyerHandInsert6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_lever_pull import SawyerLeverPull6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_dial_turn_6dof import SawyerDialTurn6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_coffee_button_6dof import SawyerCoffeeButton6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_coffee_push_6dof import SawyerCoffeePush6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_coffee_pull_6dof import SawyerCoffeePull6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_faucet_open import SawyerFaucetOpen6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_faucet_close import SawyerFaucetClose6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_peg_unplug_side_6dof import SawyerPegUnplugSide6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_soccer import SawyerSoccer6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_basketball import SawyerBasketball6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place_wall_6dof import SawyerReachPushPickPlaceWall6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_push_back_6dof import SawyerPushBack6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_pick_out_of_hole import SawyerPickOutOfHole6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_shelf_remove_6dof import SawyerShelfRemove6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_disassemble_peg_6dof import SawyerNutDisassemble6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_door_lock import SawyerDoorLock6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_door_unlock import SawyerDoorUnlock6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_sweep_tool import SawyerSweepTool6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_button_press_wall_6dof import SawyerButtonPressWall6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_button_press_topdown_wall_6dof import SawyerButtonPressTopdownWall6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_handle_press_6dof import SawyerHandlePress6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_handle_pull_6dof import SawyerHandlePull6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_handle_press_side_6dof import SawyerHandlePressSide6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_handle_pull_side_6dof import SawyerHandlePullSide6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_plate_slide_6dof import SawyerPlateSlide6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_plate_slide_back_6dof import SawyerPlateSlideBack6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_plate_slide_side_6dof import SawyerPlateSlideSide6DOFEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_plate_slide_back_side_6dof import SawyerPlateSlideBackSide6DOFEnv
'''
CoRL environment lists
Please edit metaworld.envs.mujoco.sawyer_xyz.env_lists to add new lists
'''
from metaworld.envs.mujoco.sawyer_xyz.env_lists import EASY_MODE_LIST, MEDIUM_TRAIN_LIST, MEDIUM_TRAIN_AND_TEST_LIST

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
# ENV_LIST = [SawyerReachPushPickPlace6DOFEnv, SawyerReachPushPickPlace6DOFEnv, SawyerReachPushPickPlace6DOFEnv]
# ENV_LIST = [SawyerReachPushPickPlace6DOFEnv, SawyerReachPushPickPlace6DOFEnv, SawyerReachPushPickPlace6DOFEnv, SawyerShelfPlace6DOFEnv,
# 			SawyerDrawerClose6DOFEnv, SawyerButtonPress6DOFEnv, SawyerButtonPressTopdown6DOFEnv, SawyerPegInsertionSide6DOFEnv]
# ENV_LIST = [SawyerDoor6DOFEnv, SawyerDoorClose6DOFEnv]
# ENV_LIST = [SawyerHandInsert6DOFEnv, SawyerSweep6DOFEnv, SawyerSweepIntoGoal6DOFEnv, SawyerLeverPull6DOFEnv, SawyerDialTurn6DOFEnv]
# ENV_LIST = [SawyerHandInsert6DOFEnv, SawyerSweepIntoGoal6DOFEnv, SawyerSweep6DOFEnv, SawyerDrawerOpen6DOFEnv]
# ENV_LIST = [SawyerHammer6DOFEnv, SawyerNutAssembly6DOFEnv, SawyerBinPicking6DOFEnv]
# ENV_LIST = [SawyerDrawerOpen6DOFEnv, SawyerDrawerClose6DOFEnv]
# ENV_LIST = [SawyerWindowOpen6DOFEnv, SawyerWindowClose6DOFEnv]
# ENV_LIST = [SawyerReachPushPickPlace6DOFEnv, SawyerReachPushPickPlace6DOFEnv, SawyerLeverPull6DOFEnv, SawyerDialTurn6DOFEnv,
# 			SawyerButtonPress6DOFEnv, SawyerButtonPressTopdown6DOFEnv, SawyerDoor6DOFEnv, SawyerDoorClose6DOFEnv,
# 			SawyerWindowOpen6DOFEnv, SawyerWindowClose6DOFEnv, SawyerHandInsert6DOFEnv, SawyerSweep6DOFEnv, SawyerSweepIntoGoal6DOFEnv,
# 			SawyerDrawerOpen6DOFEnv, SawyerDrawerClose6DOFEnv]
# ENV_LIST = [SawyerStack6DOFEnv, SawyerStack6DOFEnv]
# ENV_LIST = [SawyerCoffeeButton6DOFEnv, SawyerCoffeePush6DOFEnv, SawyerCoffeePull6DOFEnv,
# 			SawyerPegInsertionTopdown6DOFEnv, SawyerPegUnplugTopdown6DOFEnv, SawyerPegUnplugSide6DOFEnv, SawyerSoccer6DOFEnv, SawyerBasketball6DOFEnv]
# ENV_LIST = [SawyerReachPushPickPlaceWall6DOFEnv, SawyerReachPushPickPlaceWall6DOFEnv, SawyerReachPushPickPlaceWall6DOFEnv, SawyerFaucetOpen6DOFEnv, SawyerFaucetClose6DOFEnv]
# ENV_LIST = [SawyerReachPushPickPlaceWall6DOFEnv, SawyerReachPushPickPlaceWall6DOFEnv, SawyerReachPushPickPlaceWall6DOFEnv, SawyerUnStack6DOFEnv, SawyerPushBack6DOFEnv,
# 			SawyerPickOutOfHole6DOFEnv, SawyerShelfRemove6DOFEnv, SawyerNutDisassemble6DOFEnv, SawyerDoorLock6DOFEnv, SawyerDoorUnlock6DOFEnv, SawyerSweepTool6DOFEnv,
# 			SawyerGolfPutting6DOFEnv, SawyerFaucetOpen6DOFEnv, SawyerFaucetClose6DOFEnv]
# ENV_LIST = [SawyerCoffeePull6DOFEnv, SawyerPegInsertionTopdown6DOFEnv, SawyerPegUnplugTopdown6DOFEnv, SawyerPegUnplugSide6DOFEnv]
# ENV_LIST = [SawyerUnStack6DOFEnv, SawyerPushBack6DOFEnv, SawyerPickOutOfHole6DOFEnv, SawyerShelfRemove6DOFEnv, SawyerNutDisassemble6DOFEnv]
# ENV_LIST = [SawyerDoorLock6DOFEnv, SawyerDoorUnlock6DOFEnv]
# ENV_LIST = [SawyerFaucetOpen6DOFEnv, SawyerFaucetClose6DOFEnv]
# ENV_LIST = [SawyerCoffeePull6DOFEnv, SawyerPegUnplugSide6DOFEnv]
# ENV_LIST = [SawyerNutDisassemble6DOFEnv, SawyerPegUnplugSide6DOFEnv]
# ENV_LIST = [SawyerShelfRemove6DOFEnv, SawyerShelfRemoveFront6DOFEnv]
# ENV_LIST = [SawyerFaucetClose6DOFEnv, SawyerDoorUnlock6DOFEnv]
# ENV_LIST = [SawyerPegInsertionTopdown6DOFEnv, SawyerPegUnplugTopdown6DOFEnv]
# ENV_LIST = [SawyerReachPushPickPlaceWall6DOFEnv, SawyerReachPushPickPlaceWall6DOFEnv, SawyerReachPushPickPlaceWall6DOFEnv]
# ENV_LIST = [SawyerButtonPressWall6DOFEnv, SawyerButtonPressTopdownWall6DOFEnv]
# ENV_LIST = [SawyerNutDisassemble6DOFEnv, SawyerNutDisassemble6DOFEnv]
# ENV_LIST = [SawyerStack6DOFEnv, SawyerUnStack6DOFEnv]
# ENV_LIST = [SawyerHandlePress6DOFEnv, SawyerHandlePull6DOFEnv]
# ENV_LIST = [SawyerPlateSlide6DOFEnv, SawyerPlateSlideBack6DOFEnv]
# ENV_LIST = [SawyerPlateSlideSide6DOFEnv, SawyerPlateSlideBackSide6DOFEnv]
ENV_LIST = EASY_MODE_LIST
# ENV_LIST = MEDIUM_TRAIN_LIST
# ENV_LIST = [SawyerNutDisassemble6DOFEnv, SawyerUnStack6DOFEnv]
# ENV_LIST = [SawyerHammer6DOFEnv, SawyerDoorClose6DOFEnv]
# ENV_LIST = [SawyerSweepIntoGoal6DOFEnv, SawyerBinPicking6DOFEnv]
# ENV_LIST = [SawyerShelfPlace6DOFEnv, SawyerWindowClose6DOFEnv]
# ENV_LIST = [SawyerBoxOpen6DOFEnv, SawyerBoxClose6DOFEnv]
# ENV_LIST = [SawyerUnStack6DOFEnv, SawyerUnStack6DOFEnv]
# ENV_LIST = [SawyerStickPull6DOFEnv, SawyerStickPush6DOFEnv]
# ENV_LIST = [SawyerBoxOpen6DOFEnv, SawyerNutDisassemble6DOFEnv]
# ENV_LIST = [SawyerUnStack6DOFEnv, SawyerNutDisassemble6DOFEnv]
# ENV_LIST = [SawyerCoffeePush6DOFEnv, SawyerCoffeePull6DOFEnv]




class MultiTaskMujocoEnv(gym.Env):
	"""
	An multitask mujoco environment that contains a list of mujoco environments.
	"""
	def __init__(self,
				random_init=False,
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
			# if i < 3:
			# 	self.mujoco_envs.append(env(multitask=True, multitask_num=len(ENV_LIST), random_init=random_init, if_render=if_render, fix_task=True, task_idx=i))
			# 	# self.mujoco_envs.append(env(multitask=True, multitask_num=len(ENV_LIST), random_init=random_init, if_render=if_render, fix_task=True, task_idx=2))
			# else:
			# 	self.mujoco_envs.append(env(multitask=True, multitask_num=len(ENV_LIST), random_init=random_init, if_render=if_render))
			# if i < 2:#3:
			# 	self.mujoco_envs.append(env(multitask=True, multitask_num=len(ENV_LIST), random_init=random_init, if_render=if_render, fix_task=True, task_idx=i+1))
			# else:
			# 	self.mujoco_envs.append(env(multitask=True, multitask_num=len(ENV_LIST), random_init=random_init, if_render=if_render))
			if env is SawyerReachPushPickPlace6DOFEnv or env is SawyerReachPushPickPlaceWall6DOFEnv:
				# TODO: this could cause flaws in task_idx if SawyerReachPushPickPlace6DOFEnv/SawyerReachPushPickPlaceWall6DOFEnv is not the first environment
				self.mujoco_envs.append(env(multitask=True, multitask_num=len(ENV_LIST), random_init=random_init, fix_task=True, task_idx=i%3))
				# self.mujoco_envs.append(env(multitask=True, multitask_num=len(ENV_LIST), random_init=random_init, if_render=if_render, fix_task=True, task_idx=2))
			else:
				self.mujoco_envs.append(env(multitask=True, multitask_num=len(ENV_LIST), random_init=random_init,))
			# set the one-hot task representation
			self.mujoco_envs[i]._state_goal_idx = np.zeros((len(ENV_LIST)))
			self.mujoco_envs[i]._state_goal_idx[i] = 1.
			self.mujoco_envs[i].max_path_length = 200#150
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
		# used for alpha
		self.goal_len = self.mujoco_envs[self.task_idx].goal_space.low.shape[0]
		# self.reset()
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
