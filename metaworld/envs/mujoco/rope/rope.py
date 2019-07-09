import abc
from collections import OrderedDict
import numpy as np
from metaworld.envs.mujoco.mujoco_env import MujocoEnv
from gym.spaces import Box, Dict
from metaworld.core.serializable import Serializable
from metaworld.envs.env_util import get_stat_in_paths, \
	create_stats_ordered_dict, get_asset_full_path
from metaworld.envs.mujoco.dynamic_mjc.rope import rope
import mujoco_py
from PIL import Image
import imageio
import pickle
import tensorflow as tf


def get_beads_xy(qpos, num_beads):
	init_joint_offset = 6
	num_free_joints = 7

	xy_list = []
	for j in range(num_beads):
		offset = init_joint_offset + j*num_free_joints
		xy_list.append(qpos[offset:offset+2])

	return np.asarray(xy_list)

def get_com(xy_list):
	return np.mean(xy_list, axis=0)

def calculate_distance(qpos1, qpos2, num_beads):

	xy1 = get_beads_xy(qpos1, num_beads)
	xy2 = get_beads_xy(qpos2, num_beads)

	com1 = get_com(xy1)
	com2 = get_com(xy2)

	xy1_translate = xy1 + (com2 - com1)

	distance = np.linalg.norm(xy1_translate - xy2)

	return distance

class RopeEnv(MujocoEnv, Serializable, metaclass=abc.ABCMeta):
	"""Implements a reacher environment with visual rewards"""

	def __init__(self,
				 num_beads=7,
				 init_pos=[0.0, 0.0, 0.0],
				 substeps=50,
				 vision=True,
				 log_video=False, 
				 video_substeps = 5, 
				 video_h=500, 
				 video_w=500,
				 camera_name='overheadcam',
				 sparse=True,
				 action_penalty_const=1e-2,
				 use_latent=False,
				 latent_meta_path=None,
				 scale_and_bias_path=None,
				 goal_img_path=None,
				 goal_vec_path=None):
		self.quick_init(locals())
		
		#sim params
		self.substeps = substeps # number of intermediate positions to generate
		
		#env params
		self.num_beads = num_beads
		self.init_pos = init_pos

		#reward params
		self.sparse = sparse
		self.action_penalty_const = action_penalty_const

		#video params
		self.log_video = log_video
		self.video_substeps = video_substeps
		self.video_h = video_h
		self.video_w = video_w
		self.camera_name = camera_name
		self.vision = vision
		self.use_latent = use_latent

		if use_latent:
			tf_config = tf.ConfigProto()
			tf_config.gpu_options.allow_growth = True
			self.latent_graph = tf.Graph()
			self.latent_sess = tf.Session(config=tf_config, graph=self.latent_graph)
			with self.latent_graph.as_default():
				saver = tf.train.import_meta_graph(latent_meta_path)
				saver.restore(self.latent_sess, latent_meta_path[:-5])
				self.latent_feed_dict = {
					'ot': self.latent_graph.get_tensor_by_name('ot:0'),
					'qt': self.latent_graph.get_tensor_by_name('qt:0'),
					'og': self.latent_graph.get_tensor_by_name('og:0'),
					'xt': self.latent_graph.get_tensor_by_name('xt:0'),
					'xg': self.latent_graph.get_tensor_by_name('xg:0'),
					'eff_horizons': self.latent_graph.get_tensor_by_name('eff_horizons:0'),
					'atT_original': self.latent_graph.get_tensor_by_name('atT_0:0'),
					'plan': self.latent_graph.get_tensor_by_name('plan:0'),
				}
				self.xg = self.latent_sess.run(self.latent_feed_dict['xg'],
												feed_dict={self.latent_feed_dict['og']: np.zeros((1, 100, 100, 3))})
		# if scale_and_bias_path is not None:
			# with open(scale_and_bias_path, 'rb') as f:
			# 	data = pickle.load(f)
			# 	self.scale = data['scale']
			# 	self.bias = data['bias']
		self.scale = 0.
		self.bias = 0.

		model = rope(num_beads=self.num_beads, init_pos=self.init_pos, texture=True)
		with model.asfile() as f:
			MujocoEnv.__init__(self, f.name, 5, automatically_set_spaces=False)
		
		low = np.asarray(4*[-0.4])
		high = np.asarray(4*[0.4])
		self.action_space = Box(low=low, high=high, dtype=np.float32)
		if goal_vec_path is not None:
			with open(goal_vec_path, 'rb') as f:
				self.goal_vec = pickle.load(f)
			self.goal_img = imageio.imread(goal_img_path)
		
		#when using xmls        
		#mujoco_env.MujocoEnv.__init__(self, 'rope.xml', 5)

	def get_latent_metric(self, ot, qt=None, og=None):
		d = 0.85#1.0
		with self.latent_graph.as_default():
			# xt = self.latent_sess.run(self.latent_feed_dict['xt'],
			# 								feed_dict={self.latent_feed_dict['ot']: ot / 255.0,
			# 								self.latent_feed_dict['qt']:np.expand_dims(qt, axis=0)})
			# just use the latent representation without joint embedding seems to work a lot better!
			xt = self.latent_sess.run(self.latent_feed_dict['xg'],
											feed_dict={self.latent_feed_dict['og']: ot / 255.0})
			if og is None:
				xg = self.xg
			else:
				xg = self.latent_sess.run(self.latent_feed_dict['xg'],
												feed_dict={self.latent_feed_dict['og']: og / 255.0})
			error = np.abs(xt - xg)
			mask = (error > 1)
			dist = np.mean(np.sum(mask * (0.5 * (d**2) + d * (error - d)) + (1 - mask) * 0.5 * (error**2), axis=1))
			# dist = np.mean(mask * (0.5 * (d**2) + d * (error - d)) + (1 - mask) * 0.5 * (error**2))
		return dist

	def step(self, a):

		#make string horizontal rewards
		first_bead = self.get_body_com("bead_0")[:2]
		last_bead = self.get_body_com("bead_{}".format(self.num_beads-1))[:2]
		vec_1 = first_bead - last_bead
		vec_2 = np.asarray([1.0, 0.0]) #horizontal line
		cosine = np.dot(vec_1, vec_2)/(np.linalg.norm(vec_1) + 1e-10)
		abs_cos = np.abs(cosine)

		#compute action penalty
		#act_dim = self.action_space.shape[0]
		movement_1 = -1.0*np.linalg.norm(self.sim.data.qpos[:2] - a[:2])
		movement_2 =  -1.0*np.linalg.norm(a[:2] - a[2:])
		action_penalty = movement_1 + movement_2

		if self.vision:
			self.render()
			img, qt = self.get_current_image_obs()
			if hasattr(self, "goal_img"):
				while np.all(img == 0.):
					import time
					self.render()
					img, qt = self.get_current_image_obs()
					time.sleep(0.05)

		if self.sparse:
			if abs_cos > 0.9:
				main_rew = 1.0
			else:
				main_rew = 0.0
		else:
			main_rew = abs_cos 

		if not self.use_latent:
			reward = main_rew + self.action_penalty_const*action_penalty
		else:
			if hasattr(self, "goal_img"):
				reward = -self.get_latent_metric(np.expand_dims(img, axis=0), qt.dot(self.scale) + self.bias)
				# reward = np.exp(reward)
				# reward = 0.6*np.exp(reward) + 0.4*np.exp(2.5*reward)
				reward = np.exp(0.5*reward)
			else:
				reward = 0.

		#import IPython; IPython.embed()
		#enquire the qpos here, and then do the interpolation thing
		#self.do_simulation(a, self.frame_skip)

		#video_frames = self.do_pos_simulation_with_substeps(a)
		
		#video_frames = self.pick_place(a)
		video_frames = self.push(a)

		ob = self._get_obs()
		done = False
		is_success = False #TODO maybe fix this at some point

		if abs_cos > 0.9:
			is_success = True

		min_dist = np.amin(np.linalg.norm(get_beads_xy(self.sim.data.qpos, 7) - np.expand_dims(self.sim.data.qpos[:2], axis=0), axis=1))

		return ob, reward, done, dict(is_success=is_success, video_frames=video_frames, action_penalty=action_penalty, min_dist=min_dist)

	def pick_place(self, a):
		x_start = a[0]
		y_start = a[1]
		x_end = a[2]
		y_end = a[3]

		z_min = -0.1
		z_max = +0.05

		torque_max = +10.0
		torque_neutral = 0.0
		torque_min = -1*torque_max

		actions = np.asarray(
			   [[0.0, -0.2, z_max, 0.0, torque_neutral], #neutral position
				[x_start, y_start, z_max, 0.0, torque_neutral], #get close
				[x_start, y_start, z_min, 0.0, torque_neutral], #go down
				[x_start, y_start, z_min, 0.0, torque_max], #grasp
				[x_start, x_start, z_max,  0.0, torque_max], #go up
				[x_end,y_end,z_max,0.0,torque_max], #move
				[x_end,y_end, z_min, 0.0, torque_max], #go down
				[x_end,y_end, z_min, 0.0, torque_neutral], #drop 
				[x_end,y_end, z_max, 0.0, torque_neutral], #go back up 
				[0.0, -0.2, z_max, 0.0, torque_min],#neutral position, open gripper
				])

		video_frames = []
		for i in range(actions.shape[0]):
			video_frames.append(self.do_pos_simulation_with_substeps(actions[i]))

		return video_frames

	def push(self, a):
		x_start = a[0]
		y_start = a[1]
		x_end = a[2]
		y_end = a[3]

		x_neutral = 0.0
		y_neutral = -0.2
		z_min = -0.1
		z_max = +0.05
		torque_max = +10.0
		torque_neutral = 0.0
		torque_min = -1*torque_max

		actions = np.asarray(
			   [
				# [x_neutral, y_neutral, z_max, 0.0, torque_max], #neutral position
				[x_start, y_start, z_max, 0.0, torque_max], #get close, close gripper
				[x_start, y_start, z_min, 0.0, torque_neutral], #go down
				[x_end,y_end,z_min,0.0,torque_neutral], #move
				[x_end,y_end, z_max, 0.0, torque_max], #go back up, close gripper 
				# [x_neutral, y_neutral, z_max, 0.0, torque_max],#neutral position, open gripper
				])

		video_frames = []
		for i in range(actions.shape[0]):
			video_frames.append(self.do_pos_simulation_with_substeps(actions[i]))

		return video_frames

	def do_pos_simulation_with_substeps(self, a):
		qpos_curr = self.sim.data.qpos[:self.action_space.shape[0]]
		a_pos = a[:self.action_space.shape[0]]

		step_size = (a_pos - qpos_curr) / self.substeps
		
		if self.log_video:
			video_frames = np.zeros((int(self.substeps/self.video_substeps), self.video_h, self.video_w, 3))
		else:
			video_frames = None
		for i in range(self.substeps):
			self.sim.data.ctrl[:-1] = qpos_curr + (i+1)*step_size
			#torque control on the gripper
			self.sim.data.ctrl[-1] = a[-1]
			self.sim.step()
			if i%self.video_substeps == 0 and self.log_video :
				# video_frames[int(i/self.video_substeps)] = self.sim.render(self.video_h, self.video_w, camera_name=self.camera_name)
				video_frames[int(i/self.video_substeps)] = self.render(mode='rgb_array', width=self.video_h, height=self.video_w)

		return video_frames

	def get_current_image_obs(self):
		img = self.render(mode='rgb_array', width=self.video_h, height=self.video_w)
		pil_image = Image.fromarray(img, 'RGB')
		img = np.array(pil_image.resize((100,100), Image.ANTIALIAS))
		return img, np.concatenate([
			self.sim.data.qpos.flat[:6],
			self.sim.data.qvel.flat[:6],
		]).reshape(-1)

	def get_goal_image(self):
		assert hasattr(self, 'goal_img')
		return self.goal_img

	def viewer_setup(self):
		self.viewer.cam.trackbodyid = -1
		self.viewer.cam.distance = 1.2#2.0 #4.0
		self.viewer.cam.lookat[0] = 0.
		self.viewer.cam.lookat[1] = 0.0
		self.viewer.cam.lookat[2] = 0.1
		self.viewer.cam.elevation = -90#90
		self.viewer.cam.azimuth = 90#270

	def reset_model(self):
		qpos = self.init_qpos.copy()
		qvel = self.init_qvel.copy()
		if self.vision:
			init_joint_offset = 6
			num_free_joints = 7

			for j in range(self.num_beads):
				offset = init_joint_offset + j*num_free_joints
				qpos[offset:offset+2] = np.array([0.25-0.07*j, 0.1])
			self.set_state(qpos, qvel)
			self.goal_img, _ = self.get_current_image_obs()
			while np.all(self.goal_img == 0.):
				import time
				self.render()
				self.goal_img, _ = self.get_current_image_obs()
				time.sleep(0.05)
			if self.use_latent:
				with self.latent_graph.as_default():
					self.xg = self.latent_sess.run(self.latent_feed_dict['xg'],
													feed_dict={self.latent_feed_dict['og']: np.expand_dims(self.goal_img / 255.0, axis=0)})
		self.set_state(self.init_qpos, self.init_qvel)
		return self._get_obs()

	def _get_obs(self):
		return np.concatenate([
			self.sim.data.qpos.flat,
			self.sim.data.qvel.flat,
		])

if __name__ == "__main__": 
	
	log_video = True

	#test drive the env    
	# import matplotlib.pyplot as plt
	# time_wait = 1.5
	# torque_max = 10.0
	# torque_neutral = 0.0
	# max_z = 0.05

	# actions = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.00],
	#                 [0.0, 0.0, -0.2, 0.0, 0.00],

	#                 [0.0, 0.0, -0.2, 0.0, torque_max],
	#                 [0.0, 0.0, -0.2, 0.0, torque_max],
	#                 [0.0, 0.0, -0.2, 0.0, torque_max],

	#                 [0.0, 0.0, max_z,  0.0, torque_max],
	#                 [0.0, 0.0, max_z,  0.0, torque_max],
	#                 [0.0, 0.0, max_z,  0.0, torque_max],

	#                 [0.2, 0.0, max_z,  0.0, torque_max],
	#                 [0.2, 0.0, max_z,  0.0, torque_max],
	#                 [0.2, 0.0, max_z,  0.0, torque_max],

	#                 [0.0, -0.2, max_z, 0.0, torque_neutral],
	#                 [0.0, -0.2, max_z, 0.0, torque_neutral],

	#                 ])

	# np.set_printoptions(precision=3)


	# for i in range(actions.shape[0]):
	#     act = actions[i]
	#     o,r,d,info = env.step(act)
	#     #video = np.concatenate([video, info['video_frames']])
	#     if info['video_frames'] is not None:
	#         for j in range(info['video_frames'].shape[0]):
	#             writer.writeFrame(info['video_frames'][j])

	#     plt.figure(1); plt.clf()
	#     bla = env.sim.render(500, 500, camera_name='maincam')
	#     plt.axis('off')
	#     plt.imshow(bla)
	#     plt.pause(time_wait)
	#     #env._get_viewer().render()
	#     print(env.sim.data.qpos[:actions.shape[1]])

	# if writer is not None:
	#     writer.close()

	meta_path = '/scr/kevin/unsupervised_upn/summ/rope_random_latent_planning_ol_lr0.0003_il_lr0.25_num_plan_updates_40_horizon_1_num_train_19000__learn_lr_clip0.02_loss_coeff_100.0_n_hidden_2_latent_dim_128_dt_1_fp_23-01-2019_15-55-34/models/model_plan_test_58000.meta' #reverse data no curriculum gan


	#overheadcam, miancam, leftcam
	# env = RopeEnv(num_beads=7, substeps=50, log_video=log_video, camera_name='maincam')
	env = RopeEnv(num_beads=7, substeps=50, log_video=log_video, camera_name='maincam', 
					use_latent=True, latent_meta_path=meta_path, vision=True,
					goal_img_path='/afs/cs.stanford.edu/u/tianheyu/multiworld/multiworld/envs/mujoco/rope/rope_goal.png',
					goal_vec_path='/afs/cs.stanford.edu/u/tianheyu/multiworld/multiworld/envs/mujoco/rope/rope_goal.pkl')
	video_frames = []
	imgs = [env.get_current_image_obs()[0]]
	# video_frames += env.pick_place(np.asarray([0.,0.,-0.2,0.2]))
	# video_frames += env.pick_place(np.asarray([-0.2,0.2,-0.2,0.2]))
	# video_frames += env.pick_place(np.asarray([0.,0.,0.2,0.2]))

	for i in range(6):
		o,r,done,info = env.step(np.asarray([0.05*i,0.05*i,0.05*(i+1),0.05*(i+1)]))
		video_frames += info['video_frames']
		imgs.append(env.get_current_image_obs()[0])
		print('reward: {}'.format(r))
	# o,r,done,info = env.step(np.asarray([0.05*i,0.05*i,0.,0.]))
	# video_frames += info['video_frames']
	# imgs.append(env.get_current_image_obs()[0])
	# print('reward: {}'.format(r))
	# o,r,done,info = env.step(np.asarray([0.,0.,0.05,0.05]))
	# video_frames += info['video_frames']
	# imgs.append(env.get_current_image_obs()[0])
	# print('reward: {}'.format(r))
	# o,r,done,info = env.step(np.asarray([0.05,0.05,0.1,0.1]))
	# video_frames += info['video_frames']
	# imgs.append(env.get_current_image_obs()[0])
	# print('reward: {}'.format(r))
	# o,r,done,info = env.step(np.asarray([0.1,0.1,0.15,0.15]))
	# video_frames += info['video_frames']
	# imgs.append(env.get_current_image_obs()[0])
	# print('reward: {}'.format(r))
	#import IPython; IPython.embed()
	latent_dists = [env.get_latent_metric(np.expand_dims(imgs[i], axis=0), 0., np.expand_dims(env.get_current_image_obs()[0], axis=0)) for i in range(len(imgs))]
	import matplotlib.pyplot as plt
	plt.plot(range(len(latent_dists)), latent_dists)
	plt.xlabel('Number of time steps')
	plt.ylabel('Distance in latent space')
	plt.title('latent metric of a demo')
	plt.savefig('rope_latent_metric_dt1.png')
	import pdb; pdb.set_trace()

	if log_video: 
		import datetime, skvideo.io, imageio, pickle
		# video_name = '~/multiworld/multiworld/envs/mujoco/rope/rope_{}.gif'.format(datetime.datetime.now().strftime("%y-%m-%d-%H-%M"))
		# video_downsample_name = '~/multiworld/multiworld/envs/mujoco/rope/rope_downsample_{}.gif'.format(datetime.datetime.now().strftime("%y-%m-%d-%H-%M"))
		# goal_image_name = '~/multiworld/multiworld/envs/mujoco/rope/rope_goal_{}.png'.format(datetime.datetime.now().strftime("%y-%m-%d-%H-%M"))
		video_name = '~/multiworld/multiworld/envs/mujoco/rope/rope.gif'
		video_downsample_name = '~/multiworld/multiworld/envs/mujoco/rope/rope_downsample.gif'
		goal_image_name = '~/multiworld/multiworld/envs/mujoco/rope/rope_goal.png'
		imageio.mimwrite(video_name, np.concatenate(video_frames).astype(np.uint8))
		imageio.mimwrite(video_downsample_name, video_downsample_frames)
		imageio.imwrite(goal_image_name, env.get_current_image_obs()[0])
		with open('/afs/cs.stanford.edu/u/tianheyu/multiworld/multiworld/envs/mujoco/rope/rope_goal.pkl', 'wb') as f:
			first_bead = env.get_body_com("bead_0")[:2]
			last_bead = env.get_body_com("bead_{}".format(env.num_beads-1))[:2]
			vec_1 = first_bead - last_bead
			pickle.dump(vec_1, f)
		# video_name = 'logs/rope_{}.mp4'.format(datetime.datetime.now().strftime("%y-%m-%d-%H-%M"))
		# writer = skvideo.io.FFmpegWriter(video_name)

		# for i in range(len(video_frames)):
		# 	for j in range(video_frames[0].shape[0]):
		# 		 writer.writeFrame(video_frames[i][j])
		# writer.close()
