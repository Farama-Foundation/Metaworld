from collections import OrderedDict
import numpy as np
from gym.spaces import  Dict , Box


from metaworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from metaworld.core.multitask_env import MultitaskEnv
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv


from metaworld.envs.mujoco.utils.rotation import euler2quat

from metaworld.envs.mujoco.sawyer_xyz.base import OBS_TYPE

class SawyerWindowOpenEnv(SawyerXYZEnv):
    def __init__(
            self,
            random_init=False,
            goal_low=None,
            goal_high=None,
            obs_type='plain',
            liftThresh = 0.02,
            rewMode = 'orig',
            rotMode='fixed',
            **kwargs
    ):
        hand_low=(-0.5, 0.40, 0.05)
        hand_high=(0.5, 1, 0.5)
        obj_low=(-0.1, 0.7, 0.16)
        obj_high=(0.1, 0.9, 0.16)
        SawyerXYZEnv.__init__(
            self,
            frame_skip=5,
            action_scale=1./100,
            hand_low=hand_low,
            hand_high=hand_high,
            model_name=self.model_name,
            **kwargs
        )

        self.init_config = {
            'obj_init_angle': np.array([0.3, ], dtype=np.float32),
            'obj_init_pos': np.array([-0.1, 0.785, 0.15], dtype=np.float32),
            'hand_init_pos': np.array([0, 0.6, 0.2], dtype=np.float32),
        }
        self.goal = np.array([0.08, 0.785, 0.15])
        self.obj_init_pos = self.init_config['obj_init_pos']
        self.obj_init_angle = self.init_config['obj_init_angle']
        self.hand_init_pos = self.init_config['hand_init_pos']

        assert obs_type in OBS_TYPE
        self.obs_type = obs_type

        if goal_low is None:
            goal_low = self.hand_low
        
        if goal_high is None:
            goal_high = self.hand_high

        self.random_init = random_init
        self.max_path_length = 150
        self.rewMode = rewMode
        self.rotMode = rotMode
        self.liftThresh = liftThresh
        if rotMode == 'fixed':
            self.action_space = Box(
                np.array([-1, -1, -1, -1]),
                np.array([1, 1, 1, 1]),
            )
        elif rotMode == 'rotz':
            self.action_rot_scale = 1./50
            self.action_space = Box(
                np.array([-1, -1, -1, -np.pi, -1]),
                np.array([1, 1, 1, np.pi, 1]),
            )
        elif rotMode == 'quat':
            self.action_space = Box(
                np.array([-1, -1, -1, 0, -1, -1, -1, -1]),
                np.array([1, 1, 1, 2*np.pi, 1, 1, 1, 1]),
            )
        else:
            self.action_space = Box(
                np.array([-1, -1, -1, -np.pi/2, -np.pi/2, 0, -1]),
                np.array([1, 1, 1, np.pi/2, np.pi/2, np.pi*2, 1]),
            )
        self.obj_and_goal_space = Box(
            np.array(obj_low),
            np.array(obj_high),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        if self.obs_type == 'plain':
            self.observation_space = Box(
                np.hstack((self.hand_low, obj_low,)),
                np.hstack((self.hand_high, obj_high,)),
            )
        elif self.obs_type == 'with_goal':
            self.observation_space = Box(
                np.hstack((self.hand_low, obj_low, goal_low)),
                np.hstack((self.hand_high, obj_high, goal_high)),
            )
        else:
            raise NotImplementedError
        self.reset()

    def get_goal(self):
        return {
            'state_desired_goal': self._state_goal,
    }

    @property
    def model_name(self):     
        return get_asset_full_path('sawyer_xyz/sawyer_window_horizontal.xml')

    def step(self, action):
        if self.rotMode == 'euler':
            action_ = np.zeros(7)
            action_[:3] = action[:3]
            action_[3:] = euler2quat(action[3:6])
            self.set_xyz_action_rot(action_)
        elif self.rotMode == 'fixed':
            self.set_xyz_action(action[:3])
        elif self.rotMode == 'rotz':
            self.set_xyz_action_rotz(action[:4])
        else:
            self.set_xyz_action_rot(action[:7])
        self.do_simulation([action[-1], -action[-1]])
        # The marker seems to get reset every time you do a simulation
        # self._set_goal_marker(np.array([0., self._state_goal, 0.05]))
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        obs_dict = self._get_obs_dict()
        reward, reachDist, pickrew, pullDist = self.compute_reward(action, obs_dict, mode = self.rewMode)
        self.curr_path_length +=1
        #info = self._get_info()
        info = {'reachDist': reachDist, 'goalDist': pullDist, 'epRew' : reward, 'pickRew':pickrew, 'success': float(pullDist <= 0.05)}
        info['goal'] = self.goal
        return ob, reward, False, info

    def _get_obs(self):
        hand = self.get_endeff_pos()
        # objPos =  self.data.get_geom_xpos('handle').copy()
        # objPos[0] -= 0.01
        objPos =  self.get_site_pos('handleOpenStart')
        flat_obs = np.concatenate((hand, objPos))
        if self.obs_type == 'with_goal_and_id':
            return np.concatenate([
                    flat_obs,
                    self._state_goal,
                    self._state_goal_idx
                ])
        elif self.obs_type == 'with_goal':
            return np.concatenate([
                    flat_obs,
                    self._state_goal
                ])
        elif self.obs_type == 'plain':
            return np.concatenate([flat_obs,])  # TODO ZP do we need the concat?
        else:
            return np.concatenate([flat_obs, self._state_goal_idx])

    def _get_obs_dict(self):
        hand = self.get_endeff_pos()
        # objPos =  self.data.get_geom_xpos('handle').copy()
        # objPos[0] -= 0.01
        objPos =  self.get_site_pos('handleOpenStart')
        flat_obs = np.concatenate((hand, objPos))
        return dict(
            state_observation=flat_obs,
            state_desired_goal=self._state_goal,
            state_achieved_goal=objPos,
        )

    def _get_info(self):
        pass
    
    def _set_goal_marker(self, goal):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        self.data.site_xpos[self.model.site_name2id('goal')] = (
            goal[:3]
        )

    def _set_objCOM_marker(self):
        """
        This should be use ONLY for visualization. Use self._state_goal for
        logging, learning, etc.
        """
        objPos =  self.data.get_geom_xpos('handle')
        self.data.site_xpos[self.model.site_name2id('objSite')] = (
            objPos
        )
    




    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9] = pos
        # qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def set_goal(self, goal):
        self.task['goal'] = goal

    def reset_model(self):
        self._reset_hand()
        self._state_goal = self.goal.copy()
        self.objHeight = self.data.get_geom_xpos('handle')[2]
        self.heightTarget = self.objHeight + self.liftThresh
        if self.random_init:
            # self.obj_init_pos = np.random.uniform(-0.2, 0)
            # self._state_goal = np.squeeze(np.random.uniform(
            #     self.goal_space.low,
            #     np.array(self.data.get_geom_xpos('handle').copy()[1] + 0.05),
            # ))
            obj_pos = np.random.uniform(
                self.obj_and_goal_space.low,
                self.obj_and_goal_space.high,
                size=(self.obj_and_goal_space.low.size),
            )
            # self.obj_init_qpos = goal_pos[-1]
            self.obj_init_pos = obj_pos
            goal_pos = obj_pos.copy()
            # goal_pos[0] += 0.2
            goal_pos[0] += 0.18
            self._state_goal = goal_pos
        self._set_goal_marker(self._state_goal)
        # self._set_obj_xyz(self.obj_init_pos)
        #self._set_obj_xyz_quat(self.obj_init_pos, self.obj_init_angle)
        wall_pos = self.obj_init_pos.copy() - np.array([-0.1, 0, 0.12])
        window_another_pos = self.obj_init_pos.copy() + np.array([0.2, 0.03, 0])
        self.sim.model.body_pos[self.model.body_name2id('window')] = self.obj_init_pos
        self.sim.model.body_pos[self.model.body_name2id('window_another')] = window_another_pos
        self.sim.model.body_pos[self.model.body_name2id('wall')] = wall_pos
        self.sim.model.site_pos[self.model.site_name2id('goal')] = self._state_goal
        self.curr_path_length = 0
        self.maxPullDist = 0.2
        self.target_reward = 1000*self.maxPullDist + 1000*2
        #Can try changing this
        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)
            #self.do_simulation(None, self.frame_skip)
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.reachCompleted = False

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def compute_rewards(self, actions, obsBatch):
        #Required by HER-TD3
        assert isinstance(obsBatch, dict) == True
        obsList = obsBatch['state_observation']
        rewards = [self.compute_reward(action, obs)[0] for  action, obs in zip(actions, obsList)]
        return np.array(rewards)

    def compute_reward(self, actions, obs, mode = 'general'):
        if isinstance(obs, dict): 
            obs = obs['state_observation']

        objPos = obs[3:6]

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        pullGoal = self._state_goal

        pullDist = np.abs(objPos[0] - pullGoal[0])
        reachDist = np.linalg.norm(objPos - fingerCOM)
        heightTarget = self.heightTarget

        def reachCompleted():
            if reachDist < 0.05:
                return True
            else:
                return False

        if reachCompleted():
            self.reachCompleted = True
        else:
            self.reachCompleted = False

        # c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
        c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
        reachRew = -reachDist
        # if reachDist < 0.05:
        if self.reachCompleted:
            # pushRew = -pushDist
            pullRew = 1000*(self.maxPullDist - pullDist) + c1*(np.exp(-(pullDist**2)/c2) + np.exp(-(pullDist**2)/c3))
        else:
            pullRew = 0
        reward = reachRew + pullRew
        return [reward, reachDist, None, pullDist]

        # def reachReward():
        # 	reachRew = -reachDist# + min(actions[-1], -1)/50
        # 	reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
        # 	zRew = np.linalg.norm(fingerCOM[-1] - self.init_fingerCOM[-1])
        # 	if reachDistxy < 0.05: #0.02
        # 		reachRew = -reachDist
        # 	else:
        # 		reachRew =  -reachDistxy - 2*zRew
        # 	#incentive to close fingers when reachDist is small
        # 	if reachDist < 0.05:
        # 		reachRew = -reachDist + max(actions[-1],0)/50
        # 	return reachRew , reachDist

        # def pickCompletionCriteria():
        # 	tolerance = 0.01
        # 	if objPos[2] >= (heightTarget- tolerance):
        # 		return True
        # 	else:
        # 		return False

        # if pickCompletionCriteria():
        # 	self.pickCompleted = True


        # def objDropped():
        # 	return (objPos[2] < (self.objHeight + 0.005)) and (pullDist >0.02) and (reachDist > 0.02) 
        # 	# Object on the ground, far away from the goal, and from the gripper
        # 	#Can tweak the margin limits
       
        # def objGrasped(thresh = 0):
        # 	sensorData = self.data.sensordata
        # 	return (sensorData[0]>thresh) and (sensorData[1]> thresh)

        # def orig_pickReward():       
        # 	# hScale = 50
        # 	hScale = 100
        # 	if self.pickCompleted and not(objDropped()):
        # 		return hScale*heightTarget
        # 	# elif (reachDist < 0.1) and (objPos[2]> (self.objHeight + 0.005)) :
        # 	elif (reachDist < 0.1) and (objPos[2]> (self.objHeight + 0.005)) :
        # 		return hScale* min(heightTarget, objPos[2])
        # 	else:
        # 		return 0

        # def general_pickReward():
        # 	hScale = 50
        # 	if self.pickCompleted and objGrasped():
        # 		return hScale*heightTarget
        # 	elif objGrasped() and (objPos[2]> (self.objHeight + 0.005)):
        # 		return hScale* min(heightTarget, objPos[2])
        # 	else:
        # 		return 0

        # def pullReward():
        # 	# c1 = 1000 ; c2 = 0.03 ; c3 = 0.003
        # 	c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
        # 	if mode == 'general':
        # 		cond = self.pickCompleted and objGrasped()
        # 	else:
        # 		cond = self.pickCompleted and (reachDist < 0.1) and not(objDropped())
        # 	if cond:
        # 		pullReward = 1000*(self.maxPullDist - pullDist) + c1*(np.exp(-(pullDist**2)/c2) + np.exp(-(pullDist**2)/c3))
        # 		pullReward = max(pullReward,0)
        # 		return [pullReward , pullDist]
        # 	else:
        # 		return [0 , pullDist]

        # reachRew, reachDist = reachReward()
        # if mode == 'general':
        # 	pickRew = general_pickReward()
        # else:
        # 	pickRew = orig_pickReward()
        # pullRew , pullDist = pullReward()
        # assert ((pullRew >=0) and (pickRew>=0))
        # reward = reachRew + pickRew + pullRew
      
        # return [reward, reachDist, pickRew, pullDist] 

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        return statistics

    def log_diagnostics(self, paths = None, logger = None):
        pass
