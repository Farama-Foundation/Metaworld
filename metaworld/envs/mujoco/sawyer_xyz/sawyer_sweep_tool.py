from collections import OrderedDict
import numpy as np
from gym.spaces import  Dict , Box


from metaworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from metaworld.core.multitask_env import MultitaskEnv
from metaworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv


from metaworld.envs.mujoco.utils.rotation import euler2quat
from metaworld.envs.mujoco.sawyer_xyz.base import OBS_TYPE

class SawyerSweepToolEnv(SawyerXYZEnv):
    def __init__(
            self,
            random_init=False,
            obs_type='plain',
            tasks = [{'goal': np.array([0., 0.7, 0.02]),  'obj_init_pos':np.array([0, 0.6, 0.02]), 'obj_init_angle': 0.3}], 
            goal_low=(-0.1, 0.7, 0.02),
            goal_high=(0.1, 0.75, 0.02),
            hand_init_pos = (0, 0.6, 0.2),
            liftThresh=0.02,
            rotMode='fixed',#'fixed',
            rewMode='orig',
            multitask=False,
            multitask_num=1,
            **kwargs
    ):

        hand_low=(-0.5, 0.40, 0.05)
        hand_high=(0.5, 1, 0.5)
        obj_low=(0, 0.6, 0.02)
        obj_high=(0, 0.6, 0.02)
        SawyerXYZEnv.__init__(
            self,
            frame_skip=5,
            action_scale=1./100,
            hand_low=hand_low,
            hand_high=hand_high,
            model_name=self.model_name,
            **kwargs
        )
        assert obs_type in OBS_TYPE
        if multitask:
            obs_type = 'with_goal_and_id'
        self.obs_type = obs_type

        if goal_low is None:
            goal_low = self.hand_low
        
        if goal_high is None:
            goal_high = self.hand_high

        self.random_init = random_init
        self.max_path_length = 150#150
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.rotMode = rotMode
        self.rewMode = rewMode
        self.liftThresh = liftThresh
        self.hand_init_pos = np.array(hand_init_pos)
        self.multitask = multitask
        self.multitask_num = multitask_num
        self._state_goal_idx = np.zeros(self.multitask_num)
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
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        if not multitask and self.obs_type == 'with_goal_id':
            self.observation_space = Box(
                np.hstack((self.hand_low, obj_low, np.zeros(len(tasks)))),
                np.hstack((self.hand_high, obj_high, np.ones(len(tasks)))),
            )
        elif not multitask and self.obs_type == 'plain':
            self.observation_space = Box(
                np.hstack((self.hand_low, obj_low,)),
                np.hstack((self.hand_high, obj_high,)),
            )
        elif not multitask and self.obs_type == 'with_goal':
            self.observation_space = Box(
                np.hstack((self.hand_low, obj_low, goal_low)),
                np.hstack((self.hand_high, obj_high, goal_high)),
            )
        else:
            self.observation_space = Box(
                np.hstack((self.hand_low, obj_low, goal_low, np.zeros(multitask_num))),
                np.hstack((self.hand_high, obj_high, goal_high, np.zeros(multitask_num))),
            )
        self.reset()
        # self.observation_space = Dict([
        #     ('state_observation', self.hand_and_obj_space),
        #     ('state_desired_goal', self.goal_space),
        #     ('state_achieved_goal', self.goal_space),
        # ])


    def get_goal(self):
        return {
            'state_desired_goal': self._state_goal,
    }

    @property
    def model_name(self):     

        return get_asset_full_path('sawyer_xyz/sawyer_sweep_tool.xml')
        #return get_asset_full_path('sawyer_xyz/pickPlace_fox.xml')

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
        # self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        obs_dict = self._get_obs_dict()
        reward, reachRew, reachDist, pickRew, pushRew, pushDist, ballDist = self.compute_reward(action, ob, mode=self.rewMode)
        self.curr_path_length +=1
        #info = self._get_info()
        info = {'reachDist': reachDist, 'goalDist': pushDist, 'epRew' : reward, 'pickRew':pickRew}
        info['goal'] = self.goal
        return ob, reward, False, info

    def _get_obs(self):
        hand = self.get_endeff_pos()
        objPos =  self.data.site_xpos[self.model.site_name2id('handleStart')]
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
        objPos =  self.data.site_xpos[self.model.site_name2id('handleStart')]
        flat_obs = np.concatenate((hand, objPos))
        if self.multitask:
            assert hasattr(self, '_state_goal_idx')
            return np.concatenate([
                    flat_obs,
                    self.data.get_geom_xpos('objGeom').copy(),
                    self._state_goal_idx
                ])
        return dict(
            state_observation=flat_obs,
            state_desired_goal=self.data.get_geom_xpos('objGeom').copy(),
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
        objPos =  self.data.get_geom_xpos('objGeom')
        self.data.site_xpos[self.model.site_name2id('objSite')] = (
            objPos
        )
    




    def _set_obj_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)

    def _set_goal_xyz(self, pos):
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[16:19] = pos.copy()
        qvel[15:21] = 0
        self.set_state(qpos, qvel)


    def sample_goals(self, batch_size):
        #Required by HER-TD3
        goals = []
        for i in range(batch_size):
            task = self.tasks[np.random.randint(0, self.num_tasks)]
            goals.append(task['goal'])
        return {
            'state_desired_goal': goals,
        }


    def sample_task(self):
        self.task_idx = np.random.randint(0, self.num_tasks)
        return self.tasks[self.task_idx]

    def adjust_initObjPos(self, orig_init_pos):
        #This is to account for meshes for the geom and object are not aligned
        #If this is not done, the object could be initialized in an extreme position
        diff = self.get_body_com('obj')[:2] - self.data.get_geom_xpos('objGeom')[:2]
        adjustedPos = orig_init_pos[:2] + diff

        #The convention we follow is that body_com[2] is always 0, and geom_pos[2] is the object height
        return [adjustedPos[0], adjustedPos[1],self.data.get_geom_xpos('objGeom')[-1]]


    def reset_model(self):
        self._reset_hand()
        task = self.sample_task()
        self._state_goal = np.array(task['goal'])
        self.obj_init_pos = self.adjust_initObjPos(task['obj_init_pos'])
        self.obj_init_angle = task['obj_init_angle']
        self.clubHeight = self.data.site_xpos[self.model.site_name2id('handleStart')][2]
        self.heightTarget = self.clubHeight + self.liftThresh
        if self.random_init:
            goal_pos = np.random.uniform(
                self.obj_and_goal_space.low,
                self.obj_and_goal_space.high,
                size=(self.obj_and_goal_space.low.size),
            )
            self._state_goal = goal_pos[3:]
            while np.linalg.norm(goal_pos[1] - self._state_goal[1]) < 0.07:
                goal_pos = np.random.uniform(
                    self.obj_and_goal_space.low,
                    self.obj_and_goal_space.high,
                    size=(self.obj_and_goal_space.low.size),
                )
                self._state_goal = goal_pos[3:]
            self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))
        self._set_goal_xyz(self._state_goal)
        self._set_obj_xyz(self.obj_init_pos)
        #self._set_obj_xyz_quat(self.obj_init_pos, self.obj_init_angle)
        self.curr_path_length = 0
        self.maxPushDist = np.linalg.norm(np.array(self.obj_init_pos[:2] - self._state_goal[:2])) + \
                            np.linalg.norm(np.array(self._state_goal[:2] - self.data.site_xpos[self.model.site_name2id('goal')][:2]))
        return self._get_obs()

    def _reset_hand(self):
        for _ in range(10):
            self.data.set_mocap_pos('mocap', self.hand_init_pos)
            self.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            self.do_simulation([-1,1], self.frame_skip)
            #self.do_simulation(None, self.frame_skip)
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        self.init_fingerCOM  =  (rightFinger + leftFinger)/2
        self.pickCompleted = False

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def compute_rewards(self, actions, obsBatch):
        #Required by HER-TD3
        assert isinstance(obsBatch, dict) == True
        obsList = obsBatch['state_observation']
        rewards = [self.compute_reward(action, obs)[0] for  action, obs in zip(actions, obsList)]
        return np.array(rewards)

    def compute_reward(self, actions, obs, mode='orig'):
        if isinstance(obs, dict): 
            obs = obs['state_observation']

        graspPos = obs[3:6]
        clubPos = self.data.get_geom_xpos('clubHead')
        objPos = obs[6:9]

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        heightTarget = self.heightTarget
        pushGoal = self.data.site_xpos[self.model.site_name2id('goal')][:2]

        ballDist = np.linalg.norm(objPos - clubPos)
        pushDist = np.linalg.norm(objPos[:2] - pushGoal[:2])
        reachDist = np.linalg.norm(graspPos - fingerCOM)
        if objPos[-1] < self._state_goal[-1] - 0.05:
            reachRew = 0
            pushDist = 0
            reachDist = 0

        def reachReward():
            reachRew = -reachDist# + min(actions[-1], -1)/50
            reachDistxy = np.linalg.norm(graspPos[:-1] - fingerCOM[:-1])
            zRew = np.linalg.norm(fingerCOM[-1] - self.init_fingerCOM[-1])
            reachRew = -reachDist
            #incentive to close fingers when reachDist is small
            if reachDist < 0.05:
                reachRew = -reachDist + max(actions[-1],0)/50
            return reachRew , reachDist

        def pickCompletionCriteria():
            tolerance = 0.01
            if graspPos[2] >= (heightTarget- tolerance):
                return True
            else:
                return False

        if pickCompletionCriteria():
            self.pickCompleted = True

        def objDropped():
            return (graspPos[2] < (self.clubHeight + 0.005)) and (pushDist >0.02) and (reachDist > 0.02) 
            # Object on the ground, far away from the goal, and from the gripper
            #Can tweak the margin limits
       
        def objGrasped(thresh = 0):
            sensorData = self.data.sensordata
            return (sensorData[0]>thresh) and (sensorData[1]> thresh)

        def orig_pickReward():       
            # hScale = 50
            hScale = 100
            if self.pickCompleted and not(objDropped()):
                return hScale*heightTarget
            # elif (reachDist < 0.1) and (hammerPos[2]> (self.hammerHeight + 0.005)) :
            elif (reachDist < 0.1) and (graspPos[2]> (self.clubHeight + 0.005)) :
                return hScale* min(heightTarget, graspPos[2])
            else:
                return 0

        def general_pickReward():
            hScale = 50
            if self.pickCompleted and objGrasped():
                return hScale*heightTarget
            elif objGrasped() and (graspPos[2]> (self.clubHeight + 0.005)):
                return hScale* min(heightTarget, graspPos[2])
            else:
                return 0

        def pushReward():
            # c1 = 1000 ; c2 = 0.03 ; c3 = 0.003
            c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
            if mode == 'general':
                cond = self.pickCompleted and objGrasped()
            else:
                cond = self.pickCompleted and (reachDist < 0.1) and not(objDropped())
            if cond:
                pushRew = 1000*(self.maxPushDist - pushDist - ballDist) + c1*(np.exp(-((pushDist+ballDist)**2)/c2) + np.exp(-((pushDist+ballDist)**2)/c3))
                pushRew = max(pushRew,0)
                return [pushRew , pushDist, ballDist]
            else:
                return [0 , pushDist, ballDist]

        reachRew, reachDist = reachReward()
        if mode == 'general':
            pickRew = general_pickReward()
        else:
            pickRew = orig_pickReward()
        pushRew , pushDist, ballDist = pushReward()
        assert ((pushRew >=0) and (pickRew>=0))
        reward = reachRew + pickRew + pushRew
        return [reward, reachRew, reachDist, pickRew, pushRew, pushDist, ballDist]


    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        return statistics

    def log_diagnostics(self, paths = None, logger = None):
        pass
