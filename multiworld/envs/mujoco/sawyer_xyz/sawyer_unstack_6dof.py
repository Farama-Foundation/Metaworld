from collections import OrderedDict
import numpy as np
from gym.spaces import Box, Dict

from multiworld.envs.env_util import get_stat_in_paths, \
    create_stats_ordered_dict, get_asset_full_path
from multiworld.core.multitask_env import MultitaskEnv
from multiworld.envs.mujoco.sawyer_xyz.base import SawyerXYZEnv
from multiworld.envs.mujoco.cameras import sawyer_pick_and_place_camera, sawyer_pick_and_place_camera_slanted_angle


class SawyerUnStack6DOFEnv(MultitaskEnv, SawyerXYZEnv):
    def __init__(
            self,
            hand_low=(-0.5, 0.40, 0.05),
            hand_high=(0.5, 1, 0.5),
            obj_low=(-0.1, 0.8, 0.06),
            obj_high=(0.1, 0.9, 0.06),
            random_init=False,
            tasks = [{'goal': np.array([0, 0.6, 0.02]),  'obj_init_pos':np.array([0.1, 0.8, 0.06]), 'obj_init_angle': 0.3}], 
            goal_low=(-0.1, 0.6, 0.02),
            goal_high=(0.1, 0.7, 0.02),
            hand_init_pos = (0, 0.6, 0.2),
            liftThresh = 0.04,
            rewMode = 'dense',
            rotMode='fixed',#'fixed',
            multitask=False,
            multitask_num=1,
            if_render=False,
            **kwargs
    ):
        self.quick_init(locals())
        SawyerXYZEnv.__init__(
            self,
            frame_skip=5,
            action_scale=1./100,
            hand_low=hand_low,
            hand_high=hand_high,
            model_name=self.model_name,
            **kwargs
        )
        if obj_low is None:
            obj_low = self.hand_low

        if goal_low is None:
            goal_low = self.hand_low

        if obj_high is None:
            obj_high = self.hand_high
        
        if goal_high is None:
            goal_high = self.hand_high

        self.random_init = random_init
        self.liftThresh = liftThresh
        self.max_path_length = 200#150
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.rewMode = rewMode
        self.rotMode = rotMode
        self.hand_init_pos = np.array(hand_init_pos)
        self.multitask = multitask
        self.multitask_num = multitask_num
        self._state_goal_idx = np.zeros(self.multitask_num)
        self.if_render = if_render
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
            np.hstack((obj_low, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))
        if not multitask:
            self.observation_space = Box(
                    np.hstack((self.hand_low, obj_low, obj_low)),
                    np.hstack((self.hand_high, obj_high, obj_high)),
            )
        else:
            self.observation_space = Box(
                    np.hstack((self.hand_low, obj_low, goal_low, np.zeros(multitask_num))),
                    np.hstack((self.hand_high, obj_high, goal_high, np.zeros(multitask_num))),
            )
        # self.observation_space = Dict([
        #     ('state_observation', self.hand_and_obj_space),
        #     ('state_desired_goal', self.goal_space),
        #     ('state_achieved_goal', self.goal_space),
        # ])
        self._get_reference()
        self.reset()


    def get_goal(self):
        return {
            'state_desired_goal': self._state_goal,
    }

    @property
    def model_name(self):     

        return get_asset_full_path('sawyer_xyz/sawyer_stack.xml')
        #return get_asset_full_path('sawyer_xyz/pickPlace_fox.xml')

    def viewer_setup(self):
        # top view
        # self.viewer.cam.trackbodyid = 0
        # self.viewer.cam.lookat[0] = 0
        # self.viewer.cam.lookat[1] = 1.0
        # self.viewer.cam.lookat[2] = 0.5
        # self.viewer.cam.distance = 0.6
        # self.viewer.cam.elevation = -45
        # self.viewer.cam.azimuth = 270
        # self.viewer.cam.trackbodyid = -1
        # side view
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.lookat[0] = 0.2
        self.viewer.cam.lookat[1] = 0.75
        self.viewer.cam.lookat[2] = 0.4
        self.viewer.cam.distance = 0.4
        self.viewer.cam.elevation = -55
        self.viewer.cam.azimuth = 180
        self.viewer.cam.trackbodyid = -1

    def step(self, action):
        if self.if_render:
            self.render()
        # self.set_xyz_action_rot(action[:7])
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
        self._set_goal_marker(self._state_goal)
        ob = self._get_obs()
        obs_dict = self._get_obs_dict()
        reward, reachRew, liftRew, placeRew, reachDist, placeDist= self.compute_reward(action, obs_dict, mode=self.rewMode)
        self.curr_path_length +=1
        #info = self._get_info()
        if self.curr_path_length == self.max_path_length:
            done = True
        else:
            done = False
        return ob, reward, done, {'reachDist': reachDist, 'pickRew':liftRew, 'epRew' : reward, 'goalDist': placeDist, 'success': float(placeDist <= 0.07)}
   
    def _get_obs(self):
        hand = self.get_endeff_pos()
        objPos =  self.data.get_geom_xpos('objGeom')
        flat_obs = np.concatenate((hand, objPos))
        if self.multitask:
            assert hasattr(self, '_state_goal_idx')
            return np.concatenate([
                    flat_obs,
                    self._state_goal,
                    self._state_goal_idx
                ])
        return np.concatenate([
                flat_obs,
                self._state_goal
            ])

    def _get_obs_dict(self):
        hand = self.get_endeff_pos()
        objPos =  self.data.get_geom_xpos('objGeom')
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
        self.data.site_xpos[self.model.site_name2id('goal_unstack')] = (
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
    

    def _set_obj_xyz_quat(self, pos, angle):
        quat = Quaternion(axis = [0,0,1], angle = angle).elements
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        qpos[9:12] = pos.copy()
        qpos[12:16] = quat.copy()
        qvel[9:15] = 0
        self.set_state(qpos, qvel)


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
        task_idx = np.random.randint(0, self.num_tasks)
        return self.tasks[task_idx]

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
        self.obj_init_pos = task['obj_init_pos']
        bottom_obj_pos = self.obj_init_pos + np.array([0., 0., -0.04])
        self.obj_init_angle = task['obj_init_angle']
        self.objHeight = self.data.get_geom_xpos('objGeom')[2]
        self.heightTarget = self.objHeight + self.liftThresh
        if self.random_init:
            goal_pos = np.random.uniform(
                self.obj_and_goal_space.low,
                self.obj_and_goal_space.high,
                size=(self.obj_and_goal_space.low.size),
            )
            while np.linalg.norm(goal_pos[:2] - goal_pos[-3:-1]) < 0.15:
                goal_pos = np.random.uniform(
                    self.obj_and_goal_space.low,
                    self.obj_and_goal_space.high,
                    size=(self.obj_and_goal_space.low.size),
                )
            self._state_goal = goal_pos[-3:]
            self.obj_init_pos = np.concatenate((goal_pos[:2], [self.obj_init_pos[-1]]))
            bottom_obj_pos = self.obj_init_pos + np.array([0., 0., -0.04])
        self._set_goal_marker(self._state_goal)
        self._set_goal_xyz(bottom_obj_pos)
        self._set_obj_xyz(self.obj_init_pos)
        #self._set_obj_xyz_quat(self.obj_init_pos, self.obj_init_angle)
        self.curr_path_length = 0
        self.maxPlacingDist = np.linalg.norm(np.array([self.obj_init_pos[0], self.obj_init_pos[1], self.heightTarget]) - np.array(self._state_goal)) + self.heightTarget
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
        self.pickCompleted = False

    def get_site_pos(self, siteName):
        _id = self.model.site_names.index(siteName)
        return self.data.site_xpos[_id].copy()

    def _get_reference(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        self.obj_body_id = self.sim.model.body_name2id("obj")
        self.goal_body_id = self.sim.model.body_name2id("goal")
        self.l_finger_geom_ids = self.sim.model.geom_name2id("leftclaw_it")
        self.r_finger_geom_ids = self.sim.model.geom_name2id("rightclaw_it")
        self.obj_geom_id = self.sim.model.geom_name2id("objGeom")
        self.goal_geom_id = self.sim.model.geom_name2id("goalGeom")

    def compute_rewards(self, actions, obsBatch, mode='dense'):
        assert isinstance(obsBatch, dict) == True
        obsList = obsBatch['state_observation']
        rewards = [self.compute_reward(action, obs, mode=mode)[0] for  action, obs in zip(actions, obsList)]
        return np.array(rewards)

    def compute_reward(self, action, obs, mode='dense'):
        """
        Helper function to return staged rewards based on current physical states.
        Returns:
            r_reach (float): reward for reaching and grasping
            r_lift (float): reward for lifting and aligning
            r_stack (float): reward for stacking
        """
        # reaching is successful when the gripper site is close to
        # the center of the cube
        table_height = 0.
        obj_pos = self.sim.data.body_xpos[self.obj_body_id]
        goal_pos = self._state_goal
        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2
        dist = np.linalg.norm(fingerCOM - obj_pos)
        # horiz_dist = np.linalg.norm(
        #         np.array(obj_pos[:2]) - np.array(goal_pos[:2])
        #     )
        # stack_dist = np.linalg.norm(
        #         np.array(obj_pos) - np.array(goal_pos)
        #     )
        place_dist = np.linalg.norm(
                np.array(obj_pos) - np.array(goal_pos)
            )
        # r_reach = (1 - np.tanh(10.0 * dist)) * 0.25
        r_reach = -dist
        # dist_xy = np.linalg.norm(np.concatenate((obj_pos[:-1], [self.init_fingerCOM[-1]])) - fingerCOM)
        # if dist_xy < 0.05: #0.02
        #     r_reach = -dist + 0.1
        #     if dist < 0.04:
        #         #incentive to close fingers when reachDist is small
        #         r_reach + max(0, action[-1])/50
        # else:
        #     r_reach =  -dist_xy
        #incentive to close fingers when reachDist is small
        # if dist < 0.05:
        #     r_reach = -dist + max(action[-1],0)/50

        # collision checking
        touch_left_finger = False
        touch_right_finger = False
        touch_obj_goal = False

        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]
            if c.geom1 == self.l_finger_geom_ids and c.geom2 == self.obj_geom_id:
                touch_left_finger = True
            if c.geom1 == self.obj_geom_id and c.geom2 == self.l_finger_geom_ids:
                touch_left_finger = True
            if c.geom1 == self.r_finger_geom_ids and c.geom2 == self.obj_geom_id:
                touch_right_finger = True
            if c.geom1 == self.obj_geom_id and c.geom2 == self.r_finger_geom_ids:
                touch_right_finger = True
            if c.geom1 == self.obj_geom_id and c.geom2 == self.goal_geom_id:
                touch_obj_goal = True
            if c.geom1 == self.goal_geom_id or c.geom2 == self.obj_geom_id:
                touch_obj_goal = True

        # additional grasping reward
        if touch_left_finger and touch_right_finger:
            r_reach += 0.25

        # lifting is successful when the cube is above the table top
        # by a margin
        obj_height = obj_pos[2]
        obj_lifted = obj_height > table_height + 0.08# and (touch_right_finger or touch_left_finger)
        # r_lift = 1.0 if obj_lifted and not touch_obj_goal else 0.0
        r_lift = 100.0 if obj_lifted and not touch_obj_goal else 0.0

        # Aligning is successful when obj is right above cubeB
        r_place = 0.
        if obj_lifted and not touch_obj_goal:
            # r_lift += 0.5 * (1 - np.tanh(horiz_dist))
            # r_lift += 0.5 * (1 - np.tanh(horiz_dist*5))
            # r_place += 3.0 * (1 - np.tanh(place_dist * 10.0))
            c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
            r_place = 1000*(self.maxPlacingDist - place_dist) + c1*(np.exp(-(place_dist**2)/c2) + np.exp(-(place_dist**2)/c3))

        # stacking is successful when the block is lifted and
        # the gripper is not holding the object
        # r_stack = 0
        # not_touching = not touch_left_finger and not touch_right_finger
        # if not_touching and r_lift > 0 and touch_obj_goal:
        #     r_stack = 2.0
        #     # r_stack = 4.0

        if mode == 'dense':
            # reward = max(r_reach, r_lift, r_place)
            reward = r_reach + r_lift + r_place
        else:
            reward = 1.0 if place_dist < 0.03 else 0.0
            

        return (reward, r_reach, r_lift, r_place, dist, place_dist)

    # def compute_rewards(self, actions, obsBatch):
    #     #Required by HER-TD3
    #     assert isinstance(obsBatch, dict) == True
    #     obsList = obsBatch['state_observation']
    #     rewards = [self.compute_reward(action, obs)[0] for  action, obs in zip(actions, obsList)]
    #     return np.array(rewards)

    # def compute_reward(self, actions, obs, mode = 'general'):
    #     if isinstance(obs, dict):
    #         obs = obs['state_observation']

    #     objPos = obs[3:6]

    #     rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
    #     fingerCOM  =  (rightFinger + leftFinger)/2

    #     heightTarget = self.heightTarget
    #     placingGoal = self._state_goal

    #     reachDist = np.linalg.norm(objPos - fingerCOM)

    #     placingDist = np.linalg.norm(objPos - placingGoal)
      

    #     def reachReward():
    #         reachRew = -reachDist# + min(actions[-1], -1)/50
    #         reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
    #         zRew = np.linalg.norm(fingerCOM[-1] - self.init_fingerCOM[-1])
    #         if reachDistxy < 0.05: #0.02
    #             reachRew = -reachDist
    #         else:
    #             reachRew =  -reachDistxy - 2*zRew
    #         #incentive to close fingers when reachDist is small
    #         if reachDist < 0.05:
    #             reachRew = -reachDist + max(actions[-1],0)/50
    #         return reachRew , reachDist

    #     def pickCompletionCriteria():
    #         tolerance = 0.01
    #         if objPos[2] >= (heightTarget- tolerance):
    #             return True
    #         else:
    #             return False

    #     if pickCompletionCriteria():
    #         self.pickCompleted = True


    #     def objDropped():
    #         return (objPos[2] < (self.objHeight + 0.005)) and (placingDist >0.02) and (reachDist > 0.02) 
    #         # Object on the ground, far away from the goal, and from the gripper
    #         #Can tweak the margin limits
       
    #     def objGrasped(thresh = 0):
    #         sensorData = self.data.sensordata
    #         return (sensorData[0]>thresh) and (sensorData[1]> thresh)

    #     def orig_pickReward():       
    #         # hScale = 50
    #         hScale = 100
    #         if self.pickCompleted and not(objDropped()):
    #             return hScale*heightTarget
    #         # elif (reachDist < 0.1) and (objPos[2]> (self.objHeight + 0.005)) :
    #         elif (reachDist < 0.1) and (objPos[2]> (self.objHeight + 0.005)) :
    #             return hScale* min(heightTarget, objPos[2])
    #         else:
    #             return 0

    #     def general_pickReward():
    #         hScale = 50
    #         if self.pickCompleted and objGrasped():
    #             return hScale*heightTarget
    #         elif objGrasped() and (objPos[2]> (self.objHeight + 0.005)):
    #             return hScale* min(heightTarget, objPos[2])
    #         else:
    #             return 0

    #     def placeReward():
    #         # c1 = 1000 ; c2 = 0.03 ; c3 = 0.003
    #         c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
    #         if mode == 'general':
    #             cond = self.pickCompleted and objGrasped()
    #         else:
    #             cond = self.pickCompleted and (reachDist < 0.1) and not(objDropped())
    #         if cond:
    #             placeRew = 1000*(self.maxPlacingDist - placingDist) + c1*(np.exp(-(placingDist**2)/c2) + np.exp(-(placingDist**2)/c3))
    #             placeRew = max(placeRew,0)
    #             return [placeRew , placingDist]
    #         else:
    #             return [0 , placingDist]

    #     reachRew, reachDist = reachReward()
    #     if mode == 'general':
    #         pickRew = general_pickReward()
    #     else:
    #         pickRew = orig_pickReward()
    #     placeRew , placingDist = placeReward()
    #     assert ((placeRew >=0) and (pickRew>=0))
    #     reward = reachRew + pickRew + placeRew
    #     return [reward, reachRew, reachDist, pickRew, placeRew, placingDist] 

    def get_diagnostics(self, paths, prefix=''):
        statistics = OrderedDict()
        return statistics

    def log_diagnostics(self, paths = None, logger = None):
        pass

if __name__ == '__main__':  
    import time 
    env = SawyerUnStack6DOFEnv(random_init=True)    
    for _ in range(1000):   
        env.reset()
        for _ in range(50):
            env.render()
            env.step(env.action_space.sample())
            # env.step(np.array([np.random.uniform(low=-1., high=1.), np.random.uniform(low=-1., high=1.), 0.]))   
            time.sleep(0.05)
