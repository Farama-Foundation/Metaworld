import numpy as np
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door_6dof import SawyerDoor6DOFEnv

class SawyerDoorClose6DOFEnv(SawyerDoor6DOFEnv):
    def __init__(
            self,
            hand_low=(-0.5, 0.40, 0.05),
            hand_high=(0.5, 1, 0.5),
            obj_low=(0., 0.85, 0.1),
            obj_high=(0.1, 0.95, 0.1),
            random_init=False,
            obs_type='plain',
            tasks = [{'goal': np.array([0.2, 0.8, 0.15]),  'obj_init_pos':np.array([0.1, 0.95, 0.1]), 'obj_init_angle': 0.3}],
            goal_low=None,
            goal_high=None,
            hand_init_pos = (0, 0.6, 0.2),
            rotMode='fixed',#'fixed',
            multitask=False,
            multitask_num=1,
            if_render=False,
            **kwargs
    ):
        SawyerDoor6DOFEnv.__init__(
        self,
        hand_low=hand_low,
        hand_high=hand_high,
        obj_low=obj_low,
        obj_high=obj_high,
        random_init=random_init,
        obs_type=obs_type,
        tasks = tasks, 
        goal_low=goal_low,
        goal_high=goal_high,
        hand_init_pos=hand_init_pos,
        rotMode=rotMode,#'fixed',
        multitask=multitask,
        multitask_num=multitask_num,
        if_render=if_render,
        **kwargs)

    def reset_model(self):
        self._reset_hand()
        task = self.sample_task()
        self._state_goal = np.array(task['goal'])
        self.obj_init_pos = task['obj_init_pos']
        # self.obj_init_angle = task['obj_init_angle']
        self.objHeight = self.data.get_geom_xpos('handle')[2]
        if self.random_init:
            obj_pos = np.random.uniform(
                self.obj_and_goal_space.low,
                self.obj_and_goal_space.high,
                size=(self.obj_and_goal_space.low.size),
            )
            self.obj_init_pos = obj_pos
            goal_pos = obj_pos.copy() + np.array([0.1, -0.15, 0.05])
            self._state_goal = goal_pos
        self._set_goal_marker(self._state_goal)
        #self._set_obj_xyz_quat(self.obj_init_pos, self.obj_init_angle)
        self.sim.model.body_pos[self.model.body_name2id('door')] = self.obj_init_pos
        self.sim.model.site_pos[self.model.site_name2id('goal')] = self._state_goal
        # keep the door open after resetting initial positions
        self._set_obj_xyz(-1.5708)
        self.curr_path_length = 0
        self.maxPullDist = np.linalg.norm(self.data.get_geom_xpos('handle')[:-1] - self._state_goal[:-1])
        self.target_reward = 1000*self.maxPullDist + 1000*2
        #Can try changing this
        return self._get_obs()

    def compute_reward(self, actions, obs):
        if isinstance(obs, dict): 
            obs = obs['state_observation']

        objPos = obs[3:6]

        rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
        fingerCOM  =  (rightFinger + leftFinger)/2

        pullGoal = self._state_goal

        pullDist = np.linalg.norm(objPos[:-1] - pullGoal[:-1])
        reachDist = np.linalg.norm(objPos - fingerCOM)
        # reachDistxy = np.linalg.norm(objPos[:-1] - fingerCOM[:-1])
        # zDist = np.linalg.norm(fingerCOM[-1] - self.init_fingerCOM[-1])
        # if reachDistxy < 0.05: #0.02
        #     reachRew = -reachDist
        # else:
        #     reachRew =  -reachDistxy - zDisthand
        reachRew = -reachDist

        def reachCompleted():
            if reachDist < 0.05:
                return True
            else:
                return False

        if reachCompleted():
            self.reachCompleted = True
        else:
            self.reachCompleted = False

        def pullReward():
            c1 = 1000 ; c2 = 0.01 ; c3 = 0.001
            # c1 = 10 ; c2 = 0.01 ; c3 = 0.001
            if self.reachCompleted:
                pullRew = 1000*(self.maxPullDist - pullDist) + c1*(np.exp(-(pullDist**2)/c2) + np.exp(-(pullDist**2)/c3))
                pullRew = max(pullRew,0)
                return pullRew
            else:
                return 0
            # pullRew = 1000*(self.maxPullDist - pullDist) + c1*(np.exp(-(pullDist**2)/c2) + np.exp(-(pullDist**2)/c3))
            # pullRew = max(pullRew,0)
            # return pullRew
        # pullRew = -pullDist
        pullRew = pullReward()
        reward = reachRew + pullRew# - actions[-1]/50
        # reward = pullRew# - actions[-1]/50
      
        return [reward, reachDist, pullDist]

