
import numpy as np

from gym import GoalEnv, spaces
from gym.utils import seeding

import sys
from six import StringIO, b
import copy
import gym
from gym import utils
from gym.envs.toy_text import discrete
ACTIONS = [(-1, 0), (0, 1), (1, 0), (0, -1),(0,0)]
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3
STAY = 4
RENDER_DIR = '/home/coline/Desktop/renderings/'


class GoalGridworld(GoalEnv):
    """A goal-based environment. It functions just as any regular OpenAI Gym environment but it
    imposes a required structure on the observation_space. More concretely, the observation
    space is required to contain at least three elements, namely `observation`, `desired_goal`, and
    `achieved_goal`. Here, `desired_goal` specifies the goal that the agent should attempt to achieve.
    `achieved_goal` is the goal that it currently achieved instead. `observation` contains the
    actual observations of the environment as per usual.
    """




        
    def __init__(self,  size=[10,10], concatenated=False):
        self.nrow, self.ncol = size
        self.concatenated = concatenated
        #print("hellos")
        self.ACTIONS = ACTIONS
        nA = len(self.ACTIONS)
        nS = self.nrow * self.ncol
        self.nS = nS
        self.nA = nA
        self.lastaction=None
        self.action_space = spaces.Discrete(self.nA)
        observation_space = spaces.Box(low=0, high=1., shape=((self.nrow), self.ncol, 1))
        self.observation_space = gym.spaces.Dict({'observation': observation_space, 
                                                 'achieved_goal': observation_space,
                                                 'desired_goal': observation_space}
                                                )
        if self.concatenated:
            self.observation_space = spaces.Box(low=0, high=1., shape=((self.nrow*2), self.ncol, 1))
        self.res = 1
        self.renderres = 10
        #self.seed()
        #self.reset()

    def from_s(self, s):
        # todo: test this.
        row = int(s/self.ncol)
        return (row,s- row*self.ncol)

    def to_s(self, row, col):
        return row*self.ncol + col

    def reset(self):
        #['observation', 'achieved_goal', 'desired_goal']

        self.state = {}
        self.state['count'] = 0
        positions = np.random.permutation(self.nS)
        #positions2 = np.random.permutation([0, self.ncol-1, (self.nrow-1)*(self.ncol-1)-1,  (self.nrow-1)*(self.ncol)-1])[:len(objects)+1]
        agent_pos =  self.from_s(positions[0])
        goal_pos = self.from_s(positions[1])
        self.state['agent'] = agent_pos
        self.goal_state = {'agent': goal_pos}
        self.lastaction=None
        self.init_state = copy.deepcopy(self.state)
        obs = self.get_obs().flatten()
        goal_obs = self.imagine_obs(self.goal_state).flatten()
        #print("goal", self.goal)
        #print("obs", obs.shape)
        self.episode = np.random.randint(20)
        if self.concatenated:
            return np.concatenate((obs, goal_obs)).flatten()
        return {'observation': obs, 'desired_goal': goal_obs, 'achieved_goal': obs}

    def move_agent(self, a):
        act = ACTIONS[a]
        pos = self.state['agent']
        row, col = pos[0]+act[0], pos[1]+act[1]
        #print("action", a, "pos",self.state['agent'],"new pos", (row, col) )

        #Check bounds
        if row in range(self.nrow) and col in range(self.ncol):
            is_blocked = False
            #print("moved!")
            self.state['agent']= (row, col)
            return (row, col)
        else:
            is_blocked = True
            return pos

    def step(self, a):
        prev_state = copy.deepcopy(self.state)
        self.state['count'] +=1
        r = 0
        d = False
        #print(a)
        new_pos = self.move_agent(a)
        
        self.lastaction=a
        success = 0
        if self.state['count'] > 40:
            d = True
        #print(self.state['object_counts']['bread'])
        #success = self.reward_function(self.init_state, self.state)>0
        obs = self.get_obs().flatten()
        goal_obs = self.imagine_obs(self.goal_state).flatten()
        r = self.compute_reward(obs, goal_obs, None)
        if self.concatenated:
            return (np.concatenate((obs, goal_obs)).flatten(), r, d, 
                    {'success': success, 'count': self.state['count'], 'done':d})
        return ({'observation': obs, 'desired_goal': goal_obs, 'achieved_goal': obs}, 
                r, d, {'success': success, 'count': self.state['count'], 'done':d})


    def get_obs(self, mode='rgb'):
        if mode == 'rgb':
            img = np.zeros(((self.nrow)*self.res, self.ncol*self.res, 1))
            
            row, col = self.state['agent']
            img[row*self.res:(row+1)*self.res, col*self.res:(col+1)*self.res, :] += 1
            for x in [-1,1]:
                for y in [-1,1]:
                    if row+x in range(self.nrow) and col+y in range(self.ncol):
                        img[row+x:(row+x+1), col+y:(col+y+1), :] += 0.2
            return img.flatten()
        
    def imagine_obs(self, state, mode='rgb'):
        if mode == 'rgb':
            img = np.zeros(((self.nrow)*self.res, self.ncol*self.res, 1))
            
            row, col = state['agent']
            
            for x in [-1,0,1]:
                for y in [-1,0,1]:
                    if row+x in range(self.nrow) and col+y in range(self.ncol):
                        img[row+x:(row+x+1), col+y:(col+y+1), :] += 0.2
            img[row*self.res:(row+1)*self.res, col*self.res:(col+1)*self.res, :] = 1
            return img.flatten()

#     def get_diagnostics(self,paths, **kwargs):
#         successes = [p['env_infos'][-1]['success'] for p in paths]
#         success_rate = sum(successes)/len(successes)
#         lengths = [p['env_infos'][-1]['count'] for p in paths]
#         length_rate = sum(lengths)/len(lengths)
#         return {'SuccessRate': success_rate, 'PathLengthMean': length_rate, 'PathLengthMin':min(lengths)}


    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on an a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in info and compute it accordingly.
        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information
        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['goal'], info)
        """
        error = np.sum(np.square(achieved_goal - desired_goal))
        return -error
        
        
    def render(self, mode='rgb'):
        import cv2
        #res = self.res
        self.renderres = 10
        img = np.zeros(((self.nrow)*self.renderres, self.ncol*self.renderres, 3))
            
        row, col = self.state['agent']
        #print("row, col", row, col)
        for x in [-1,0,1]:
            for y in [-1,0,1]:
                if row+x in range(self.nrow) and col+y in range(self.ncol):
                    img[(row+x)*self.renderres:(row+x+1)*self.renderres, (col+y)*self.renderres:(col+y+1)*self.renderres, 1] += 0.2
        img[row*self.renderres:(row+1)*self.renderres, col*self.renderres:(col+1)*self.renderres, 1] = 1


        w,h,c = img.shape
        #print("hunger",self.state['hunger'] )
        row, col = self.goal_state['agent']
        img[row*self.renderres:(row+1)*self.renderres, col*self.renderres:(col+1)*self.renderres, 2] = 1

        cv2.imwrite(RENDER_DIR+'img{:04d}_{:04d}.png'.format(self.episode, self.state['count']), img*255)
