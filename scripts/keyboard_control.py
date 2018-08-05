"""
Use this script to control the env with your keyboard.
For this script to work, you need to have the PyGame window in focus.

See/modify `char_to_action` to set the key-to-action mapping.
"""
import sys

import numpy as np

from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import \
    SawyerPickAndPlaceEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import \
    SawyerPushAndReachXYEnv, SawyerPushAndReachXYZEnv

import pygame
from pygame.locals import QUIT, KEYDOWN

from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach import SawyerReachXYEnv, \
    SawyerReachXYZEnv

pygame.init()
screen = pygame.display.set_mode((400, 300))


char_to_action = {
    'w': np.array([0, -1, 0, 0]),
    'a': np.array([1, 0, 0, 0]),
    's': np.array([0, 1, 0, 0]),
    'd': np.array([-1, 0, 0, 0]),
    'q': np.array([1, -1, 0, 0]),
    'e': np.array([-1, -1, 0, 0]),
    'z': np.array([1, 1, 0, 0]),
    'c': np.array([-1, 1, 0, 0]),
    'k': np.array([0, 0, 1, 0]),
    'j': np.array([0, 0, -1, 0]),
    'h': 'close',
    'l': 'open',
    'x': 'toggle',
    'r': 'reset',
    'p': 'put obj in hand',
}


env = SawyerPushAndReachXYEnv(
            hand_low=(-0.275, 0.3, 0.05),
            hand_high=(0.275, 0.9, 0.3),
            puck_low=(-.31, .33),
            puck_high=(.31, 85),
            goal_low=(-0.25, 0.3, 0.05, -.22, .38),
            goal_high=(0.26, 0.875, 0.3, .22, .8)
)
# env = SawyerPushAndReachXYZEnv()
# env = SawyerReachXYEnv()
# env = SawyerReachXYZEnv()
# env = SawyerPickAndPlaceEnv()
NDIM = env.action_space.low.size
lock_action = False
obs = env.reset()
action = np.zeros(10)
while True:
    done = False
    if not lock_action:
        action[:3] = 0
    for event in pygame.event.get():
        event_happened = True
        if event.type == QUIT:
            sys.exit()
        if event.type == KEYDOWN:
            char = event.dict['key']
            new_action = char_to_action.get(chr(char), None)
            if new_action == 'toggle':
                lock_action = not lock_action
            elif new_action == 'reset':
                done = True
            elif new_action == 'close':
                action[3] = 1
            elif new_action == 'open':
                action[3] = -1
            # elif new_action == 'put obj in hand':
            #     print("putting obj in hand")
            #     env.put_obj_in_hand()
            #     action[3] = 1
            elif new_action is not None:
                action[:3] = new_action[:3]
            else:
                action = np.zeros(10)
    obs, reward, _, info = env.step(action[:NDIM])
    print('Puck_pos ', env.get_puck_pos())
    print('Hand_pos ', env.get_endeff_pos())
    env.render()
    if done:
        obs = env.reset()
