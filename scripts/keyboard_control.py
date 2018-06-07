import sys

import numpy as np

from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import \
    SawyerPushAndReachXYEnv

import pygame
from pygame.locals import QUIT, KEYDOWN

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
    'x': 'toggle',
    'r': 'reset',
}


env = SawyerPushAndReachXYEnv()
lock_action = False
obs = env.reset()
action = np.array([0, 0, 0, 0])
while True:
    done = False
    if not lock_action:
        action = np.array([0, 0, 0, 0])
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
            elif new_action is not None:
                action = new_action / 10
            else:
                action = np.array([0, 0, 0, 0])
            print("got char:", char)
            print("action", action)
            print("angles", env.data.qpos.copy())
    obs, reward, _, info = env.step(action)

    env.render()
    if done:
        break
