"""
Use this script to control the env with your keyboard.
For this script to work, you need to have the PyGame window in focus.

See/modify `char_to_action` to set the key-to-action mapping.
"""
import sys
import gym

import numpy as np
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door_hook import SawyerDoorHookEnv

from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import \
    SawyerPickAndPlaceEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import \
    SawyerPushAndReachXYEnv, SawyerPushAndReachXYZEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env_two_pucks import (
    SawyerPushAndReachXYDoublePuckEnv,
    SawyerPushAndReachXYZDoublePuckEnv,
)

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


import gym
import multiworld
import pygame
# env = gym.make('SawyerPushAndReachEnvEasy-v0')
# env = SawyerPushAndReachXYEnv(
#     goal_low=(-0.15, 0.4, 0.02, -.1, .5),
#     goal_high=(0.15, 0.75, 0.02, .1, .7),
#     puck_low=(-.3, .25),
#     puck_high=(.3, .9),
#     hand_low=(-0.15, 0.4, 0.05),
#     hand_high=(0.15, .75, 0.3),
#     norm_order=2,
#     xml_path='sawyer_xyz/sawyer_push_puck_small_arena.xml',
#     reward_type='state_distance',
#     reset_free=False,
# )
env = SawyerReachXYEnv()
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
            elif new_action == 'put obj in hand':
                print("putting obj in hand")
                env.put_obj_in_hand()
                action[3] = 1
            elif new_action is not None:
                action[:3] = new_action[:3]
            else:
                action = np.zeros(3)
    env.step(action[:2])
    if done:
        obs = env.reset()
    env.render()
