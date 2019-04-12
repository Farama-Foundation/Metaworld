"""
Use this script to control the env with your keyboard.
For this script to work, you need to have the PyGame window in focus.

See/modify `char_to_action` to set the key-to-action mapping.
"""
import sys
import gym

import numpy as np
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door_hook import SawyerDoorHookEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door import SawyerDoorEnv

from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import \
    SawyerPickAndPlaceEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env_two_pucks import (
    SawyerPushAndReachXYDoublePuckEnv,
    SawyerPushAndReachXYZDoublePuckEnv,
)

from multiworld.envs.mujoco.sawyer_xyz.sawyer_stack_6dof import SawyerStack6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_dial_turn_6dof import SawyerDialTurn6DOFEnv


from robosuite.devices import SpaceMouse
from multiworld.envs.mujoco.utils import rotation
from robosuite.utils.transform_utils import rotation_matrix



import gym
import multiworld

space_mouse = SpaceMouse()
env = SawyerDialTurn6DOFEnv(rotMode = 'rotz')
NDIM = env.action_space.low.size
lock_action = False
obs = env.reset()
action = np.zeros(10)
while True:
    done = False
    env.render()
    # print(space_mouse.control)
    print(space_mouse.get_controller_state())
    env.step(np.hstack([space_mouse.control, space_mouse.control_gripper]))

    if done:
        obs = env.reset()
