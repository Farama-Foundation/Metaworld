"""
Use this script to control the env with your keyboard.
For this script to work, you need to have the PyGame window in focus.

See/modify `char_to_action` to set the key-to-action mapping.
"""

import numpy as np
# from metaworld.envs.mujoco.sawyer_xyz.sawyer_door_hook import SawyerDoorHookEnv

# from metaworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import \
    # SawyerPickAndPlaceEnv
# from metaworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env_two_pucks import (
    # SawyerPushAndReachXYDoublePuckEnv,
    # SawyerPushAndReachXYZDoublePuckEnv,
# )

# from metaworld.envs.mujoco.sawyer_xyz.sawyer_stack import SawyerStackEnv
# from metaworld.envs.mujoco.sawyer_xyz.sawyer_laptop_close import SawyerLaptopCloseEnv
# from metaworld.envs.mujoco.sawyer_xyz.sawyer_box_open import SawyerBoxOpenEnv
# from metaworld.envs.mujoco.sawyer_xyz.sawyer_peg_insertion_topdown import SawyerPegInsertionTopdownEnv


from robosuite.devices import SpaceMouse

space_mouse = SpaceMouse()
env = SawyerPegInsertionTopdownEnv()
NDIM = env.action_space.low.size
lock_action = False
obs = env.reset()
action = np.zeros(10)
closed = False

while True:
    done = False
    env.render()

    state = space_mouse.get_controller_state()
    dpos, rotation, grasp, reset = (
        state["dpos"],
        state["rotation"],
        state["grasp"],
        state["reset"],
    )

    # convert into a suitable end effector action for the environment
    # current = env.get_mocap_quat()

    # desired_quat = mat2quat(rotation)
    # current_z = quat_to_zangle(current)
    # desired_z = quat_to_zangle(desired_quat)

    # # drotation = current.T.dot(rotation)  # relative rotation of desired from current
    # # dquat = T.mat2quat(drotation)

    # print('current', current_z)
    # print('desired', desired_z)


    # print('diff unclipped', desired_z - current_z)
    # diff = desired_z - current_z
    # print('diff', diff)

    gripper = grasp
    if gripper == 1:
        closed = not closed

    obs, reward, done, _ = env.step(np.hstack([dpos/.005, 0, closed]))
    # print(obs)

    # if done:
    #     obs = env.reset()
