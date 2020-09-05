#!/usr/bin/env python3

import time
import glfw
import numpy as np
import argparse

from metaworld.envs.mujoco.sawyer_xyz.v1.sawyer_assembly_peg import SawyerNutAssemblyEnv
from metaworld.envs.mujoco.sawyer_xyz.v1.sawyer_bin_picking import SawyerBinPickingEnv
from metaworld.envs.mujoco.sawyer_xyz.v1.sawyer_box_close import SawyerBoxCloseEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_box_open import SawyerBoxOpenEnv
from metaworld.envs.mujoco.sawyer_xyz.v1.sawyer_button_press import SawyerButtonPressEnv
from metaworld.envs.mujoco.sawyer_xyz.v1.sawyer_button_press_topdown import SawyerButtonPressTopdownEnv
from metaworld.envs.mujoco.sawyer_xyz.v1.sawyer_dial_turn import SawyerDialTurnEnv
from metaworld.envs.mujoco.sawyer_xyz.v1.sawyer_door import SawyerDoorEnv
from metaworld.envs.mujoco.sawyer_xyz.v1.sawyer_door_close import SawyerDoorCloseEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_door_hook import SawyerDoorHookEnv
from metaworld.envs.mujoco.sawyer_xyz.v1.sawyer_drawer_close import SawyerDrawerCloseEnv
from metaworld.envs.mujoco.sawyer_xyz.v1.sawyer_drawer_open import SawyerDrawerOpenEnv
from metaworld.envs.mujoco.sawyer_xyz.v1.sawyer_hammer import SawyerHammerEnv
from metaworld.envs.mujoco.sawyer_xyz.v1.sawyer_hand_insert import SawyerHandInsertEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_laptop_close import SawyerLaptopCloseEnv
from metaworld.envs.mujoco.sawyer_xyz.v1.sawyer_lever_pull import SawyerLeverPullEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_multiple_objects import MultiSawyerEnv
from metaworld.envs.mujoco.sawyer_xyz.v1.sawyer_peg_insertion_side import SawyerPegInsertionSideEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import SawyerPickAndPlaceEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place_wsg import SawyerPickAndPlaceWsgEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import SawyerPushAndReachXYEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env_two_pucks import SawyerPushAndReachXYZDoublePuckEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj import SawyerTwoObjectEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_push_nips import SawyerPushAndReachXYEasyEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_reach import SawyerReachXYZEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_reach import SawyerReachEnv
from metaworld.envs.mujoco.sawyer_xyz.v1.sawyer_reach_push_pick_place import SawyerReachPushPickPlaceEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_rope import SawyerRopeEnv
from metaworld.envs.mujoco.sawyer_xyz.v1.sawyer_shelf_place import SawyerShelfPlaceEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_stack import SawyerStackEnv
from metaworld.envs.mujoco.sawyer_xyz.v1.sawyer_stick_pull import SawyerStickPullEnv
from metaworld.envs.mujoco.sawyer_xyz.v1.sawyer_stick_push import SawyerStickPushEnv
from metaworld.envs.mujoco.sawyer_xyz.v1.sawyer_sweep import SawyerSweepEnv
from metaworld.envs.mujoco.sawyer_xyz.v1.sawyer_sweep_into_goal import SawyerSweepIntoGoalEnv
from metaworld.envs.mujoco.sawyer_xyz.sawyer_throw import SawyerThrowEnv
from metaworld.envs.mujoco.sawyer_xyz.v1.sawyer_window_close import SawyerWindowCloseEnv
from metaworld.envs.mujoco.sawyer_xyz.v1.sawyer_window_open import SawyerWindowOpenEnv


# function that closes the render window
def close(env):
    if env.viewer is not None:
        # self.viewer.finish()
        glfw.destroy_window(env.viewer.window)
    env.viewer = None


def sample_sawyer_assembly_peg():
    env = SawyerNutAssemblyEnv()
    for _ in range(1):
        env.reset()
        for _ in range(50):
            env.render()
            env.step(env.action_space.sample())
            # env.step(np.array([np.random.uniform(low=-1., high=1.), np.random.uniform(low=-1., high=1.), 0.]))
            time.sleep(0.05)
    close(env)


def sample_sawyer_bin_picking():
    env = SawyerBinPickingEnv()
    for _ in range(1):
        env.reset()
        for _ in range(50):
            env.render()
            env.step(env.action_space.sample())
            # env.step(np.array([np.random.uniform(low=-1., high=1.), np.random.uniform(low=-1., high=1.), 0.]))
            time.sleep(0.05)
    close(env)


def sample_sawyer_box_close():
    env = SawyerBoxCloseEnv()
    for _ in range(1):
        env.reset()
        # for _ in range(10):
        #     env.data.set_mocap_pos('mocap', np.array([0, 0.8, 0.05]))
        #     env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        #     env.do_simulation([-1,1], env.frame_skip)
        #     #self.do_simulation(None, self.frame_skip)
        for _ in range(10):
            env.data.set_mocap_pos('mocap', np.array([0, 0.8, 0.25]))
            env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            env.do_simulation([-1, 1], env.frame_skip)
            # self.do_simulation(None, self.frame_skip)
        for _ in range(100):
            env.render()
            # env.step(env.action_space.sample())
            # env.step(np.array([0, -1, 0, 0, 0]))
            if _ < 10:
                env.step(np.array([0, 0, -1, 0, 0]))
            elif _ < 50:
                env.step(np.array([0, 0, 0, 0, 1]))
            else:
                env.step(np.array([0, 0, 1, 0, 1]))
            # env.step(np.array([np.random.uniform(low=-1., high=1.), np.random.uniform(low=-1., high=1.), 0.]))
            time.sleep(0.05)
    close(env)


def sample_sawyer_box_open():
    env = SawyerBoxOpenEnv()
    for _ in range(1):
        env.reset()
        # for _ in range(10):
        #     env.data.set_mocap_pos('mocap', np.array([0, 0.8, 0.05]))
        #     env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        #     env.do_simulation([-1,1], env.frame_skip)
        #     #self.do_simulation(None, self.frame_skip)
        for _ in range(10):
            env.data.set_mocap_pos('mocap', np.array([0, 0.8, 0.25]))
            # env.data.set_mocap_pos('mocap', np.array([0, 0.6, 0.25]))
            env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            env.do_simulation([-1, 1], env.frame_skip)
            # self.do_simulation(None, self.frame_skip)
        for _ in range(100):
            env.render()
            if _ < 10:
                env.step(np.array([0, 0, -1, 0, 0]))
            elif _ < 50:
                env.step(np.array([0, 0, 0, 0, 1]))
            else:
                env.step(np.array([0, 0, 1, 0, 1]))
                # env.step(np.array([0, 1, 0, 0, 0]))
            # env.step(np.array([np.random.uniform(low=-1., high=1.), np.random.uniform(low=-1., high=1.), 0.]))
            time.sleep(0.05)
    close(env)


def sample_sawyer_button_press_6d0f():
    env = SawyerButtonPressEnv()
    for _ in range(1):
        env.reset()
        # for _ in range(10):
        #     env.data.set_mocap_pos('mocap', np.array([0, 0.8, 0.05]))
        #     env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        #     env.do_simulation([-1,1], env.frame_skip)
        #     #self.do_simulation(None, self.frame_skip)
        # for _ in range(10):
        #     env.data.set_mocap_pos('mocap', np.array([0, 0.8, 0.25]))
        #     # env.data.set_mocap_pos('mocap', np.array([0, 0.6, 0.25]))
        #     env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        #     env.do_simulation([-1,1], env.frame_skip)
        #     #self.do_simulation(None, self.frame_skip)
        for _ in range(100):
            print(env.data.site_xpos[env.model.site_name2id('buttonStart')])
            env.render()
            # env.step(env.action_space.sample())
            # if _ < 10:
            #     env.step(np.array([0, 0, -1, 0, 0]))
            # elif _ < 50:
            #     env.step(np.array([0, 0, 0, 0, 1]))
            # env.step(np.array([0, 1, 0, 0, 1]))
            # env.step(np.array([0, 1, 0, 0, 0]))
            env.step(np.array([0, 1, 0, 0, 1]))
            # env.step(np.array([np.random.uniform(low=-1., high=1.), np.random.uniform(low=-1., high=1.), 0.]))
            time.sleep(0.05)
    close(env)


def sample_sawyer_button_press_topdown_6d0f():
    env = SawyerButtonPressTopdownEnv()
    for _ in range(1):
        env.reset()
        # for _ in range(10):
        #     env.data.set_mocap_pos('mocap', np.array([0, 0.8, 0.05]))
        #     env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        #     env.do_simulation([-1,1], env.frame_skip)
        #     #self.do_simulation(None, self.frame_skip)
        # for _ in range(10):
        #     env.data.set_mocap_pos('mocap', np.array([0, 0.8, 0.25]))
        #     # env.data.set_mocap_pos('mocap', np.array([0, 0.6, 0.25]))
        #     env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        #     env.do_simulation([-1,1], env.frame_skip)
        #     #self.do_simulation(None, self.frame_skip)
        for _ in range(100):
            print(env.data.site_xpos[env.model.site_name2id('buttonStart')])
            env.render()
            # env.step(env.action_space.sample())
            # if _ < 10:
            #     env.step(np.array([0, 0, -1, 0, 0]))
            # elif _ < 50:
            #     env.step(np.array([0, 0, 0, 0, 1]))
            # env.step(np.array([0, 1, 0, 0, 1]))
            # env.step(np.array([0, 1, 0, 0, 0]))
            env.step(np.array([0, 0, -1, 0, 1]))
            # env.step(np.array([np.random.uniform(low=-1., high=1.), np.random.uniform(low=-1., high=1.), 0.]))
            time.sleep(0.05)
    close(env)


def sample_sawyer_dial_turn():
    env = SawyerDialTurnEnv()
    for _ in range(1):
        env.reset()
        # for _ in range(10):
        #     env.data.set_mocap_pos('mocap', np.array([0, 0.8, 0.05]))
        #     env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        #     env.do_simulation([-1,1], env.frame_skip)
        #     #self.do_simulation(None, self.frame_skip)
        # for _ in range(10):
        #     env.data.set_mocap_pos('mocap', np.array([0, 0.8, 0.25]))
        #     # env.data.set_mocap_pos('mocap', np.array([0, 0.6, 0.25]))
        #     env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        #     env.do_simulation([-1,1], env.frame_skip)
        #     #self.do_simulation(None, self.frame_skip)
        for _ in range(100):
            print(env.data.site_xpos[env.model.site_name2id('dialStart')])
            env.render()
            # env.step(env.action_space.sample())
            # if _ < 10:
            #     env.step(np.array([0, 0, -1, 0, 0]))
            # elif _ < 50:
            #     env.step(np.array([0, 0, 0, 0, 1]))
            # env.step(np.array([0, 1, 0, 0, 1]))
            # env.step(np.array([0, 1, 0, 0, 0]))
            env.step(np.array([0, 0, -1, 0, 1]))
            # env.step(np.array([np.random.uniform(low=-1., high=1.), np.random.uniform(low=-1., high=1.), 0.]))
            time.sleep(0.05)
    close(env)


def sample_sawyer_door():
    env = SawyerDoorEnv()
    for _ in range(100):
        env.render()
        action = env.action_space.sample()
        env.step(action)
        time.sleep(0.05)
    close(env)


def sample_sawyer_door_close():
    env = SawyerDoorCloseEnv()
    for _ in range(100):
        env.render()
        action = env.action_space.sample()
        env.step(action)
        time.sleep(0.05)
    close(env)


def sample_sawyer_door_hook():
    env = SawyerDoorHookEnv()
    for _ in range(100):
        env.render()
        action = env.action_space.sample()
        env.step(action)
        time.sleep(0.05)
    close(env)


def sample_sawyer_door():
    env = SawyerDoorEnv()
    for _ in range(100):
        env.render()
        action = env.action_space.sample()
        env.step(action)
        time.sleep(0.05)
    close(env)


def sample_sawyer_drawer_close():
    env = SawyerDrawerCloseEnv()
    for _ in range(1):
        env.reset()
        # for _ in range(10):
        #     env.data.set_mocap_pos('mocap', np.array([0, 0.8, 0.05]))
        #     env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        #     env.do_simulation([-1,1], env.frame_skip)
        #     #self.do_simulation(None, self.frame_skip)
        env._set_obj_xyz(np.array([-0.2, 0.8, 0.05]))
        for _ in range(10):
            env.data.set_mocap_pos('mocap', np.array([0, 0.5, 0.05]))
            env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            env.do_simulation([-1, 1], env.frame_skip)
            # self.do_simulation(None, self.frame_skip)
        for _ in range(50):
            env.render()
            # env.step(env.action_space.sample())
            # env.step(np.array([0, -1, 0, 0, 0]))
            env.step(np.array([0, 1, 0, 0, 0]))
            # env.step(np.array([np.random.uniform(low=-1., high=1.), np.random.uniform(low=-1., high=1.), 0.]))
            time.sleep(0.05)
    close(env)


def sample_sawyer_drawer_open():
    env = SawyerDrawerOpenEnv()
    for _ in range(1):
        env.reset()
        # for _ in range(10):
        #     env.data.set_mocap_pos('mocap', np.array([0, 0.8, 0.05]))
        #     env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        #     env.do_simulation([-1,1], env.frame_skip)
        #     #self.do_simulation(None, self.frame_skip)
        env._set_obj_xyz(np.array([-0.2, 0.8, 0.05]))
        for _ in range(10):
            env.data.set_mocap_pos('mocap', np.array([0, 0.5, 0.05]))
            env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
            env.do_simulation([-1, 1], env.frame_skip)
        # self.do_simulation(None, self.frame_skip)
        for _ in range(50):
            env.render()
            # env.step(env.action_space.sample())
            # env.step(np.array([0, -1, 0, 0, 0]))
            env.step(np.array([0, 1, 0, 0, 0]))
            # env.step(np.array([np.random.uniform(low=-1., high=1.), np.random.uniform(low=-1., high=1.), 0.]))
            time.sleep(0.05)
    close(env)


def sample_sawyer_hammer():
    env = SawyerHammerEnv()
    for _ in range(1):
        env.reset()
        # for _ in range(10):
        #     env.data.set_mocap_pos('mocap', np.array([0, 0.8, 0.05]))
        #     env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        #     env.do_simulation([-1,1], env.frame_skip)
        #     #self.do_simulation(None, self.frame_skip)
        # for _ in range(10):
        #     env.data.set_mocap_pos('mocap', np.array([0, 0.8, 0.25]))
        #     # env.data.set_mocap_pos('mocap', np.array([0, 0.6, 0.25]))
        #     env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        #     env.do_simulation([-1,1], env.frame_skip)
        #     #self.do_simulation(None, self.frame_skip)
        for _ in range(100):
            env.render()
            # env.step(env.action_space.sample())
            # if _ < 10:
            #     env.step(np.array([0, 0, -1, 0, 0]))
            # elif _ < 50:
            #     env.step(np.array([0, 0, 0, 0, 1]))
            if _ < 10:
                env.step(np.array([0, 0, -1, 0, 0]))
            else:
                env.step(np.array([0, 1, 0, 0, 1]))
                # env.step(np.array([0, 1, 0, 0, 0]))
            # env.step(np.array([np.random.uniform(low=-1., high=1.), np.random.uniform(low=-1., high=1.), 0.]))
            time.sleep(0.05)
    close(env)


def sample_sawyer_hand_insert():
    env = SawyerHandInsertEnv(fix_goal=True)
    for i in range(100):
        if i % 100 == 0:
            env.reset()
        env.step(np.array([0, 1, 1]))
        env.render()
    close(env)


def sample_sawyer_laptop_close():
    env = SawyerLaptopCloseEnv()
    for _ in range(1):
        env.reset()
        # for _ in range(10):
        #     env.data.set_mocap_pos('mocap', np.array([0, 0.9, 0.22]))
        #     env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        #     # env.do_simulation([-1,1], env.frame_skip)
        #     env.do_simulation([1,-1], env.frame_skip)
        # env._set_obj_xyz(np.array([-0.2, 0.8, 0.05]))
        # for _ in range(10):
        #     env.data.set_mocap_pos('mocap', np.array([0, 0.5, 0.05]))
        #     env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        #     env.do_simulation([-1,1], env.frame_skip)
        #     #self.do_simulation(None, self.frame_skip)
        for _ in range(100):
            env.render()
            # env.step(env.action_space.sample())
            # env.step(np.array([0, -1, 0, 0, 1]))
            env.step(np.array([0, 0, 0, 0, 1]))
            print(env.get_laptop_angle())
            # env.step(np.array([0, 1, 0, 0, 0]))
            # env.step(np.array([np.random.uniform(low=-1., high=1.), np.random.uniform(low=-1., high=1.), 0.]))
            time.sleep(0.05)
    close(env)


def sample_sawyer_lever_pull():
    env = SawyerLeverPullEnv()
    for _ in range(1):
        env.reset()
        # for _ in range(10):
        #     env.data.set_mocap_pos('mocap', np.array([0, 0.8, 0.05]))
        #     env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        #     env.do_simulation([-1,1], env.frame_skip)
        #     #self.do_simulation(None, self.frame_skip)
        # for _ in range(10):
        #     env.data.set_mocap_pos('mocap', np.array([0, 0.8, 0.25]))
        #     # env.data.set_mocap_pos('mocap', np.array([0, 0.6, 0.25]))
        #     env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        #     env.do_simulation([-1,1], env.frame_skip)
        #     #self.do_simulation(None, self.frame_skip)
        for _ in range(100):
            print(env.data.site_xpos[env.model.site_name2id('basesite')])
            env.render()
            # env.step(env.action_space.sample())
            # if _ < 10:
            #     env.step(np.array([0, 0, -1, 0, 0]))
            # elif _ < 50:
            #     env.step(np.array([0, 0, 0, 0, 1]))
            # env.step(np.array([0, 1, 0, 0, 1]))
            # env.step(np.array([0, 1, 0, 0, 0]))
            env.step(np.array([0, 0, -1, 0, 1]))
            # env.step(np.array([np.random.uniform(low=-1., high=1.), np.random.uniform(low=-1., high=1.), 0.]))
            time.sleep(0.05)
    close(env)


# sawyer_multiple_objects doesn't work
def sample_sawyer_multiple_objects():
    # env = MultiSawyerEnv(
    #     do_render=False,
    #     finger_sensors=False,
    #     num_objects=3,
    #     object_meshes=None,
    #     randomize_initial_pos=False,
    #     fix_z=True,
    #     fix_gripper=True,
    #     fix_rotation=True,
    # )
    # env = ImageEnv(env,
    #     non_presampled_goal_img_is_garbage=True,
    #     recompute_reward=False,
    #     init_camera=sawyer_pusher_camera_upright_v2,
    # )
    # for i in range(10000):
    #     a = np.random.uniform(-1, 1, 5)
    #     o, _, _, _ = env.step(a)
    #     if i % 10 == 0:
    #         env.reset()

    #     img = o["image_observation"].transpose().reshape(84, 84, 3)
    #     cv2.imshow('window', img)
    #     cv2.waitKey(100)

    size = 0.1
    low = np.array([-size, 0.4 - size, 0])
    high = np.array([size, 0.4 + size, 0.1])
    env = MultiSawyerEnv(
        do_render=False,
        finger_sensors=False,
        num_objects=1,
        object_meshes=None,
        # randomize_initial_pos=True,
        fix_z=True,
        fix_gripper=True,
        fix_rotation=True,
        cylinder_radius=0.03,
        maxlen=0.03,
        workspace_low=low,
        workspace_high=high,
        hand_low=low,
        hand_high=high,
        init_hand_xyz=(0, 0.4 - size, 0.089),
    )
    for i in range(100):
        a = np.random.uniform(-1, 1, 5)
        o, r, _, _ = env.step(a)
        if i % 100 == 0:
            o = env.reset()
        # print(i, r)
        # print(o["state_observation"])
        # print(o["state_desired_goal"])
        env.render()
    close(env)

    # from robosuite.devices import SpaceMouse

    # device = SpaceMouse()
    # size = 0.1
    # low = np.array([-size, 0.4 - size, 0])
    # high = np.array([size, 0.4 + size, 0.1])
    # env = MultiSawyerEnv(
    #     do_render=False,
    #     finger_sensors=False,
    #     num_objects=1,
    #     object_meshes=None,
    #     workspace_low = low,
    #     workspace_high = high,
    #     hand_low = low,
    #     hand_high = high,
    #     fix_z=True,
    #     fix_gripper=True,
    #     fix_rotation=True,
    #     cylinder_radius=0.03,
    #     maxlen=0.03,
    #     init_hand_xyz=(0, 0.4-size, 0.089),
    # )
    # for i in range(10000):
    #     state = device.get_controller_state()
    #     dpos, rotation, grasp, reset = (
    #         state["dpos"],
    #         state["rotation"],
    #         state["grasp"],
    #         state["reset"],
    #     )

    #     # convert into a suitable end effector action for the environment
    #     # current = env._right_hand_orn
    #     # drotation = current.T.dot(rotation)  # relative rotation of desired from current
    #     # dquat = T.mat2quat(drotation)
    #     # grasp = grasp - 1.  # map 0 to -1 (open) and 1 to 0 (closed halfway)
    #     # action = np.concatenate([dpos, dquat, [grasp]])

    #     a = dpos * 10 # 200

    #     # a[:3] = np.array((0, 0.7, 0.1)) - env.get_endeff_pos()
    #     # a = np.array([np.random.uniform(-0.05, 0.05), np.random.uniform(-0.05, 0.05), 0.1, 0 ,  1])
    #     o, _, _, _ = env.step(a)
    #     if i % 100 == 0:
    #         env.reset()
    #     # print(env.sim.data.qpos[:7])
    #     env.render()


def sample_sawyer_peg_insertion_side():
    env = SawyerPegInsertionSideEnv()
    for _ in range(1):
        env.reset()
        # for _ in range(10):
        #     env.data.set_mocap_pos('mocap', np.array([0, 0.8, 0.05]))
        #     env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        #     env.do_simulation([-1,1], env.frame_skip)
        #     #self.do_simulation(None, self.frame_skip)
        # for _ in range(10):
        #     env.data.set_mocap_pos('mocap', np.array([0, 0.8, 0.25]))
        #     # env.data.set_mocap_pos('mocap', np.array([0, 0.6, 0.25]))
        #     env.data.set_mocap_quat('mocap', np.array([1, 0, 1, 0]))
        #     env.do_simulation([-1,1], env.frame_skip)
        #     #self.do_simulation(None, self.frame_skip)
        for _ in range(100):
            print('Before:', env.sim.model.site_pos[env.model.site_name2id('hole')] + env.sim.model.body_pos[
                env.model.body_name2id('box')])
            env.sim.model.body_pos[env.model.body_name2id('box')] = np.array([-0.3, np.random.uniform(0.5, 0.9), 0.05])
            print("After: ", env.sim.model.site_pos[env.model.site_name2id('hole')] + env.sim.model.body_pos[
                env.model.body_name2id('box')])
            env.render()
            env.step(env.action_space.sample())
            # if _ < 10:
            #     env.step(np.array([0, 0, -1, 0, 0]))
            # elif _ < 50:
            #     env.step(np.array([0, 0, 0, 0, 1]))
            # if _ < 10:
            #     env.step(np.array([0, 0, -1, 0, 0]))
            # else:
            #     env.step(np.array([0, 1, 0, 0, 1]))
            # env.step(np.array([0, 1, 0, 0, 0]))
            # env.step(np.array([np.random.uniform(low=-1., high=1.), np.random.uniform(low=-1., high=1.), 0.]))
            time.sleep(0.05)
    close(env)


def sample_sawyer_pick_and_place():
    env = SawyerPickAndPlaceEnv()
    env.reset()
    for _ in range(200):
        env.render()
        env.step(env.action_space.sample())
        time.sleep(0.05)
    glfw.destroy_window(env.viewer.window)


def sample_sawyer_pick_and_place():
    env = SawyerPickAndPlaceEnv()
    env.reset()
    for _ in range(50):
        env.render()
        env.step(env.action_space.sample())
        time.sleep(0.05)
    glfw.destroy_window(env.viewer.window)


def sample_sawyer_pick_and_place_wsg():
    env = SawyerPickAndPlaceWsgEnv()
    env.reset()
    for _ in range(100):
        env.render()
        env.step(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
        time.sleep(0.05)
    glfw.destroy_window(env.viewer.window)


def sample_sawyer_push_and_reach_env():
    env = SawyerPushAndReachXYEnv()
    for i in range(100):
        if i % 100 == 0:
            env.reset()
        env.step([0, 1])
        env.render()
    glfw.destroy_window(env.viewer.window)


def sample_sawyer_push_and_reach_two_pucks():
    env = SawyerPushAndReachXYZDoublePuckEnv()
    env.reset()
    for i in range(100):
        env.render()
        env.set_goal({'state_desired_goal': np.array([1, 1, 1, 1, 1, 1, 1])})
        env.step(env.action_space.sample())
    glfw.destroy_window(env.viewer.window)


def sample_sawyer_push_multiobj():
    env = SawyerTwoObjectEnv()
    for _ in range(100):
        env.render()
        env.step(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
        time.sleep(0.05)
    glfw.destroy_window(env.viewer.window)


def sample_sawyer_push_multiobj():
    env = SawyerTwoObjectEnv()
    env.reset()
    for _ in range(50):
        env.render()
        env.step(env.action_space.sample())
        time.sleep(0.05)
    glfw.destroy_window(env.viewer.window)


def sample_sawyer_push_nips():
    env = SawyerPushAndReachXYEasyEnv()
    for _ in range(100):
        env.render()
        env.step(env.action_space.sample())
        time.sleep(0.05)
    glfw.destroy_window(env.viewer.window)


def sample_sawyer_reach():
    env = SawyerReachXYZEnv()
    for i in range(100):
        if i % 100 == 0:
            env.reset()
        env.step(np.array([0, 1, 1]))
        env.render()
    glfw.destroy_window(env.viewer.window)


def sample_sawyer_reach():
    env = SawyerReachEnv()
    for i in range(100):
        if i % 100 == 0:
            env.reset()
        env.step(env.action_space.sample())
        env.render()
    glfw.destroy_window(env.viewer.window)


def sample_sawyer_reach_push_pick_place():
    env = SawyerReachPushPickPlaceEnv()
    for i in range(100):
        if i % 100 == 0:
            env.reset()
        env.step(np.array([0, 1, 1]))
        env.render()
    glfw.destroy_window(env.viewer.window)


def sample_sawyer_rope():
    env = SawyerRopeEnv()
    env.reset()
    for _ in range(50):
        env.render()
        env.step(env.action_space.sample())
        time.sleep(0.05)
    glfw.destroy_window(env.viewer.window)


def sample_sawyer_shelf_place():
    env = SawyerShelfPlaceEnv()
    env.reset()
    for _ in range(100):
        env.render()
        env.step(env.action_space.sample())
        time.sleep(0.05)
    glfw.destroy_window(env.viewer.window)


def sample_sawyer_stack():
    env = SawyerStackEnv()
    env.reset()
    for _ in range(50):
        env.render()
        env.step(env.action_space.sample())
        time.sleep(0.05)
    glfw.destroy_window(env.viewer.window)


def sample_sawyer_stick_pull():
    env = SawyerStickPullEnv()
    env.reset()
    for _ in range(100):
        env.render()
        env.step(env.action_space.sample())
        time.sleep(0.05)
    glfw.destroy_window(env.viewer.window)


def sample_sawyer_stick_push():
    env = SawyerStickPushEnv()
    env.reset()
    for _ in range(100):
        env.render()
        env.step(env.action_space.sample())
        if _ < 10:
            env.step(np.array([0, 0, -1, 0, 0]))
        elif _ < 20:
            env.step(np.array([0, 0, 0, 0, 1]))
        else:
            env.step(np.array([1, 0, 0, 0, 1]))
        time.sleep(0.05)
    glfw.destroy_window(env.viewer.window)


def sample_sawyer_sweep():
    env = SawyerSweepEnv(fix_goal=True)
    for i in range(200):
        if i % 100 == 0:
            env.reset()
        env.step(env.action_space.sample())
        env.render()
    glfw.destroy_window(env.viewer.window)


def sample_sawyer_sweep_into_goal():
    env = SawyerSweepIntoGoalEnv(fix_goal=True)
    for i in range(1000):
        if i % 100 == 0:
            env.reset()
        env.step(np.array([0, 1, 1]))
        env.render()
    glfw.destroy_window(env.viewer.window)


def sample_sawyer_throw():
    env = SawyerThrowEnv()
    for i in range(1000):
        if i % 100 == 0:
            env.reset()
        env.step(np.array([0, 0, 0, 1]))
        env.render()
    glfw.destroy_window(env.viewer.window)


def sample_sawyer_window_close():
    env = SawyerWindowCloseEnv()
    env.reset()
    for _ in range(100):
        env.render()
        env.step(np.array([1, 0, 0, 1]))
        time.sleep(0.05)
    glfw.destroy_window(env.viewer.window)


def sample_sawyer_window_open():
    env = SawyerWindowOpenEnv()
    env.reset()
    for _ in range(100):
        env.render()
        env.step(np.array([1, 0, 0, 1]))
        time.sleep(0.05)
    glfw.destroy_window(env.viewer.window)


demos = {
    SawyerNutAssemblyEnv: sample_sawyer_assembly_peg,
    SawyerBinPickingEnv: sample_sawyer_bin_picking,
    SawyerBoxCloseEnv: sample_sawyer_box_close,
    SawyerBoxOpenEnv: sample_sawyer_box_open,
    SawyerButtonPressEnv: sample_sawyer_button_press_6d0f,
    SawyerButtonPressTopdownEnv: sample_sawyer_button_press_topdown_6d0f,
    SawyerDialTurnEnv: sample_sawyer_dial_turn,
    SawyerDoorEnv: sample_sawyer_door,
    SawyerDoorCloseEnv: sample_sawyer_door_close,
    SawyerDoorHookEnv: sample_sawyer_door_hook,
    SawyerDoorEnv: sample_sawyer_door,
    SawyerDrawerCloseEnv: sample_sawyer_drawer_close,
    SawyerDrawerOpenEnv: sample_sawyer_drawer_open,
    SawyerHammerEnv: sample_sawyer_hammer,
    SawyerHandInsertEnv: sample_sawyer_hand_insert,
    SawyerLaptopCloseEnv: sample_sawyer_laptop_close,
    SawyerLeverPullEnv: sample_sawyer_lever_pull,
    MultiSawyerEnv: sample_sawyer_multiple_objects,
    SawyerPegInsertionSideEnv: sample_sawyer_peg_insertion_side,
    SawyerPickAndPlaceEnv: sample_sawyer_pick_and_place,
    SawyerPickAndPlaceEnv: sample_sawyer_pick_and_place,
    SawyerPickAndPlaceWsgEnv: sample_sawyer_pick_and_place_wsg,
    SawyerPushAndReachXYEnv: sample_sawyer_push_and_reach_env,
    SawyerPushAndReachXYZDoublePuckEnv: sample_sawyer_push_and_reach_two_pucks,
    SawyerTwoObjectEnv: sample_sawyer_push_multiobj,
    SawyerTwoObjectEnv: sample_sawyer_push_multiobj,
    SawyerPushAndReachXYEasyEnv: sample_sawyer_push_nips,
    SawyerReachXYZEnv: sample_sawyer_reach,
    SawyerReachEnv: sample_sawyer_reach,
    SawyerReachPushPickPlaceEnv: sample_sawyer_reach_push_pick_place,
    SawyerRopeEnv: sample_sawyer_rope,
    SawyerShelfPlaceEnv: sample_sawyer_shelf_place,
    SawyerStackEnv: sample_sawyer_stack,
    SawyerStickPullEnv: sample_sawyer_stick_pull,
    SawyerStickPushEnv: sample_sawyer_stick_push,
    SawyerSweepEnv: sample_sawyer_sweep,
    SawyerSweepIntoGoalEnv: sample_sawyer_sweep_into_goal,
    SawyerThrowEnv: sample_sawyer_throw,
    SawyerWindowCloseEnv: sample_sawyer_window_close,
    SawyerWindowOpenEnv: sample_sawyer_window_open,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run sample test of one specific environment!')
    parser.add_argument('--env', help='The environment name wanted to be test.')
    env_cls = globals()[parser.parse_args().env]
    demos[env_cls]()
