#!/usr/bin/env python3

import time
import glfw
import numpy as np
import argparse

from multiworld.envs.mujoco.sawyer_xyz.sawyer_assembly_peg_6dof import SawyerNutAssembly6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_bin_picking_6dof import SawyerBinPicking6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_box_close_6dof import SawyerBoxClose6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_box_open_6dof import SawyerBoxOpen6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_button_press_6dof import SawyerButtonPress6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_button_press_topdown_6dof import SawyerButtonPressTopdown6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_dial_turn_6dof import SawyerDialTurn6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door import SawyerDoorEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door_6dof import SawyerDoor6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door_close import SawyerDoorCloseEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door_hook import SawyerDoorHookEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_drawer_close_6dof import SawyerDrawerClose6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_drawer_open_6dof import SawyerDrawerOpen6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_hammer_6dof import SawyerHammer6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_hand_insert import SawyerHandInsertEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_laptop_close_6dof import SawyerLaptopClose6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_lever_pull import SawyerLeverPull6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_multiple_objects import MultiSawyerEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_peg_insertion_side_6dof import SawyerPegInsertionSide6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import SawyerPickAndPlaceEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place_6dof import SawyerPickAndPlace6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place_wsg_6dof import SawyerPickAndPlaceWsg6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import SawyerPushAndReachXYEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env_two_pucks import SawyerPushAndReachXYZDoublePuckEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj import SawyerTwoObjectEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_multiobj_6dof import SawyerTwoObject6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_nips import SawyerPushAndReachXYEasyEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach import SawyerReachXYZEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach_6dof import SawyerReach6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach_push_pick_place_6dof import SawyerReachPushPickPlace6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_rope_6dof import SawyerRope6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_shelf_place_6dof import SawyerShelfPlace6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_stack_6dof import SawyerStack6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_stick_pull_6dof import SawyerStickPull6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_stick_push_6dof import SawyerStickPush6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_sweep import SawyerSweepEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_sweep_into_goal import SawyerSweepIntoGoalEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_throw import SawyerThrowEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_window_close_6dof import SawyerWindowClose6DOFEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_window_open_6dof import SawyerWindowOpen6DOFEnv


# function that closes the render window
def close(env):
    if env.viewer is not None:
        # self.viewer.finish()
        glfw.destroy_window(env.viewer.window)
    env.viewer = None


def sample_sawyer_assembly_peg_6dof():
    env = SawyerNutAssembly6DOFEnv()
    for _ in range(1):
        env.reset()
        for _ in range(50):
            env.render()
            env.step(env.action_space.sample())
            # env.step(np.array([np.random.uniform(low=-1., high=1.), np.random.uniform(low=-1., high=1.), 0.]))
            time.sleep(0.05)
    close(env)


def sample_sawyer_bin_picking_6dof():
    env = SawyerBinPicking6DOFEnv()
    for _ in range(1):
        env.reset()
        for _ in range(50):
            env.render()
            env.step(env.action_space.sample())
            # env.step(np.array([np.random.uniform(low=-1., high=1.), np.random.uniform(low=-1., high=1.), 0.]))
            time.sleep(0.05)
    close(env)


def sample_sawyer_box_close_6dof():
    env = SawyerBoxClose6DOFEnv()
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


def sample_sawyer_box_open_6dof():
    env = SawyerBoxOpen6DOFEnv()
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
    env = SawyerButtonPress6DOFEnv()
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
    env = SawyerButtonPressTopdown6DOFEnv()
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


def sample_sawyer_dial_turn_6dof():
    env = SawyerDialTurn6DOFEnv()
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


def sample_sawyer_door_6dof():
    env = SawyerDoor6DOFEnv()
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


def sample_sawyer_drawer_close_6dof():
    env = SawyerDrawerClose6DOFEnv()
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


def sample_sawyer_drawer_open_6dof():
    env = SawyerDrawerOpen6DOFEnv()
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


def sample_sawyer_hammer_6dof():
    env = SawyerHammer6DOFEnv()
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


def sample_sawyer_laptop_close_6dof():
    env = SawyerLaptopClose6DOFEnv()
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
    env = SawyerLeverPull6DOFEnv()
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


def sample_sawyer_peg_insertion_side_6dof():
    env = SawyerPegInsertionSide6DOFEnv()
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


def sample_sawyer_pick_and_place_6dof():
    env = SawyerPickAndPlace6DOFEnv()
    env.reset()
    for _ in range(50):
        env.render()
        env.step(env.action_space.sample())
        time.sleep(0.05)
    glfw.destroy_window(env.viewer.window)


def sample_sawyer_pick_and_place_wsg_6dof():
    env = SawyerPickAndPlaceWsg6DOFEnv()
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


def sample_sawyer_push_multiobj_6dof():
    env = SawyerTwoObject6DOFEnv()
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


def sample_sawyer_reach_6dof():
    env = SawyerReach6DOFEnv()
    for i in range(100):
        if i % 100 == 0:
            env.reset()
        env.step(env.action_space.sample())
        env.render()
    glfw.destroy_window(env.viewer.window)


def sample_sawyer_reach_push_pick_place_6dof():
    env = SawyerReachPushPickPlace6DOFEnv()
    for i in range(100):
        if i % 100 == 0:
            env.reset()
        env.step(np.array([0, 1, 1]))
        env.render()
    glfw.destroy_window(env.viewer.window)


def sample_sawyer_rope_6dof():
    env = SawyerRope6DOFEnv()
    env.reset()
    for _ in range(50):
        env.render()
        env.step(env.action_space.sample())
        time.sleep(0.05)
    glfw.destroy_window(env.viewer.window)


def sample_sawyer_shelf_place_6dof():
    env = SawyerShelfPlace6DOFEnv()
    env.reset()
    for _ in range(100):
        env.render()
        env.step(env.action_space.sample())
        time.sleep(0.05)
    glfw.destroy_window(env.viewer.window)


def sample_sawyer_stack_6dof():
    env = SawyerStack6DOFEnv()
    env.reset()
    for _ in range(50):
        env.render()
        env.step(env.action_space.sample())
        time.sleep(0.05)
    glfw.destroy_window(env.viewer.window)


def sample_sawyer_stick_pull_6dof():
    env = SawyerStickPull6DOFEnv()
    env.reset()
    for _ in range(100):
        env.render()
        env.step(env.action_space.sample())
        time.sleep(0.05)
    glfw.destroy_window(env.viewer.window)


def sample_sawyer_stick_push_6dof():
    env = SawyerStickPush6DOFEnv()
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


def sample_sawyer_window_close_6dof():
    env = SawyerWindowClose6DOFEnv()
    env.reset()
    for _ in range(100):
        env.render()
        env.step(np.array([1, 0, 0, 1]))
        time.sleep(0.05)
    glfw.destroy_window(env.viewer.window)


def sample_sawyer_window_open_6dof():
    env = SawyerWindowOpen6DOFEnv()
    env.reset()
    for _ in range(100):
        env.render()
        env.step(np.array([1, 0, 0, 1]))
        time.sleep(0.05)
    glfw.destroy_window(env.viewer.window)


demos = {
    SawyerNutAssembly6DOFEnv: sample_sawyer_assembly_peg_6dof,
    SawyerBinPicking6DOFEnv: sample_sawyer_bin_picking_6dof,
    SawyerBoxClose6DOFEnv: sample_sawyer_box_close_6dof,
    SawyerBoxOpen6DOFEnv: sample_sawyer_box_open_6dof,
    SawyerButtonPress6DOFEnv: sample_sawyer_button_press_6d0f,
    SawyerButtonPressTopdown6DOFEnv: sample_sawyer_button_press_topdown_6d0f,
    SawyerDialTurn6DOFEnv: sample_sawyer_dial_turn_6dof,
    SawyerDoor6DOFEnv: sample_sawyer_door_6dof,
    SawyerDoorCloseEnv: sample_sawyer_door_close,
    SawyerDoorHookEnv: sample_sawyer_door_hook,
    SawyerDoorEnv: sample_sawyer_door,
    SawyerDrawerClose6DOFEnv: sample_sawyer_drawer_close_6dof,
    SawyerDrawerOpen6DOFEnv: sample_sawyer_drawer_open_6dof,
    SawyerHammer6DOFEnv: sample_sawyer_hammer_6dof,
    SawyerHandInsertEnv: sample_sawyer_hand_insert,
    SawyerLaptopClose6DOFEnv: sample_sawyer_laptop_close_6dof,
    SawyerLeverPull6DOFEnv: sample_sawyer_lever_pull,
    MultiSawyerEnv: sample_sawyer_multiple_objects,
    SawyerPegInsertionSide6DOFEnv: sample_sawyer_peg_insertion_side_6dof,
    SawyerPickAndPlaceEnv: sample_sawyer_pick_and_place,
    SawyerPickAndPlace6DOFEnv: sample_sawyer_pick_and_place_6dof,
    SawyerPickAndPlaceWsg6DOFEnv: sample_sawyer_pick_and_place_wsg_6dof,
    SawyerPushAndReachXYEnv: sample_sawyer_push_and_reach_env,
    SawyerPushAndReachXYZDoublePuckEnv: sample_sawyer_push_and_reach_two_pucks,
    SawyerTwoObjectEnv: sample_sawyer_push_multiobj,
    SawyerTwoObject6DOFEnv: sample_sawyer_push_multiobj_6dof,
    SawyerPushAndReachXYEasyEnv: sample_sawyer_push_nips,
    SawyerReachXYZEnv: sample_sawyer_reach,
    SawyerReach6DOFEnv: sample_sawyer_reach_6dof,
    SawyerReachPushPickPlace6DOFEnv: sample_sawyer_reach_push_pick_place_6dof,
    SawyerRope6DOFEnv: sample_sawyer_rope_6dof,
    SawyerShelfPlace6DOFEnv: sample_sawyer_shelf_place_6dof,
    SawyerStack6DOFEnv: sample_sawyer_stack_6dof,
    SawyerStickPull6DOFEnv: sample_sawyer_stick_pull_6dof,
    SawyerStickPush6DOFEnv: sample_sawyer_stick_push_6dof,
    SawyerSweepEnv: sample_sawyer_sweep,
    SawyerSweepIntoGoalEnv: sample_sawyer_sweep_into_goal,
    SawyerThrowEnv: sample_sawyer_throw,
    SawyerWindowClose6DOFEnv: sample_sawyer_window_close_6dof,
    SawyerWindowOpen6DOFEnv: sample_sawyer_window_open_6dof,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run sample test of one specific environment!')
    parser.add_argument('--env', help='The environment name wanted to be test.')
    env_cls = globals()[parser.parse_args().env]
    demos[env_cls]()
