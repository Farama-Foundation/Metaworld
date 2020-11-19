import numpy as np
from gym.spaces import Box

from metaworld.envs import reward_utils
from metaworld.envs.mujoco.sawyer_xyz.sawyer_xyz_env import _assert_task_is_set
from metaworld.envs.mujoco.sawyer_xyz.visual.visual_sawyer_sandbox_env import VisualSawyerSandboxEnv
from metaworld.envs.mujoco.sawyer_xyz.visual.tools import ButtonBox

from .library import TOOLSETS


class ButtonPress(VisualSawyerSandboxEnv):
    BUTTON_TRAVEL = 0.1

    def __init__(self):
        super().__init__()
        self.init_config = {
            'hand_init_pos': self.hand_init_pos,
        }

        obj_low = (-1, -1, -1)
        obj_high = (1, 1, 1)
        goal_low = (-1., -1., -1.)
        goal_high = (1., 1., 1.)

        self._target_pos = np.zeros(3)
        self._random_reset_space = Box(
            np.hstack((obj_low, goal_low)),
            np.hstack((obj_high, goal_high)),
        )
        self.goal_space = Box(np.array(goal_low), np.array(goal_high))

        self._toolset_required = TOOLSETS[type(self).__name__]
        self.randomize_extra_toolset(5)

    def reset_model(self):
        ob = super().reset_model()
        self._target_pos = self._get_site_pos('ButtonBoxEnd')
        return ob

    def _reset_required_tools(self, world, solver):
        button = ButtonBox()
        x = 0.0
        y = 0.0
        z = button.resting_pos_z
        if self.random_init:
            x = world.size[0]/2.0 + 0.8 * (np.random.random() - 0.5)
            y = world.size[1]/8.0 + 0.3 * (np.random.random() - 0.0)
        button.specified_pos = np.array([x, y, z])
        solver.did_manual_set(button)

    @_assert_task_is_set
    def step(self, action):
        ob = super().step(action)
        (
            reward,
            tcp_to_obj,
            tcp_open,
            obj_to_target,
            near_button,
            button_pressed
        ) = self.compute_reward(action, ob)

        info = {
            'success': float(obj_to_target <= 0.02),
            'near_object': float(tcp_to_obj <= 0.05),
            'grasp_success': float(tcp_open > 0),
            'grasp_reward': near_button,
            'in_place_reward': button_pressed,
            'obj_to_target': obj_to_target,
            'unscaled_reward': reward,
        }

        self.curr_path_length += 1
        return ob, reward, False, info

    def _get_pos_objects(self):
        return self.get_body_com('button') + np.array([.0, .0, .193])

    def _get_quat_objects(self):
        return self.sim.data.get_body_xquat('button')

    def compute_reward(self, action, obs):
        del action
        obj = obs[4:7]
        tcp = self.tcp_center

        tcp_to_obj = np.linalg.norm(obj - tcp)
        tcp_to_obj_init = np.linalg.norm(obj - self.init_tcp)
        obj_to_target = abs(self._target_pos[2] - obj[2])

        tcp_closed = 1 - obs[3]
        near_button = reward_utils.tolerance(
            tcp_to_obj,
            bounds=(0, 0.01),
            margin=tcp_to_obj_init,
            sigmoid='long_tail',
        )
        button_pressed = reward_utils.tolerance(
            obj_to_target,
            bounds=(0, 0.005),
            margin=self.BUTTON_TRAVEL,
            sigmoid='long_tail',
        )

        reward = 5 * reward_utils.hamacher_product(tcp_closed, near_button)
        if tcp_to_obj <= 0.03:
            reward += 5 * button_pressed

        return (
            reward,
            tcp_to_obj,
            obs[3],
            obj_to_target,
            near_button,
            button_pressed
        )