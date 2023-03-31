import abc
import copy
import os.path
import warnings

import glfw
import mujoco
from gymnasium import error
from gymnasium.utils import seeding
import numpy as np
from os import path
import gymnasium
import mujoco
from PIL import Image
import time


def _assert_task_is_set(func):
    def inner(*args, **kwargs):
        env = args[0]
        if not env._set_task_called:
            raise RuntimeError(
                'You must call env.set_task before using env.'
                + func.__name__
            )
        return func(*args, **kwargs)
    return inner


DEFAULT_SIZE = 500


class MujocoEnv(gymnasium.Env, abc.ABC):
    """
    This is a simplified version of the gymnasium MujocoEnv class.

    Some differences are:
     - Do not automatically set the observation/action space.
    """

    max_path_length = 500

    def __init__(self, model_path, frame_skip):
        import mujoco
        if not path.exists(model_path):
            raise IOError("File %s does not exist" % model_path)

        self.frame_skip = frame_skip
        self.model = mujoco.MjModel.from_xml_path(filename=model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self._viewers = {}
        self.renderer = None
        self.scene = None
        self.metadata = {
            'render.modes': ['human'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.init_qvel = self.data.qvel.ravel().copy()
        self.init_qpos = self.data.qpos.ravel().copy()
        self._did_see_sim_exception = False

        self.np_random, _ = seeding.np_random(None)

    def seed(self, seed):
        assert seed is not None
        self.np_random, seed = seeding.np_random(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        self.goal_space.seed(seed)
        return [seed]

    @abc.abstractmethod
    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        pass

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    @_assert_task_is_set
    def reset(self):
        self._did_see_sim_exception = False
        # self.sim.reset()
        mujoco.mj_resetData(self.model, self.data)
        ob = self.reset_model()
        if self.viewer is not None:
            self.viewer_setup()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.data.qvel = qvel
        self.data.qpos = qpos
        """TODO: Reggie look into if this sets the state correctly"""
        print(self.data.time)
        mujoco.mj_step(self.model, self.data)

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames=None):
        if getattr(self, 'curr_path_length', 0) > self.max_path_length:
            raise ValueError('Maximum path length allowed by the benchmark has been exceeded')
        if self._did_see_sim_exception:
            return

        if n_frames is None:
            n_frames = self.frame_skip

        for _ in range(n_frames):
            try:
                mujoco.mj_step1(self.model, self.data)
                self.data.ctrl[:] = ctrl
                mujoco.mj_step2(self.model, self.data)
            except mujoco.mjr_getError() as err:
                warnings.warn(str(err), category=RuntimeWarning)
                self._did_see_sim_exception = True

    def render(self, offscreen=False, camera_name="corner2", resolution=(640, 480)):
        assert_string = ("camera_name should be one of ",
                "corner3, corner, corner2, topview, gripperPOV, behindGripper")
        assert camera_name in {"corner3", "corner", "corner2", 
            "topview", "gripperPOV", "behindGripper"}, assert_string
        if not offscreen:
            if not self.renderer:
                self.renderer = mujoco.Renderer(self.model, 480, 640)
            self.renderer.update_scene(self.data)
            img = Image.fromarray(self.renderer.render())
            img.show()
            time.sleep(2)
            img.close()
            exit(0)
            # self._get_viewer('human').render()
        else:
            return self.sim.render(
                *resolution,
                mode='offscreen',
                camera_name=camera_name
            )

    def close(self):
        if self.viewer is not None:
            glfw.destroy_window(self.viewer.window)
            self.viewer = None

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                print("Huh")
                #self.viewer = mujoco.MjVisual(mode)
                # self.viewer = mujoco_py.MjViewer(self.sim)
            self.viewer_setup()
            self._viewers[mode] = self.viewer
        self.viewer_setup()
        return self.viewer

    def get_body_com(self, body_name):
        try:
            return self.data.geom(body_name + '_geom').xpos
        except:
            try:
                return self.data.geom(body_name + 'Geom').xpos
            except:
                print(body_name + ' not found')
                assert 1 == 2, "Something is wrong"

