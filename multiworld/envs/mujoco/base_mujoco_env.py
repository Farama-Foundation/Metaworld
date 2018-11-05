from mujoco_py import load_model_from_path, MjSim
import numpy as np



class BaseMujocoEnv():
    def __init__(self, height=480, width=640):
        self._frame_height = height
        self._frame_width = width

        self._base_adim, self._base_sdim = None, None                 #state/action dimension of Mujoco control
        self._adim, self._sdim = None, None   #state/action dimension presented to agent
        self.num_objects, self._n_joints = None, None

    def render(self, mode='dual'):
        """ Renders the enviornment.
        Implements custom rendering support. If mode is:

        - dual: renders both left and main cameras
        - left: renders only left camera
        - main: renders only main (front) camera
        :param mode: Mode to render with (dual by default)
        :return: uint8 numpy array with rendering from sim
        """
        cameras = ['maincam']
        if mode == 'dual':
            cameras = ['maincam', 'leftcam']
        elif mode == 'leftcam':
            cameras = ['leftcam']

        images = np.zeros((len(cameras), self._frame_height, self._frame_width, 3), dtype=np.uint8)
        for i, cam in enumerate(cameras):
            images[i] = self.sim.render(self._frame_width, self._frame_height, camera_name=cam)
        return images
