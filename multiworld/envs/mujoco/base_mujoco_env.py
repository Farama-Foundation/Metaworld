from mujoco_py import load_model_from_path, MjSim
import numpy as np



class BaseMujocoEnv():
    def __init__(self,  model_path, height=480, width=640):
        self._frame_height = height
        self._frame_width = width
        print(model_path)
        self._reset_sim(model_path)

        self._base_adim, self._base_sdim = None, None                 #state/action dimension of Mujoco control
        self._adim, self._sdim = None, None   #state/action dimension presented to agent
        self.num_objects, self._n_joints = None, None

    def _reset_sim(self, model_path):
        """
        Creates a MjSim from passed in model_path
        :param model_path: Absolute path to model file
        :return: None
        """
        self._model_path = model_path
        self.sim = MjSim(load_model_from_path(self._model_path))

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

    def project_point(self, point, camera):
        model_matrix = np.zeros((4, 4))
        model_matrix[:3, :3] = self.sim.data.get_camera_xmat(camera).T
        model_matrix[-1, -1] = 1

        fovy_radians = np.deg2rad(self.sim.model.cam_fovy[self.sim.model.camera_name2id(camera)])
        uh = 1. / np.tan(fovy_radians / 2)
        uw = uh / (self._frame_width / self._frame_height)
        extent = self.sim.model.stat.extent
        far, near = self.sim.model.vis.map.zfar * extent, self.sim.model.vis.map.znear * extent
        view_matrix = np.array([[uw, 0., 0., 0.],                        #matrix definition from
                                [0., uh, 0., 0.],                        #https://stackoverflow.com/questions/18404890/how-to-build-perspective-projection-matrix-no-api
                                [0., 0., far / (far - near), -1.],
                                [0., 0., -2*far*near/(far - near), 0.]]) #Note Mujoco doubles this quantity

        MVP_matrix = view_matrix.dot(model_matrix)
        world_coord = np.ones((4, 1))
        world_coord[:3, 0] = point - self.sim.data.get_camera_xpos(camera)

        clip = MVP_matrix.dot(world_coord)
        ndc = clip[:3] / clip[3]  # everything should now be in -1 to 1!!
        col, row = (ndc[0] + 1) * self._frame_width / 2, (-ndc[1] + 1) * self._frame_height / 2

        return self._frame_height - row, col                 #rendering flipped around in height

    def get_desig_pix(self, cams, target_width, round=True):
        qpos_dim = self._n_joints      # the states contains pos and vel
        assert self.sim.data.qpos.shape[0] == qpos_dim + 7 * self.num_objects
        desig_pix = np.zeros([len(cams), self.num_objects, 2], dtype=np.int)
        ratio = self._frame_width / target_width
        for icam, cam in range(cams):
            for i in range(self.num_objects):
                fullpose = self.sim.data.qpos[i * 7 + qpos_dim:(i + 1) * 7 + qpos_dim].squeeze()
                d = self.project_point(fullpose[:3], cam)
                d = np.stack(d) / ratio
                if round:
                    d = np.around(d).astype(np.int)
                desig_pix[icam, i] = d
        return desig_pix

    def get_goal_pix(self, cams, target_width, goal_obj_pose, round=True):
        goal_pix = np.zeros([len(cams), self.num_objects, 2], dtype=np.int)
        ratio = self._frame_width / target_width
        for icam, cam in range(cams):
            for i in range(self.num_objects):
                g = self.project_point(goal_obj_pose[i, :3], cam)
                g = np.stack(g) / ratio
                if round:
                    g= np.around(g).astype(np.int)
                goal_pix[icam, i] = g
        return goal_pix

    def snapshot_noarm(self):
        raise NotImplementedError

    @property
    def adim(self):
        return self._adim

    @property
    def sdim(self):
        return self._sdim
