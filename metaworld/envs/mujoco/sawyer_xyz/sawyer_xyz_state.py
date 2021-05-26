import numpy as np


class SawyerXYZState:

    def __init__(self):
        self._timestep = 0

        self._action = np.zeros(4, dtype='float')

        self._pos_objs: np.ndarray | None = None
        self._quat_objs: np.ndarray | None = None

        self._pos_hand = np.zeros(3, dtype='float')
        self._pos_pad_l = np.zeros(3, dtype='float')
        self._pos_pad_r = np.zeros(3, dtype='float')

        self._pos_pads_center: np.ndarray | None = None
        self._inter_pad_distance: np.ndarray | None = None

    def populate(self,
                 action: np.ndarray,
                 pos_objs: np.ndarray,
                 quat_objs: np.ndarray,
                 mjsim):
        self._action = action

        self._pos_objs = pos_objs
        self._quat_objs = quat_objs

        self._pos_hand = mjsim.data.get_body_xpos('hand').copy()
        self._pos_pad_l = mjsim.model.site_pos[
            mjsim.model.site_name2id('leftEndEffector')
        ]
        self._pos_pad_r = mjsim.model.site_pos[
            mjsim.model.site_name2id('rightEndEffector')
        ]

        self._timestep += 1

    @property
    def timestep(self):
        return self._timestep

    @property
    def action(self):
        # TODO One could easily add .copy() to this and other properties to
        # avoid accidental modification. Going without .copy() for now in case
        # someone is making use of undocumented behavior somewhere
        return self._action

    @property
    def pos_objs(self):
        return self._pos_objs

    @property
    def quat_objs(self):
        return self.quat_objs

    @property
    def pos_hand(self):
        return self._pos_hand

    @property
    def pos_pad_l(self):
        return self._pos_pad_l

    @property
    def pos_pad_r(self):
        return self._pos_pad_r

    @property
    def pos_pads_center(self):
        if self._pos_pads_center is not None:
            return self._pos_pads_center

        # Lazy loading
        self._pos_pads_center = (self._pos_pad_l + self._pos_pad_r) / 2.0
        return self._pos_pads_center

    @property
    def inter_pad_distance(self):
        if self._inter_pad_distance is not None:
            return self._inter_pad_distance

        # Lazy loading
        self._inter_pad_distance = np.linalg.norm(
            self._pos_pad_l - self._pos_pad_r
        )
        return self._inter_pad_distance

    @property
    def normalized_inter_pad_distance(self):
        return np.clip(self.inter_pad_distance / 0.1, 0., 1.)
