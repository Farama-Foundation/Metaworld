import numpy as np
from scipy.interpolate import CubicSpline
import copy

class QuinticSpline:
    _solver_mat = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0],
                           [0, 0, 2, 0, 0, 0],
                           [1, 1, 1, 1, 1, 1],
                           [0, 1, 2, 3, 4, 5],
                           [0, 0, 2, 6, 12, 20]], dtype = np.float32)

    def __init__(self, p_1, p_2, t = 1):
        solver_mat = self._solver_mat.copy()
        solver_mat[3:] *= np.array([1, t, t**2, t**3, t**4, t**5])
        self._p1 = p_1.reshape(-1)
        self._p2 = p_2.reshape(-1)
        self._orig_shape = copy.deepcopy(p_1.shape)

        self._poly_coeff = np.zeros((self._p1.shape[0], 6))

        for i in xrange(self._p1.shape[0]):
            vec = np.array([self._p1[i], 0, 0, self._p2[i], 0, 0]).reshape((-1, 1))
            self._poly_coeff[i] = np.linalg.solve(solver_mat, vec).reshape(-1)

        self._first_order = self._poly_coeff[:, 1:] * np.array([[1, 2, 3, 4, 5]])
        self._second_order = self._first_order[:, 1:] * np.array([[1, 2, 3, 4]])

    def get(self, t):
        if isinstance(t, np.ndarray):
            t_array = np.array([np.ones_like(t), t, t ** 2, t** 3, t** 4, t**5]).T[:, None, :]
        else:
            t_array = np.array([1, t, t**2, t**3, t**4, t**5]).reshape(1, 1, 6)

        eval_0 = np.sum(t_array * self._poly_coeff[None], -1)
        eval_1 = np.sum(t_array[:, :, :-1] * self._first_order[None], -1)
        eval_2 = np.sum(t_array[:, :, :-2] * self._second_order[None], -1)

        return eval_0, eval_1, eval_2


class TwoPointCSpline(object):
    def __init__(self, p_1, p_2):
        self.cs = CubicSpline(np.array([0.0, 1.0]), np.array([p_1, p_2]), bc_type='clamped')

    def get(self, t):
        # assert 0 <= t <= 1, "Time should be in [0, 1] but is {}".format(t)
        t = np.array(t)

        ret = (self.cs(t), self.cs(t, nu=1), self.cs(t, nu=2))

        return ret


class CSpline:
    def __init__(self, points, duration=1., bc_type='clamped'):
        n_points = points.shape[0]
        self._duration = duration
        self._cs = CubicSpline(np.linspace(0, duration, n_points), points, bc_type=bc_type)

    def get(self, t):
        t = np.array(min(t, self._duration))

        return self._cs(t), self._cs(t, nu=1), self._cs(t, nu=2)