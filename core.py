import numpy as np
import logging
from scipy.integrate import quad

import utils


class Plane3d(object):
    """
    ax + by + cz + d = 0
    """
    def __init__(self, a, b, c, d):
        self.logger = logging.getLogger(type(self).__name__)
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)

    @property
    def normal(self):
        vec = np.asarray([self.a, self.b, self.c])
        return vec / np.linalg.norm(vec)

    def pt_dist(self, pt):
        assert len(pt) == 3
        _pt = np.asarray(pt)
        return np.dot(self.normal, _pt) + self.d

    @classmethod
    def fit(cls, pts):
        _pts = np.asarray(pts)
        centroid = _pts.mean(axis=0)
        u, s, vh = np.linalg.svd(_pts - centroid)
        [a, b, c] = vh[-1]
        d = -np.dot(vh[-1], centroid)
        return cls(a, b, c, d), s

    def xy2z(self, xy):
        assert self.c != 0
        _xy = np.asarray(xy)
        return (-self.d - np.dot(_xy, np.array([self.a, self.b]))) / self.c


class Ellipsoid(object):
    """
    x^2/rx^2 + y^2/ry^2 + z^2/rz^2 = 1
    """
    def __init__(self, rx, ry, rz):
        assert rx > 0 and ry > 0 and rz > 0
        self.logger = logging.getLogger(type(self).__name__)
        self.rx = float(rx)
        self.ry = float(ry)
        self.rz = float(rz)

    def xy2z(self, xy):
        """
        CAUTION: ONLY RETURNS POSITIVE Z's
        :param xy:
        :return:
        """
        _xy = np.asarray(xy)
        rxy = np.array([[self.rx, self.ry]])
        assert np.all(np.abs(_xy) <= rxy)
        return self.rz * np.sqrt(1 - np.sum(_xy**2 / rxy**2, axis=1))

    def _get_plane_intersect_params(self, plane):
        rx, ry, rz = self.rx, self.ry, self.rz
        a, b, c, d = plane.a, plane.b, plane.c, plane.d
        zc_denom = rz**2 * c**2
        A = (1/rz**2 + a**2/zc_denom)
        B = (1/ry**2 + b**2/zc_denom)
        C = (2*a*b/zc_denom)
        D = (2*a*d/zc_denom)
        E = (2*b*d/zc_denom)
        F = (d**2/zc_denom)
        return A, B, C, D, E, F

    def plane_intersect_arc(self, plane, sample_x):
        """
        Given plane and x, output pt(x, y, z) that's in the intersection.
        x out of intersection are discarded.
        :param plane: Plane3d obj
        :param sample_x: list of floats
        :return:
        """
        A, B, C, D, E, F = self._get_plane_intersect_params(plane)
        res = []
        for x in sample_x:
            try:
                ys = utils.real_quad_roots(B, C*x + E, A * x**2 + D * x + F - 1)
                self.logger.debug("get y: {}".format(ys))
                y = ys[0]
            except AssertionError as e:
                self.logger.error(e)
                continue
            res.append((x, y))
        xy = np.asarray(res)
        z = plane.xy2z(xy)
        return np.hstack((xy, z[:, None]))

    def arc_length(self, x2ptfunc, start_x, end_x):
        """
        Given parametric function on x, calculate the integration of arc length
        :param x2ptfunc:
        :param start_x:
        :param end_x:
        :return:
        """
        rs = np.array([self.rx, self.ry, self.rz])
        return quad(lambda x: 2*np.linalg.norm(x2ptfunc(x) / rs), start_x, end_x)
