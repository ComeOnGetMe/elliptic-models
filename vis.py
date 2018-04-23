from core import Plane3d, Ellipsoid
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging


def vis_plane():
    n_pts = 22
    plane = Plane3d(1, 2, 3, 1)
    x = np.linspace(-4, 4, n_pts)
    X, Y = np.meshgrid(x, x)
    X = X.reshape((-1, 1))
    Y = Y.reshape((-1, 1))
    pt_xy = np.hstack((X, Y))
    Z = plane.xy2z(pt_xy).reshape((-1, 1))

    noise_pts = np.hstack((X, Y, Z)) + np.random.normal(0, 1, (n_pts**2, 3))
    plane_fit, s = Plane3d.fit(noise_pts)
    print "singular value", s
    print plane_fit.normal / plane_fit.a
    noise_Z = plane_fit.xy2z(pt_xy)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c='b')
    ax.scatter(X, Y, noise_Z, c='g')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def vis_ellipsoid():
    n_pts = 55
    elps = Ellipsoid(5, 10, 5)
    x = np.linspace(-4, 4, n_pts)
    y = np.linspace(-8, 8, n_pts)
    X, Y = np.meshgrid(x, y)
    X = X.reshape((-1, 1))
    Y = Y.reshape((-1, 1))
    pt_xy = np.hstack((X, Y))
    Z = elps.xy2z(pt_xy)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z)
    ax.scatter(X, Y, -Z)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def vis_both():
    n_pts = 22
    elps = Ellipsoid(5, 10, 5)
    plane = Plane3d(1, 2, 3, 1)

    x = np.linspace(-4, 4, n_pts)
    y = np.linspace(-8, 8, n_pts)
    X, Y = np.meshgrid(x, y)
    X = X.reshape((-1, 1))
    Y = Y.reshape((-1, 1))
    pt_xy = np.hstack((X, Y))

    elps_Z = elps.xy2z(pt_xy)
    plane_Z = plane.xy2z(pt_xy)
    inside_idx = np.where(np.abs(plane_Z) < elps_Z)[0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, elps_Z, c='b')
    ax.scatter(X, Y, -elps_Z, c='b')
    ax.scatter(X[inside_idx], Y[inside_idx], plane_Z[inside_idx], c='m')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    return fig, ax


def vis_intersection():
    n_samples = 77
    elps = Ellipsoid(5, 10, 5)
    plane = Plane3d(1, 2, 3, 1)

    sample_x = np.linspace(-4, 4, n_samples)
    intersect_pts = elps.plane_intersect_arc(plane, sample_x)
    param_func = lambda x: elps.plane_intersect_arc(plane, [x]).squeeze()
    arc_len = elps.arc_length(param_func, -4, 4)

    fig, ax = vis_both()
    ax.scatter(intersect_pts[:, 0], intersect_pts[:, 1], intersect_pts[:, 2], c='g')
    plt.title("arc length: ".format(arc_len))
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    vis_intersection()
