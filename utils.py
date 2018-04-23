import numpy as np


def real_quad_roots(a, b, c):
    assert b**2 > 4*a*c, "{}, {}, {}".format(a, b, c)
    roots = np.roots([a, b, c])
    return np.real(roots)
