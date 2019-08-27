
def _XOR(a, b):
    return bool(a) ^ bool(b)


def general_sl_coeffs(P, N=None, T=None):
    x0, y0 = P
    assert _XOR(N, T), "either of N or T must be set!"
    if N:
        A, B = N
    elif T:
        A = T[1] ; B = -T[0]
    assert A or B, "vector must be non-zero!"
    return A, B, - A * x0 - B * y0


def _extr_points(A, B, C, xlim, ylim):
    xL, xR = xlim ; yD, yU = ylim  # left-right, down-up
    if A == 0:
        assert B
        P0 = (xL, -C/B) ; P1 = (xR, -C/B)
    elif B == 0:
        assert A
        P0 = (-C/A, yD) ; P1 = (-C/A, yU)
    else:
        yL = (-C - A * xL) / B ; yR = (-C - A * xR) / B
        xD = (-C - B * yD) / A ; xU = (-C - B * yU) / A
        expoints = set()
        if yD <= yL <= yU: expoints.add((xL, yL))
        if xL <= xD <= xR: expoints.add((xD, yD))
        if yD <= yR <= yU: expoints.add((xR, yR))
        if xL <= xU <= xR: expoints.add((xU, yU))
        assert len(expoints) == 2
        P0, P1 = list(expoints)
    return P0, P1


def plt_line(ax, xlim, ylim, ABC, **kwargs):
    A, B, C = ABC
    assert A**2 + B**2 > 0.0
    P0, P1 = _extr_points(A, B, C, xlim, ylim)
    ax.plot(*zip(P0,P1), **kwargs)
    return A, B, C


