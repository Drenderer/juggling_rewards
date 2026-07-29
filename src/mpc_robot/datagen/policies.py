"""
    Some simple definitions of movement primitives to compose open loop movements.

    mail@kaiploeger.ntext
"""

import numpy as np
from abc import ABC, abstractmethod


class CubicSpline:
    """ Cubic spline interpolation.
    """
    order = 3

    def __init__(self, x0, dx0, xT, dxT, T):
            self.x0 = x0
            self.dx0 = dx0
            self.xT = xT
            self.dxT = dxT
            self.T = T
            self.num_dim = x0.shape[0]
            self.fit_splines()

    def fit_splines(self):
        self.a = np.zeros((self.order+1, self.num_dim))
        self.a[0, :] = self.x0
        self.a[1, :] = self.dx0
        self.a[2, :] = 3 * (self.xT - self.x0) / self.T ** 2 - (self.dxT + 2 * self.dx0) / self.T
        self.a[3, :] = 2 * (self.x0 - self.xT) / self.T ** 3 + (self.dxT + self.dx0) / self.T ** 2

    def eval(self, t):
        t = np.clip(t, 0, self.T)
        x = self.a[0, :] + self.a[1, :] * t + self.a[2, :] * t ** 2 + self.a[3, :] * t ** 3
        dx = self.a[1, :] + 2 * self.a[2, :] * t + 3 * self.a[3, :] * t ** 2
        return x, dx


class MovementPrimitive(ABC):
    """ Abstract class for movement primitives.
    """
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, time):
        pass


class CubicMP(MovementPrimitive):
    """ Interpolates between via-points using cubic splines. Can be cyclic.
    """
    def __init__(self, pos, vel, times, cyclic=False):

        MovementPrimitive.__init__(self)

        self.pos = pos
        self.vel = vel
        self.times = times
        self.cyclic = cyclic
        self.cycle_duration = self.times[-1]
        if self.cyclic:
            self.duration = np.inf
            self.pos  = np.concatenate((self.pos, [self.pos[0, :]]), axis=0)
            self.vel = np.concatenate((self.vel, [self.vel[0, :]]), axis=0)
        else:
            self.duration = self.times[-1]

        self.times = np.concatenate(([0], self.times))
        self.dm_act = self.pos.shape[1]
        self.num_splines = len(self.times) - 1

        self.splines = [CubicSpline(self.pos[i], self.vel[i], self.pos[i + 1], self.vel[i + 1], self.times[i + 1] - self.times[i]) for i in range(self.num_splines)]


    def __call__(self, time):
        if self.cyclic:
            time = time % self.times[-1]
        else:
            time = np.clip(time, self.times[0], self.times[-1])

        spline_idx = np.searchsorted(self.times, time, side='right') - 1
        pos, vel = self.splines[spline_idx].eval(time - self.times[spline_idx])
        return pos, vel


class ConstantMP(MovementPrimitive):
    """ Returns a constant value.
    """
    def __init__(self, pos, duration, vel=None):
        self.pos = pos
        self.vel = np.zeros_like(pos) if vel is None else vel
        self.duration = duration

    def __call__(self, time):
        return self.pos, self.vel


class PiecewiseMP(MovementPrimitive):
    """ Concatenates multiple movement primitives.
    """
    def __init__(self, pieces):
        self.n_pieces = len(pieces)
        self.pieces = pieces
        self.durations = np.array([piece.duration for piece in self.pieces])
        self.intervals = np.array([np.sum(self.durations[:i+1]) for i in range(self.n_pieces)])

    def __call__(self, time):
        current_piece = np.argmax(time<self.intervals)
        if current_piece == 0:
            pos, vel = self.pieces[current_piece](time)
        else:
            pos, vel = self.pieces[current_piece](time-self.intervals[current_piece-1])
        return pos, vel
