import numpy as np
import jax
from jax import numpy as jnp

def find_throwing(ball_f , rest_time):
    idx_throw = np.where(ball_f[rest_time:] < 1e-04)[0]
    flag = False
    i=0
    while flag == False:
        if idx_throw[i+10] - idx_throw[i] == 10:
            idx_throw = int(rest_time + idx_throw[i])
            flag = True
        i=i+1
    return idx_throw


def hitting_ground(ball_x):
    idx_floor = np.where(ball_x[:, 2] - 0.038 < 1e-4)[0]
    return idx_floor[0]

def ball_free_flight_trajecotry (s0 , ts , gs=9.81):
    x0, y0, z0, vx0, vy0, vz0 = s0
    t = ts
    x = x0 + vx0 * t
    y = y0 + vy0 * t
    z = z0 + vz0 * t - 0.5 * gs * t**2
    vx = jnp.full_like(t, vx0)
    vy = jnp.full_like(t, vy0)
    vz = vz0 - gs * t

    ball_q=jnp.stack([x, y, z, vx, vy, vz], axis=-1)
    time_hit = hitting_ground(ball_q)
    return ball_q , time_hit