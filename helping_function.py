import numpy as np
import jax
from jax import numpy as jnp

def find_throwing(ball_c , rest_time):
    idx_throw = np.where(ball_c[rest_time:] == 0)[0]
    idx=None
    for i in range(len(idx_throw) - 100):
        if idx_throw[i+100] - idx_throw[i] == 100:
            idx = int(rest_time + idx_throw[i])
            break
    
    return idx


def hitting(ball_c , idx):
    idx_hit=None
    if idx is not None:
        idx_hit = np.where(ball_c[idx+1:] == 1)[0]
        t_end = int(idx +1 + idx_hit[0])
    return t_end

def hitting_ground(ball_q):
    hit_mask = ball_q[:, 2] - 0.038 < 1e-4
    return jnp.argmax(hit_mask)

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


def Forward_kinematic (q):
    L = jnp.array([-2.3360866e-05  ,-1.8214112e-05, 1.1860114e+00 ,5.4999387e-01 ,
                   2.5295885e-05 ,-4.5004494e-02 ,4.4200087e-01,2.5333025e-05 , 8.8627271e-02 ])
    Lx0 , Ly0 ,L0z ,Lx1 ,Ly1, Lz1, Lx2 ,Ly2 , Lz2 = L[0],L[1] ,L[2] ,L[3] , L[4] , L[5] , L[6] , L[7] , L[8]
    q1 , q2 , q3 , q4 = q[0] , q[1] , q[2] , q[3]
    pi = jnp.pi
    D0 = jnp.array([[1,0,0,Lx0],
                   [0,1,0,Ly0],
                   [0,0,1,L0z],
                   [0,0,0,1]])
    
    
    R1 = jnp.array([[jnp.cos(-q1) , -jnp.sin(-q1) , 0 ,0],
                    [jnp.sin(-q1) , jnp.cos(-q1) , 0 , 0],
                    [0             ,0              ,1   ,0],
                    [0             ,0              ,0   ,1]])
    
    R2 = jnp.array([[jnp.cos(-((pi/2) - q2)) , 0 , jnp.sin(-((pi/2) - q2)) , 0],
                   [0                    , 1 ,0                     , 0],
                   [-jnp.sin(-((pi/2)-q2))  , 0 , jnp.cos(-((pi/2)-q2))   ,0],
                   [0                    , 0 , 0                    ,1]])
    
    D1 = jnp.array([[1,0,0,Lx1],
                   [0,1,0,Ly1],
                   [0,0,1,Lz1],
                   [0,0,0,1]])
    
    R3 = jnp.array([[1,0           ,0            ,0],
                    [0,jnp.cos(q3),-jnp.sin(q3),0],
                    [0,jnp.sin(q3), jnp.cos(q3),0],
                    [0,0           ,0            ,1]])
    
    R4=jnp.array([[jnp.cos(q4) , 0 , jnp.sin(q4) , 0],
                  [0            , 1 ,0             , 0],
                  [-jnp.sin(q4)  , 0 , jnp.cos(q4) ,0],
                  [0              , 0 , 0            ,1]])
    
    D2= jnp.array([[1,0,0,Lx2],
                   [0,1,0,Ly2],
                   [0,0,1,Lz2],
                   [0,0,0,1]])
    
    return D0@R1@R2@D1@R3@R4@D2