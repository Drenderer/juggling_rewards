
from jax import numpy as jnp


def Forward_kinematic (L,q):
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