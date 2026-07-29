import klax
from jax import random as jr

from dynax import ISPHS, ConvexLyapunov, ODESolver
from klax.nn import (
    FICNN,
    ConstantMatrix,
    ConstantSkewSymmetricMatrix,
    ConstantSPDMatrix,
)

def make_sphnn(key):
    ficnn_key , h_key , j_key , r_key , g_key = jr.split(key , 5)

    ficnn =FICNN(in_size=8 , out_size='scalar' , width_sizes=[64,128,64] , key = ficnn_key)
    H = ConvexLyapunov(ficnn , state_size= 8 , minimum_learnable=True , key = h_key )
    J = ConstantSkewSymmetricMatrix((8,8) , key = j_key)
    R = ConstantSPDMatrix ((8,8) ,key=r_key)
    G = ConstantMatrix ( (8,4) , key =g_key)

    isphs = ISPHS( H , J , R, G)
    sphnn = ODESolver(isphs)
    sphnn_ = klax.finalize(sphnn)

    return sphnn_