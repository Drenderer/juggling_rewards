#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import jax
from jax import random as jr
from jax import numpy as jnp
from jaxtyping import Array, PyTree
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.append("..") 
from helping_function import hitting_ground, find_throwing
from normalize import Normalization, coefficients

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
base_path = Path(__file__).resolve().parent
data_path = base_path.parent / "Data/clean_data/second dataset/robot_throwing.npz"
data_robot = np.load(data_path)
time = data_robot['time']
robot_q = data_robot['robot_q']
robot_dq = data_robot['robot_dq']
robot_ddq = data_robot['robot_ddq']
robot_u = data_robot['robot_u']

print (time.shape,robot_q.shape , robot_ddq.shape , robot_u.shape)


base_path = Path(__file__).resolve().parent
data_path = base_path.parent / "Data/clean_data/second dataset/ball_throwing.npz"
data_ball = np.load(data_path)

ball_x = data_ball['ball_q']
ball_dx = data_ball['ball_dq']
ball_f = data_ball['ball_f']
ball_c = data_ball['ball_c']


print (ball_x.shape , ball_dx.shape , ball_f.shape)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
robot_q_forward = jnp.array(robot_q[: , 0 , :])
robot_x = jnp.array(ball_x[: , 0 , :])
robot_x = robot_x.at[:,2].add(-0.01)
print (robot_q_forward.shape , robot_x.shape)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def calculation (L , q):
    matrix = Forward_kinematic(L,q)
    x = matrix [0,3]
    y = matrix [1,3]
    z = matrix [2,3]
    robot_x = jnp.array([x , y , z])
    return robot_x

def loss(L , data , batch_axis):
    robot_batch , true_x = data
    pred_x = jax.vmap(calculation , in_axes=(None,0))(L , robot_batch)
    return jnp.mean(jnp.square(pred_x - true_x))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
L = jnp.array([0.1,0.1 ,0.1 ,0.1 , 0.1 , 0.1 , 0.1 ,0.1 , 0.1])
import klax
import optax
L , hist = klax.fit(
    L,
    (robot_q_forward[:5000] , robot_x[:5000]),
    optimizer=optax.adam(4e-4),
    loss_fn=loss,
    steps=30000,
    key=jr.key(0)
)

hist.plot()
plt.show()
print(L)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pred = jax.vmap(calculation, in_axes=(None, 0))(L, robot_q_forward[5000:10000 ,])

err_vec = pred - robot_x[5000:10000]
err_norm = jnp.linalg.norm(err_vec, axis=1)

print("Mean error (m):", jnp.mean(err_norm))
print("Max error  (m):", jnp.max(err_norm))
print("RMSE       (m):", jnp.sqrt(jnp.mean(err_norm**2)))
# %%
