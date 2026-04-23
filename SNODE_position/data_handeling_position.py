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
from helping_function import  find_throwing , Forward_kinematic
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
def prepare_dataset_position(time ,robot_q , robot_dq , robot_ddq , robot_u,
                  ball_x , ball_dx , ball_c , step=2 , window_size = 20 , rest_time =50):
    """"
    this function build dataset for training state_less NODE which can predict only position


    return
    robot_y : (N , window_size , 16)      robot data [q ,dq , ddq , u] for last window size
    tobot_t :(N , window_size)            time of last window size
    ball_y : (N , 6)                      ball position and velocity from t=T_throw 
    index : (N ,)                         time of throw
    """

    key = jr.PRNGKey(0)
    N_total, T, _ = robot_q.shape
    robot_y , robot_t , ball_y , index  = [] , [] , [] , []
    for i in range(N_total):

        idx = find_throwing(ball_c[i] , rest_time )
        robot_data = np.concatenate([robot_q[i] , robot_dq[i] , robot_ddq[i] , robot_u[i]] , axis =-1)
        robot_data = robot_data [idx - window_size + 1 : idx  + 1]             # cutting the last window_size steps for robot 
        ball_data = np.concatenate([ball_x[i] , ball_dx[i]], axis=-1)
        ball_data = ball_data[idx - window_size + 1 : idx  + 1]
        t = time[i , idx - window_size  + 1 : idx + 1]
        
        #build dilated indices
        #offsets = np.arange(window_size)[::-1] * step
        #indices = idx - offsets
        #robot_data = robot_data[indices]
        #ball_data = ball_data[indices]
        #t = time[i, indices]

        
        robot_t.append(t)
        robot_y.append(robot_data)
        ball_y.append(ball_data)
        index.append(idx)
    
    return robot_t, robot_y , ball_y , index

    
robot_t , robot_y , ball_y, index = prepare_dataset_position(time,robot_q , robot_dq , robot_ddq 
                                                             , robot_u,ball_x , ball_dx , ball_c)
robot_t = np.array(robot_t)
robot_y = np.array(robot_y)
ball_y = np.array(ball_y)
index = np.array (index)

print (robot_t.shape, robot_y.shape , ball_y.shape , index.shape)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
''''
def _semi_flatten(x: Array) -> Array:
            return x.reshape(-1, x.shape[-1])


mean_x = _semi_flatten(robot_y[:,:,0:4]).mean(axis=0)
std_x = _semi_flatten(robot_y[:,:,0:4]).std(axis=0)
std_dx = _semi_flatten(robot_y[:,:,4:8]).std(axis=0)
std_ddx = _semi_flatten(robot_y[:,:,8:12]).std(axis=0)
mean_u = _semi_flatten(robot_y[:,:,12:16]).mean(axis=0)
std_u = _semi_flatten(robot_y[:,:,12:16]).std(axis=0)

alpha_x, tau_x , alpha_u = coefficients (mean_x , std_x , std_u ,std_dx , std_ddx)

norm = Normalization (mean_q=mean_x, alpha_q=alpha_x, tau_q=tau_x,
                      mean_u=mean_u, alpha_u=alpha_u)

robot_y[:,:,0:4] = norm.transform_qs(robot_y[:,:,0:4])
robot_y[:,:,4:8] = norm.transform_q_ts(robot_y[:,:,4:8])
robot_y[:,:,8:12] = norm.transform_q_tts(robot_y[:,:,8:12])
robot_y[:,:,12:16] = norm.transform_taus(robot_y[:,:,12:16])
robot_t = norm.transform_ts(robot_t)
'''

robot_y = robot_y[: , : , :8]

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def fk_position(q):
    matrix = Forward_kinematic(q)
    return matrix[:3, 3]

def fk_coordinate(q):
    matrix = Forward_kinematic(q)
    nz = matrix[:3,2]
    ny = matrix[:3,1]
    nx = matrix[:3,0]
    return nx , ny , nz

def fk_nx(q):
    matrix = Forward_kinematic(q)
    return matrix[:3, 0]

def fk_ny(q):
    matrix = Forward_kinematic(q)
    return matrix[:3, 1]

def fk_nz(q):
    matrix = Forward_kinematic(q)
    return matrix[:3, 2]

jac_fn = jax.jacobian(fk_position)
jac_nx_fn = jax.jacobian(fk_nx)
jac_ny_fn = jax.jacobian(fk_ny)
jac_nz_fn = jax.jacobian(fk_nz)

def compute_velocity(q, dq):
    J = jac_fn(q)
    return J @ dq

def compute_nx_velocity(q, dq):
    J_nx = jac_nx_fn(q)
    return J_nx @ dq

def compute_ny_velocity(q, dq):
    J_ny = jac_ny_fn(q)
    return J_ny @ dq

def compute_nz_velocity(q, dq):
    J_nz = jac_nz_fn(q)
    return J_nz @ dq

def full_state(robot_data):
    q = robot_data[:, :, :4]    # (N, T, 4)
    dq = robot_data[:, :, 4:8]  # (N, T, 4)

    pos = jax.vmap(jax.vmap(fk_position))(q)                 # (N, T, 3)
    vel = jax.vmap(jax.vmap(compute_velocity))(q, dq)        # (N, T, 3)

    nx, ny, nz = jax.vmap(jax.vmap(fk_coordinate))(q)        # each (N, T, 3)

    dnx = jax.vmap(jax.vmap(compute_nx_velocity))(q, dq)     # (N, T, 3)
    dny = jax.vmap(jax.vmap(compute_ny_velocity))(q, dq)     # (N, T, 3)
    dnz = jax.vmap(jax.vmap(compute_nz_velocity))(q, dq)     # (N, T, 3)

    coord = jnp.concatenate([nx, ny, nz, dnx, dny, dnz], axis=-1)   # (N, T, 18)
    robot_x = jnp.concatenate([pos, vel], axis=-1)                  # (N, T, 6)

    return robot_x, coord

robot_x, robot_coord = full_state(robot_y)
print(robot_x.shape, robot_coord.shape)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train_ratio = 0.8
N = robot_y.shape[0]
rng = np.random.default_rng(seed=42)   
perm = rng.permutation(N)

N_train = int(train_ratio * N)

train_idx = perm[:N_train]
test_idx  = perm[N_train:]

robot_time_train = robot_t[train_idx]
robot_train_q = robot_y[train_idx]
robot_train_x = robot_x[train_idx]
robot_train_coord = robot_coord[train_idx]
ball_train  = ball_y[train_idx]
index_train = index[train_idx]

robot_time_test = robot_t[test_idx]
robot_test_q = robot_y[test_idx]
robot_test_x = robot_x[test_idx]
robot_test_coord = robot_coord[test_idx]
ball_test  = ball_y[test_idx]
index_test = index[test_idx]

print("Train:", robot_train_x.shape, ball_train.shape)
print("Test :", robot_test_x.shape, ball_test.shape)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
np.savez ('prepared_samples/train_data.npz' , robot_t= robot_time_train ,robot_q = robot_train_q , 
                                            robot_x = robot_train_x , robot_coord = robot_train_coord,
                                            ball_y = ball_train , index = index_train)

np.savez ('prepared_samples/test_data.npz' , robot_t= robot_time_test ,robot_q = robot_test_q , 
                                            robot_x = robot_test_x , robot_coord = robot_test_coord,
                                            ball_y = ball_test , index = index_test)
# %%
