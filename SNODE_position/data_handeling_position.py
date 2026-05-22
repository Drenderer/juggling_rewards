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
from helping_function import  find_throwing , Forward_kinematic , hitting
from normalize import Normalization, coefficients

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%import train Dataset
base_path = Path(__file__).resolve().parent
data_path = base_path.parent / "Data/clean_data/third dataset/robot_throwing_train.npz"
data_train_robot = np.load(data_path)
time_train = data_train_robot['time']
robot_train_q = data_train_robot['robot_q']
robot_train_dq = data_train_robot['robot_dq']
robot_train_ddq = data_train_robot['robot_ddq']
robot_train_u = data_train_robot['robot_u']

print (time_train.shape,robot_train_q.shape , robot_train_ddq.shape , robot_train_u.shape)


base_path = Path(__file__).resolve().parent
data_path = base_path.parent / "Data/clean_data/third dataset/ball_throwing_train.npz"
data_train_ball = np.load(data_path)

ball_train_x = data_train_ball['ball_q']
ball_train_dx = data_train_ball['ball_dq']
ball_train_f = data_train_ball['ball_f']
ball_train_c = data_train_ball['ball_c']


print (ball_train_x.shape , ball_train_dx.shape , ball_train_f.shape)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%import test Dataset
base_path = Path(__file__).resolve().parent
data_path = base_path.parent / "Data/clean_data/third dataset/robot_throwing_test.npz"
data_test_robot = np.load(data_path)
time_test = data_test_robot['time']
robot_test_q = data_test_robot['robot_q']
robot_test_dq = data_test_robot['robot_dq']
robot_test_ddq = data_test_robot['robot_ddq']
robot_test_u = data_test_robot['robot_u']

print (time_test.shape,robot_test_q.shape , robot_test_ddq.shape , robot_test_u.shape)


base_path = Path(__file__).resolve().parent
data_path = base_path.parent / "Data/clean_data/third dataset/ball_throwing_test.npz"
data_test_ball = np.load(data_path)

ball_test_x = data_test_ball['ball_q']
ball_test_dx = data_test_ball['ball_dq']
ball_test_f = data_test_ball['ball_f']
ball_test_c = data_test_ball['ball_c']


print (ball_test_x.shape , ball_test_dx.shape , ball_test_f.shape)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def find_biggest_throw_index(ball_c, rest_time=50):
    indices = []

    for i in range(ball_c.shape[0]):
        idx = find_throwing(ball_c[i], rest_time)
        indices.append(idx)

    return max(indices)

max_train = find_biggest_throw_index(ball_train_c)
max_test = find_biggest_throw_index(ball_test_c)

print (max_train , max_test)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def find_smallest_flight(ball_c, rest_time=50):
    indices = []

    for i in range(ball_c.shape[0]):
        idx = find_throwing(ball_c[i], rest_time)
        idx_hit = hitting(ball_c[i], idx)
        indices.append(idx_hit)

    return max(indices)

max_train = find_smallest_flight(ball_train_c)
max_test = find_smallest_flight(ball_test_c)

print(max_train, max_test)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def prepare_dataset_position(time ,robot_q , robot_dq , robot_ddq , robot_u,
                  ball_x , ball_dx , ball_c , step=10  , window_size = 150 , rest_time =50):

    N_total, T, _ = robot_q.shape
    robot_y , robot_t , ball_y , index  = [] , [] , [] , []
    for i in range(N_total):

        idx = find_throwing(ball_c[i] , rest_time )
        idx_hit = hitting(ball_c[i] , idx)
        robot_full_data = np.concatenate([robot_q[i] , robot_dq[i] , robot_ddq[i] , robot_u[i]] , axis =-1) 
        ball_full_data = np.concatenate([ball_x[i] , ball_dx[i]], axis=-1)

        robot_dim = robot_full_data.shape[-1]
        ball_dim = ball_full_data.shape[-1]

        robot_data = np.zeros((window_size, robot_dim))
        ball_data = np.zeros((window_size, ball_dim))
        t = np.zeros((window_size,))
        
        end_idx = idx_hit

        cell_throw = end_idx // step
        sample_indices = end_idx - np.arange(cell_throw, -1, -1) * step
        n_samples = len(sample_indices)

        robot_data[:n_samples] = robot_full_data[sample_indices]
        ball_data[:n_samples] = ball_full_data[sample_indices]
        t[:n_samples] = time[i, sample_indices]

    

        robot_t.append(t)
        robot_y.append(robot_data)
        ball_y.append(ball_data)
        index.append(idx)
    
    return robot_t, robot_y , ball_y , index

    
robot_train_t , robot_train_y , ball_train_y, index_train = prepare_dataset_position(time_train,robot_train_q , 
                                                            robot_train_dq , 
                                                            robot_train_ddq , robot_train_u,ball_train_x , 
                                                            ball_train_dx , ball_train_c)
robot_train_t = np.array(robot_train_t)
robot_train_y = np.array(robot_train_y)
ball_train_y = np.array(ball_train_y)
index_train = np.array (index_train)


robot_test_t , robot_test_y , ball_test_y, index_test = prepare_dataset_position(time_test,robot_test_q , 
                                                            robot_test_dq , 
                                                            robot_test_ddq , robot_test_u,ball_test_x , 
                                                            ball_test_dx , ball_test_c)
robot_test_t = np.array(robot_test_t)
robot_test_y = np.array(robot_test_y)
ball_test_y = np.array(ball_test_y)
index_test = np.array (index_test)


print (robot_train_t.shape, robot_train_y.shape , ball_train_y.shape , index_train.shape)
print (robot_test_t.shape, robot_test_y.shape , ball_test_y.shape , index_test.shape)

robot_train_y = robot_train_y[: , : , :8]
robot_test_y = robot_test_y[: , : , :8]


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

    valid_mask = jnp.any(robot_data != 0, axis=-1, keepdims=True)  # (N,T,1)


    pos = jax.vmap(jax.vmap(fk_position))(q)                 # (N, T, 3)
    vel = jax.vmap(jax.vmap(compute_velocity))(q, dq)        # (N, T, 3)

    nx, ny, nz = jax.vmap(jax.vmap(fk_coordinate))(q)        # each (N, T, 3)

    dnx = jax.vmap(jax.vmap(compute_nx_velocity))(q, dq)     # (N, T, 3)
    dny = jax.vmap(jax.vmap(compute_ny_velocity))(q, dq)     # (N, T, 3)
    dnz = jax.vmap(jax.vmap(compute_nz_velocity))(q, dq)     # (N, T, 3)

    coord = jnp.concatenate([nx, ny, nz, dnx, dny, dnz], axis=-1)   # (N, T, 18)
    robot_x = jnp.concatenate([pos, vel], axis=-1)                  # (N, T, 6)

    robot_x = jnp.where(valid_mask, robot_x, 0.0)
    coord = jnp.where(valid_mask, coord, 0.0)
    
    return robot_x, coord

robot_train_x, robot_train_coord = full_state(robot_train_y)
print(robot_train_x.shape, robot_train_coord.shape)

robot_test_x, robot_test_coord = full_state(robot_test_y)
print(robot_test_x.shape, robot_test_coord.shape)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
np.savez ('prepared_samples/train_data.npz' , robot_t= robot_train_t ,robot_q = robot_train_q , 
                                            robot_x = robot_train_x , robot_coord = robot_train_coord,
                                            ball_y = ball_train_y , index = index_train)

np.savez ('prepared_samples/test_data.npz' , robot_t= robot_test_t ,robot_q = robot_test_q , 
                                            robot_x = robot_test_x , robot_coord = robot_test_coord,
                                            ball_y = ball_test_y , index = index_test)
# %%
