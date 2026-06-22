#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import jax
import optax
import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, PyTree
import matplotlib.pyplot as plt
from dynax import normalization_coefficients 
import sys
from pathlib import Path
sys.path.append("..") 
from helping_function import  ball_free_flight_trajecotry 
from normalize import Normalization, coefficients

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = np.load('prepared_samples/train_data.npz')
robot_time_train  = data['robot_t'][  : ]
robot_train_q = data['robot_q'][: , :  , :]
robot_train_x = data['robot_x'][: , :  , :]
robot_train_coord = data['robot_coord'][: , :  , :]
ball_train= data['ball_y'][:, : ]
index_train = data['index'][:]

data = np.load('prepared_samples/test_data.npz')
robot_time_test  = data['robot_t'][: , : ]
robot_test_q = data['robot_q'][: , :  , :]
robot_test_x = data['robot_x'][: , :  , :]
robot_test_coord = data['robot_coord'][: , :  , :]
ball_test= data['ball_y'][: , : ]
index_test = data['index'][:]

print (robot_time_train.shape ,robot_train_q.shape , robot_train_x.shape ,
        robot_train_coord.shape,ball_train.shape , index_train.shape)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
coord_train_z = robot_train_coord [: , : , 6:9]
coord_train_dz = robot_train_coord[: , : ,15:18]

coord_test_z = robot_test_coord [: , : , 6:9]
coord_test_dz = robot_test_coord[: , : ,15:18]

robot_train_coord1 = jnp.concatenate([coord_train_z , coord_train_dz] , axis=-1)
robot_test_coord = jnp.concatenate([coord_test_z , coord_test_dz] , axis=-1)


robot_train_input = jnp.concatenate([robot_train_x , robot_train_coord1] , axis = -1)
robot_test_input = jnp.concatenate([robot_test_x , robot_test_coord] , axis = -1)

mask_train = jnp.any(robot_train_input != 0, axis=-1)
mask_test = jnp.any(robot_test_input != 0, axis=-1)


print (robot_train_input.shape , mask_train.shape)

#%%%%%%%%%%%%%%%%%%% normalize coefficient with dynax %%%%%%%%%%%%%%%%%%%%%%%%%
def masked_std(a, mask):
    valid = a[mask]
    return jnp.std(valid, axis=0)

def masked_mean(x, mask):
    valid = x[mask]
    return jnp.mean(valid, axis=0)


# split train
x_train  = robot_train_input[..., 0:3]
dx_train = robot_train_input[..., 3:6]
n_train  = robot_train_input[..., 6:9]
dn_train = robot_train_input[..., 9:12]
x_ball_train = ball_train[..., 0:3]
dx_ball_train = ball_train[..., 3:6]

# split train
x_test  = robot_test_input[..., 0:3]
dx_test = robot_test_input[..., 3:6]
n_test  = robot_test_input[..., 6:9]
dn_test = robot_test_input[..., 9:12]
x_ball_test = ball_test[..., 0:3]
dx_ball_test = ball_test[..., 3:6]

# y contains position-like variables v contains their velocities
y_train = jnp.concatenate([x_train, n_train , x_ball_train], axis=-1)
v_train = jnp.concatenate([dx_train, dn_train , dx_ball_train], axis=-1)

# y contains position-like variables v contains their velocities
y_test = jnp.concatenate([x_test, n_test , x_ball_test], axis=-1)
v_test = jnp.concatenate([dx_test, dn_test , dx_ball_test], axis=-1)

mean_y = masked_mean(y_train, mask_train)
std_y = masked_std(y_train, mask_train)
std_v = masked_std(v_train, mask_train)

alpha, tau = normalization_coefficients(
    std_y=std_y,
    std_v=std_v,
    verbosity=0,
)

print (alpha)
print (tau)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

norm = Normalization(
    mean_q=mean_y[:6],
    alpha_q=alpha[:6],
    tau_q=tau,
    mean_u=jnp.zeros((1,)),
    alpha_u=jnp.ones((1,))
)

ball_norm = Normalization(
    mean_q=mean_y[6:9],
    alpha_q=alpha[6:9],
    tau_q=norm.tau_q,   # important: same time scale as robot
    mean_u=jnp.zeros((1,)),
    alpha_u=jnp.ones((1,))
)


#%%%%%%%%%%%%%%%%%%%%%% normalizing train and test for robot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

robot_time_train_norm = norm.transform_ts (robot_time_train)
y_train_norm = norm.transform_qs(y_train[... , :6])
v_train_norm = norm.transform_q_ts(v_train[... ,:6])


robot_time_test_norm = norm.transform_ts (robot_time_test)
y_test_norm = norm.transform_qs(y_test[... , :6])
v_test_norm = norm.transform_q_ts(v_test[... ,:6])

robot_train_input_norm = jnp.concatenate([y_train_norm[... ,0:3] ,v_train_norm[... ,0:3],
                                        y_train_norm[... ,3:6],v_train_norm[... ,3:6]] , axis = -1)

robot_test_input_norm = jnp.concatenate([y_test_norm[... ,0:3] ,v_test_norm[... ,0:3],
                                        y_test_norm[... ,3:6],v_test_norm[... ,3:6]] , axis = -1)

print (robot_train_input_norm.shape)

#%%%%%%%%%%%%%%%%%%%%%% normalizing train and test for ball %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ball_pos_train_norm = ball_norm.transform_qs(x_ball_train)
ball_vel_train_norm = ball_norm.transform_q_ts(dx_ball_train)

ball_pos_test_norm = ball_norm.transform_qs(x_ball_test)
ball_vel_test_norm = ball_norm.transform_q_ts(dx_ball_test)

ball_train_norm = jnp.concatenate(
    [ball_pos_train_norm, ball_vel_train_norm],
    axis=-1
)

ball_test_norm = jnp.concatenate(
    [ball_pos_test_norm, ball_vel_test_norm],
    axis=-1
)

#%%%%%%%%%%%%%%%%%%% after mask must everything be 0 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
robot_train_input_norm = jnp.where(
    mask_train[..., None],
    robot_train_input_norm,
    0.0
)

robot_test_input_norm = jnp.where(
    mask_test[..., None],
    robot_test_input_norm,
    0.0
)

ball_train_norm = jnp.where(mask_train[..., None], ball_train_norm, 0.0)
ball_test_norm  = jnp.where(mask_test[..., None], ball_test_norm, 0.0)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

def make_time_start_zero(robot_time, mask):
    """
    Keep original valid time values.
    Only replace padded zeros with safe increasing values.
    """

    N, T = robot_time.shape

    valid_len = jnp.sum(mask.astype(jnp.int32), axis=1)
    last_valid_idx = valid_len - 1

    dt = robot_time[:, 1] - robot_time[:, 0]

    grid = jnp.arange(T)[None, :]

    last_valid_time = robot_time[jnp.arange(N), last_valid_idx]

    time_safe = last_valid_time[:, None] + (
        grid - last_valid_idx[:, None]
    ) * dt[:, None]

    time_final = jnp.where(mask, robot_time, time_safe)

    return time_final

time_train = make_time_start_zero(robot_time_train_norm, mask_train)
time_test  = make_time_start_zero(robot_time_test_norm, mask_test)
print (time_test[1])


# %%%%%%%%%%%%%%%%%%%%%% double check the normalization for robot x , n

idx =420
# normalized time, shifted so throw = 0
t = time_test[idx]
m = mask_test[idx]

true_x = robot_test_input[idx , : , 0:3]
true_n = robot_test_input[idx , : , 6:9]
true_dx = robot_test_input[idx , : , 3:6]
true_dn = robot_test_input[idx , : , 9:12]

check_x =robot_test_input_norm [ idx , : , 0:3]
check_n = robot_test_input_norm [ idx , : , 6:9]
check_dx =robot_test_input_norm [ idx , : , 3:6]
check_dn = robot_test_input_norm [ idx , : , 9:12]

check_position = jnp.concatenate ([check_x , check_n] , axis = -1)
check_velocity = jnp.concatenate ([check_dx , check_dn] , axis = -1)

check_position_inverse = norm.inverse_transform_qs(check_position)
check_velocity_inverse = norm.inverse_transform_q_ts(check_velocity)


print (true_dx[:10 , 0])
print (check_velocity_inverse [:10 , 0])
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% check the physical relationship for robot

valid_pair = m[1:] & m[:-1]
dt = t[1:] - t[:-1]

# finite-difference velocity in normalized domain
dx_from_x = (check_x[1:] - check_x[:-1]) / (t[1:] - t[:-1])[:, None]
dn_from_n = (check_n[1:] - check_n[:-1]) / (t[1:] - t[:-1])[:, None]

# compare with stored normalized velocity
dx_mid = 0.5 * (check_dx[1:] + check_dx[:-1])
dn_mid = 0.5 * (check_dn[1:] + check_dn[:-1])


t_mid = 0.5 * (t[1:] + t[:-1])
t_plot = t_mid[valid_pair]

for d in range(3):
    plt.figure(figsize=(8, 4))
    plt.plot(t_plot, dx_from_x[valid_pair, d], label="Reconstructed Vx ")
    plt.plot(t_plot, dx_mid[valid_pair, d], "--", label="Vx")
    plt.title(f"Normalized x-dx relation, dim {d}")
    plt.xlabel("normalized time")
    plt.ylabel("normalized velocity")
    plt.legend()
    plt.grid(True)
    plt.show()

for d in range(3):
    plt.figure(figsize=(8, 4))
    plt.plot(t_plot, dn_from_n[valid_pair, d], label="Reconstructed Vn")
    plt.plot(t_plot, dn_mid[valid_pair, d], "--", label="Vn")
    plt.title(f"Normalized n-dn relation, dim {d}")
    plt.xlabel("normalized time")
    plt.ylabel("normalized velocity")
    plt.legend()
    plt.grid(True)
    plt.show()

# %%%%%%%%%%%%%%%%%%%%%% double check the normalization for ball x 
true_ball_x = ball_test[idx , : , 0:3]
true_ball_dx = ball_test[idx , : , 3:6]

check_ball_x = ball_test_norm[idx , : , 0:3]
check_ball_dx = ball_test_norm[idx , : , 3:6]

check_position_inverse = ball_norm.inverse_transform_qs(check_ball_x)
check_velocity_inverse = ball_norm.inverse_transform_q_ts(check_ball_dx)

print (true_ball_x[:10 , 0])
print (check_position_inverse[:10 , 0])


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% check the physical relationship for ball

dx_from_x = (check_ball_x[1:] - check_ball_x[:-1]) / (t[1:] - t[:-1])[:, None]
dx_mid = 0.5 * (check_ball_dx[1:] + check_ball_dx[:-1])

for d in range(3):
    plt.figure(figsize=(8, 4))
    plt.plot(t_plot, dx_from_x[valid_pair, d], label="Reconstructed Vx for ball ")
    plt.plot(t_plot, dx_mid[valid_pair, d], "--", label="Vx for ball")
    plt.title(f"Normalized x-dx relation, dim {d}")
    plt.xlabel("normalized time")
    plt.ylabel("normalized velocity")
    plt.legend()
    plt.grid(True)
    plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%% check the general condition after normalization

mean_robot_pos = masked_mean(robot_train_input_norm[... , 0:3], mask_train)
std_robot_pos = masked_std (robot_train_input_norm[... , 0:3] , mask_train)

print (mean_robot_pos , std_robot_pos)

mean_robot_vel = masked_mean(robot_train_input_norm[... , 3:6], mask_train)
std_robot_vel = masked_std (robot_train_input_norm[... , 3:6] , mask_train)

print (mean_robot_vel , std_robot_vel)

mean_coord_pos = masked_mean(robot_train_input_norm[... , 6:9], mask_train)
std_coord_pos = masked_std (robot_train_input_norm[... , 6:9] , mask_train)

print (mean_coord_pos , std_coord_pos)

mean_coord_vel = masked_mean(robot_train_input_norm[... , 9:12], mask_train)
std_coord_vel = masked_std (robot_train_input_norm[... , 9:12] , mask_train)

print (mean_coord_vel , std_coord_vel)

mean_ball_pos = masked_mean(ball_train_norm[...,0:3] , mask_train)
std_ball_pos = masked_std(ball_train_norm[...,0:3] , mask_train)

print (mean_ball_pos , std_ball_pos)

mean_ball_vel = masked_mean(ball_train_norm[...,3:6] , mask_train)
std_ball_vel = masked_std(ball_train_norm[...,3:6] , mask_train)

print (mean_ball_vel , std_ball_vel)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nz = robot_train_input_norm [: , : , 6:9]
nz_norm = jnp.linalg.norm(nz, axis=-1)   # (N,T)

mask = mask_train
if mask.ndim == 3:
    mask = mask[..., 0]                  # (N,T)

print("nz_norm shape:", nz_norm.shape)
print("mask shape:", mask.shape)

valid_nz_norm = jnp.where(mask, nz_norm, jnp.nan)

print("mean norm:", jnp.nanmean(valid_nz_norm))
print("min norm:", jnp.nanmin(valid_nz_norm))
print("max norm:", jnp.nanmax(valid_nz_norm))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
np.savez ('prepared_samples/train_data_norm.npz' , time = robot_time_train_norm
                                                 , robot_norm = robot_train_input_norm
                                                 , ball_norm = ball_train_norm)


np.savez ('prepared_samples/test_data_norm.npz' , time = robot_time_test_norm
                                                 , robot_norm = robot_test_input_norm
                                                 , ball_norm = ball_test_norm)

np.savez('prepared_samples/norm_value.npz' , alpha = alpha[:6] , alpha_ball = alpha[6:9] , 
                                            tau = tau , mean_y = mean_y[:6] , mean_ball = mean_y[6:9])
# %%
