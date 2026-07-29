

import numpy as np
from src.datagen.normalize import coefficients
from jax import numpy as jnp

data_robot = np.load("Data_MPC/robot_data.npz")
time = data_robot['ts']
robot_q = data_robot['qs']
robot_dq = data_robot['qs_t']
robot_ddq = data_robot['qs_tt']
robot_u = data_robot['us']



mean_y = jnp.mean(robot_q ,axis=(0, 1))
std_y = jnp.std(robot_q , axis=(0, 1))
std_v = jnp.std(robot_dq , axis=(0, 1))
std_a = jnp.std(robot_ddq , axis=(0, 1))
mean_u = jnp.mean (robot_u , axis=(0, 1))
std_u = jnp.std(robot_u , axis=(0, 1))


alpha_q, tau_q , alpha_u = coefficients(mean_q=mean_y , std_q=std_y,
                                        std_u=std_u , std_v = std_v,std_a=std_a)

train_ratio = 0.8
N = robot_q.shape[0]
rng = np.random.default_rng(seed=42)   
perm = rng.permutation(N)
N_train = int(train_ratio * N)

train_idx = perm[:N_train]
test_idx  = perm[N_train:]

time_train = time[train_idx]
robot_train_q = robot_q[train_idx]
robot_train_dq = robot_dq[train_idx]
robot_train_ddq = robot_ddq[train_idx]
robot_train_u = robot_u[train_idx]

time_test = time[test_idx]
robot_test_q = robot_q[test_idx]
robot_test_dq = robot_dq[test_idx]
robot_test_ddq = robot_ddq[test_idx]
robot_test_u = robot_u[test_idx]



np.savez('Data_MPC/robot_train.npz', time =time_train , robot_q =robot_train_q , 
                              robot_dq = robot_train_dq , robot_ddq = robot_train_ddq , robot_u = robot_train_u)


np.savez('Data_MPC/robot_test.npz', time =time_test , robot_q =robot_test_q , 
                              robot_dq = robot_test_dq , robot_ddq = robot_test_ddq , robot_u = robot_test_u)


np.savez('Data_MPC/norm_value.npz', mean_q = mean_y , alpha_q=alpha_q , tau_q = tau_q , 
                                mean_u = mean_u , alpha_u=alpha_u)

print ("data is devided into test and train samples , coefficients for normalization is saved")
# %%
