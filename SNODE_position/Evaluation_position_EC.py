#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import jax
import numpy as np
import matplotlib.pyplot as plt
import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, PyTree
from dynax import ODESolver
import klax
import sys
from pathlib import Path
sys.path.append("..") 
from node import NODE
from helping_function import hitting_ground , find_throwing , ball_free_flight_trajecotry , Forward_kinematic
from normalize import Normalization, coefficients

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = np.load('prepared_samples/train_data.npz')
robot_time_train  = data['robot_t'][ : ,  : ]
robot_train_q = data['robot_q'][: , :  , :]
robot_train_x = data['robot_x'][: , :  , :]
robot_train_coord = data['robot_coord'][: , :  ,:]
ball_train= data['ball_y'][:, : ]
index_train = data['index'][:]

data = np.load('prepared_samples/test_data.npz')
robot_time_test  = data['robot_t'][: ,: ]
robot_test_q = data['robot_q'][: ,:  , :]
robot_test_x = data['robot_x'][: , :  ,:]
robot_test_coord = data['robot_coord'][: , :  , :]
ball_test= data['ball_y'][: , : ]
index_test = data['index'][:]

print (robot_time_train.shape ,robot_train_q.shape , robot_train_x.shape ,
        robot_train_coord.shape,ball_train.shape , index_train.shape)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = np.load('prepared_samples/train_data_norm.npz')
robot_time_train_norm = data['time']
robot_train_input_norm = data['robot_norm']
ball_train_norm = data['ball_norm']

data = np.load('prepared_samples/test_data_norm.npz')
robot_time_test_norm = data['time']
robot_test_input_norm = data['robot_norm']
ball_test_norm = data['ball_norm']

data = np.load('prepared_samples/norm_value.npz')
alpha = data['alpha']
alpha_ball = data['alpha_ball']
tau = data['tau']
mean_y = data['mean_y']
mean_ball = data['mean_ball']

print (robot_time_train_norm.shape , robot_train_input_norm.shape)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

coord_train_z = robot_train_coord [: , : , 6:9]
coord_train_dz = robot_train_coord[: , : ,15:18]

coord_test_z = robot_test_coord [: , : , 6:9]
coord_test_dz = robot_test_coord[: , : ,15:18]

robot_train_coord = jnp.concatenate([coord_train_z , coord_train_dz] , axis=-1)
robot_test_coord = jnp.concatenate([coord_test_z , coord_test_dz] , axis=-1)


robot_train_input = jnp.concatenate([robot_train_x , robot_train_coord] , axis = -1)
robot_test_input = jnp.concatenate([robot_test_x , robot_test_coord] , axis = -1)

mask_train = jnp.any(robot_train_input != 0, axis=-1)
mask_test = jnp.any(robot_test_input != 0, axis=-1)


print (robot_train_input.shape , mask_train.shape)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
norm = Normalization(
    mean_q=mean_y,
    alpha_q=alpha,
    tau_q=tau,
    mean_u=jnp.zeros((1,)),
    alpha_u=jnp.ones((1,))
)

ball_norm = Normalization(
    mean_q=mean_ball,
    alpha_q=alpha_ball,
    tau_q=norm.tau_q,  
    mean_u=jnp.zeros((1,)),
    alpha_u=jnp.ones((1,))
)

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


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Model (eqx.Module):
    node: NODE
    ode: ODESolver
    encoder:eqx.nn.MLP
    decoder:eqx.nn.MLP
    latent_dim: int = eqx.field(static=True)

    def __init__ (self, node  , ode , key , latent_dim):
        key1 , key2 = jr.split(key , 2)
        self.node = node
        self.ode = ode
        self.latent_dim = latent_dim

        self.encoder = eqx.nn.MLP(
             in_size = 6,
             out_size = latent_dim,
             width_size = 32,
             depth = 2,
             activation=jax.nn.softplus,
            key = key1,
        )
        self.decoder = eqx.nn.MLP(
            in_size = latent_dim,
            out_size= 6,
            width_size = 32,
            depth=2,
            activation=jax.nn.softplus,
            key=key2,
        )

    def __call__(self, ts_robot, u_robot , mask):

        h0 = self.encoder(u_robot[0,:6])
        ball0 = self.decoder(h0)

        h = self.ode(ts_robot, h0, us=u_robot)

        last_valid_idx = jnp.sum(mask.astype(jnp.int32)) - 1
        h_last_valid = h[last_valid_idx]

        y_ball = self.decoder(h_last_valid)
        #y_ball = jax.vmap(self.decoder)(h)

    
        return y_ball, ball0

    

latent_dim = 16
key = jr.key(0)

encoder = NODE(state_size=latent_dim, input_size=12, width_sizes=[64,64, 64], key=key)
ode = ODESolver(encoder)
model_template = Model(encoder, ode, latent_dim=latent_dim, key=key)

model_loaded = eqx.tree_deserialise_leaves("trained_model_position10_2.eqx", model_template)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model_ = klax.finalize(model_loaded)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DT = 0.002
ts = jnp.arange(2500) * DT

last_valid_idx = jnp.sum(mask_test.astype(jnp.int32), axis=1) - 1

N_eval = 2000

time_eval_norm = time_test[:N_eval]
robot_eval_norm = robot_test_input_norm[:N_eval]
mask_eval = mask_test[:N_eval]
ball_eval = ball_test[:N_eval]
last_valid_eval = last_valid_idx[:N_eval]


def one_sample(time_i, robot_i, mask_i, ball_i, throw_idx_i):
    pred_traj_i, init_i = model_(time_i, robot_i, mask_i)

    pred_x = ball_norm.inverse_transform_qs (pred_traj_i[:3])
    pred_dx = ball_norm.inverse_transform_q_ts(pred_traj_i[3:6])

    pred = jnp.concat([pred_x , pred_dx])

    
    true_throw_i = ball_i[throw_idx_i, :]

    q_true_i, true_time_i = ball_free_flight_trajecotry(true_throw_i, ts)
    q_pred_i, pred_time_i = ball_free_flight_trajecotry(pred, ts)

    return q_true_i, q_pred_i, true_time_i, pred_time_i


batched_eval = jax.jit(jax.vmap(one_sample, in_axes=(0, 0, 0, 0, 0)))

q_total_true, q_total_pred, total_true_time, total_pred_time = batched_eval(
    time_eval_norm,
    robot_eval_norm,
    mask_eval,
    ball_eval,
    last_valid_eval,
)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
error = []
for idx in range(2000):
    time = total_true_time[idx]
    error_x = jnp.abs(q_total_true[idx ,time , 0] - q_total_pred[idx ,time , 0]) 
    error_y = jnp.abs(q_total_true[idx ,time , 1] - q_total_pred[idx ,time , 1]) 
    error_z = jnp.abs(q_total_true[idx ,time , 2] - q_total_pred[idx ,time , 2])

    err1 = jnp.sqrt (error_x**2 + error_y**2 + error_z**2)

    error. append(err1)

error = jnp.array(error)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

D = 0.075  # Diameter of ball
R = D/2
error = jnp.asarray(error)

error_small = error[error <= D]
error_large = error[error > D]

plt.figure(figsize=(10, 5))
bins = jnp.linspace(error.min(), error.max(), 50)

plt.hist(error_small, bins=bins, color="tab:blue", alpha=0.7, label=f"Error ≤ {D}")
plt.hist(error_large, bins=bins, color="tab:red",  alpha=0.7, label=f"Error > {D}")

plt.axvline(D, color="black", linestyle="--", linewidth=2, label="Threshold")

plt.xlabel("Distance error [m]")
plt.ylabel("Number of samples")
plt.title("Distribution of position errors at Hitting Ground ")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_total = error.shape[0]
num_below = jnp.sum(error <= D)
percentage_below = 100.0 * num_below / num_total

print(f"Threshold R = {D} m")
print(f"Samples below R: {int(num_below)} / {num_total}")
print(f"Percentage below R: {float(percentage_below):.2f}%")




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

error0 = []
for idx in range(2000):
    error_x = jnp.abs(q_total_true[idx ,0 , 0] - q_total_pred[idx ,0 , 0]) 
    error_y = jnp.abs(q_total_true[idx ,0, 1] - q_total_pred[idx ,0 , 1]) 
    error_z = jnp.abs(q_total_true[idx ,0 , 2] - q_total_pred[idx ,0 , 2])

    err0 = jnp.sqrt (error_x**2 + error_y**2 + error_z**2)
    error0. append(err0)

error0 = jnp.array(error0)
error0_small = error0[error0 <= D]
error0_large = error0[error0 > D]
bins = jnp.linspace(error0.min(), error0.max(), 50)

plt.figure(figsize=(10, 5))
plt.hist(error0_small, bins=bins, color="tab:blue", alpha=0.7, label=f"Error ≤ {D}")
plt.hist(error0_large, bins=bins, color="tab:red",  alpha=0.7, label=f"Error > {D}")
plt.axvline(D, color="black", linestyle="--", linewidth=2, label="Threshold")
plt.xlabel("Distance error [m]")
plt.ylabel("Number of samples")
plt.title("Distribution of position errors  at Throwing Time")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_total = error0.shape[0]
num_below = jnp.sum(error0 <= D)
percentage_below = 100.0 * num_below / num_total

print(f"Threshold D = {D} m")
print(f"Samples below D: {int(num_below)} / {num_total}")
print(f"Percentage below D: {float(percentage_below):.2f}%")



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

indices_bad = jnp.where(error > 6*D)[0]
print ("bad ones" , indices_bad)

indices_good = jnp.where(error<R/2)[0]
print ("good ones" , indices_good)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


idx = 3
pred , init = model_(time_test[idx] ,robot_test_input[idx] , mask_test[idx])

# ---- find throw index
last_valid_idx = jnp.sum(mask_test.astype(jnp.int32), axis=1) - 1
throw_idx = int(last_valid_idx[idx])

# ---- extract throw states
true_traj = ball_test[idx]   # important
true_throw = ball_test[idx, throw_idx, :]
#pred_throw = pred_position[throw_idx, :]

print("true throw:", true_throw)
#print("pred throw:", pred_throw)

# ---- free flight
DT = 0.002
ts = jnp.arange(2500) * DT

ball_q_true, true_time = ball_free_flight_trajecotry(true_throw, ts)
ball_q_pred, pred_time = ball_free_flight_trajecotry(pred, ts)

labels = ["x", "y", "z", "vx", "vy", "vz"]

plt.figure(figsize=(12, 6))

for d in range(6):
    plt.subplot(2, 3, d + 1)
    valid = mask_test[idx]

    plt.plot(time_test[idx][valid], robot_test_input[idx, :, d][valid], label="cup")
    plt.plot(time_test[idx][valid], true_traj[valid, d], label="true ball")
    #plt.plot(time_test[idx][valid], pred_position[valid, d], label="pred ball")

    plt.xlabel("time [s]")
    plt.ylabel(labels[d])
    plt.title(labels[d])
    plt.grid(alpha=0.3)
    plt.legend()

plt.suptitle(f"Ball vs Cup (before throw) - sample {idx}", fontsize=14)
plt.tight_layout()
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plt.figure(figsize=(12, 6))
for d in range(6):
    plt.subplot(2, 3, d + 1)
    plt.plot(ts[:true_time], ball_q_pred[:true_time, d], label="Prediction")
    plt.plot(ts[:true_time], ball_q_true[:true_time, d], label="True value")
    plt.xlabel("time [s]")          # <-- real time
    plt.ylabel(labels[d])
    plt.title(labels[d])
    plt.grid(alpha=0.3)
    plt.legend()

plt.suptitle(f"Ball state prediction - sample {idx}", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

error_x = jnp.abs(ball_q_pred[true_time , 0] - ball_q_true[true_time , 0]) 
error_y = jnp.abs(ball_q_pred[true_time , 1] - ball_q_true[true_time , 1]) 
error_z = jnp.abs(ball_q_pred[true_time , 2] - ball_q_true[true_time , 2]) 
print ("error in x:" , error_x)
print ("error in y:" , error_y)
print ("error in z:" , error_z)

error = jnp.sqrt (error_x**2 + error_y**2 + error_z**2)
print ("distance error" , error)

# %%
