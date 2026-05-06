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
def make_time_throw_zero(robot_time, mask):
    """
    robot_time: (N, T)
    mask:       (N, T), True for valid data, False for padding

    Output:
    - valid part uses real measured time
    - throw moment is exactly t = 0
    - padded part continues increasing safely for ODE
    """

    N, T = robot_time.shape

    last_valid_idx = jnp.sum(mask.astype(jnp.int32), axis=1) - 1

    t_throw = robot_time[jnp.arange(N), last_valid_idx]

    # real shifted time
    time_shifted = robot_time - t_throw[:, None]

    # estimate dt from valid data
    dt = robot_time[:, 1] - robot_time[:, 0]

    grid = jnp.arange(T)[None, :]

    # safe increasing time for padded part
    time_safe = (grid - last_valid_idx[:, None]) * dt[:, None]

    # use real time for valid part, safe time for padding
    time_final = jnp.where(mask, time_shifted, time_safe)

    return time_final

time_train = make_time_throw_zero(robot_time_train, mask_train)
time_test  = make_time_throw_zero(robot_time_test, mask_test)

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

    def __call__(self, ts_robot, u_robot , ball0):

        h0 = self.encoder(ball0)
        ball0 = self.decoder(h0)

        h = self.ode(ts_robot, h0, us=u_robot)

        y_ball = jax.vmap(self.decoder)(h)

    
        return y_ball, ball0

    

latent_dim = 16
key = jr.key(0)

encoder = NODE(state_size=latent_dim, input_size=12, width_sizes=[64,64, 64], key=key)
ode = ODESolver(encoder)
model_template = Model(encoder, ode, latent_dim=latent_dim, key=key)

model_loaded = eqx.tree_deserialise_leaves("trained_model_position13.eqx", model_template)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model_ = klax.finalize(model_loaded)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DT = 0.002
ts = jnp.arange(2500) * DT

last_valid_idx = jnp.sum(mask_test.astype(jnp.int32), axis=1) - 1

N_eval = 2000

time_eval = time_test[:N_eval]
robot_eval = robot_test_input[:N_eval]
mask_eval = mask_test[:N_eval]
ball_eval = ball_test[:N_eval]
ball0_eval = ball_eval[: , 0 ,:]
last_valid_eval = last_valid_idx[:N_eval]


def one_sample(time_i, robot_i, ball0_i, ball_i, throw_idx_i):
    pred_traj_i, init_i = model_(time_i, robot_i, ball0_i)

    true_throw_i = ball_i[throw_idx_i, :]
    pred_throw_i = pred_traj_i[throw_idx_i, :]

    q_true_i, true_time_i = ball_free_flight_trajecotry(true_throw_i, ts)
    q_pred_i, pred_time_i = ball_free_flight_trajecotry(pred_throw_i, ts)

    return q_true_i, q_pred_i, true_time_i, pred_time_i


batched_eval = jax.jit(jax.vmap(one_sample, in_axes=(0, 0, 0, 0, 0)))

q_total_true, q_total_pred, total_true_time, total_pred_time = batched_eval(
    time_eval,
    robot_eval,
    ball0_eval,
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
idx = 1481
valid = mask_test[idx]
pred_position , init = model_(time_test[idx] ,robot_test_input[idx] , mask_test[idx])
labels = ["x", "y", "z", "vx", "vy", "vz"]

plt.figure(figsize=(12, 6))

for i in range (6):
    plt.subplot(2, 3, i + 1)
    plt.plot(time_test[idx,][valid] , ball_test[idx, :,i][valid] , lw=1.6 , label = "True ball " , color = 'g')
    plt.plot(time_test[idx,][valid] , robot_test_x[ idx ,:,i][valid], lw=1.6 , label = "robot", color ='b')
    plt.plot(time_test[idx][valid], pred_position[valid, i], label="pred ball" , color='r')
    
    plt.xlabel("time [s]")
    plt.ylabel(labels[i])
    plt.title(labels[i])
    plt.grid(alpha=0.3)
    plt.legend()

plt.suptitle(f"Ball vs Cup (before throw) - sample {idx}", fontsize=14)
plt.tight_layout()
plt.show()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
time = total_true_time[idx]
labels = ["x", "y", "z" , "vx" , "vy" , "vz"]

                  
plt.figure(figsize=(12, 6))
for d in range(6):
    plt.subplot(2, 3, d + 1)
    plt.plot(ts[:time], q_total_pred[idx ,:time, d], label="Prediction")
    plt.plot(ts[:time], q_total_true[idx , :time, d], label="True value")
    plt.xlabel("time [s]")          # <-- real time
    plt.ylabel(labels[d])
    plt.title(labels[d])
    plt.grid(alpha=0.3)
    plt.legend()

plt.suptitle(f"Ball state prediction - sample {idx}", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

error_x = jnp.abs(q_total_true[idx ,time , 0] - q_total_pred[idx ,time , 0]) 
error_y = jnp.abs(q_total_true[idx ,time, 1] - q_total_pred[idx ,time, 1]) 
error_z = jnp.abs(q_total_true[idx ,time , 2] - q_total_pred[idx ,time , 2])

err_idx = jnp.sqrt (error_x**2 + error_y**2 + error_z**2)

print ("error in x:" , error_x)
print ("error in y:" , error_y)
print ("error in z:" , error_z)
print ("distance error" , err_idx)

# %%
print (time)
print (q_total_true[idx, 0 , :])
# %%
