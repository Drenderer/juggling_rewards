#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import jax
import optax
import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, PyTree
import matplotlib.pyplot as plt
from dynax import ODESolver ,normalization_coefficients , ISPHS
import klax
from klax.nn import MLP , ConstantSPDMatrix , ConstantMatrix , ConstantSkewSymmetricMatrix
import sys
from pathlib import Path
sys.path.append("..") 
from node import NODE
from helping_function import  ball_free_flight_trajecotry  ,ball_free_flight_trajecotry_training
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

#%%%%%%%%%%%%%%%%%%%% import norm data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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



#%%%%%%%%%%%%%%%%%%%%%% build contact dataset for test and train %%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_test , T , _ = ball_test.shape
N_train , T, _  =ball_train.shape

contact_train = jnp.zeros((N_train , T))
contact_test = jnp.zeros((N_test , T))

throw_idx_train = (index_train // 10) +1
throw_idx_test = (index_test // 10) +1

# grid of time indices
t_grid = jnp.arange(T)[None, :]   # shape: (1, T)

# 1 before throw, 0 after throw
contact_train = (t_grid < throw_idx_train[:, None]).astype(jnp.float32)
contact_test = (t_grid < throw_idx_test[:, None]).astype(jnp.float32)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

idx = 120
throw_index = (index_test[idx] // 10 ) +1
throw_idx_test = jnp.sum(contact_test.astype(jnp.int32), axis=1) 

print (throw_index , throw_idx_test[idx])

#%%
k=-2
for i in range(4):
    print (k)
    print (contact_test[idx,throw_index+k])
    print (ball_test[idx,throw_index+ k , 3:6])
    i=i+1
    k=k+1

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

traj = robot_test_input[idx]   # shape (84, 12)

x = traj[:, 0]
y = traj[:, 1]
z = traj[:, 2]

t = range(len(x))

plt.figure(figsize=(8,5))

plt.plot(t, x, label='x')
plt.plot(t, y, label='y')
plt.plot(t, z, label='z')
plt.plot(t, contact_test[idx] , label='contact')
plt.plot(t , mask_test[idx], label='mask')


plt.xlabel('index')
plt.ylabel('value')
plt.title(f'Sample {idx} (x, y, z vs index)')
plt.legend()

plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class GRUThrowTimeModel(eqx.Module):
    gru: eqx.nn.GRUCell
    head: eqx.nn.MLP
    hidden_size: int = eqx.field(static=True)

    def __init__(self, input_size, hidden_size, key):
        key1, key2 = jr.split(key, 2)

        self.hidden_size = hidden_size

        self.gru = eqx.nn.GRUCell(
            input_size=input_size,
            hidden_size=hidden_size,
            key=key1,
        )

        self.head = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=1,
            width_size=64,
            depth=2,
            activation=jax.nn.softplus,
            key=key2,
        )

    def __call__(self, robot_seq):
        # robot_seq: (T, input_size)

        h0 = jnp.zeros((self.hidden_size,))

        def step(h, x):
            h_new = self.gru(x, h)
            return h_new, h_new

        h_final, h_all = jax.lax.scan(step, h0, robot_seq)

        t_hat = jnp.ravel(self.head(h_final))[0]
        return t_hat
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
input_size = robot_train_input.shape[-1]   # probably 12
hidden_size = 128

key = jr.key(11)

model_gru = GRUThrowTimeModel(
    input_size=input_size,
    hidden_size=hidden_size,
    key=key,
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


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t_throw_train = time_train[
    jnp.arange(time_train.shape[0]),
    throw_idx_train
]
t_throw_test = time_test[
    jnp.arange(time_test.shape[0]),
    throw_idx_test
]


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@klax.loss
def loss_gru_throw_time(model, data, batch_axis):
    robot_batch, t_throw_batch = data

    t_pred = jax.vmap(model)(robot_batch)

    loss = jnp.mean((t_pred - t_throw_batch) ** 2)

    return loss
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class RunStateUpdater(klax.Callback):
    """Updates the run_state to be the training step."""
    
    def on_training_step(self, context):
        context.state.run_state = context.state.step

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

model_gru, hist_gru = klax.fit(
    model_gru,
    (robot_train_input_norm, t_throw_train),
    validation_data=(robot_test_input_norm, t_throw_test),
    run_state=0,
    batch_size=64,
    optimizer=optax.adam(1e-4),
    loss=loss_gru_throw_time,
    steps=20000,
    verbose=True,
    callbacks=[RunStateUpdater()],
    log_every=50,
    key=jr.key(5),
)

hist_gru.plot()
plt.show()

model_gru, hist_gru = klax.fit(
    model_gru,
    (robot_train_input_norm, t_throw_train),
    validation_data=(robot_test_input_norm, t_throw_test),
    run_state=0,
    batch_size=64,
    optimizer=optax.adam(3e-5),
    loss=loss_gru_throw_time,
    steps=20000,
    verbose=True,
    callbacks=[RunStateUpdater()],
    log_every=50,
    key=jr.key(5),
)

hist_gru.plot()
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model_gru_ = klax.finalize(model_gru)
idx=6
pred = model_gru(robot_test_input_norm[idx])
print (pred , t_throw_test[idx])

pred_denorm = ball_norm.inverse_transform_ts(pred)
true_denorm = ball_norm.inverse_transform_ts(t_throw_test[idx])

print (pred_denorm , true_denorm)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_eval = 2000

robot_eval = robot_test_input_norm[:N_eval]
throw_eval = t_throw_test[:N_eval]

throw_pred = jax.jit(jax.vmap(model_gru_))(robot_eval)
throw_true = throw_eval

pred_denorm = ball_norm.inverse_transform_ts(throw_pred)
true_denorm = ball_norm.inverse_transform_ts(throw_eval)

diff = np.array(jnp.abs(pred_denorm - true_denorm))

D = 0.04

error_small = diff[diff <= D]
error_large = diff[diff > D]

plt.figure(figsize=(8, 4))
bins = jnp.linspace(diff.min(), diff.max(), 50)
plt.hist(error_small, bins=bins, color="tab:blue", alpha=0.7, label=f"Error ≤ {D}")
plt.hist(error_large, bins=bins, color="tab:red",  alpha=0.7, label=f"Error > {D}")

plt.axvline(D, color="black", linestyle="--", linewidth=2, label="Threshold")
plt.xlabel("throw time error: pred - true [s]")
plt.ylabel("number of samples")
plt.title("Histogram of throw-time error")
plt.grid(True , axis="y")
plt.legend()
plt.tight_layout()
plt.show()

print(f"% within {D:.3f}s = {100*np.mean(diff <= D):.1f}%")



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# --- de-normalize time
#time_denorm_train = ball_norm.inverse_transform_ts(time_train)

plt.figure(figsize=(8,5))

plt.plot(time_test[idx], contact_test[idx], label='true throw' , linestyle='--', color='b')
#plt.plot(time_denorm_train[idx], contact_pred, label='predict throw' , linestyle='--', color ='g')
plt.plot(time_test[idx], mask_test[idx], label='valid mask' , linestyle='--', color ='r')

plt.xlabel('time')
plt.ylabel('value')
plt.title(f'Sample {idx} (contact learning)')
plt.legend()

plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eqx.tree_serialise_leaves("saved_models/time/trained_model_time_6.eqx", model_gru_)

np.save(
    "saved_models/time/training_history_time_6.npy",
    hist_gru,
    allow_pickle=True,
)

# %%
