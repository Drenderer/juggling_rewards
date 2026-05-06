#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import jax
import optax
import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, PyTree
import matplotlib.pyplot as plt
from dynax import ODESolver ,normalization_coefficients
import klax
import sys
from pathlib import Path
sys.path.append("..") 
from node import NODE
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

robot_train_coord = jnp.concatenate([coord_train_z , coord_train_dz] , axis=-1)
robot_test_coord = jnp.concatenate([coord_test_z , coord_test_dz] , axis=-1)


robot_train_input = jnp.concatenate([robot_train_x , robot_train_coord] , axis = -1)
robot_test_input = jnp.concatenate([robot_test_x , robot_test_coord] , axis = -1)

mask_train = jnp.any(robot_train_input != 0, axis=-1)
mask_test = jnp.any(robot_test_input != 0, axis=-1)


print (robot_train_input.shape , mask_train.shape)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
idx = 302  # choose one sample 

traj = robot_train_input[idx]   # shape (84, 12)

x = traj[:, 0]
y = traj[:, 1]
z = traj[:, 2]

t = range(len(x))

plt.figure(figsize=(8,5))

plt.plot(t, x, label='x')
plt.plot(t, y, label='y')
plt.plot(t, z, label='z')

plt.xlabel('index')
plt.ylabel('value')
plt.title(f'Sample {idx} (x, y, z vs index)')
plt.legend()

plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

    def __call__(self, ts_robot, u_robot , ball_init):

        h0 = self.encoder(ball_init)
        ball0 = self.decoder(h0)

        h = self.ode(ts_robot, h0, us=u_robot)

        y_ball = jax.vmap(self.decoder)(h)

    
        return y_ball, ball0


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
latent_dim =16
key = jr.key(0)
nn = NODE(state_size=latent_dim , input_size=12, width_sizes=[64,64,64], key=key)
ode = ODESolver(nn)
model = Model(nn, ode, latent_dim=latent_dim , key=key)

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

    time_shifted = robot_time - t_throw[:, None]
    dt = robot_time[:, 1] - robot_time[:, 0]

    grid = jnp.arange(T)[None, :]

    time_safe = (grid - last_valid_idx[:, None]) * dt[:, None]

    time_final = jnp.where(mask, time_shifted, time_safe)

    return time_final

time_train = make_time_throw_zero(robot_time_train, mask_train)
time_test  = make_time_throw_zero(robot_time_test, mask_test)




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def make_curriculum_weights(t: float, size: int, transition_width: float = 0.2):
    """Generate adaptive temporal weights for trajectory fitting.

    Computes weights w(tau):
        tau<t: 1
        tau>t+transition_width: 0
        else: smooth cosine transition
    for tau = linspace(0, 1, size)
    Returns the normalized weights (softmax). 

    Args:
        t: Normalized training time as *positive* float. 
            If t=0 then only the first `round(size * transition_width)` weights are non-zero.
            If t>1 then all weights are equal.
        size: Size of the output weights vector
        transition_width: Ratio of the transition length to size. Defaults to 0.2.

    Returns:
        Normalized weight vector of size `size`.

    """
    ts = jnp.linspace(0, 1, size)

    weights = jnp.where(
        ts < t,
        1.0,
        jnp.where(
            ts < t + transition_width,
            0.5 + 0.5 * jnp.cos((ts - t) * jnp.pi / transition_width),
            0.0,
        ),
    )
    return jax.nn.softmax(weights)

class RunStateUpdater(klax.Callback):
    """Updates the run_state to be the training step."""
    
    def on_training_step(self, context):
        context.state.run_state = context.state.step

@klax.loss
def curriculum_loss(model, batch, run_state):
    robot_ts, robot_batch, mask_batch , ball_batch = batch
    step = run_state
    ball0 = ball_batch[:,0,:]
    pred , init = jax.vmap(model , in_axes=(0,0,0))(robot_ts , robot_batch , ball0)
    mask = mask_batch[..., None]  # (B, T, 1)

    t_schedule = step / 5000
    weights = make_curriculum_weights(t_schedule, robot_ts.shape[-1])

    dynamic_loss_per_timestamp = jnp.sum(mask * jnp.square(pred - ball_batch), axis=(0, 2))
    dynamic_loss = jnp.dot(weights, dynamic_loss_per_timestamp) / jnp.sum(mask)

    landa1 = 0.2
    loss_enc_dec = landa1 * jnp.mean(jnp.square(init - ball0))

    return dynamic_loss + loss_enc_dec
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w =make_curriculum_weights(1 , 84)
plt.plot(w, label=f"t={t}")

plt.xlabel("Trajectory time index")
plt.ylabel("Weight")
plt.title("Curriculum Weights")
plt.legend()
plt.grid(True)

plt.show()
print (robot_time_test.shape[-1])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
''''
@klax.loss
def loss_Trajectory(model , data, batch_axis):
    robot_ts , robot_batch, mask_batch, ball_batch = data
    ball0 = ball_batch[:,0,:]
    pred , init = jax.vmap(model , in_axes=(0,0,0))(robot_ts , robot_batch , ball0)
    mask = mask_batch[..., None]  # (B, T, 1)
    loss_pred = jnp.sum(mask * jnp.square(pred - ball_batch)) / (
        jnp.sum(mask) * ball_batch.shape[-1]
    )
    loss_enc_dec = jnp.mean(jnp.square(init - ball0))
    landa1 = 0.2
    return loss_pred + landa1 *loss_enc_dec 
'''
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


model , hist_traj = klax.fit(
    model,
    (time_train, robot_train_input, mask_train , ball_train),
    validation_data=(time_test, robot_test_input,mask_test, ball_test),
    run_state=0,
    batch_size=32,
    optimizer=optax.adam(3e-4),
    loss=curriculum_loss,
    steps=5000,
    callbacks=[RunStateUpdater()],
    log_every=1,
    key=jr.key(0)
)

hist_traj.plot()
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model_ = klax.finalize(model)
idx = 103
pred_position , init = model_(time_test[idx] ,robot_test_input[idx] , ball_test[idx,0])

# ---- find throw index
last_valid_idx = jnp.sum(mask_test.astype(jnp.int32), axis=1) - 1
throw_idx = int(last_valid_idx[idx])

# ---- extract throw states
true_traj = ball_test[idx]   
true_throw = ball_test[idx, throw_idx, :]
pred_throw = pred_position[throw_idx, :]

print("true throw:", true_throw)
print("pred throw:", pred_throw)

# ---- free flight
DT = 0.002
ts = jnp.arange(2500) * DT

ball_q_true, true_time = ball_free_flight_trajecotry(true_throw, ts)
ball_q_pred, pred_time = ball_free_flight_trajecotry(pred_throw, ts)

labels = ["x", "y", "z", "vx", "vy", "vz"]

plt.figure(figsize=(12, 6))

for d in range(6):
    plt.subplot(2, 3, d + 1)
    valid = mask_test[idx]

    plt.plot(time_test[idx][valid], robot_test_input[idx, :, d][valid], label="cup")
    plt.plot(time_test[idx][valid], true_traj[valid, d], label="true ball")
    plt.plot(time_test[idx][valid], pred_position[valid, d], label="pred ball")

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
