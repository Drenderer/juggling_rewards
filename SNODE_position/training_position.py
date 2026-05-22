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
from klax.nn import MLP , ConstantMatrix , ConstantSkewSymmetricMatrix , ConstantSPDMatrix
import sys
from pathlib import Path
sys.path.append("..") 
from node import NODE
from helping_function import  ball_free_flight_trajecotry 
from normalize import Normalization

#%%%%%%%%%%%%%%%%%%%% import real data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
#%%%%%%%%%%%%%%%%%% building [ x , dx , n , dn] %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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


#%%%%%%%%%%%%%%%%%%%%%% normalizing train and test for robot %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
    tau_q=norm.tau_q,   # important: same time scale as robot
    mean_u=jnp.zeros((1,)),
    alpha_u=jnp.ones((1,))
)


#%%%%%%%%%%%%%%%%%%% check the data before training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
idx =103 # choose sample
print (index_test[idx])
traj = robot_test_input[idx]   # shape (90, 12)
traj_ball = ball_test[idx]
x = traj[:, 0]
y = traj[:, 1]
z = traj[:, 2]
ball_x = traj_ball[:, 0]
ball_y = traj_ball[:, 1]
ball_z = traj_ball[:, 2]

t = range(len(x))

plt.figure(figsize=(8,5))

plt.plot(t, x, label='robot_x' , linestyle='--', color='b')
plt.plot(t, y, label='robot_y' , linestyle='--', color ='g')
plt.plot(t, z, label='robot_z' , linestyle='--', color = 'r')
plt.plot(t, ball_x, label='ball_x' , color='b')
plt.plot(t, ball_y, label='ball_y', color='g')
plt.plot(t, ball_z, label='ball_z' , color='r')


plt.xlabel('index')
plt.ylabel('value')
plt.title(f'Sample {idx} (x, y, z vs index)')
plt.legend()

plt.show()

#%%%%%%%%%%%%%%%%%%%%%% Build the Model ( Encoder + ode solver + decoder) %%%%%%%%%%%%

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



class Augmented_Model(eqx.Module):
    node: NODE
    ode: ODESolver
    state_dim: int =eqx.field(static=True)
    aug_dim: int=eqx.field(static=True)

    def __init__ (self , node, ode , state_dim , aug_dim):
        self.node = node
        self.ode = ode
        self.state_dim = state_dim
        self.aug_dim = aug_dim
    
    def __call__(self , ts_robot , u_robot , ball_init):
        aug0 = jnp.zeros((self.aug_dim,))
        h0 = jnp.concatenate([ball_init, aug0], axis=0)
        h = self.ode(ts_robot, h0, us=u_robot)
        y_ball = h[:, :6]
        
        return y_ball


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
state_size =6
aug_size = 12
total_size = state_size+ aug_size
key = jr.key(42)
key_h , key_G , key_R , key_J = jr.split(key, 4)
#nn = NODE(state_size=latent_dim , input_size=12, width_sizes=[64,64,64], key=key)

class Bounded_Energy(eqx.Module):
    mlp: eqx.nn.MLP

    def __call__(self, h):
        x = self.mlp(h)
        return  jax.nn.softplus(x).squeeze() + jnp.sum(h**2)

H = MLP(in_size=total_size , out_size=1 , width_sizes=[64,64,64] , key=key_h)
J= ConstantSkewSymmetricMatrix((total_size,total_size) , key = key_J)
R = ConstantSPDMatrix((total_size,total_size) , key = key_R)
G = ConstantMatrix((total_size , 12) , key=key_G)

H_bound = Bounded_Energy(H)
bphnn = ISPHS(H_bound , J , R , G)


ode = ODESolver(bphnn)
model = Augmented_Model(bphnn, ode, state_dim=state_size , aug_dim=aug_size)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def make_time_throw_zero(robot_time, mask, after_throw_steps=0):
    """
    robot_time: (N, T)
    mask:       (N, T), True for valid data, False for padding

    Output:
    - throw moment is exactly t = 0
    - time remains strictly increasing
    - padded part continues increasing safely for ODE
    """

    N, T = robot_time.shape

    last_valid_idx = jnp.sum(mask.astype(jnp.int32), axis=1) - 1

    throw_idx = last_valid_idx - after_throw_steps

    t_throw = robot_time[jnp.arange(N), throw_idx]

    # real shifted time, throw becomes zero
    time_shifted = robot_time - t_throw[:, None]

    # estimate dt
    dt = robot_time[:, 1] - robot_time[:, 0]

    grid = jnp.arange(T)[None, :]

    # continue after last valid shifted time
    last_valid_time = time_shifted[jnp.arange(N), last_valid_idx]

    time_safe = last_valid_time[:, None] + (
        grid - last_valid_idx[:, None]
    ) * dt[:, None]

    # use real shifted time for valid part, safe time for padding
    time_final = jnp.where(mask, time_shifted, time_safe)

    return time_final

time_train = make_time_throw_zero(robot_time_train_norm, mask_train)
time_test  = make_time_throw_zero(robot_time_test_norm, mask_test)
print (time_train[119])


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

    #t_schedule = step / 5000
    t_schedule = jnp.minimum(run_state / 20000.0, 1)
    
    weights = make_curriculum_weights(t_schedule, robot_ts.shape[-1])

    dynamic_loss_per_timestamp = jnp.sum(mask * jnp.square(pred - ball_batch), axis=(0, 2))
    dynamic_loss = jnp.dot(weights, dynamic_loss_per_timestamp) / jnp.sum(mask)

    landa1 = 0.01
    loss_enc_dec = landa1 * jnp.mean(jnp.square(init - ball0))

    return dynamic_loss + loss_enc_dec

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
weight1 = make_curriculum_weights(0.01 , 150)
weight2 = make_curriculum_weights(0.1 , 150)
weight3 = make_curriculum_weights(0.2 , 150 )
weight4 = make_curriculum_weights(0.5 , 150 )
weight5 = make_curriculum_weights(0.75 , 150 )
weight6 = make_curriculum_weights(1 , 150 )
plt.figure(figsize=(8,5))

plt.plot(t, weight1*mask_train[idx], label='x')
plt.plot(t, weight2*mask_train[idx], label='x')
plt.plot(t, weight3*mask_train[idx], label='x')
plt.plot(t, weight4*mask_train[idx], label='x')
plt.plot(t, weight5*mask_train[idx], label='x')
plt.plot(t, weight6*mask_train[idx], label='x')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@klax.loss
def loss_Trajectory(model , data, batch_axis):
    robot_ts , robot_batch, mask_batch, ball_batch = data
    ball0 = ball_batch[:,0,:]
    pred = jax.vmap(model , in_axes=(0,0,0))(robot_ts , robot_batch , ball0)
    mask = mask_batch[..., None]  # (B, T, 1)
    loss_pred = jnp.sum(mask * jnp.square(pred - ball_batch)) / (
        jnp.sum(mask) * ball_batch.shape[-1]
    )
    return loss_pred 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model_ = klax.finalize(model)
idx = 1
pred_position_norm = model_(time_test[idx] ,robot_test_input_norm[idx] , ball_test_norm[idx,0])

# ---- find throw index
last_valid_idx = jnp.sum(mask_test.astype(jnp.int32), axis=1) - 1
hitting_idx = int(last_valid_idx[idx])
throw_idx = int (index_test[idx]/10)

# ---- de- normalize the prediction
pred_x = ball_norm.inverse_transform_qs(pred_position_norm[..., 0:3])
pred_dx = ball_norm.inverse_transform_q_ts(pred_position_norm[..., 3:6])
denorm_time_test = norm.inverse_transform_ts(time_test)


pred_position = jnp.concatenate([pred_x , pred_dx]  , axis = -1)

# ---- extract throw states
true_traj = ball_test[idx]   # important
true_hitting = ball_test[idx, hitting_idx, :]
pred_hitting= pred_position[hitting_idx, :]

true_throw = ball_test[idx, throw_idx, :]
pred_throw= pred_position[throw_idx, :]

print("true throw:", true_throw)
print("pred throw:", pred_throw)

print("true hitting:", true_hitting)
print("pred hitting:", pred_hitting)


labels = ["x", "y", "z", "vx", "vy", "vz"]

plt.figure(figsize=(12, 6))

for d in range(6):
    plt.subplot(2, 3, d + 1)
    valid = mask_test[idx]

    plt.plot(denorm_time_test[idx][valid], robot_test_input[idx, :, d][valid], label="cup")
    plt.plot(denorm_time_test[idx][valid], true_traj[valid, d], label="true ball")
    plt.plot(denorm_time_test[idx][valid], pred_position[valid, d], label="pred ball")
    plt.axvline(denorm_time_test[idx , throw_idx], linestyle='--', linewidth=1.0 , color='black')

    plt.xlabel("time [s]")
    plt.ylabel(labels[d])
    plt.title(labels[d])
    plt.grid(alpha=0.3)
    plt.legend()

plt.suptitle(f"Ball vs Cup before training - sample {idx}", fontsize=14)
plt.tight_layout()
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

model , hist_traj = klax.fit(
    model,
    (time_train[:10], robot_train_input_norm[:10], mask_train[:10] , ball_train_norm[:10]),
    validation_data=(time_test, robot_test_input_norm, mask_test, ball_test_norm),
    run_state=0,
    batch_size=2,
    optimizer=optax.adam(3e-4),
    loss= loss_Trajectory,
    steps=15000,
    verbose=True,
    callbacks=[RunStateUpdater()],
    log_every=50,
    key=jr.key(0)
)

hist_traj.plot()
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model_ = klax.finalize(model)
idx = 9
pred_position_norm = model_(time_train[idx] ,robot_train_input_norm[idx] , ball_train_norm[idx,0])

# ---- find throw index
last_valid_idx = jnp.sum(mask_train.astype(jnp.int32), axis=1) - 1
hitting_idx = int(last_valid_idx[idx])
throw_idx = int (index_train[idx]/10)

# ---- de- normalize the prediction
pred_x = ball_norm.inverse_transform_qs(pred_position_norm[..., 0:3])
pred_dx = ball_norm.inverse_transform_q_ts(pred_position_norm[..., 3:6])
denorm_time_train = norm.inverse_transform_ts(time_train)


pred_position = jnp.concatenate([pred_x , pred_dx]  , axis = -1)

# ---- extract throw states
true_traj = ball_train[idx]   # important
true_hitting = ball_train[idx, hitting_idx, :]
pred_hitting= pred_position[hitting_idx, :]

true_throw = ball_train[idx, throw_idx, :]
pred_throw= pred_position[throw_idx, :]

print("true throw:", true_throw)
print("pred throw:", pred_throw)

print("true hitting:", true_hitting)
print("pred hitting:", pred_hitting)


labels = ["x", "y", "z", "vx", "vy", "vz"]

plt.figure(figsize=(12, 6))

for d in range(6):
    plt.subplot(2, 3, d + 1)
    valid = mask_train[idx]

    plt.plot(denorm_time_train[idx][valid], robot_train_input[idx, :, d][valid], label="cup")
    plt.plot(denorm_time_train[idx][valid], true_traj[valid, d], label="true ball")
    plt.plot(denorm_time_train[idx][valid], pred_position[valid, d], label="pred ball")
    plt.axvline(denorm_time_train[idx , throw_idx], linestyle='--', linewidth=1.0 , color='black')

    plt.xlabel("time [s]")
    plt.ylabel(labels[d])
    plt.title(labels[d])
    plt.grid(alpha=0.3)
    plt.legend()

plt.suptitle(f"Ball vs Cup after training - sample {idx}", fontsize=14)
plt.tight_layout()
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
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
'''
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eqx.tree_serialise_leaves("trained_model_position5.eqx", model_)


# %%
