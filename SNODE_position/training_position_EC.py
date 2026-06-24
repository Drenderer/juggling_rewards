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



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
idx = 3 # choose sample

traj = robot_train_input_norm[idx]   # shape (84, 12)

x = traj[:, 0]
y = traj[:, 1]
z = traj[:, 2]

mask = mask_train[idx]

t = range(len(x))

plt.figure(figsize=(8,5))

plt.plot(t, x, label='x')
plt.plot(t, y, label='y')
plt.plot(t, z, label='z')
plt.plot(t , mask , label='mask')

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

    def __call__(self, ts_robot, u_robot , mask):

        h0 = self.encoder(u_robot[0,:6])
        ball0 = self.decoder(h0)

        h = self.ode(ts_robot, h0, us=u_robot)

        last_valid_idx = jnp.sum(mask.astype(jnp.int32)) - 1
        h_last_valid = h[last_valid_idx]

        y_ball = self.decoder(h_last_valid)
        #y_ball = jax.vmap(self.decoder)(h)

    
        return y_ball, ball0


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
latent_dim =24
key = jr.key(1)
n_key , j_key , r_key , g_key , m_key = jr.split(key , 5)

nn = NODE(state_size=latent_dim , input_size=12, width_sizes=[64,128,64], key=key)

''''
H = MLP (in_size=latent_dim , out_size=1 , width_sizes=[64,64,64] , key=n_key)
class Bounded_Energy(eqx.Module):
    mlp: eqx.nn.MLP

    def __call__(self, h):
        x = self.mlp(h)
        return jax.nn.softplus(x).squeeze() + 0.1*jnp.sum(h**2)
    
H_bounded = Bounded_Energy(H)

J = ConstantSkewSymmetricMatrix((latent_dim, latent_dim) ,key=j_key)
R = ConstantSPDMatrix((latent_dim, latent_dim), key=r_key)
G = ConstantMatrix((latent_dim, 12), key=g_key)


nn =ISPHS(H_bounded , J , R , G)
'''
ode = ODESolver(nn)
model = Model( nn , ode, latent_dim=latent_dim , key=m_key)



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

def gather_last_valid(x, mask):
    # x: (B, T, D), mask: (B, T)
    last_idx = jnp.sum(mask.astype(jnp.int32), axis=1) - 1
    return x[jnp.arange(x.shape[0]), last_idx]

def free_flight_loss(pred_throw, true_throw):
    DT = 0.002
    H = 250
    ts = jnp.arange(H) * DT

    pred_x = ball_norm.inverse_transform_qs (pred_throw[:3])
    pred_dx = ball_norm.inverse_transform_q_ts(pred_throw[3:6])
    pred = jnp.concat([pred_x , pred_dx])

    true_x = ball_norm.inverse_transform_qs (true_throw[:3])
    true_dx = ball_norm.inverse_transform_q_ts(true_throw[3:6])
    true = jnp.concat([true_x , true_dx])

    true_q = ball_free_flight_trajecotry_training(true, ts)
    pred_q = ball_free_flight_trajecotry_training(pred ,ts)

    sq_err = (pred_q - true_q) ** 2        # (H, 6)

    loss = jnp.mean(sq_err)

    return loss

@klax.loss
def loss_Trajectory(model, data, batch_axis):
    robot_ts, robot_batch, mask_batch, ball_batch = data

    pred_throw, pred_init = jax.vmap(model, in_axes=(0, 0, 0))(
        robot_ts, robot_batch, mask_batch
    )

    true_throw = gather_last_valid(ball_batch, mask_batch)

    loss_pred = jnp.mean(
        jax.vmap(free_flight_loss)(pred_throw, true_throw)
    )

    loss_init = jnp.mean((pred_init - ball_batch[:, 0, :]) ** 2)
    '''
    jax.debug.print("loss_pred {}", loss_pred)
    jax.debug.print("weighted_init {}", 0.1 * loss_init)
    '''
    return loss_pred +  0.1 * loss_init 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class RunStateUpdater(klax.Callback):
    """Updates the run_state to be the training step."""
    
    def on_training_step(self, context):
        context.state.run_state = context.state.step

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model , hist_traj = klax.fit(
    model,
    (time_train, robot_train_input_norm, mask_train, ball_train_norm),
    validation_data=(time_test, robot_test_input_norm,mask_test, ball_test_norm),
    run_state=0,
    batch_size=64,
    optimizer=optax.adam(3e-4),
    loss= loss_Trajectory,
    steps=10000,
    verbose=True,
    callbacks=[RunStateUpdater()],
    log_every=50,
    key=jr.key(0)
)

hist_traj.plot()
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model_ = klax.finalize(model)
idx = 24
pred_norm , init = model_(time_test[idx] ,robot_test_input_norm[idx] , mask_test[idx])

# ---- find throw index
last_valid_idx = jnp.sum(mask_test.astype(jnp.int32), axis=1) - 1
throw_idx = int(last_valid_idx[idx])

# ---- extract throw states
true_traj = ball_test[idx]   # important
true_throw = ball_test[idx, throw_idx, :]
#pred_throw = pred_position[throw_idx, :]

# ----- denormalize the prediction
time_test_denorm1 = ball_norm.inverse_transform_ts(time_test)
time_test_denorm2 = norm.inverse_transform_ts(time_test)
pred_x = ball_norm.inverse_transform_qs (pred_norm[:3])
pred_dx = ball_norm.inverse_transform_q_ts(pred_norm[3:6])

pred = jnp.concat([pred_x , pred_dx])

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

    plt.plot(time_test_denorm1[idx][valid], robot_test_input[idx, :, d][valid], label="cup")
    plt.plot(time_test_denorm1[idx][valid], true_traj[valid, d], label="true ball")
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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eqx.tree_serialise_leaves("trained_model_position10_2.eqx", model_)


# %%
