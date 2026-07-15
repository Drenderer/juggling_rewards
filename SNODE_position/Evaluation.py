#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import jax
import numpy as np
import matplotlib.pyplot as plt
import equinox as eqx
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, PyTree
from dynax import ODESolver , ISPHS , normalization_coefficients
from klax.nn import MLP , ConstantMatrix ,ConstantSkewSymmetricMatrix , ConstantSPDMatrix
import klax
import sys
from pathlib import Path
sys.path.append("..") 
from node import NODE
from Modified_isphs import contact_ISPHS
from helping_function import hitting_ground , find_throwing , ball_free_flight_trajecotry , Forward_kinematic
from normalize import Normalization, coefficients

#%%%%%%%%%%%%%%%%%%%% import original data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

#%%%%%%%%%%%%%%%%%%%% build norm class %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
N_test , T , _ = ball_test_norm.shape
N_train , T, _  =ball_train_norm.shape

contact_train = jnp.zeros((N_train , T))
contact_test = jnp.zeros((N_test , T))

throw_idx_train = (index_train // 10) +1
throw_idx_test = (index_test // 10) +1

# grid of time indices
t_grid = jnp.arange(T)[None, :]   # shape: (1, T)

# 1 before throw, 0 after throw
contact_train = (t_grid < throw_idx_train[:, None]).astype(jnp.float32)
contact_test = (t_grid < throw_idx_test[:, None]).astype(jnp.float32)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

#%%%%%%%%%%%%%%%%%%%%%% build model for time prediciton %%%%%%%%%%%%%%%%%%%%%

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
    
input_size = robot_train_input.shape[-1]   # probably 12
hidden_size = 128

key = jr.key(0)

model_gru_template = GRUThrowTimeModel(
    input_size=input_size,
    hidden_size=hidden_size,
    key=key,
)

model_time = eqx.tree_deserialise_leaves('saved_models/time/trained_model_time_3.eqx', model_gru_template)
model_time_ = klax.finalize(model_time)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N_eval = 2000

robot_eval = robot_test_input_norm[:N_eval]
throw_eval = t_throw_test[:N_eval]

throw_pred = jax.jit(jax.vmap(model_time_))(robot_eval)
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

    def __call__(self, ts_robot, u_robot , contact):

        h0 = self.encoder(u_robot[0,:6])
        ball0 = self.decoder(h0)

        h = self.ode(ts_robot, h0, us=u_robot)

        last_valid_idx = jnp.sum(contact.astype(jnp.int32)) 
        h_last_valid = h[last_valid_idx]

        y_ball = self.decoder(h_last_valid)

    
        return y_ball, ball0


latent_dim =16
key = jr.key(15)
n_key , j_key , r_key , g_key , m_key = jr.split(key , 5)

nn = NODE(state_size=latent_dim , input_size=12, width_sizes=[64,64,64], key=key)

ode = ODESolver(nn)
model_template = Model( nn , ode, latent_dim=latent_dim , key=m_key)

model_position = eqx.tree_deserialise_leaves('saved_models/position/trained_model_position10_3_3.eqx', model_template)
model_position_ = klax.finalize(model_position)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DT = 0.002
ts = jnp.arange(2500) * DT

N_eval = 2000


time_eval_norm = time_test[:N_eval]
robot_eval_norm = robot_test_input_norm[:N_eval]
contact_eval = contact_test[:N_eval]
ball_eval = ball_test[:N_eval]
last_valid_idx_eval = throw_idx_test[:N_eval]



def one_sample(time_i, robot_i, contact_i, ball_i, throw_idx_i ,pred_idx_i):
    #idx = jnp.round(pred_idx_i / 0.02).astype(jnp.int32)
    idx = jnp.asarray(pred_idx_i / 0.02, dtype=jnp.int32)
    new_contact_i = (jnp.arange(contact_i.shape[0]) < idx).astype(contact_i.dtype)

    pred_traj_i, init_i = model_position_(time_i, robot_i, new_contact_i)
    pred_x = ball_norm.inverse_transform_qs (pred_traj_i[:3])
    pred_dx = ball_norm.inverse_transform_q_ts(pred_traj_i[3:6])

    pred = jnp.concat([pred_x , pred_dx])

    
    true_throw_i = ball_i[throw_idx_i, :]

    q_true_i, true_time_i = ball_free_flight_trajecotry(true_throw_i, ts)
    q_pred_i, pred_time_i = ball_free_flight_trajecotry(pred, ts)
    '''
    jax.debug.print(
        "pred_idx: {},throw_idx: {}, throw_true: {} , pred_throw: {} , model_pred:{}",
        idx, throw_idx_i , ball_i[throw_idx_i,0] , ball_i[idx , 0] , pred[0]
    )
    '''
    return q_true_i, q_pred_i, true_time_i, pred_time_i


batched_eval = jax.jit(jax.vmap(one_sample, in_axes=(0, 0, 0, 0, 0,0)))

q_total_true, q_total_pred, total_true_time, total_pred_time = batched_eval(
    time_eval_norm,
    robot_eval_norm,
    contact_eval,
    ball_eval,
    last_valid_idx_eval,
    pred_denorm[:2000],
)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

idx = 0
error = []
for idx in range(2000):
    hitting_idx= total_true_time[idx]
    error_x = jnp.abs(q_total_pred[idx ,hitting_idx , 0] - q_total_true[idx ,hitting_idx , 0]) 
    error_y = jnp.abs(q_total_pred[idx ,hitting_idx, 1] - q_total_true[idx ,hitting_idx , 1]) 
    error_z = jnp.abs(q_total_pred[idx ,hitting_idx, 2] - q_total_true[idx ,hitting_idx , 2])

    err1 = jnp.sqrt (error_x**2 + error_y**2 + error_z**2)

    error. append(err1)
    #print (idx , err1)

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

num_total = error.shape[0]
num_below = jnp.sum(error <= D)
percentage_below = 100.0 * num_below / num_total

print(f"Threshold R = {D} m")
print(f"Samples below R: {int(num_below)} / {num_total}")
print(f"Percentage below R: {float(percentage_below):.2f}%")




#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

error0 = []
for idx in range(2000):
    throw_idx = int (index_test[idx]/10) +1
    error_x = jnp.abs(q_total_pred[idx ,0 , 0] - q_total_true[idx ,0 , 0]) 
    error_y = jnp.abs(q_total_pred[idx ,0, 1] - q_total_true[idx ,0 , 1]) 
    error_z = jnp.abs(q_total_pred[idx ,0, 2] - q_total_true[idx ,0 , 2])

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

num_total = error0.shape[0]
num_below = jnp.sum(error0 <= D)
percentage_below = 100.0 * num_below / num_total

print(f"Threshold D = {D} m")
print(f"Samples below D: {int(num_below)} / {num_total}")
print(f"Percentage below D: {float(percentage_below):.2f}%")




# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
idx = 184
pred_time = model_time_(robot_test_input_norm[idx])

pred_denorm = ball_norm.inverse_transform_ts(pred_time)
true_denorm = ball_norm.inverse_transform_ts(t_throw_test[idx])


idx_time = jnp.asarray(pred_denorm / 0.02, dtype=jnp.int32)


print ("true time:" , true_denorm , "true index:" , throw_idx_test[idx])
print ("pred time:" , pred_denorm , "pred index:" , pred_denorm / 0.02, idx_time )


new_contact_i = (jnp.arange(contact_test[idx].shape[0]) < idx_time).astype(contact_test[idx].dtype)

pred, init_i = model_position_(time_test[idx], robot_test_input_norm[idx], contact_test[idx])
pred_total, init_i = model_position_(time_test[idx], robot_test_input_norm[idx], new_contact_i)

throw_idx = int(throw_idx_test[idx])

# ---- extract throw states
true_traj = ball_test[idx]   # important
true_throw = ball_test[idx, throw_idx, :]

# ----- denormalize the time
time_test_denorm1 = ball_norm.inverse_transform_ts(time_test)

#------ denormalize the position 
pred_x = ball_norm.inverse_transform_qs (pred[:3])
pred_dx = ball_norm.inverse_transform_q_ts(pred[3:6])

pred = jnp.concat([pred_x , pred_dx])


#------ denormalize the position 
pred_total_x = ball_norm.inverse_transform_qs (pred_total[:3])
pred_total_dx = ball_norm.inverse_transform_q_ts(pred_total[3:6])

pred_total = jnp.concat([pred_total_x , pred_total_dx])

print ("true throw:" , true_throw)
print ("throw when we have time:" ,pred)
print ("throw when we pred time:" ,pred_total)

DT = 0.002
ts = jnp.arange(2500) * DT

ball_q_true, true_time = ball_free_flight_trajecotry(true_throw, ts)
ball_q_pred, pred_time = ball_free_flight_trajecotry(pred, ts)

error_x = jnp.abs(ball_q_pred[true_time , 0] - ball_q_true[true_time , 0]) 
error_y = jnp.abs(ball_q_pred[true_time , 1] - ball_q_true[true_time , 1]) 
error_z = jnp.abs(ball_q_pred[true_time , 2] - ball_q_true[true_time , 2]) 
#print ("error in x:" , error_x)
#print ("error in y:" , error_y)
#print ("error in z:" , error_z)

error = jnp.sqrt (error_x**2 + error_y**2 + error_z**2)
print ("distance error when we have time" , error)

ball_q_pred_total, pred_time = ball_free_flight_trajecotry(pred_total, ts)
error_x = jnp.abs(ball_q_pred_total[true_time , 0] - ball_q_true[true_time , 0]) 
error_y = jnp.abs(ball_q_pred_total[true_time , 1] - ball_q_true[true_time , 1]) 
error_z = jnp.abs(ball_q_pred_total[true_time , 2] - ball_q_true[true_time , 2]) 
#print ("error in x:" , error_x)
#print ("error in y:" , error_y)
#print ("error in z:" , error_z)

error = jnp.sqrt (error_x**2 + error_y**2 + error_z**2)
print ("distance error when we pred time" , error)

# %%

for k in range(6):
    print ("the value of true ball at " , k-3)
    print (ball_test[idx , throw_idx_test[idx] + (k-3) , : ])
    
# %%
