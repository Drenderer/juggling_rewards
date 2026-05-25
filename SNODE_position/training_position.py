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
from Modified_ISPHS import Modified_ISPHS
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
#%%%%%%%%%%%%%%%%%%%%%% build contact dataset for test and train %%%%%%%%%%%%%%%%%%%%%%%%%%%%
N_test , T , _ = ball_test_norm.shape
N_train , T, _  =ball_train_norm.shape

contact_train = jnp.zeros((N_train , T))
contact_test = jnp.zeros((N_test , T))

throw_idx_train = index_train // 10
throw_idx_test = index_test // 10

# grid of time indices
t_grid = jnp.arange(T)[None, :]   # shape: (1, T)

# 1 before throw, 0 after throw
contact_train = (t_grid < throw_idx_train[:, None]).astype(jnp.float32)
contact_test = (t_grid < throw_idx_test[:, None]).astype(jnp.float32)

#%%%%%%%%%%%%%%%%%%% check the data before training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

idx =103 # choose sample
print (index_test[idx])
print (contact_test[idx])
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
plt.plot(t , contact_test[idx] , label='contact' )


plt.xlabel('index')
plt.ylabel('value')
plt.title(f'Sample {idx} (x, y, z vs index)')
plt.legend()

plt.show()

#%%%%%%%%%%%%%%%%%%%%%% Build the Model %%%%%%%%%%%%

class ContactHead(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key):
        self.mlp = eqx.nn.MLP(
            in_size=12,      
            out_size=1,
            width_size=16,
            depth=2,
            key=key,
        )

    def __call__(self, x, u):
        y_ball = x[:6]

        ball_pos = y_ball[:3]
        ball_vel = y_ball[3:6]

        cup_pos = u[:3]
        cup_vel = u[3:6]
        cup_n = u[6:9]
        cup_dn = u[9:12]

        rel_pos = ball_pos - cup_pos
        rel_vel = ball_vel - cup_vel

        contact_input = jnp.concatenate(
            [rel_pos, rel_vel, cup_n, cup_dn],
            axis=0,
        )

        logit = self.mlp(contact_input).squeeze()
        return jax.nn.sigmoid(logit)



class Augmented_Model(eqx.Module):
    node: NODE
    ode: ODESolver
    state_dim: int = eqx.field(static=True)
    aug_dim: int = eqx.field(static=True)

    def __init__(self, node, ode, state_dim, aug_dim):
        self.node = node
        self.ode = ode
        self.state_dim = state_dim
        self.aug_dim = aug_dim

    def __call__(self, ts_robot, u_robot, ball_init):
        aug0 = jnp.zeros((self.aug_dim,))
        h0 = jnp.concatenate([ball_init, aug0], axis=0)

        h = self.ode(ts_robot, h0, us=u_robot)

        y_ball = h[:, :6]

    
        c = jax.vmap(self.node.contact)(h, u_robot)

        return y_ball, c


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
state_size =6
aug_size = 12
total_size = state_size+ aug_size
key = jr.key(42)
key_h , key_G , key_R , key_J  , key_c= jr.split(key, 5)

class Bounded_Energy(eqx.Module):
    mlp: eqx.nn.MLP

    def __call__(self, h):
        x = self.mlp(h)
        return  jax.nn.softplus(x).squeeze() + jnp.sum(h**2)


H = MLP(in_size=total_size , out_size=1 , width_sizes=[64,64,64] , key=key_h)
J= ConstantSkewSymmetricMatrix((total_size,total_size) , key = key_J)
R = ConstantSPDMatrix((total_size,total_size) , key = key_R)
G = ConstantMatrix((total_size , 12) , key=key_G)
contact = ContactHead(key=key_c,)
H_bound = Bounded_Energy(H)
bphnn = Modified_ISPHS(H_bound , J , R , G , contact)


ode = ODESolver(bphnn)



model = Augmented_Model(bphnn, ode, state_dim=state_size , aug_dim=aug_size)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
print (time_test[idx])


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class RunStateUpdater(klax.Callback):
    """Updates the run_state to be the training step."""
    
    def on_training_step(self, context):
        context.state.run_state = context.state.step

@klax.loss
def loss_Trajectory(model , data, batch_axis):
    robot_ts , robot_batch, mask_batch, ball_batch , contact_batch = data
    ball0 = ball_batch[:,0,:]
    pred , c_pred = jax.vmap(model , in_axes=(0,0,0))(robot_ts , robot_batch , ball0)
    mask = mask_batch[..., None]  # (B, T, 1)
    loss_pred = jnp.sum(mask * jnp.square(pred - ball_batch)) / (
        jnp.sum(mask) * ball_batch.shape[-1]
    )
    loss_contact = jnp.sum(mask_batch*jnp.square(c_pred - contact_batch)) / (jnp.sum(mask_batch) + 1e-8)
    landa = 0.2
    return loss_pred + landa*loss_contact


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model_ = klax.finalize(model)
idx = 1
pred_position_norm , c = model_(time_test[idx] ,robot_test_input_norm[idx] , ball_test_norm[idx,0])
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

t = range(len(x))

plt.figure(figsize=(8,5))

plt.plot(t, contact_test[idx], label='true contact' , linestyle='--', color='b')
plt.plot(t, c, label='predict contact' , linestyle='--', color ='g')

plt.xlabel('index')
plt.ylabel('value')
plt.title(f'Sample {idx} (x, y, z vs index)')
plt.legend()

plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

model , hist_traj = klax.fit(
    model,
    (time_train, robot_train_input_norm, mask_train , ball_train_norm , contact_train),
    validation_data=(time_test, robot_test_input_norm, mask_test, ball_test_norm , contact_test),
    run_state=0,
    batch_size=32,
    optimizer=optax.adam(3e-4),
    loss= loss_Trajectory,
    steps=20000,
    verbose=True,
    callbacks=[RunStateUpdater()],
    log_every=50,
    key=jr.key(0)
)

hist_traj.plot()
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model_ = klax.finalize(model)
idx = 360
pred_position_norm , c = model_(time_test[idx] ,robot_test_input_norm[idx] , ball_test_norm[idx,0])

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

plt.suptitle(f"Ball vs Cup after training - sample {idx}", fontsize=14)
plt.tight_layout()
plt.show()


#%%%%%
t = range(len(x))

plt.figure(figsize=(8,5))

plt.plot(t, contact_test[idx], label='true contact' , linestyle='--', color='b')
plt.plot(t, c*mask_test[idx], label='predict contact' , linestyle='--', color ='g')

plt.xlabel('index')
plt.ylabel('value')
plt.title(f'Sample {idx} (x, y, z vs index)')
plt.legend()

plt.show()

#%%%
print (contact_train[103])
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
