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
print (time_test[1])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class ContactHead(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key):
        self.mlp = eqx.nn.MLP(
            in_size=12,      
            out_size=1,
            width_size=32,
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


state_size =6
aug_size = 6
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
bphnn = contact_ISPHS(H_bound, J , R , G , contact)

ode = ODESolver(bphnn)

model_template = Augmented_Model(bphnn, ode, state_dim=state_size , aug_dim=aug_size)

model_loaded = eqx.tree_deserialise_leaves("trained_model_position6.eqx", model_template)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model_ = klax.finalize(model_loaded)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DT = 0.002
ts = jnp.arange(2500) * DT

last_valid_idx = jnp.sum(mask_test.astype(jnp.int32), axis=1) - 1

N_eval = 2000

time_eval = time_test[:N_eval]
robot_eval = robot_test_input_norm[:N_eval]
mask_eval = mask_test[:N_eval]
ball_eval = ball_test[:N_eval]
ball0_eval = ball_test_norm[:N_eval,0,:]
last_valid_eval = last_valid_idx[:N_eval]
index_eval = index_test[:N_eval]
contact_eval = contact_test[:N_eval]


def one_sample(time_i, robot_i, ball0_i):
    pred_traj_i, c_i = model_(time_i, robot_i, ball0_i)

    # ---- de- normalize the prediction
    pred_x = ball_norm.inverse_transform_qs(pred_traj_i[..., 0:3])
    pred_dx = ball_norm.inverse_transform_q_ts(pred_traj_i[..., 3:6])



    pred_position = jnp.concatenate([pred_x , pred_dx]  , axis = -1)

    #true_hit = ball_i[hitting_idx_i, :]
    #pred_hit = pred_position[hitting_idx_i, :]

    #throw_idx_i = int(index_i /10)
    #true_throw = ball_i [throw_idx_i , :]
    #pred_throw = pred_position[throw_idx_i , :]

    

    return pred_position , c_i


batched_eval = jax.jit(jax.vmap(one_sample, in_axes=(0, 0, 0)))

q_total_pred , c_total_pred = batched_eval(
    time_eval,
    robot_eval,
    ball0_eval,
)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
last_valid_idx = jnp.sum(mask_eval.astype(jnp.int32), axis=1) - 1

error = []
for idx in range(2000):
    hitting_idx = int(last_valid_idx[idx])
    throw_idx = int (index_test[idx]/10)
    error_x = jnp.abs(q_total_pred[idx ,hitting_idx , 0] - ball_eval[idx ,hitting_idx , 0]) 
    error_y = jnp.abs(q_total_pred[idx ,hitting_idx, 1] - ball_eval[idx ,hitting_idx , 1]) 
    error_z = jnp.abs(q_total_pred[idx ,hitting_idx, 2] - ball_eval[idx ,hitting_idx , 2])

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
    hitting_idx = int(last_valid_idx[idx])
    throw_idx = int (index_test[idx]/10)
    error_x = jnp.abs(q_total_pred[idx ,throw_idx , 0] - ball_eval[idx ,throw_idx , 0]) 
    error_y = jnp.abs(q_total_pred[idx ,throw_idx, 1] - ball_eval[idx ,throw_idx , 1]) 
    error_z = jnp.abs(q_total_pred[idx ,throw_idx, 2] - ball_eval[idx ,throw_idx , 2])

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
error_throw = []
for idx in range (2000):
    pred_throw = jnp.where (c_total_pred[idx]<0.2)[0]
    true_throw = jnp.where (contact_test[idx]<0.2)[0]

    error_throw.append (jnp.abs(pred_throw[0] - true_throw[0]))

error_throw = jnp.array(error_throw)
bins = jnp.linspace(error_throw.min(), error_throw.max(), 50)

plt.figure(figsize=(10, 5))
plt.hist(error_throw, bins=bins, color="tab:blue", alpha=0.7, label=f"Error ≤ {D}")
plt.xlabel("Distance error [m]")
plt.ylabel("Number of samples")
plt.title("Distribution of position errors  at Throwing Time")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
idx = 6
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

t = range(150)

plt.figure(figsize=(8,5))

plt.plot(t, contact_test[idx], label='true contact' , linestyle='--', color='b')
plt.plot(t, c, label='predict contact' , linestyle='--', color ='g')
plt.plot(t, mask_test[idx], label='valid mask' , linestyle='--', color ='r')

plt.xlabel('index')
plt.ylabel('value')
plt.title(f'Sample {idx} (x, y, z vs index)')
plt.legend()

plt.show()





error_x = jnp.abs(ball_eval[idx ,hitting_idx , 0] - q_total_pred[idx ,hitting_idx , 0]) 
error_y = jnp.abs(ball_eval[idx ,hitting_idx, 1] - q_total_pred[idx ,hitting_idx, 1]) 
error_z = jnp.abs(ball_eval[idx ,hitting_idx, 2] - q_total_pred[idx ,hitting_idx, 2])

err_idx = jnp.sqrt (error_x**2 + error_y**2 + error_z**2)

error_x0 = jnp.abs(ball_eval[idx ,throw_idx, 0] - q_total_pred[idx ,throw_idx , 0]) 
error_y0 = jnp.abs(ball_eval[idx ,throw_idx, 1] - q_total_pred[idx ,throw_idx, 1]) 
error_z0 = jnp.abs(ball_eval[idx ,throw_idx, 2] - q_total_pred[idx ,throw_idx, 2])



err0_idx = jnp.sqrt (error_x0**2 + error_y0**2 + error_z0**2)

print ("error0 in x:" , error_x0)
print ("error0 in y:" , error_y0)
print ("error0 in z:" , error_z0)

print ("distance error0 :" , err0_idx)



print ("distance error at hitting point" , err_idx)




# %%
