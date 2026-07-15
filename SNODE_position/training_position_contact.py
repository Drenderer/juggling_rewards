#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import numpy as np
import jax
import optax
import equinox as eqx
from collections.abc import Callable
from jax import numpy as jnp
from jax import random as jr
from jaxtyping import Array, PyTree ,  Float, Scalar
import matplotlib.pyplot as plt
from dynax import ODESolver ,normalization_coefficients , ISPHS
import klax
from klax.nn import MLP , ConstantMatrix , ConstantSkewSymmetricMatrix , ConstantSPDMatrix
import sys
from pathlib import Path
sys.path.append("..") 
#from Modified_isphs import contact_ISPHS
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

robot_train_n = jnp.concatenate([coord_train_z , coord_train_dz] , axis=-1)
robot_test_n = jnp.concatenate([coord_test_z , coord_test_dz] , axis=-1)


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

#%%%%%%%%%%%%%%%%%%% check the data before training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

idx =89 # choose sample

throw_idx = int (index_train[idx]/10)+1

nx = robot_train_coord[: , : , 0:3]
ny = robot_train_coord[: , : , 3:6]
nz = robot_train_coord [: , : , 6:9]
R = jnp.stack([nx[idx] , ny[idx] , nz[idx]] , axis = -1 )
traj = robot_train_input[idx]   
traj_ball = ball_train[idx]

x = traj[:, 0]
y = traj[:, 1]
z = traj[:, 2]
ball_x = traj_ball[:, 0]
ball_y = traj_ball[:, 1]
ball_z = traj_ball[:, 2]

rel = traj_ball[:, 0:3] - traj[:, 0:3]      # (T,3)
r = jnp.matmul(
    jnp.swapaxes(R, -1, -2),
    rel[..., None]
).squeeze(-1)

rel_x = r[: ,0]
rel_y = r[: ,1]
rel_z = r[: ,2]
dis = jnp.sqrt(rel_x **2 + rel_y **2 + rel_z**2)
t = range(len(x))

valid = contact_train[idx]
plt.figure(figsize=(8,5))

#plt.plot(t, z, label='robot_z' , linestyle='--', color='b')
#plt.plot(t, y, label='robot_y' , linestyle='--', color ='g')
plt.plot(t, z, label='robot_z' , linestyle='--', color = 'r')
#plt.plot(t, ball_z, label='ball_z' , color='b')
#plt.plot(t, ball_y, label='ball_y', color='g')
plt.plot(t, ball_z, label='ball_z' , color='r')
#plt.plot(t, rel_z, label='z_local' , color='b')
#plt.plot(t, rel[:,2], label='rel_z_global' , color='g')
plt.plot(t , contact_train[idx] , label='contact' )


plt.xlabel('index')
plt.ylabel('value')
plt.title(f'Sample {idx} (x, y, z vs index)')
plt.legend()

plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
print (throw_idx)
print (contact_train[idx])
k=0
for k in range (10):
    print (traj_ball[throw_idx+k-1,3:])
    k=k+1


#%%%%%%%%%%%%%%%%%%%%%% Build the Naive Model %%%%%%%%%%%%
class Naive_Model(eqx.Module):
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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class contact_ISPHS (eqx.Module):

    hamiltonian: Callable[[Array], Scalar]
    #structure_matrix: Callable[[Float[Array, "n"]], Float[Array, "n n"]]  
    structure_matrix: Array = eqx.field(static=True)
    dissipation_matrix: (
        Callable[[Float[Array, "n"]], Float[Array, "n n"]] | None
    )  
    input_matrix: Callable[[Float[Array, "n"]], Float[Array, "n m"]] | None  
    contact: Callable[[Array, Array], Scalar]

    def __init__ (
            self,
        hamiltonian: Callable[[Array], Scalar],
        structure_matrix,
        dissipation_matrix: Callable[[Float[Array, "n"]], Float[Array, "n n"]]  # noqa: F722, F821
        | None = None,
        input_matrix: Callable[[Float[Array, "n"]], Float[Array, "n m"]]
        | None = None,  # noqa: F722, F821
        contact: Callable[[Array, Array], Scalar] | None = None
    ):
        
        self.hamiltonian = hamiltonian
        self.structure_matrix = structure_matrix
        self.dissipation_matrix = dissipation_matrix
        self.input_matrix = input_matrix
        self.contact = contact

    def __call__(self, t: Scalar, x: Array, u: Array | None = None, args=None) -> Array:
        
        if self.contact is None:
                c = 0.0
        else:
                c = self.contact(x, u)    
        
        structure_matrix = self.structure_matrix

        if self.dissipation_matrix is not None:
            dissipation_matrix = self.dissipation_matrix(x)
            structure_matrix = structure_matrix - c*dissipation_matrix

        x_t = structure_matrix @ jax.grad(self.hamiltonian)(x)
        

        if self.input_matrix is not None:
            if u is None:
                raise ValueError(
                    "The ISPHS has an input matrix but no input u was provided."
                )

            input_matrix = self.input_matrix(x , u)

            #x_t = x_t + c * input_matrix
            x_t = x_t.at[3:6].add(c*input_matrix)
   
        return x_t
    

#%%%%%%%%%%%%%%%%%%%%%% Build the Model %%%%%%%%%%%%

class Augmented_Model(eqx.Module):
    #bphnn: eqx.Module
    ode: ODESolver
    state_dim: int = eqx.field(static=True)
    aug_dim: int = eqx.field(static=True)

    #def __init__(self, bphnn, ode, state_dim, aug_dim):
    def __init__(self, ode, state_dim, aug_dim):
        #self.bphnn = bphnn
        self.ode = ode
        self.state_dim = state_dim
        self.aug_dim = aug_dim

    def __call__(self, ts_robot, u_robot, ball_init):
        aug0 = jnp.zeros((self.aug_dim,))
        
        h0 = jnp.concatenate([ball_init, aug0], axis=0)

        h = self.ode(ts_robot, h0, us=u_robot)

        y_ball = h[:, :self.state_dim]

    
        #c = jax.vmap(self.bphnn.contact)(h, u_robot)
        c = jax.vmap(self.ode.func.contact)(h, u_robot)
        return y_ball, c

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Model initialization   %%%%%%%%%%%%%%%%%%%%%
state_size =6
aug_size = 0
total_size = state_size + aug_size
key = jr.key(0)
key_h , key_G , key_R , key_J  , key_c= jr.split(key, 5)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% build G and contact   %%%%%%%%%%%%%%%%%%%%%
class ContactHead(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key):
        self.mlp = eqx.nn.MLP(
            in_size=6,      
            out_size=1,
            width_size=32,
            depth=2,
            key=key,
        )

    def __call__(self, x, u):

        ball_pos = x[:3]
        ball_vel = x[3:6]

        cup_pos = u[:3]
        cup_vel = u[3:6]

        nx = u[6:9]
        ny = u[9:12]
        nz = u[12:15]

        R = jnp.stack([nx, ny, nz], axis=-1)

        r_c = R.T @ (ball_pos - cup_pos)
        v_c = R.T @ (ball_vel - cup_vel)

        
        input = jnp.concatenate([r_c , v_c], axis=0)

        logit = self.mlp(input).squeeze()
        return jax.nn.sigmoid(logit)



class InputNN(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key):
        self.mlp = eqx.nn.MLP(
            in_size=15,
            out_size=3,  
            width_size=64,
            depth=3,
            key=key,
        )

    def __call__(self, x, u):
        q = x[:3]
        v = x[3:6]

        cup_pos = u[:3]
        cup_vel = u[3:6]

        nx = u[6:9]
        ny = u[9:12]
        nz = u[12:15]

        R = jnp.stack([nx, ny, nz], axis=-1)

        r_c = R.T @ (q - cup_pos)
        v_c = R.T @ (v - cup_vel)
        input = jnp.concatenate([r_c , v_c , nx , ny , nz], axis=0)

        return self.mlp(input)



G = InputNN(key = key_G)
contact = ContactHead(key=key_c)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% build J   %%%%%%%%%%%%%%%%%%%%%
def make_J_free(state_dim=6):
    J = jnp.zeros((state_dim, state_dim))

    I3 = jnp.eye(3)

    
    J = J.at[0:3, 3:6].set(I3)
    J = J.at[3:6, 0:3].set(-I3)

    return J

'''
class AugmentedJ(eqx.Module):
    J_aug: eqx.Module
    J_coupling: eqx.Module
    contact: eqx.Module
    state_dim: int = eqx.field(static=True)
    aug_dim: int = eqx.field(static=True)
    coupling_scale: float = eqx.field(static=True, default=1e-3)
    aug_scale: float = eqx.field(static=True, default=1e-3)

    def __call__(self, h, u=None):
        a = h[self.state_dim:]

        J_free = make_J_free(state_dim=self.state_dim)

        c = self.contact(h, u)

        Jc = c* self.coupling_scale * self.J_coupling(a).reshape(
            self.state_dim, self.aug_dim
        )

        J_aug = self.aug_scale * self.J_aug(a)

        top = jnp.concatenate([J_free, Jc], axis=1)
        bottom = jnp.concatenate([-Jc.T, J_aug], axis=1)

        return jnp.concatenate([top, bottom], axis=0)


J_aug= ConstantSkewSymmetricMatrix((aug_size,aug_size) , key = key_J)
J_coupling = MLP(in_size=aug_size , out_size=state_size*aug_size , width_sizes=[16,16] , key=key_J)

J_total = AugmentedJ(
    J_aug=J_aug,
    J_coupling=J_coupling,
    contact=contact,
    state_dim=state_size,
    aug_dim=aug_size,
    coupling_scale=1e-4,
    aug_scale=1e-4,
)
'''

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% build H   %%%%%%%%%%%%%%%%%%%%%
class FreeBallEnergy(eqx.Module):
    g: float = eqx.field(static=True, default=9.81)

    def __call__(self, y):
        q = y[:3]
        v = y[3:6]

        z = q[2]

        kinetic = 0.5 * jnp.sum(v**2)
        potential = self.g * z

        return kinetic + potential

'''
class Bounded_Energy(eqx.Module):
    mlp: eqx.nn.MLP

    def __call__(self, h):
        x = self.mlp(h)
        return jax.nn.softplus(x).squeeze() + jnp.sum(h**2)

class TotalEnergy(eqx.Module):
    H_free: eqx.Module
    H_aug: eqx.Module
    state_dim: int = eqx.field(static=True)

    def __call__(self, h):
        x = h[:self.state_dim]
        a = h[self.state_dim:]

        return self.H_free(x) + self.H_aug(a)

H_free = FreeBallEnergy()
H_aug = MLP(in_size=aug_size , out_size=1 , width_sizes=[16,16] , key=key_h)

H_bounded = Bounded_Energy(H_aug)

H_total = TotalEnergy(H_free=H_free , H_aug= H_bounded , state_dim=state_size)
'''

H_total = FreeBallEnergy()
J_total = make_J_free()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Wraping the augmented model   %%%%%%%%%%%%%%%%%%%%%
bphnn = contact_ISPHS(
    hamiltonian=H_total,
    structure_matrix=J_total,
    dissipation_matrix=None,
    input_matrix=G,
    contact=contact,
)

ode = ODESolver(bphnn)

#model = Augmented_Model(bphnn, ode, state_dim=state_size , aug_dim=aug_size)
model = Augmented_Model(ode, state_dim=state_size , aug_dim=aug_size)


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

time_train = make_time_start_zero(robot_time_train ,mask_train)
time_test  = make_time_start_zero(robot_time_test, mask_test)



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
    
    return loss_pred +  0.5 * loss_contact


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model_ = klax.finalize(model)
idx = 9
pred_position , c  = model_(time_train[idx] ,robot_train_input[idx] , ball_train[idx,0])
# ---- find throw index
last_valid_idx = jnp.sum(mask_train.astype(jnp.int32), axis=1) - 1
hitting_idx = int(last_valid_idx[idx])
throw_idx = int (index_train[idx]/10) +1

# ---- de- normalize the prediction
#pred_x = ball_norm.inverse_transform_qs(pred_position_norm[..., 0:3])
#pred_dx = ball_norm.inverse_transform_q_ts(pred_position_norm[..., 3:6])
#denorm_time_test = norm.inverse_transform_ts(time_test)


#pred_position = jnp.concatenate([pred_x , pred_dx]  , axis = -1)

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

    plt.plot(time_train[idx][valid], robot_train_input[idx, :, d][valid], label="cup")
    plt.plot(time_train[idx][valid], true_traj[valid, d], label="true ball")
    plt.plot(time_train[idx][valid], pred_position[valid, d], label="pred ball")
    plt.axvline(time_train[idx , throw_idx], linestyle='--', linewidth=1.0 , color='black')

    plt.xlabel("time [s]")
    plt.ylabel(labels[d])
    plt.title(labels[d])
    plt.grid(alpha=0.3)
    plt.legend()

plt.suptitle(f"Ball vs Cup before training - sample {idx}", fontsize=14)
plt.tight_layout()
plt.show()


t= time_train[idx]

plt.figure(figsize=(8,5))

plt.plot(t, contact_train[idx], label='true contact' , linestyle='--', color='b')
plt.plot(t, c, label='predict contact' , linestyle='--', color ='g')
plt.plot(t, mask_train[idx], label='valid mask' , linestyle='--', color ='r')

plt.xlabel('time')
plt.ylabel('value')
plt.title(f'Sample {idx} contact')
plt.legend()

plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

model , hist_traj = klax.fit(
    model,
    (time_train, robot_train_input[:, : , :15] ,mask_train, ball_train, contact_train),
    validation_data=(time_test, robot_test_input[: , : , :15], mask_test, ball_test , contact_test),
    run_state=0,
    batch_size=32,
    optimizer=optax.adam(2e-4),
    loss= loss_Trajectory,
    steps=40000,
    verbose=True,
    callbacks=[RunStateUpdater()],
    log_every=50,
    key=jr.key(0)
)

hist_traj.plot()
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model_ = klax.finalize(model)
idx =32
pred_position , c  = model_(time_test[idx] ,robot_test_input[idx ,:,:15] , ball_test[idx,0])


# ---- find throw index
last_valid_idx = jnp.sum(mask_test.astype(jnp.int32), axis=1) - 1
hitting_idx = int(last_valid_idx[idx])
throw_idx = int (index_test[idx]/10) +1

drop = contact_test[idx ,:-1] - contact_test[idx , 1:]
idx_last_contact = jnp.argmax(drop)


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

    plt.plot(time_test[idx][valid], robot_test_input[idx, :, d][valid], label="cup")
    plt.plot(time_test[idx][valid], true_traj[valid, d], label="true ball" )
    plt.plot(time_test[idx][valid], pred_position[valid, d], label="pred ball")
    plt.axvline(time_test[idx , throw_idx], linestyle='--', linewidth=1.0 , color='black')

    plt.xlabel("time [s]")
    plt.ylabel(labels[d])
    plt.title(labels[d])
    plt.grid(alpha=0.3)
    plt.legend()

plt.suptitle(f"Ball vs Cup after training - sample {idx}", fontsize=14)
plt.tight_layout()
plt.show()



plt.figure(figsize=(8,5))

plt.plot(time_test[idx], contact_test[idx], label='true contact' , linestyle='--', color='b')
plt.plot(time_test[idx], c, label='predict contact' , linestyle='--', color ='g')
plt.plot(time_test[idx], mask_test[idx], label='valid mask' , linestyle='--', color ='r')

plt.xlabel('time')
plt.ylabel('value')
plt.title(f'Sample {idx} (contact learning)')
plt.legend()

plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DT = 0.002
ts = jnp.arange(2500) * DT

last_valid_idx = jnp.sum(mask_test.astype(jnp.int32), axis=1) - 1

N_eval = 2000

time_eval = time_test[:N_eval]
robot_eval = robot_test_input[:N_eval]
mask_eval = mask_test[:N_eval]
ball_eval = ball_test[:N_eval]
last_valid_eval = last_valid_idx[:N_eval]
index_eval = index_test[:N_eval]
contact_eval = contact_test[:N_eval]


def one_sample(time_i, robot_i, ball0_i , contact_i):
    pred_traj_i, c_i = model_(time_i, robot_i, ball0_i)
    throw_pred_i = c_i[:-1] - c_i[1:]
    throw_true_i = contact_i[:-1] - contact_i[1:]

    return pred_traj_i , c_i , throw_pred_i , throw_true_i


batched_eval = jax.jit(jax.vmap(one_sample, in_axes=(0, 0, 0 , 0)))

q_total_pred , c_total_pred  , throw_pred , throw_true= batched_eval(
    time_eval,
    robot_eval,
    ball_eval[:,0,:],
    contact_eval,
)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
idx =151
last_valid_idx = jnp.sum(mask_test.astype(jnp.int32), axis=1) - 1
hitting_idx = int(last_valid_idx[idx])
plt.figure(figsize=(8,5))

plt.plot(time_test[idx,:hitting_idx], throw_pred[idx , :hitting_idx], label='pred_throw' , linestyle='--', color='b')
plt.plot(time_test[idx,:hitting_idx], throw_true[idx , :hitting_idx], label='true_throw' , linestyle='--', color ='g')


plt.xlabel('time')
plt.ylabel('value')
plt.title(f'Sample {idx} (contact learning)')
plt.legend()

plt.show()

print (jnp.argmax(throw_pred[idx]))
print (jnp.argmax(throw_true[idx]))


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
idx_pred = jnp.argmax(throw_pred, axis=1)
idx_true = jnp.argmax(throw_true, axis=1)

abs_diff_idx = np.array(jnp.abs(idx_pred - idx_true))

plt.figure(figsize=(8, 4))
plt.hist(
    abs_diff_idx,
    bins=np.arange(abs_diff_idx.max() + 2) - 0.5,
    edgecolor="black"
)

plt.xticks(np.arange(abs_diff_idx.max() + 1))
plt.xlabel("absolute difference in throw index")
plt.ylabel("number of samples")
plt.title("Histogram of throw-index error")
plt.grid(True, axis="y")
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# descending order
worst_order = jnp.argsort(abs_diff_idx)[::-1]

# top 50
worst50 = worst_order[:50]

print("Worst 50 sample ids:")
print(worst50)

print("Corresponding errors:")
print(abs_diff_idx[worst50])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
eqx.tree_serialise_leaves("trained_model_position14.eqx", model_)


# %%
