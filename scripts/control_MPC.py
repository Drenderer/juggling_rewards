#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import equinox as eqx
import jax
import klax
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from jax import numpy as jnp
from jax import random as jr


from pathlib import Path
from mpc_robot.datagen.normalize import Normalization
from mpc_robot.control.mpc_optax import mpc_optax
from mpc_robot.control.mpc_scipy import mpc_scipy
from mpc_robot.models.sphnn import make_sphnn
from mpc_robot.evaluation.evaluation_functions import tracking_summary_MPC , input_summary_MPC , loss_history_mpc

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ROOT = Path(__file__).resolve().parents[1]
data_robot = np.load(ROOT/"Data_MPC/robot_train.npz")
train_t = data_robot['time']
train_q = data_robot['robot_q']
train_dq = data_robot['robot_dq']
train_ddq = data_robot['robot_ddq']
train_u = data_robot['robot_u']

print (train_t.shape,train_q.shape , train_ddq.shape , train_u.shape)

data_robot = np.load(ROOT/"Data_MPC/robot_test.npz")
test_t = data_robot['time']
test_q = data_robot['robot_q']
test_dq = data_robot['robot_dq']
test_ddq = data_robot['robot_ddq']
test_u = data_robot['robot_u']

print (test_t.shape,test_q.shape , test_ddq.shape , test_u.shape)

data = np.load(ROOT/'Data_MPC/norm_value.npz')
mean_q = data['mean_q']
alpha_q = data['alpha_q']
tau_q = data['tau_q']
mean_u = data['mean_u']
alpha_u = data['alpha_u']

print (mean_q , alpha_q , tau_q , mean_u , alpha_u)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

norm = Normalization(mean_q=mean_q , alpha_q=alpha_q , tau_q=tau_q,
                     mean_u=mean_u , alpha_u=alpha_u)

train_t_norm = norm.transform_ts(train_t)
train_q_norm = norm.transform_qs(train_q)
train_dq_norm = norm.transform_q_ts(train_dq)
train_ddq_norm = norm.transform_q_tts(train_ddq)
train_u_norm = norm.transform_taus(train_u)

test_t_norm = norm.transform_ts(test_t)
test_q_norm = norm.transform_qs(test_q)
test_dq_norm = norm.transform_q_ts(test_dq)
test_ddq_norm = norm.transform_q_tts(test_ddq)
test_u_norm = norm.transform_taus(test_u)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

key = jr.key(0)
sphnn_template = make_sphnn(key)

sphnn = eqx.tree_deserialise_leaves(ROOT/"saved_models/sphnn_selected1.eqx" , sphnn_template)
sphnn_ = klax.finalize(sphnn)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
horizon = 20
opt_steps = 40
lr = 3e-2
    
u_min_norm = jnp.min(test_u_norm,axis=(0, 1))

u_max_norm = jnp.max(test_u_norm,axis=(0, 1))

mpc_model = mpc_optax(
    model=sphnn_,
    horizon=horizon,
    opt_steps=opt_steps,
    learning_rate=lr,
    u_min=u_min_norm,
    u_max=u_max_norm,
    warm_start=False
)
'''
mpc_model = mpc_scipy(
    model=sphnn_,
    horizon=horizon,
    maxiter=opt_steps,
    u_min=u_min_norm,
    u_max=u_max_norm,
    warm_start=False
)
'''
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model = sphnn_


def prediction_sample(t_norm , q_norm , dq_norm , u_norm):

    state_norm_mpc , input_norm_mpc , loss_mpc = mpc_model(t_norm , q_norm , dq_norm)
    q_mpc = norm.inverse_transform_qs(state_norm_mpc[..., :4])
    dq_mpc = norm.inverse_transform_q_ts(state_norm_mpc[..., 4:])
    state_mpc = jnp.concatenate([q_mpc , dq_mpc] , axis = -1)
    input_mpc = norm.inverse_transform_taus(input_norm_mpc)

    init_norm = jnp.concatenate([q_norm[0] , dq_norm[0]])
    state_norm_direct = model (t_norm , init_norm , u_norm)
    q_direct = norm.inverse_transform_qs(state_norm_direct[...,:4])
    dq_direct = norm.inverse_transform_q_ts(state_norm_direct[...,4:])
    state_direct = jnp.concatenate([q_direct , dq_direct] , axis =-1)


    return state_mpc , input_mpc , loss_mpc , state_direct 

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
idx = 10
size = 500
state_mpc, input_mpc, loss_mpc ,state_direct = prediction_sample(
    test_t_norm[idx ],
    test_q_norm[idx ],
    test_dq_norm[idx ],
    test_u_norm[idx]
)


fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharex=True, sharey=False)

for i, ax in enumerate(axes):
    ax.plot(test_t[idx , :], test_q[idx, :, i], lw=2, label="True")
    ax.plot(test_t[idx ,:], state_mpc[:, i], "--", lw=2, label="MPC")
    ax.plot(test_t[idx ,:], state_direct[:, i], "--", lw=2, label="Prediction Direct")
    ax.set_title(f"$q_{i+1}$")
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Joint Position [rad]")

for ax in axes:
    ax.set_xlabel("Time [s]")

axes[0].legend()

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharex=True, sharey=False)

for i, ax in enumerate(axes):
    ax.plot(test_t[idx , :size-1], test_u[idx, :size-1, i], lw=2, label="True")
    ax.plot(test_t[idx ,:size -1], input_mpc[:, i], "--", lw=2, label="MPC")
    ax.set_title(f"$input_{i+1}$")
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Joint input")

for ax in axes:
    ax.set_xlabel("Time [s]")

axes[0].legend()

plt.tight_layout()
plt.show()

mse_sphnn = np.mean((state_direct[:,:4] - test_q[idx,:,])**2, axis=0)        # per joint
rmse_sphnn = np.sqrt(mse_sphnn)      

mse_mpc = np.mean((state_mpc[:,:4] - test_q[idx,:,])**2, axis=0)        # per joint
rmse_mpc = np.sqrt(mse_mpc)     

print("Error metrics for SPHNN prediction per joint:")
for i in range(4):
    print(f"q{i+1}:RMSE={rmse_sphnn[i]:.4e}")

print("Error metrics for MPC - SPHNN prediction per joint:")
for i in range(4):
    print(f"q{i+1}:RMSE={rmse_mpc[i]:.4e}")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

loss_history_mpc(loss_mpc)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N = 50
batch_eval = jax.jit(jax.vmap(prediction_sample , in_axes = (0,0,0,0)))
state_mpc , input_mpc , loss_mpc , state_direct = batch_eval(
    test_t_norm[:N],
    test_q_norm[:N],
    test_dq_norm[:N],
    test_u_norm[:N]
)

print (state_mpc.shape)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

state_true = jnp.concatenate([test_q[:N], test_dq[:N]],axis=-1)
tracking_summary_MPC(state_direct,state_mpc,state_true)
input_summary_MPC(input_mpc,test_u[:N])




# %%
