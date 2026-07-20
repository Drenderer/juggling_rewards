#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import equinox as eqx
import jax
import klax
import optax
import numpy as np
from jax import numpy as jnp
from jax import random as jr

from dynax import ISPHS, ConvexLyapunov, ODESolver
from klax.nn import (
    FICNN,
    ConstantMatrix,
    ConstantSkewSymmetricMatrix,
    ConstantSPDMatrix,
)

from pathlib import Path
from normalize import Normalization, coefficients
from matplotlib import pyplot as plt
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_robot = np.load("Data_MPC/robot_train.npz")
train_t = data_robot['time']
train_q = data_robot['robot_q']
train_dq = data_robot['robot_dq']
train_ddq = data_robot['robot_ddq']
train_u = data_robot['robot_u']

print (train_t.shape,train_q.shape , train_ddq.shape , train_u.shape)

data_robot = np.load("Data_MPC/robot_test.npz")
test_t = data_robot['time']
test_q = data_robot['robot_q']
test_dq = data_robot['robot_dq']
test_ddq = data_robot['robot_ddq']
test_u = data_robot['robot_u']

print (test_t.shape,test_q.shape , test_ddq.shape , test_u.shape)

data = np.load('Data_MPC/norm_value.npz')
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
ficnn_key , h_key , j_key , r_key , g_key = jr.split(key , 5)

ficnn =FICNN(in_size=8 , out_size='scalar' , width_sizes=[64,128,64] , key = ficnn_key)
H = ConvexLyapunov(ficnn , state_size= 8 , minimum_learnable=True , key = h_key )
J = ConstantSkewSymmetricMatrix((8,8) , key = j_key)
R = ConstantSPDMatrix ((8,8) ,key=r_key)
G = ConstantMatrix ( (8,4) , key =g_key)

isphs = ISPHS( H , J , R, G)
sphnn = ODESolver(isphs)


sphnn_ = klax.finalize(sphnn)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
idx =10
init = jnp.concatenate([test_q_norm[idx , 0] , test_dq_norm[idx,0]])
pred_norm = sphnn_ (test_t_norm[idx] , init , test_u_norm[idx])

pred_x_norm = pred_norm[:,:4]
pred_dx_norm = pred_norm[:,4:]

pred_x = norm.inverse_transform_qs(pred_x_norm)
pred_dx = norm.inverse_transform_q_ts(pred_dx_norm)

for i in range(4):
    plt.figure()
    plt.plot(test_t[idx,:], test_q[idx, :, i], lw=1.6, label="True")
    plt.plot(test_t[idx,:], pred_x[:, i], "--", lw=1.2, label="Direct Predicted SPHNN")
    plt.xlabel("t [s]")
    plt.ylabel(rf"$q_{i+1}$")
    plt.title(f"DoF {i+1} Position: True vs Predicted")
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
state = jnp.concat([train_q_norm , train_dq_norm] , axis = -1)
state_deriv= jnp.concat([train_dq_norm , train_ddq_norm] , axis = -1)

print (state.shape , state_deriv.shape)
state_flat = jnp.reshape(state, (-1, state.shape[-1]))
state_deriv_flat = jnp.reshape(state_deriv, (-1, state_deriv.shape[-1]))
input_flat = jnp.reshape(train_u_norm, (-1, train_u_norm.shape[-1]))

print (state_flat.shape)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@klax.loss
def derivative_loss(model, data, batch_axis):
    ys, y_ts, us = data

    pred = jax.vmap(
        lambda y, u: model.func(0.0, y, u)
    )(ys, us)

    return jnp.mean(jnp.square(pred - y_ts))

class RunStateUpdater(klax.Callback):
    """Updates the run_state to be the training step."""
    
    def on_training_step(self, context):
        context.state.run_state = context.state.step

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sphnn_, hist = klax.fit(
    sphnn_,
    (state_flat, state_deriv_flat, input_flat),
    batch_size=32,
    optimizer=optax.adam(2e-4),
    loss=derivative_loss,
    steps=100_000,
    verbose=True,
    callbacks=[RunStateUpdater()],
    log_every=100,
    key=jr.key(0)
)

hist.plot()
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
idx = 35
init = jnp.concatenate([test_q_norm[idx , 0] , test_dq_norm[idx,0]])
pred_norm = sphnn_ (test_t_norm[idx] , init , test_u_norm[idx])

pred_x_norm = pred_norm[:,:4]
pred_dx_norm = pred_norm[:,4:]

pred_x = norm.inverse_transform_qs(pred_x_norm)
pred_dx = norm.inverse_transform_q_ts(pred_dx_norm)

for i in range(4):
    plt.figure()
    plt.plot(test_t[idx,:], test_q[idx, :, i], lw=1.6, label="True")
    plt.plot(test_t[idx,:], pred_x[:, i], "--", lw=1.2, label="Direct Predicted SPHNN")
    plt.xlabel("t [s]")
    plt.ylabel(rf"$q_{i+1}$")
    plt.title(f"DoF {i+1} Position: True vs Predicted")
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.show()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@klax.loss
def trajectory_loss(model, data, batch_axis):
    ts, ys, us = data
    ys_pred = jax.vmap(model, in_axes=(0,0,0))(ts, ys[:,0], us)
    return jnp.mean(jnp.square(ys_pred - ys))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t_list = [50,100,200,400,500]

state_test = jnp.concat([test_q_norm , test_dq_norm] , axis = -1)


for i in range(len(t_list)):
    sphnn, hist = klax.fit(
    sphnn,
    (train_t_norm[:10,:t_list[i]], state[:10,:t_list[i]], train_u_norm[:10,:t_list[i]]),
    validation_data=(test_t_norm[:,:t_list[i]], state_test[:,:t_list[i]], test_u_norm[:,:t_list[i]]),
    batch_size=10,
    optimizer=optax.adam(2e-4),
    loss=trajectory_loss,
    steps=5000,
    verbose=True,
    callbacks=[RunStateUpdater()],
    log_every=50,
    key=jr.key(0)
)
    hist.plot()
    plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
idx =3
sphnn_ = klax.finalize(sphnn)
init = jnp.concatenate([train_q_norm[idx , 0] , train_dq_norm[idx,0]])
pred_norm = sphnn_ (train_t_norm[idx] , init , train_u_norm[idx])

pred_x_norm = pred_norm[:,:4]
pred_dx_norm = pred_norm[:,4:]

pred_x = norm.inverse_transform_qs(pred_x_norm)
pred_dx = norm.inverse_transform_q_ts(pred_dx_norm)

fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharex=True, sharey=False)

for i, ax in enumerate(axes):
    ax.plot(train_t[idx], train_q[idx, :, i], lw=2, label="True")
    ax.plot(train_t[idx], pred_x[:, i], "--", lw=2, label="Prediction")

    ax.set_title(f"$q_{i+1}$")
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Joint Position [rad]")

for ax in axes:
    ax.set_xlabel("Time [s]")

axes[0].legend()

plt.tight_layout()
plt.show()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


plt.figure(figsize=(10, 6))

colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

for i in range(4):
    plt.plot(
        test_t[idx],
        test_q[idx, :, i],
        color=colors[i],
        lw=2,
        label=f"True $q_{i+1}$",
    )

    plt.plot(
        test_t[idx],
        pred_x[:, i],
        "--",
        color=colors[i],
        lw=2,
        label=f"Pred $q_{i+1}$",
    )

plt.xlabel("Time [s]")
plt.ylabel("Joint Position [rad]")
plt.title("Robot Joint Positions: True vs Predicted")
plt.grid(alpha=0.3)

plt.legend(ncol=2)
plt.tight_layout()
plt.show()

#%%%%
eqx.tree_serialise_leaves("saved_models/sphnn_over.eqx", sphnn_)



# %%
