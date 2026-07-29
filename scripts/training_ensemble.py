#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import equinox as eqx
import jax
import klax
import optax
import numpy as np
from jax import numpy as jnp
from jax import random as jr
from matplotlib import pyplot as plt

from pathlib import Path
from mpc_robot.models.sphnn import make_sphnn
from mpc_robot.datagen.normalize import Normalization


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
n_models = 5
model_keys = jr.split(key, n_models)
sphnn_ensemble = eqx.filter_vmap(make_sphnn)(model_keys)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@eqx.filter_vmap(
    in_axes=(eqx.if_array(0), None, None, None)
)
def evaluate_ensemble(model, ts, x0, u):
    return model(ts, x0, u)

idx =10
init = jnp.concatenate([test_q_norm[idx , 0] , test_dq_norm[idx,0]])
pred_norm = evaluate_ensemble(
    sphnn_ensemble,
    test_t_norm[idx],
    init,
    test_u_norm[idx],
)

pred_x_norm = pred_norm[:,:,:4]
pred_dx_norm = pred_norm[:,:,4:]

pred_x = norm.inverse_transform_qs(pred_x_norm)
pred_dx = norm.inverse_transform_q_ts(pred_dx_norm)

for i in range(4):

    plt.figure(figsize=(8, 4))

    plt.plot(test_t[idx],test_q[idx, :, i],color="black",linewidth=2,label="True")

    for m in range(n_models):
        plt.plot(test_t[idx],pred_x[m, :, i],"--",linewidth=1.2,alpha=0.8,label=f"model {m+1}")

    plt.xlabel("t [s]")
    plt.ylabel(rf"$q_{i+1}$")
    plt.title(f"DoF {i+1}")
    plt.grid(alpha=0.3)
    plt.legend()

plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
state = jnp.concat([train_q_norm , train_dq_norm] , axis = -1)
state_deriv= jnp.concat([train_dq_norm , train_ddq_norm] , axis = -1)


state_flat = jnp.reshape(state, (-1, state.shape[-1]))
state_deriv_flat = jnp.reshape(state_deriv, (-1, state_deriv.shape[-1]))
input_flat = jnp.reshape(train_u_norm, (-1, train_u_norm.shape[-1]))


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@klax.loss
def derivative_loss(model, data, batch_axis):
    ys, y_ts, us = data

    pred = jax.vmap(
        lambda y, u: model.func(0.0, y, u)
    )(ys, us)

    return jnp.mean(jnp.square(pred - y_ts))

class RunStateUpdater(klax.Callback):
    """Update run_state with the current training step."""

    def on_training_step(self, context):
        context.state.run_state = context.step

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
sphnn_ensemble, hist = klax.fit(
    sphnn_ensemble,
    (state_flat, state_deriv_flat, input_flat),
    batch_size=32,
    optimizer=optax.adam(2e-4),
    loss=derivative_loss,
    steps=50_000,
    verbose=True,
    callbacks=[RunStateUpdater()],
    log_every=100,
    vmap_ensemble=True,
    key=jr.key(1)
)

hist.plot()
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
idx = 10
init = jnp.concatenate([test_q_norm[idx , 0] , test_dq_norm[idx,0]])
pred_norm = evaluate_ensemble(
    sphnn_ensemble,
    test_t_norm[idx],
    init,
    test_u_norm[idx],
)


pred_x_norm = pred_norm[:,:,:4]
pred_dx_norm = pred_norm[:,:,4:]

pred_x = norm.inverse_transform_qs(pred_x_norm)
pred_dx = norm.inverse_transform_q_ts(pred_dx_norm)

for i in range(4):

    plt.figure(figsize=(8, 4))

    plt.plot(test_t[idx],test_q[idx, :, i],color="black",linewidth=2,label="True")

    for m in range(n_models):
        plt.plot(test_t[idx],pred_x[m, :, i],"--",linewidth=1.2,alpha=0.8,label=f"model {m+1}")

    plt.xlabel("t [s]")
    plt.ylabel(rf"$q_{i+1}$")
    plt.title(f"DoF {i+1}")
    plt.grid(alpha=0.3)
    plt.legend()

plt.show()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@klax.loss
def trajectory_loss(model, data, batch_axis):
    ts, ys, us = data
    ys_pred = jax.vmap(model, in_axes=(0,0,0))(ts, ys[:,0], us)
    return jnp.mean(jnp.square(ys_pred - ys))

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t_list = [50,100,200,400,500]
step_list = [5000 , 5000 , 5000 , 3000 , 2000]

state_test = jnp.concat([test_q_norm , test_dq_norm] , axis = -1)


for i in range(len(t_list)):
    sphnn_ensemble, hist = klax.fit(
    sphnn_ensemble,
    (train_t_norm[:,:t_list[i]], state[:,:t_list[i]], train_u_norm[:,:t_list[i]]),
    validation_data=(test_t_norm[:,:t_list[i]], state_test[:,:t_list[i]], test_u_norm[:,:t_list[i]]),
    batch_size=32,
    optimizer=optax.adam(2e-4),
    loss=trajectory_loss,
    steps=step_list[i],
    verbose=True,
    callbacks=[RunStateUpdater()],
    log_every=50,
    vmap_ensemble=True,
    key=jr.key(0)
)
    hist.plot()
    plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
idx = 10
init = jnp.concatenate([test_q_norm[idx , 0] , test_dq_norm[idx,0]])
pred_norm = evaluate_ensemble(
    sphnn_ensemble,
    test_t_norm[idx],
    init,
    test_u_norm[idx],
)


pred_x_norm = pred_norm[:,:,:4]
pred_dx_norm = pred_norm[:,:,4:]

pred_x = norm.inverse_transform_qs(pred_x_norm)
pred_dx = norm.inverse_transform_q_ts(pred_dx_norm)

for i in range(4):

    plt.figure(figsize=(8, 4))

    plt.plot(test_t[idx],test_q[idx, :, i],color="black",linewidth=2,label="True")

    for m in range(n_models):
        plt.plot(test_t[idx],pred_x[m, :, i],"--",linewidth=1.2,alpha=0.8,label=f"model {m+1}")

    plt.xlabel("t [s]")
    plt.ylabel(rf"$q_{i+1}$")
    plt.title(f"DoF {i+1}")
    plt.grid(alpha=0.3)
    plt.legend()

plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

eqx.tree_serialise_leaves("saved_models/sphnn_ensemble1.eqx",sphnn_ensemble)



# %%
