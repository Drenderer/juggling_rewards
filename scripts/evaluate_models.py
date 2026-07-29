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
from mpc_robot.evaluation.evaluation_functions import evaluate_ensemble_training


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
sphnn_ensemble_temp = eqx.filter_vmap(make_sphnn)(model_keys)



sphnn = eqx.tree_deserialise_leaves(ROOT/"saved_models/sphnn_ensemble1.eqx",sphnn_ensemble_temp)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@eqx.filter_vmap(
    in_axes=(eqx.if_array(0), None, None, None)
)
def evaluate_ensemble(model, ts, x0, u):
    return model(ts, x0, u)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
idx =10
init = jnp.concatenate([test_q_norm[idx , 0] , test_dq_norm[idx,0]])
pred_norm = evaluate_ensemble(
    sphnn,
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
def prediction_sample(t_norm , q_norm , dq_norm , u_norm):

    init_norm = jnp.concatenate([q_norm[0] , dq_norm[0]])
    state_norm_direct = evaluate_ensemble (sphnn,t_norm , init_norm , u_norm)
    q_direct = norm.inverse_transform_qs(state_norm_direct[...,:4])
    dq_direct = norm.inverse_transform_q_ts(state_norm_direct[...,4:])
    state_direct = jnp.concatenate([q_direct , dq_direct] , axis =-1)


    return  state_direct 

N = 100

batch_eval = eqx.filter_jit(jax.vmap(prediction_sample,in_axes=(0, 0, 0, 0)))

state_direct = batch_eval(
    test_t_norm[:N],
    test_q_norm[:N],
    test_dq_norm[:N],
    test_u_norm[:N],
)

print (state_direct.shape)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rmse, selected_idx = evaluate_ensemble_training(
    state_pred=state_direct,
    q_true=test_q[:N],
    bins="auto",
)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
selected_model = jax.tree.map(
    lambda x: x[selected_idx] if eqx.is_array(x) else x,
    sphnn,
)

eqx.tree_serialise_leaves(ROOT/"saved_models/sphnn_selected1.eqx",selected_model)
# %%
