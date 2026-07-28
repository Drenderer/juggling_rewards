#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import time
import equinox as eqx
import jax
import klax
import optax
import numpy as np
from scipy.optimize import minimize
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
from evaluation_functions import tracking_summary , input_summary

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
state_size =8
input_size=4

key = jr.key(0)
ficnn_key , h_key , j_key , r_key , g_key = jr.split(key , 5)

ficnn =FICNN(in_size=state_size , out_size='scalar' , width_sizes=[64,128,64] , key = ficnn_key)
H = ConvexLyapunov(ficnn , state_size= state_size , minimum_learnable=True , key = h_key )
J = ConstantSkewSymmetricMatrix((state_size,state_size) , key = j_key)
R = ConstantSPDMatrix ((state_size,state_size) ,key=r_key)
G = ConstantMatrix ( (state_size,input_size) , key =g_key)

isphs = ISPHS( H , J , R, G)
sphnn_template = ODESolver(isphs)

sphnn = eqx.tree_deserialise_leaves("saved_models/sphnn_1.eqx" , sphnn_template)
sphnn_ = klax.finalize(sphnn)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%starting MPC

def mpc_scipy(model, horizon, opt_steps, u_min, u_max):

    input_dim = len(u_min)

    def rollout(ts, x0, u):
        u_sequence = jnp.concatenate([u, u[-1:]], axis=0)
        return model(ts, x0, u_sequence)[1:]

    def loss_function(u_flat, ts, x0, reference):
        u = u_flat.reshape(horizon, input_dim)
        prediction = rollout(ts, x0, u)
        return jnp.mean((prediction - reference) ** 2)

    loss_and_grad = jax.jit(jax.value_and_grad(loss_function))

    def objective(u_flat, ts, x0, reference):
        loss, grad = loss_and_grad(jnp.asarray(u_flat),ts,x0,reference)
        return float(loss), np.asarray(grad, dtype=np.float64)

    bounds = list(zip(
        np.tile(np.asarray(u_min), horizon),
        np.tile(np.asarray(u_max), horizon),
    ))

    def solve_mpc(ts, x0, reference, u_initial):
        start = time.perf_counter()
        result = minimize(
            objective,
            np.asarray(u_initial).reshape(-1),
            args=(ts, x0, reference),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": opt_steps},
        )
        elapsed = time.perf_counter() - start

        print(
            f"time={elapsed:.3f}s | "
            f"nit={result.nit} | "
            f"nfev={result.nfev} | "
            f"njev={result.njev} | "
            f"success={result.success}"
        )

        u_opt = result.x.reshape(horizon, input_dim)

        return jnp.asarray(u_opt), result.fun

    def run_mpc(ts, true_q, true_dq):

        true_state = jnp.concatenate([true_q, true_dq], axis=-1)
        number_of_steps = len(ts)

        dt = ts[-1] - ts[-2]

        ts_padded = jnp.concatenate([
            ts,ts[-1] + dt * jnp.arange(1, horizon + 1)])

        state_padded = jnp.concatenate([
            true_state,jnp.repeat(true_state[-1:], horizon, axis=0)])

        x_current = true_state[0]

        previous_u = jnp.clip(jnp.zeros((horizon, input_dim)),u_min,u_max)

        x_mpc = [x_current]
        u_mpc = []
        losses = []

        for k in range(number_of_steps - 1):
            ts_horizon = ts_padded[k:k + horizon + 1]
            reference = state_padded[k + 1:k + horizon + 1]

            u_opt, loss = solve_mpc(
                ts_horizon,
                x_current,
                reference,
                previous_u,
            )

            u_current = u_opt[0]

            x_current = model(ts_horizon[:2],x_current,jnp.stack([u_current, u_current]))[-1]

            previous_u = jnp.concatenate([
                u_opt[1:],
                u_opt[-1:],
            ])

            x_mpc.append(x_current)
            u_mpc.append(u_current)
            losses.append(loss)

        return(jnp.stack(x_mpc),jnp.stack(u_mpc),jnp.asarray(losses))

    return run_mpc

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
def mpc_optax(model,horizon,opt_steps,learning_rate,u_min,u_max,):
   
    optimizer = optax.adam(learning_rate)
    #optimizer = optax.sgd(learning_rate , momentum=0.9 , nesterov=True)
    
    def rollout(ts, x0, u):
        u_sequence = jnp.concatenate([u, u[-1:]], axis=0)
        prediction = model(ts, x0, u_sequence)

        return prediction[1:]


    def loss_function(u, ts, x0, reference):
            prediction = rollout(ts, x0, u)
            return jnp.mean((prediction - reference) ** 2)

    loss_and_grad = jax.value_and_grad(loss_function)

    
    def solve_mpc(ts, x0, reference, u_initial):
    
        optimizer_state = optimizer.init(u_initial)

        def optimization_step(_, carry):
            u, optimizer_state = carry
            loss, gradient = loss_and_grad(u,ts,x0,reference)
            updates, optimizer_state = optimizer.update(gradient,optimizer_state,u)
            u = optax.apply_updates(u, updates)
            u = jnp.clip(u, u_min, u_max)

            return u, optimizer_state

        u_opt, optimizer_state = jax.lax.fori_loop(0,opt_steps,optimization_step,
                                                   (u_initial, optimizer_state),)
        
        final_loss = loss_function(u_opt,ts,x0,reference,)

        return u_opt, final_loss

    @jax.jit
    def run_mpc(ts, true_q, true_dq):

        true_state = jnp.concatenate([true_q, true_dq],axis=-1)
        number_of_steps = ts.shape[0]

        dt = ts[-1] - ts[-2]
        extra_time = ts[-1] + dt * jnp.arange(1,horizon + 1)

        ts_padded = jnp.concatenate([ts, extra_time],axis=0)
        final_reference = jnp.repeat(true_state[-1:],horizon,axis=0) #repeat final state for pading
        state_padded = jnp.concatenate([true_state, final_reference],axis=0)

        x_initial = true_state[0]

        u_initial = jnp.zeros((horizon, u_min.shape[0]))
        u_initial = jnp.clip(u_initial,u_min,u_max )

    
        
        def mpc_step(carry, k):

            x_current, previous_u = carry

            ts_horizon = jax.lax.dynamic_slice(ts_padded,(k,),(horizon + 1,))
            reference = jax.lax.dynamic_slice(state_padded,(k + 1, 0),(horizon, true_state.shape[1]))

            u_opt, loss = solve_mpc(ts_horizon,x_current,reference,previous_u)

            u_current = u_opt[0]

            ts_step = ts_horizon[:2]

            u_step = jnp.stack([u_current, u_current],axis=0)

            x_next = model(ts_step,x_current,u_step)[-1]

            shifted_u = jnp.concatenate([u_opt[1:], u_opt[-1:]],axis=0)

            next_carry = (x_next, shifted_u)
            output = (x_next, u_current, loss)

            return next_carry, output
        
        _, outputs = jax.lax.scan(mpc_step,(x_initial, u_initial),jnp.arange(number_of_steps - 1))

        x_future, u_mpc, losses = outputs

        x_mpc = jnp.concatenate([x_initial[None, :], x_future],axis=0)

        return x_mpc, u_mpc, losses

    return run_mpc

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def dummy_model(ts , x0 , u):
    return jnp.repeat(x0[None,:] , len(ts) , axis = 0)

horizon = 20
opt_steps = 10
lr = 1e-2
    
u_min_norm = jnp.min(test_u_norm,axis=(0, 1))

u_max_norm = jnp.max(test_u_norm,axis=(0, 1))

mpc_model = mpc_optax(
    model=sphnn_,
    horizon=horizon,
    opt_steps=opt_steps,
    learning_rate=lr,
    u_min=u_min_norm,
    u_max=u_max_norm,
)

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
idx = 0
size = 500
state_mpc, input_mpc, loss_mpc ,state_direct = prediction_sample(
    test_t_norm[idx],
    test_q_norm[idx],
    test_dq_norm[idx],
    test_u_norm[idx],
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

N = 20
batch_eval = jax.jit(jax.vmap(prediction_sample , in_axes = (0,0,0,0)))
state_mpc , input_mpc , loss_mpc , state_direct = batch_eval(
    test_t_norm[:N],
    test_q_norm[:N],
    test_dq_norm[:N],
    test_u_norm[:N]
)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

state_true = jnp.concatenate([test_q[:N], test_dq[:N]],axis=-1)
tracking_summary(state_direct,state_mpc,state_true)
input_summary(input_mpc,test_u[:N])




# %%
