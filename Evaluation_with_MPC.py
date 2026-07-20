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
sphnn_template = ODESolver(isphs)

sphnn = eqx.tree_deserialise_leaves("saved_models/sphnn_over.eqx" , sphnn_template)
#sphnn =  eqx.tree_deserialise_leaves("experiment.db-x-artifact-2-content.bin" , sphnn_template)
sphnn_ = klax.finalize(sphnn)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
idx =20
init = jnp.concatenate([test_q_norm[idx , 0] , test_dq_norm[idx,0]])
pred_norm = sphnn_ (test_t_norm[idx] , init , test_u_norm[idx])

pred_x_norm = pred_norm[:,:4]
pred_dx_norm = pred_norm[:,4:]

pred_x = norm.inverse_transform_qs(pred_x_norm)
pred_dx = norm.inverse_transform_q_ts(pred_dx_norm)

fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharex=True, sharey=False)

for i, ax in enumerate(axes):
    ax.plot(test_t[idx], test_q[idx, :, i], lw=2, label="True")
    ax.plot(test_t[idx], pred_x[:, i], "--", lw=2, label="Prediction Direct")

    ax.set_title(f"$q_{i+1}$")
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Joint Position [rad]")

for ax in axes:
    ax.set_xlabel("Time [s]")

axes[0].legend()

plt.tight_layout()
plt.show()

mse_sphnn = np.mean((pred_x[:,] - test_q[idx,:,])**2, axis=0)        # per joint
rmse_sphnn = np.sqrt(mse_sphnn)                              

print("Error metrics for SPHNN prediction per joint:")
for i in range(4):
    print(f"q{i+1}:RMSE={rmse_sphnn[i]:.4e}")


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%starting MPC
horizon = 20
state =8
input=4
opt_step = 10
lr = 1e-2
optimizer = optax.adam(lr)       


def mpc(model,horizon,opt_steps,learning_rate,u_min,u_max,):
    """
    Creates a compiled MPC controller.
    model:Trained dynamics model.
    horizon:Number of future MPC steps.
    opt_steps:Number of Adam iterations per MPC step.
    """
    optimizer = optax.adam(learning_rate)

    # Predict states over one MPC horizon
    
    def rollout(ts, x0, u):
        """
        ts : (horizon + 1,)
        x0 : (8,)
        u  : (horizon, 4)

        returns:
            predicted future states: (horizon, 8)
        """
        u_sequence = jnp.concatenate([u, u[-1:]], axis=0)
        prediction = model(ts, x0, u_sequence)

        return prediction[1:]

    

    # Tracking objective

    def loss_function(u, ts, x0, reference):
        prediction = rollout(ts, x0, u)
        return jnp.mean((prediction - reference) ** 2)

    loss_and_grad = jax.value_and_grad(loss_function)

   

    # Solve one MPC optimization problem
    
    def solve_mpc(ts, x0, reference, u_initial):

        optimizer_state = optimizer.init(u_initial)

        def optimization_step(_, carry):
            u, optimizer_state = carry

            loss, gradient = loss_and_grad(u,ts,x0,reference)

            updates, optimizer_state = optimizer.update(gradient,optimizer_state,u)

            u = optax.apply_updates(u, updates)

            # Keep controls inside the training-data range
            u = jnp.clip(u, u_min, u_max)

            return u, optimizer_state


        u_opt, optimizer_state = jax.lax.fori_loop(0,opt_steps,optimization_step,
                                                   (u_initial, optimizer_state),)

        final_loss = loss_function(u_opt,ts,x0,reference,)

        return u_opt, final_loss


    
    # Complete closed-loop MPC
    @jax.jit
    def run_mpc(ts, true_q, true_dq):

        true_state = jnp.concatenate([true_q, true_dq],axis=-1)
        number_of_steps = ts.shape[0]

        # -----------------------------------------------------
        # Padding keeps every optimization horizon the same size
        # -----------------------------------------------------
        dt = ts[-1] - ts[-2]
        extra_time = ts[-1] + dt * jnp.arange(1,horizon + 1)

        ts_padded = jnp.concatenate([ts, extra_time],axis=0)

        final_reference = jnp.repeat(true_state[-1:],horizon,axis=0) #repeat final state for pading

        state_padded = jnp.concatenate([true_state, final_reference],axis=0)

        x_initial = true_state[0]

        # Initial control guess
        u_initial = jnp.zeros((horizon, u_min.shape[0]))

        u_initial = jnp.clip(u_initial,u_min,u_max )

        
        # One closed-loop MPC time step
        
        def mpc_step(carry, k):

            x_current, previous_u = carry

            # Fixed-size time window
            ts_horizon = jax.lax.dynamic_slice(ts_padded,(k,),(horizon + 1,))

            # Fixed-size desired trajectory
            reference = jax.lax.dynamic_slice(state_padded,(k + 1, 0),(horizon, true_state.shape[1]))

            # Optimize future controls
            u_opt, loss = solve_mpc(ts_horizon,x_current,reference,previous_u)

            # Apply only the first control
            u_current = u_opt[0]

            # Predict one closed-loop step
            ts_step = ts_horizon[:2]

            u_step = jnp.stack([u_current, u_current],axis=0)

            x_next = model(ts_step,x_current,u_step)[-1]

            # Shift optimized sequence for warm-starting
            shifted_u = jnp.concatenate([u_opt[1:], u_opt[-1:]],axis=0)

            next_carry = (x_next, shifted_u)
            output = (x_next, u_current, loss)

            return next_carry, output

        # Compile and execute the complete MPC loop
        _, outputs = jax.lax.scan(mpc_step,(x_initial, u_initial),jnp.arange(number_of_steps - 1))

        x_future, u_mpc, losses = outputs

        x_mpc = jnp.concatenate([x_initial[None, :], x_future],axis=0)

        return x_mpc, u_mpc, losses

    return run_mpc
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

horizon = 20
opt_steps = 10
learning_rate = 1e-2

u_min_norm = jnp.min(train_u_norm,axis=(0, 1))

u_max_norm = jnp.max(train_u_norm,axis=(0, 1))

mpc_model = mpc(
    model=sphnn_,
    horizon=horizon,
    opt_steps=opt_steps,
    learning_rate=learning_rate,
    u_min=u_min_norm,
    u_max=u_max_norm,
)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
idx =0
init = jnp.concatenate([train_q_norm[idx , 0] , train_dq_norm[idx,0]])
pred_norm = sphnn_ (train_t_norm[idx] , init , train_u_norm[idx])

pred_x_norm = pred_norm[:,:4]
pred_dx_norm = pred_norm[:,4:]

pred_x = norm.inverse_transform_qs(pred_x_norm)
pred_dx = norm.inverse_transform_q_ts(pred_dx_norm)

x_mpc_norm, u_mpc_norm, mpc_losses = mpc_model(
    train_t_norm[idx, :],
    train_q_norm[idx, :],
    train_dq_norm[idx, :],
)

# Wait until JAX computation is finished
x_mpc_norm.block_until_ready()


print(x_mpc_norm.shape)
print(u_mpc_norm.shape)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
q_mpc_norm = x_mpc_norm[:, :4]
dq_mpc_norm = x_mpc_norm[:, 4:]

q_mpc = norm.inverse_transform_qs(q_mpc_norm)
dq_mpc = norm.inverse_transform_q_ts(dq_mpc_norm)

fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharex=True, sharey=False)

for i, ax in enumerate(axes):
    ax.plot(train_t[idx , :], train_q[idx, :, i], lw=2, label="True")
    ax.plot(train_t[idx ,:], pred_x[:, i], "--", lw=2, label="Prediction Direct")
    ax.plot(train_t[idx ,:], q_mpc[:, i], "--", lw=2, label="MPC")
    ax.set_title(f"$q_{i+1}$")
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Joint Position [rad]")

for ax in axes:
    ax.set_xlabel("Time [s]")

axes[0].legend()

plt.tight_layout()
plt.show()

mse_sphnn = np.mean((pred_x[:,] - train_q[idx,:,])**2, axis=0)        # per joint
rmse_sphnn = np.sqrt(mse_sphnn)      

mse_mpc = np.mean((q_mpc[:,] - train_q[idx,:,])**2, axis=0)        # per joint
rmse_mpc = np.sqrt(mse_mpc)     

print("Error metrics for SPHNN prediction per joint:")
for i in range(4):
    print(f"q{i+1}:RMSE={rmse_sphnn[i]:.4e}")

print("Error metrics for MPC - SPHNN prediction per joint:")
for i in range(4):
    print(f"q{i+1}:RMSE={rmse_mpc[i]:.4e}")
# %%
