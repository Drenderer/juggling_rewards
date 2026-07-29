
import jax
import optax
import numpy as np
from jax import numpy as jnp
from scipy.optimize import minimize

def mpc_scipy(model,horizon,maxiter,u_min,u_max,warm_start=True):

    input_dim = u_min.shape[0]
    jax_dtype = jnp.asarray(u_min).dtype

    u_min_np = np.asarray(u_min, dtype=np.float64)
    u_max_np = np.asarray(u_max, dtype=np.float64)

    bounds = list(zip(np.tile(u_min_np, horizon),np.tile(u_max_np, horizon)))

    def rollout(ts, x0, u):
        u_sequence = jnp.concatenate([u, u[-1:]],axis=0)
        return model(ts, x0, u_sequence)[1:]

    def loss_function(u, ts, x0, reference):
        prediction = rollout(ts,x0,u)
        return jnp.mean((prediction - reference) ** 2)

    loss_and_grad = jax.jit(jax.value_and_grad(loss_function))

    def solve_mpc(ts,x0,reference,u_initial):

        evaluation_losses = []

        def objective(u_flat):
            u = jnp.asarray(u_flat,dtype=jax_dtype,).reshape(horizon, input_dim)
            loss, gradient = loss_and_grad(u,ts,x0,reference)
            loss.block_until_ready()
            evaluation_losses.append(float(loss))
            return (float(loss),np.asarray(gradient,dtype=np.float64).ravel())

        result = minimize(objective,np.asarray(u_initial,dtype=np.float64,).ravel(),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": maxiter,
            },
        )

        u_opt = jnp.asarray(result.x,dtype=jax_dtype,).reshape(horizon, input_dim)

        return (u_opt,np.asarray(evaluation_losses))



    def run_mpc(ts,true_q,true_dq):

        true_state = jnp.concatenate([true_q, true_dq],axis=-1,)
        number_of_steps = ts.shape[0]
        dt = ts[-1] - ts[-2]
        extra_time = (ts[-1]+ dt * jnp.arange( 1,horizon + 1))
        ts_padded = jnp.concatenate([ts, extra_time],axis=0,)

        final_reference = jnp.repeat(true_state[-1:],horizon,axis=0)

        state_padded = jnp.concatenate([true_state, final_reference],axis=0)

        x_current = true_state[0]
        u_zero = jnp.zeros((horizon, input_dim),dtype=jax_dtype)
        previous_u = u_zero

        # Compile the objective once before entering the SciPy MPC loop.
        initial_ts = ts_padded[:horizon + 1]
        initial_reference = state_padded[1:horizon + 1]

        initial_loss, initial_gradient = loss_and_grad(u_zero,initial_ts,x_current,initial_reference,)
        jax.block_until_ready((initial_loss, initial_gradient))

        x_history = [x_current]
        u_history = []
        loss_history = []

        for k in range(number_of_steps - 1):

            ts_horizon = ts_padded[k : k + horizon + 1]
            reference = state_padded[k + 1 : k + horizon + 1]

            u_initial = (previous_u if warm_start else u_zero)

            u_opt, step_loss_history = solve_mpc(ts_horizon,x_current,reference,u_initial)

            u_current = u_opt[0]

            x_next = model(ts_horizon[:2],x_current,jnp.stack([u_current, u_current]))[-1]

            previous_u = jnp.concatenate([u_opt[1:], u_opt[-1:]],axis=0)

            x_history.append(x_next)
            u_history.append(u_current)
            loss_history.append(step_loss_history)

            x_current = x_next

        x_mpc = jnp.stack(x_history,axis=0)
        u_mpc = jnp.stack(u_history,axis=0)

        max_length = max(len(losses) for losses in loss_history)
        loss_history = np.stack([np.pad(losses,(0, max_length - len(losses)),mode="edge",)for losses in loss_history])
        loss_history = jnp.asarray(loss_history)


        return (x_mpc,u_mpc,loss_history)

    return run_mpc
