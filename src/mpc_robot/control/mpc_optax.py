
import jax
import optax
from jax import numpy as jnp

def mpc_optax(model, horizon, opt_steps, learning_rate, u_min, u_max , warm_start=True):

    optimizer = optax.adam(learning_rate)
    
    def rollout(ts, x0, u):
        u = jnp.concatenate([u, u[-1:]], axis=0)
        return model(ts, x0, u)[1:]

    def loss_function(u, ts, x0, reference):
        prediction = rollout(ts, x0, u)
        return jnp.mean((prediction - reference) ** 2)

    loss_and_grad = jax.value_and_grad(loss_function)

    def solve_mpc(ts, x0, reference, u_initial):

        optimizer_state = optimizer.init(u_initial)

        def optimization_step(carry, _):

            u, optimizer_state = carry
            loss, gradient = loss_and_grad(u, ts, x0, reference)
            updates, optimizer_state = optimizer.update(gradient, optimizer_state, u)
            u = optax.apply_updates(u, updates)

            return (u, optimizer_state), loss

        (u_opt, _), loss_history = jax.lax.scan(optimization_step,(u_initial, optimizer_state),xs=None,length=opt_steps)

        return u_opt, loss_history

    @jax.jit
    def run_mpc(ts, true_q, true_dq):

        true_state = jnp.concatenate([true_q, true_dq], axis=-1)
        number_of_steps = ts.shape[0]
        dt = ts[-1] - ts[-2]

        extra_time = (ts[-1]+ dt * jnp.arange(1, horizon + 1))
        ts_padded = jnp.concatenate([ts, extra_time])
        final_reference = jnp.repeat(true_state[-1:],horizon,axis=0)
        state_padded = jnp.concatenate([true_state, final_reference],axis=0)

        x_initial = true_state[0]
        u_zero = jnp.zeros((horizon, u_min.shape[0]))
        
        def mpc_step(carry, k):

            x_current, previous_u = carry

            ts_horizon = jax.lax.dynamic_slice(ts_padded,(k,),(horizon + 1,))

            reference = jax.lax.dynamic_slice(state_padded,(k + 1, 0),(horizon, true_state.shape[1]))

            u_initial = jax.lax.cond(
                warm_start,
                lambda _: previous_u,   # if warm up is True
                lambda _: u_zero,       # if warm up is False
                operand=None,
            )


            u_opt, loss_history = solve_mpc(ts_horizon,x_current, reference, u_initial)

            u_current = u_opt[0]

            x_next = model(ts_horizon[:2],x_current,jnp.stack([u_current, u_current]))[-1]

            shifted_u = jnp.concatenate([u_opt[1:], u_opt[-1:]],axis=0)

            
            carry = (x_next, shifted_u)

            output = (x_next,u_current,loss_history)

            return carry, output
         
        
    
        _, outputs = jax.lax.scan(mpc_step,(x_initial, u_zero),jnp.arange(number_of_steps - 1))
        
        x_future, u_mpc, loss_history = outputs

        x_mpc = jnp.concatenate([x_initial[None], x_future],axis=0)

        return x_mpc, u_mpc, loss_history
    

    return run_mpc