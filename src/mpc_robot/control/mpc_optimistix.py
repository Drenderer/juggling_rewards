
import jax
import optimistix
from jax import numpy as jnp

def mpc_optimistix(model, horizon, loss_fn , opt_steps, u_min, u_max , warm_start=True):

    solver = optimistix.LBFGS(history_length=10 ,rtol=1e-5 , atol=1e-5)

    @jax.jit
    def solve_mpc(ts, x0 , reference , u_initial):
        args = (model,ts,x0,reference)
        init_loss , init_grad = jax.value_and_grad(loss_fn)(u_initial , args)
        solution = optimistix.minimise (fn=loss_fn , solver=solver , y0=u_initial , args=args, max_steps=opt_steps , throw=False)
        u_opt = solution.value
        final_loss , final_grad = jax.value_and_grad(loss_fn)(u_opt , args)
        loss = jnp.stack([init_loss , final_loss])
        grad = jnp.stack([jnp.linalg.norm(init_grad),jnp.linalg.norm(final_grad)])
        return u_opt, loss ,grad
    
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
            ts_horizon = jax.lax.dynamic_slice(ts_padded,(k,),(horizon,))
            reference = jax.lax.dynamic_slice(state_padded,(k, 0),(horizon, true_state.shape[1]))
            
            u_initial = jax.lax.cond(
                warm_start,
                lambda _: previous_u,   # if warm up is True
                lambda _: u_zero,       # if warm up is False
                operand=None,
            )

            u_opt, loss_history , grad_history = solve_mpc(ts_horizon,x_current, reference, u_initial)

            u_current = u_opt[0]
            x_next = model(ts_horizon[:2],x_current,jnp.stack([u_current, u_current]))[-1]
            shifted_u = jnp.concatenate([u_opt[1:], u_opt[-1:]],axis=0)

            
            carry = (x_next, shifted_u)
            output = (x_next,u_current,loss_history , grad_history)

            return carry, output
        

        _, outputs = jax.lax.scan(mpc_step,(x_initial, u_zero),jnp.arange(number_of_steps - 1))
        
        x_future, u_mpc, loss_history , grad_history = outputs

        x_mpc = jnp.concatenate([x_initial[None], x_future],axis=0)

        return x_mpc, u_mpc, loss_history , grad_history
    

    return run_mpc

            