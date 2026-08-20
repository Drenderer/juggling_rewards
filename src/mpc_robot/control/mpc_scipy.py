

import numpy as np
from scipy.optimize import minimize

def mpc_scipy(model,horizon,loss_fn,maxiter,u_min,u_max,warm_start=True):

    input_dim = u_min.shape[0]

    def solve_mpc(ts,x0,reference,u_initial):

        evaluation_losses = []

        def objective(u_flat):
            u = np.asarray(u_flat,dtype=np.float64).reshape(horizon, input_dim)
            args = (model , ts ,x0,reference)
            loss = loss_fn(u , args)
            evaluation_losses.append(float(loss))
            return float(loss)

        result = minimize(objective,np.asarray(u_initial,dtype=np.float64).ravel(),
            method="L-BFGS-B",
            options={"maxiter": maxiter , "eps": 1e-2})
        u_opt = np.asarray(result.x,dtype=np.float64).reshape(horizon, input_dim)
        
        return (u_opt,np.asarray(evaluation_losses))


    def run_mpc(ts,true_q,true_dq):

        true_state = np.concatenate([true_q, true_dq],axis=-1,)
        number_of_steps = ts.shape[0]
        dt = ts[-1] - ts[-2]
        extra_time = (ts[-1]+ dt * np.arange( 1,horizon + 1))
        ts_padded = np.concatenate([ts, extra_time],axis=0,)

        final_reference = np.repeat(true_state[-1:],horizon,axis=0)

        state_padded = np.concatenate([true_state, final_reference],axis=0)

        x_current = true_state[0]
        u_zero = np.zeros((horizon, input_dim),dtype=np.float64)
        previous_u = u_zero

        x_history = [x_current]
        u_history = []
        loss_history = []

        for k in range(number_of_steps - 1):
            print (k)
            ts_horizon = ts_padded[k+1 : k + horizon + 1]
            reference = state_padded[k+1  : k + horizon + 1]

            u_initial = (previous_u if warm_start else u_zero)

            u_opt, step_loss_history = solve_mpc(ts_horizon,x_current,reference,u_initial)

            u_current = u_opt[0]
        
            #x_next = model(ts_horizon[:2],x_current,np.stack([u_current, u_current]))[-1]
            x_next = model(ts_horizon[:1],x_current,u_current[None, :])[-1]
            previous_u = np.concatenate([u_opt[1:], u_opt[-1:]],axis=0)

            x_history.append(x_next)
            u_history.append(u_current)
            loss_history.append(step_loss_history)

            x_current = x_next

        x_mpc = np.stack(x_history,axis=0)
        u_mpc = np.stack(u_history,axis=0)

        max_length = max(len(losses) for losses in loss_history)
        loss_history = np.stack([np.pad(losses,(0, max_length - len(losses)),mode="edge",)for losses in loss_history])

        return (x_mpc,u_mpc,loss_history)

    return run_mpc
