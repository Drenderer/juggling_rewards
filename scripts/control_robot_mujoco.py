#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import equinox as eqx
import jax
import klax
import numpy as np
from matplotlib import pyplot as plt
from jax import numpy as jnp
from jax import random as jr
from pathlib import Path
import mujoco as mj
import numpy as np
import time


import imageio

from matplotlib.animation import FuncAnimation
from IPython.display import HTML

from mpc_robot.datagen.mujoco_environment import MjEnvironment, Arm
from mpc_robot.datagen.normalize import Normalization
from mpc_robot.datagen.main_with_policy import get_policy ,get_viwer

from mpc_robot.control.mpc_optax import mpc_optax
from mpc_robot.control.mpc_optimistix import mpc_optimistix
from mpc_robot.control.mpc_scipy import mpc_scipy

from mpc_robot.models.sphnn import make_sphnn
from mpc_robot.kinematic.forward import Forward_kinematic
from mpc_robot.evaluation.evaluation_functions import  loss_history_mpc



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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
key = jr.key(0)
sphnn_template = make_sphnn(key)

sphnn = eqx.tree_deserialise_leaves(ROOT/"saved_models/sphnn_selected1.eqx" , sphnn_template)
sphnn_ = klax.finalize(sphnn)

norm = Normalization(mean_q=mean_q , alpha_q=alpha_q , tau_q=tau_q,
                     mean_u=mean_u , alpha_u=alpha_u)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DT = 0.002 
t_rest = int(0.1/DT)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
XML_PATH = PROJECT_ROOT / "robot_description" / "one_arm.xml"

Kp  = np.array([200.0, 300.0, 100.0, 100.0])
Kd  = np.array([  7.0,  15.0,   5.0,   2.5])
MAX_CTRL = np.array([150.0, 125.0,  40.0,  60.0]) #


def pd_control(y, q_des, dq_des, kp, kd):
    q = y[:4]
    dq = y[4:]
    tau = kp * (q_des - q) + kd * (dq_des - dq)
    return  jnp.clip(tau, -MAX_CTRL, MAX_CTRL)





#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Forward kinematic constant %%%%%%%%%%%%%%%%%
L = jnp.array([-2.3360866e-05  ,-1.8214112e-05, 1.1860114e+00 ,5.4999387e-01 ,
                2.5295885e-05 ,-4.5004494e-02 ,4.4200087e-01,2.5333025e-05 , 8.8627271e-02 ])


def calculation (L , q):
    matrix = Forward_kinematic(L,q)
    x = matrix [0,3]
    y = matrix [1,3]
    z = matrix [2,3]
    robot_x = jnp.array([x , y , z])
    return robot_x

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% build mujoco model for dynamics of robot %%%%%%%
def build_mujoco_model(XML_PATH):

    mj_model = mj.MjModel.from_xml_path(str(XML_PATH))
    data = mj.MjData(mj_model)

    def model(ts, x0, u):
        horizon = len(u)
        mj.mj_resetData(mj_model, data)
        data.qpos[:4] = np.asarray(x0[:4])
        data.qvel[:4] = np.asarray(x0[4:])
        mj.mj_forward(mj_model, data)
        states = np.empty((horizon, 8))

        for k in range(horizon):
            data.ctrl[:4] = np.asarray(u[k])
            mj.mj_step(mj_model, data)
            states[k, :4] = data.qpos[:4]
            states[k, 4:] = data.qvel[:4]

        return states

    return model

mujoco_dynamics = build_mujoco_model(XML_PATH)



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% sphnn + pd + open loop %%%%%%%%%%%%%%%%%%%%%%%%
@eqx.filter_jit
def one_step_pd(y, q_des, dq_des,model):

    tau = pd_control( y , q_des , dq_des ,Kp , Kd)
    tau_norm = norm.transform_taus(tau)
    us_norm = jnp.stack([tau_norm, tau_norm], axis=0) 
    ts = jnp.array([0.0, DT], dtype=jnp.float32)   # (2,)
    ts_norm = norm.transform_ts(ts)
    y_norm = norm.normalize_state(y)
    y_next_norm = model(ts_norm, y_norm, us_norm)[-1]        # (8,)
    y_next = norm.de_normalize_state(y_next_norm)

    return y_next, tau

@eqx.filter_jit
def run_pd(model, q_reference, dq_reference):
    y_initial = jnp.concatenate([q_reference[0],dq_reference[0]])
    def pd_step(y, reference):
        q_des, dq_des = reference
        y_next, tau = one_step_pd(y,q_des,dq_des,model)

        return y_next, (y_next, tau)

    _, outputs = jax.lax.scan(pd_step,y_initial,(q_reference, dq_reference))
    y_future, tau_pd = outputs
    y_pd = jnp.concatenate([y_initial[None], y_future],axis=0)

    return y_pd, tau_pd
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% sphnn + mpc + open loop %%%%%%%%%%%%%%%%%%%%%%%%

def loss_function (u , args):
    model , ts , x0 , reference = args
    prediciton = model (ts , x0 , u)
    return jnp.mean(jnp.square(reference - prediciton))  # + 0.05 * jnp.mean(u**2)



run_mpc = mpc_optax(
    model=sphnn_,
    horizon=20,
    loss_fn=loss_function,
    opt_steps=50,
    learning_rate=1e-2,
    u_min=-MAX_CTRL,
    u_max=MAX_CTRL,
    warm_start=True
)

run_mpc2 = mpc_optimistix(
    model=sphnn_,
    horizon=20,
    loss_fn=loss_function,
    opt_steps=50,
    u_min=-MAX_CTRL,
    u_max=MAX_CTRL,
    warm_start=True
)

run_mpc3 = mpc_scipy(
    model = mujoco_dynamics,
    horizon =20,
    loss_fn=loss_function,
    maxiter=10,
    u_min=-MAX_CTRL,
    u_max=MAX_CTRL,
    warm_start=True
)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Mujoco simulation %%%%%%%%%%%%%%%%%%%%%%%%

def main_mujoco(policy , q0 , dq0, motion_mode , input = None,T_end=1.0 , render =True):

    if motion_mode not in {"mujoco_pd_close" , "mujoco_mpc_open" , "sphnn_pd_open" , "sphnn_mpc_open"}:
        raise ValueError(
            "motion_mode must be 'mujoco_pd_close' ,'mujoco_mpc_open' or 'sphnn_pd_open' ,''sphnn_mpc_open' "
        )
    
    if motion_mode in {"Mujoco_mpc_open","sphnn_pd_open" , "sphnn_mpc_open"}:
        if input is None:
            raise ValueError(f"{motion_mode} requires input sequence")
    
    
    mj_model = mj.MjModel.from_xml_path(str(XML_PATH))     
    data_simulation = mj.MjData(mj_model)  

    if render:
        viewer = get_viwer(mj_model, data_simulation)
    else:
        viewer = None                             
                   
    env = MjEnvironment(mj_model, data_simulation, viewer)  
    arm = Arm(mj_model, data_simulation, "wam")    

    N = int(round(T_end / DT))

    q_hist  = np.empty((N+1, 4),dtype=jnp.float32 );  q_hist[0]  = np.array(q0)
    dq_hist = np.empty((N+1, 4),dtype=jnp.float32 );  dq_hist[0] = np.array(dq0)
    tau_hist= np.empty((N+1, 4),dtype=jnp.float64) ;  

    arm.q = np.asarray(q0, np.float64)
    arm.dq = np.asarray(dq0, np.float64)
    
    mj.mj_forward(mj_model, data_simulation)   

    for k in range(N):
        qd, dqd = policy(k * DT)
        q_des  = jnp.asarray(qd,  jnp.float32)             
        dq_des = jnp.asarray(dqd, jnp.float32)    

        if motion_mode == "mujoco_pd_close" :
            y_true = jnp.concatenate([
                    jnp.asarray(arm.q, dtype=jnp.float32),
                    jnp.asarray(arm.dq, dtype=jnp.float32),
                ])
            tau_np = pd_control(y_true,q_des,dq_des, Kp , Kd)


        if motion_mode == "sphnn_pd_open":
            tau_np = np.asarray(input[k],dtype=np.float64)

        
        if motion_mode =="sphnn_mpc_open":
            tau_np = np.asarray(input[k],dtype=np.float64)

        if motion_mode =="mujoco_mpc_open":
            tau_np = np.asarray(input[k],dtype=np.float64)

        arm.tau = tau_np
        tau_hist[k] = tau_np

        if render:
            env.step()
            env.render()
        else:
            mj.mj_step(mj_model, data_simulation)

        q_hist[k+1] = arm.q
        dq_hist[k+1] = arm.dq

    if viewer is not None:
        viewer.close()
    
    return q_hist, dq_hist, tau_hist
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% build policy %%%%%%%%%%%%%%%%%%%%%%%%
seed = int(time.time())
key = jr.PRNGKey(seed)
k1, k2, k3 ,k4 = jr.split(key, 4)
q1 = jr.uniform(k1 , shape=(2,) , minval=-0.3 , maxval=0.3)
q2 = jr.uniform(k2 , shape=(2,) , minval= 0.65 , maxval=1.45)
q4 = jr.uniform(k3 , shape=(2,) , minval=0.65 , maxval=1.45)

print (q1 , q2 , q4)
policy = get_policy(q1  , q2 , q4 , 0.3)
q0 , dq0 = policy(0.0)

ts = jnp.arange(0.0, 1.0 + DT, DT)

q_policy = []
dq_policy = []

for t in ts:
    q, dq = policy(t)
    q_policy.append(q)
    dq_policy.append(dq)

q_policy = jnp.stack(q_policy)
dq_policy = jnp.stack(dq_policy)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% run mpc %%%%%%%%%%%%%%%%%%%%%%%%
ts_norm = norm.transform_ts(ts)
q_policy_norm = norm.transform_qs(q_policy)
dq_policy_norm = norm.transform_q_ts(dq_policy)

state_sphnn_mpc_norm, tau_sphnn_mpc_norm, loss_sphnn_mpc , grad_sphnn_mpc = run_mpc2(ts_norm,q_policy_norm,dq_policy_norm)
state_mujco_mpc, tau_mujoco_mpc, loss_mujoco_mpc = run_mpc3(ts,q_policy,dq_policy)
tau_sphnn_mpc = norm.inverse_transform_taus(tau_sphnn_mpc_norm)

#loss_history_mpc(loss_mujoco_mpc)

state_sphnn_pd, tau_sphnn_pd = run_pd(sphnn_,q_policy,dq_policy)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% run simulation %%%%%%%%%%%%%%%%%

q_mujoco_pd, dq_mujoco_pd, tau_mujoco_pd = main_mujoco(policy, q0, dq0,motion_mode="mujoco_pd_close", T_end=1.0)


'''
ts_norm = norm.transform_ts(ts)
q_hist_norm = norm.transform_qs(q_hist_true)
dq_hist_norm = norm.transform_q_ts(dq_hist_true)

state_mpc_norm, tau_mpc_norm, loss_mpc = run_mpc(ts_norm,q_hist_norm,dq_hist_norm)

tau_mpc = norm.inverse_transform_taus(tau_mpc_norm)
'''
q_sphnn_pd, dq_sphnn_pd, tau_sphnn_pd = main_mujoco(policy, q0,dq0,motion_mode="sphnn_pd_open",input = tau_sphnn_pd, T_end=1.0 )
q_sphnn_mpc, dq_sphnn_mpc, tau_sphnn_mpc = main_mujoco(policy, q0,dq0,motion_mode="sphnn_mpc_open",input = tau_sphnn_mpc, T_end=1.0 )
q_mujoco_mpc, dq_mujoco_mpc, tau_mujoco_mpc = main_mujoco( policy, q0, dq0,motion_mode="mujoco_mpc_open" ,input=tau_mujoco_mpc, T_end=1.0)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% evaluation in angles domain  %%%%%%%%%%%%%%%%%
color_policy = 'red'
color_mujoco_pd = 'blue'
color_mujoco_mpc = 'green'
color_sphnn_pd = 'purple'
color_sphnn_mpc = 'orange'

t_true = np.asarray(test_t)
fig, axes = plt.subplots(2,2,figsize=(12, 8),sharex=True)
axes = axes.ravel()

for i, ax in enumerate(axes):
    ax.plot(t_true[0],q_policy[:500, i],lw=1.6,label="policy" , color = color_policy)
    ax.plot(t_true[0],q_mujoco_pd[:500, i],lw=1.6,label="mujoco_pd" , color = color_mujoco_pd)
    ax.plot(t_true[0],q_mujoco_mpc[:500, i],lw=1.6,label="mujoco_mpc" , color = color_mujoco_mpc)
    ax.plot(t_true[0],q_sphnn_pd[:500, i],lw=1.2,label="sphnn_pd" , color = color_sphnn_pd)
    ax.plot(t_true[0],q_sphnn_mpc[:500, i],lw=1.2,label="sphnn_mpc" , color = color_sphnn_mpc)
    ax.set_xlabel("t [s]")
    ax.set_ylabel(rf"$q_{i+1}$")
    ax.set_title(f"DoF {i+1} Position")
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.suptitle("Joint Robot Positions",fontsize=14)

fig.tight_layout()
plt.show()

fig, axes = plt.subplots(2,2,figsize=(12, 8),sharex=True)
axes = axes.ravel()

for i, ax in enumerate(axes):
    ax.plot(t_true[0],tau_mujoco_pd[:500, i],lw=1.6,label="mujoco_pd" , color = color_mujoco_pd)
    ax.plot(t_true[0],tau_mujoco_mpc[:500, i],lw=1.6,label="mujoco_mpc" , color =color_mujoco_mpc)
    ax.plot(t_true[0],tau_sphnn_pd[:500, i],lw=1.2,label="sphnn_pd" , color = color_sphnn_pd)
    ax.plot(t_true[0],tau_sphnn_mpc[:500, i],lw=1.2,label="sphnn_mpc" , color = color_sphnn_mpc)
    
    ax.set_xlabel("t [s]")
    ax.set_ylabel(rf"$input_{i+1}$")
    ax.set_title(f"DoF {i+1} input")
    ax.grid(True, alpha=0.3)
    ax.legend()

fig.suptitle("Joint inputs",fontsize=14)

fig.tight_layout()
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% evaluation in x-y-z domain  %%%%%%%%%%%%%%%%%

to_xyz = jax.jit(jax.vmap(lambda q: calculation(L, q)))

xyz_mujoco_pd  = to_xyz(np.asarray(q_mujoco_pd))
xyz_sphnn_pd = to_xyz(np.asarray(q_sphnn_pd))
xyz_sphnn_mpc   = to_xyz(np.asarray(q_sphnn_mpc))
xyz_mujoco_mpc  = to_xyz(np.asarray(q_mujoco_mpc))
xyz_policy = to_xyz(np.asarray(q_policy))

print(xyz_mujoco_pd.shape)   # (N, 3)


fig, axes = plt.subplots(1, 3, figsize=(15, 4))

labels = ["x", "y", "z"]

for i, ax in enumerate(axes):
    ax.plot(ts[:len(xyz_policy)], xyz_policy[:, i], label="policy" , color = color_policy)
    ax.plot(ts[:len(xyz_mujoco_pd)], xyz_mujoco_pd[:, i], label="mujoco_pd" , color = color_mujoco_pd)
    ax.plot(ts[:len(xyz_mujoco_mpc)], xyz_mujoco_mpc[:, i], label="mujoco_mpc" , color = color_mujoco_mpc)
    ax.plot(ts[:len(xyz_sphnn_pd)], xyz_sphnn_pd[:, i], label="sphnn_pd" , color = color_sphnn_pd)
    ax.plot(ts[:len(xyz_sphnn_mpc)], xyz_sphnn_mpc[:, i], label="sphnn_mpc" , color = color_sphnn_mpc)
    

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"{labels[i]} [m]")
    ax.set_title(labels[i])
    ax.grid()
    ax.legend()

plt.tight_layout()
plt.show()

# %%
