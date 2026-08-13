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
import imageio.v2 as imageio
import imageio_ffmpeg

from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import copy
from mpc_robot.datagen.mujoco_environment import MjEnvironment, MjViewer, Arm, Ball
from mpc_robot.datagen.normalize import Normalization
from mpc_robot.datagen.main_with_policy import get_policy , get_viwer

from mpc_robot.control.mpc_optax import mpc_optax
from mpc_robot.control.mpc_optimistix import mpc_optimistix
from mpc_robot.models.sphnn import make_sphnn
from mpc_robot.kinematic.forward import Forward_kinematic
from mpc_robot.evaluation.evaluation_functions import  loss_history_mpc

print(imageio_ffmpeg.get_ffmpeg_exe())

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
DT = 0.002 #simulation time step
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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% robot + pd + close loop %%%%%%%%%%%%%%%%%%%%%%%%

def true_control_step(y_true, q_des, dq_des):
    tau = pd_control(y_true,q_des,dq_des, Kp , Kd)
    return tau

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
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Mujoco simulation %%%%%%%%%%%%%%%%%%%%%%%%

def main_mujoco(policy , q0 , dq0, motion_mode , input = None,T_end=1.0 , render =True ,name="mujoco_run.mp4"):

    if motion_mode not in {"true" , "sphnn_pd_open" , "sphnn_mpc_open"}:
        raise ValueError(
            "motion_mode must be 'true' or 'sphnn_pd_open' ,''sphnn_mpc_open' "
        )
    
    if motion_mode in {"sphnn_pd_open" , "sphnn_mpc_open"}:
        if input is None:
            raise ValueError(f"{motion_mode} requires input sequence")
    
    
    mj_model = mj.MjModel.from_xml_path(str(XML_PATH))     
    data = mj.MjData(mj_model)  
    if render:
        viewer = get_viwer(mj_model, data)
    else:
        viewer = None                             
                   
    env = MjEnvironment(mj_model, data, viewer)  
    arm = Arm(mj_model, data, "wam")    
   
    if name is not None:
        width = mj_model.vis.global_.offwidth
        height = mj_model.vis.global_.offheight
        renderer = mj.Renderer(mj_model,height=height,width=width)
        if viewer is not None:camera = copy.deepcopy(viewer.cam)
        else:camera = None
        writer = imageio.get_writer(name,fps=50,codec="libx264")
    else:
        renderer = None  
        writer = None 
        camera = None
    
    N = int(round(T_end / DT))

    q_hist  = np.empty((N+1, 4),dtype=jnp.float32 );  q_hist[0]  = np.array(q0)
    dq_hist = np.empty((N+1, 4),dtype=jnp.float32 );  dq_hist[0] = np.array(dq0)
    tau_hist= np.empty((N+1, 4),dtype=jnp.float64) ;  

    arm.q = np.asarray(q0, np.float64)
    arm.dq = np.asarray(dq0, np.float64)
    
    mj.mj_forward(mj_model, data)   
    skip = 10
    for k in range(N):
        qd, dqd = policy(k * DT)
        q_des  = jnp.asarray(qd,  jnp.float32)             
        dq_des = jnp.asarray(dqd, jnp.float32)    

        if motion_mode == "true" :
            y_true = jnp.concatenate([
                    jnp.asarray(arm.q, dtype=jnp.float32),
                    jnp.asarray(arm.dq, dtype=jnp.float32),
                ])
            tau_np = true_control_step(y_true,q_des,dq_des)


        if motion_mode == "sphnn_pd_open":
            tau_np = np.asarray(input[k],dtype=np.float64)

        
        if motion_mode =="sphnn_mpc_open":
            tau_np = np.asarray(input[k],dtype=np.float64)


        arm.tau = tau_np
        tau_hist[k] = tau_np

        if render:
            env.step()
            env.render()
        else:
            mj.mj_step(mj_model, data)

        q_hist[k+1] = arm.q
        dq_hist[k+1] = arm.dq
        
        if name is not None and k % skip == 0:
            renderer.update_scene(data,camera=camera)
            writer.append_data(renderer.render())

    if writer is not None:
        writer.close()
        renderer.close()
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

#q1 = jnp.array([0.3 , 0.3])
#q2 = jnp.array([0.85 , 0.85])
#q4 = jnp.array([0.75, 1.35])
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

state_mpc_norm, tau_mpc_norm, loss_mpc , grad_mpc = run_mpc2(ts_norm,q_policy_norm,dq_policy_norm)

tau_mpc = norm.inverse_transform_taus(tau_mpc_norm)

loss_history_mpc(loss_mpc)

plt.figure(figsize=(10, 5))

plt.plot(grad_mpc[:, 0], label="Initial gradient norm")
plt.plot(grad_mpc[:, -1], label="Final gradient norm")

plt.yscale("log")
plt.xlabel("MPC step")
plt.ylabel("Gradient norm")
plt.title("Initial and final MPC gradient norm")
plt.grid()
plt.legend()

plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% run pd %%%%%%%%%%%%%%%%%%%%%%%%
state_pd, tau_pd = run_pd(sphnn_,q_policy,dq_policy)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% run simulation %%%%%%%%%%%%%%%%%

q_hist_true, dq_hist_true, tau_hist_true = main_mujoco(policy, q0, dq0,motion_mode="true", T_end=1.0 , name="mujoco_run.mp4")
#%%
'''
ts_norm = norm.transform_ts(ts)
q_hist_norm = norm.transform_qs(q_hist_true)
dq_hist_norm = norm.transform_q_ts(dq_hist_true)

state_mpc_norm, tau_mpc_norm, loss_mpc = run_mpc(ts_norm,q_hist_norm,dq_hist_norm)

tau_mpc = norm.inverse_transform_taus(tau_mpc_norm)
'''
q_hist_sphnn, dq_hist_sphnn, tau_hist_sphnn = main_mujoco(policy, q0,dq0,motion_mode="sphnn_pd_open",input = tau_pd, T_end=1.0 , name=None)
q_hist_mpc, dq_hist_mpc, tau_hist_mpc = main_mujoco( policy, q0, dq0,motion_mode="sphnn_mpc_open" ,input=tau_mpc, T_end=1.0 , name=None)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% evaluation in angles domain  %%%%%%%%%%%%%%%%%

t_true = np.asarray(test_t)
fig, axes = plt.subplots(2,2,figsize=(12, 8),sharex=True)
axes = axes.ravel()

for i, ax in enumerate(axes):
    ax.plot(t_true[0],q_hist_true[:500, i],lw=1.6,label="True")
    ax.plot(t_true[0],q_hist_sphnn[:500, i],lw=1.2,label="sphnn_pd")
    ax.plot(t_true[0],q_hist_mpc[:500, i],lw=1.6,label="sphnn_mpc")
    ax.plot(t_true[0],q_policy[:500, i],lw=1.6,label="policy")
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
    ax.plot(t_true[0],tau_hist_true[:500, i],lw=1.6,label="True")
    ax.plot(t_true[0],tau_hist_sphnn[:500, i],lw=1.2,label="sphnn_pd")
    ax.plot(t_true[0],tau_hist_mpc[:500, i],lw=1.6,label="sphnn_mpc")
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

xyz_true  = to_xyz(np.asarray(q_hist_true))
xyz_sphnn = to_xyz(np.asarray(q_hist_sphnn))
xyz_mpc   = to_xyz(np.asarray(q_hist_mpc))
xyz_policy = to_xyz(np.asarray(q_policy))

print(xyz_true.shape)   # (N, 3)


fig, axes = plt.subplots(1, 3, figsize=(15, 4))

labels = ["x", "y", "z"]

for i, ax in enumerate(axes):

    ax.plot(ts[:len(xyz_true)], xyz_true[:, i], label="True")
    ax.plot(ts[:len(xyz_sphnn)], xyz_sphnn[:, i], label="sphnn + PD")
    ax.plot(ts[:len(xyz_mpc)], xyz_mpc[:, i], label="sphnn + MPC")
    ax.plot(ts[:len(xyz_mpc)], xyz_policy[:, i], label="policy")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(f"{labels[i]} [m]")
    ax.set_title(labels[i])
    ax.grid()
    ax.legend()

plt.tight_layout()
plt.show()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% videos  %%%%%%%%%%%%%%%%%

N = len(xyz_true)
t = np.linspace(0, 1.0, N)

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")

# fixed axis limits
all_xyz = np.concatenate([xyz_true, xyz_sphnn, xyz_mpc ,xyz_policy], axis=0)

ax.set_xlim(all_xyz[:, 0].min(), all_xyz[:, 0].max())
ax.set_ylim(all_xyz[:, 1].min(), all_xyz[:, 1].max())
ax.set_zlim(all_xyz[:, 2].min(), all_xyz[:, 2].max())
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("z [m]")


line_true,  = ax.plot([], [], [], label="True" , color ='blue')
line_policy,  = ax.plot([], [], [], label="policy" , color ='red')
line_sphnn, = ax.plot([], [], [], label="SPHNN + PD" , color ='orange',)
line_mpc,   = ax.plot([], [], [], label="SPHNN + MPC" , color='green')

point_true,  = ax.plot([], [], [], "o" , color ='blue')
point_policy,  = ax.plot([], [], [], "o" , color ='red')
point_sphnn, = ax.plot([], [], [], "o" , color ='orange')
point_mpc,   = ax.plot([], [], [], "o" , color='green')

ax.legend()

title = ax.set_title("t = 0.000 s")


def update(k):

    # trajectories until current time
    line_true.set_data(xyz_true[:k+1, 0],xyz_true[:k+1, 1])
    line_true.set_3d_properties(xyz_true[:k+1, 2])

    line_sphnn.set_data(xyz_sphnn[:k+1, 0],xyz_sphnn[:k+1, 1])
    line_sphnn.set_3d_properties(xyz_sphnn[:k+1, 2])

    line_mpc.set_data(xyz_mpc[:k+1, 0],xyz_mpc[:k+1, 1])
    line_mpc.set_3d_properties(xyz_mpc[:k+1, 2])

    line_policy.set_data(xyz_policy[:k+1, 0],xyz_policy[:k+1, 1])
    line_policy.set_3d_properties(xyz_policy[:k+1, 2])

    # moving end-effector points
    point_true.set_data([xyz_true[k, 0]],[xyz_true[k, 1]])
    point_true.set_3d_properties([xyz_true[k, 2]])

    point_sphnn.set_data([xyz_sphnn[k, 0]],[xyz_sphnn[k, 1]])
    point_sphnn.set_3d_properties([xyz_sphnn[k, 2]])

    point_mpc.set_data([xyz_mpc[k, 0]],[xyz_mpc[k, 1]])
    point_mpc.set_3d_properties([xyz_mpc[k, 2]])

    point_policy.set_data([xyz_policy[k, 0]],[xyz_policy[k, 1]])
    point_policy.set_3d_properties([xyz_policy[k, 2]])

    title.set_text(f"t = {t[k]:.3f} s")

    return (line_true,line_sphnn,line_mpc,line_policy,point_true,
            point_sphnn,point_mpc,point_policy,title)

skip = 10
fps = 1 / (DT * skip)
frames = np.arange(0, N, skip)

ani = FuncAnimation(
    fig,
    update,
    frames=frames,
    interval=DT * skip * 1000,
    blit=False
)

plt.close()

HTML(ani.to_jshtml())

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% save videos  %%%%%%%%%%%%%%%%%
ani.save(
    "end_effector_trajectory.gif",
    writer="pillow",
    fps=20,
    dpi=100
)
# %%
