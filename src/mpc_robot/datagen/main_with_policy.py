from pathlib import Path

import mujoco as mj
import numpy as np
import time
from policies import CubicMP, ConstantMP, PiecewiseMP
from mujoco_environment import MjEnvironment, MjViewer, Arm, Ball
import matplotlib.pyplot as plt
from dynax import bandlimited_noise
import jax.random as jr
from jax import random as jr



DT = 0.002 #simulation time step
t_rest = int(0.1/DT)
#XML_PATH = Path(__file__).parent / 'robot_description' / 'one_arm.xml'
PROJECT_ROOT = Path(__file__).resolve().parents[2]
XML_PATH = PROJECT_ROOT / "robot_description" / "one_arm.xml"

Kp  = np.array([200.0, 300.0, 100.0, 100.0])
Kd  = np.array([  7.0,  15.0,   5.0,   2.5])
MAX_CTRL = np.array([150.0, 125.0,  40.0,  60.0]) #

def get_policy(q1 , q2 , q4 , t):                      
    
    q_via_stroke = np.array([[ q1[0] , q2[0] , 0 , q4[0]],
                            [q1[1],  q2[1] , 0 , q4[1]]])
                
    
    dq_via_stroke = np.array([[0,  0,  0  ,  0],
                              [0,  0,  0  ,  0],])
    
    times_stroke = np.array([t])
    
    policy_wait = ConstantMP(pos=q_via_stroke[0], duration=t_rest*DT)   #the original duration was 0.1
    policy_stroke = CubicMP(q_via_stroke, dq_via_stroke, times_stroke, cyclic=False) 
    policy_hold   = ConstantMP(pos=q_via_stroke[-1],     duration=5.0)
    policy = PiecewiseMP([policy_wait, policy_stroke , policy_hold]) 
    return policy


def get_viwer(model, data):
    viewer = MjViewer(model, data)
    viewer.vopt.geomgroup[0] = True
    viewer.vopt.geomgroup[1] = True
    viewer.vopt.geomgroup[2] = True
    viewer.vopt.geomgroup[3] = False
    viewer.vopt.geomgroup[4] = False
    viewer.vopt.geomgroup[5] = False
    viewer._hide_menu = True
    viewer._run_speed = 1.0
    # viewer._run_speed = 0.1
    viewer.cam.distance = 3.1
    viewer.cam.lookat[2] += 1.4
    viewer.cam.elevation = -15
    viewer.cam.azimuth = -135
    return viewer


def pd_control(robot, q_des, dq_des):
    q = robot.q
    dq = robot.dq
    tau = Kp * (q_des - q) + Kd * (dq_des - dq)
    return np.clip(tau, -MAX_CTRL, MAX_CTRL)

'''
def get_ball_contact_force(model, data, ball_body_id):
    f1 = 0.0
    f2 = 0.0
    fn = 0.0
    for i in range(data.ncon):
        contact = data.contact[i]
        b1 = model.geom_bodyid[contact.geom1]
        b2 = model.geom_bodyid[contact.geom2]
        if b1 == ball_body_id or b2 == ball_body_id:
            f = np.zeros(6, dtype=np.float64)
            mj.mj_contactForce(model, data, i, f)
            if ((f[0]**2 +f[1] **2 + f[2]**2) > (fn**2 +f1 **2 + f2**2)):
                fn = f[0]
                f1 = f[1]
                f2 = f[2]
    return np.array([fn])


def get_ball_contact(model, data, ball_body_id):
    c=0
    for i in range(data.ncon):
        contact = data.contact[i]
        b1 = model.geom_bodyid[contact.geom1]
        b2 = model.geom_bodyid[contact.geom2]
        if b1 == ball_body_id or b2 == ball_body_id:
           c=c+1
    if c>0:
        return 1
    else:
        return 0 


'''

def main():
    seed = int(time.time())
    key = jr.PRNGKey(seed)
    k1, k2, k3 ,k4 = jr.split(key, 4)

    q1 = jr.uniform(k1 , shape=(2,) , minval=-0.3 , maxval=0.3)
    q2 = jr.uniform(k2 , shape=(2,) , minval= 0.65 , maxval=1.45)
    q4 = jr.uniform(k3 , shape=(2,) , minval=0.65 , maxval=1.45)
    #t = jr.uniform(k4, shape=(1,) , minval=0.2 , maxval=0.4 )


    #q1 = np.array([-0.7 , -0.7 , 0.087 , -0.088])
    #q2 = np.array([1.2 ,1.2 , 1.2 , 1.2])
    #q4 = np.array([0.856 , 1.006 , 0.564 , 0.629])
    
    print (q1 , q2 , q4 )
    policy = get_policy(q1  , q2 , q4 , 0.3)
    model = mj.MjModel.from_xml_path(str(XML_PATH))
    data = mj.MjData(model)
    viewer = get_viwer(model, data)
    env = MjEnvironment(model, data, viewer)

    arm = Arm(model, data, 'wam')
    #ball0 = Ball(model, data, 0)
   
    # find the ball_id and ball size
    ball_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "balls/ball0")
    adr = model.body_geomadr[ball_body_id]
    num = model.body_geomnum[ball_body_id]

    #print(f"Body '{"balls/ball0"}' (id={ball_body_id}) has {num} geoms")

    for geom_id in range(adr, adr + num):
        geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geom_id)
        geom_type = model.geom_type[geom_id]
        geom_size = model.geom_size[geom_id].copy()   # (3,)
        #print(f"  geom_id={geom_id}, name={geom_name}, type={int(geom_type)}, size={geom_size}")


    # reset env
    q, dq = policy(time=0)
    arm.q = q
    arm.dq = dq
    arm.tau = np.zeros(arm.num_dof)

    mj.mj_forward(model, data)
    #ball0.x = arm.x + np.array([0.0, 0.0, 0.01])
    #ball0.x = arm.x + np.array([0.0, 0.0, 0.0])

    key = jr.key(33)  
    ts = np.linspace(0, 10, 5000)

    k = 0
    ts ,us, ys, ys_t ,ys_tt = [],[],[],[],[]
    ball_force , ball_contact = [],[]

    while env.time <= 1.0:
        q, dq = policy(k * DT)
        tau = pd_control(arm, q, dq)
        arm.tau = tau  
        #ball0.record_state()
        env.step()
        env.render()
    
        #contact_forces = get_ball_contact_force(model , data , ball_body_id)
        #contact = get_ball_contact(model , data , ball_body_id)
        k += 1
    
        ts.append(env.time)
        us.append(arm.tau)
        ys.append(arm.q)
        ys_t.append(arm.dq)
        ys_tt.append(arm.ddq)
        #ball_force.append (contact_forces)
        #ball_contact.append (contact)
        

    #xb0, dxb0 = ball0.get_recording()
    ts = np.array(ts)
    us = np.array(us)
    ys = np.array(ys)
    ys_t = np.array(ys_t)
    ys_tt = np.array(ys_tt)
    #xb0 = np.array(xb0)
    #dxb0 = np.array(dxb0)
    #ball_force = np.array(ball_force)
    #ball_contact = np.array(ball_contact)
    
    '''
    idx_throw = np.where(ball_contact[t_rest:] == 0)[0]
    idx=None
    for i in range(len(idx_throw) - 100):
        if idx_throw[i+100] - idx_throw[i] == 100:
            idx = int(t_rest + idx_throw[i])
            break
    
    if idx is not None:
        print ("the time of throwing is" , idx*DT)
        idx_hit = np.where(ball_contact[idx+1:] == 1)[0]
        t_end = int(idx +1 + idx_hit[0])
        print ("the time of the ball touch floor or hit " , (t_end)* DT)
        print ("average of contact befor idx of throw", np.mean(ball_contact[t_rest:idx]))
    else:
        print ("no throwing in this sample")
        t_end = 2500
        idx = 2500


    fig, axs = plt.subplots(3, 1, sharex=True)
    labels = ["x [m]", "y [m]", "z [m]"]
    for j, ax in enumerate(axs):
        ax.plot(ts[:t_end], xb0[:t_end, j] ,'-')
        ax.plot(ts[:t_end:10], xb0[:t_end:10, j], 'o')
        ax.axvline(ts[idx], linestyle='--', color = 'r' ,linewidth=1.5)
        ax.set_ylabel(labels[j])
        ax.grid(True)

    axs[-1].set_xlabel("time [s]")
    plt.tight_layout()
    plt.show()


    fig, ax = plt.subplots() 
    ax.plot(ts, ball_force[:,0])
    ax.axvline(ts[idx], linestyle='--', color = 'r' ,linewidth=1.5)
    ax.set_ylabel("fn (normal force)")
    ax.set_xlabel("time [s]")
    ax.grid(True)
    plt.show()

    fig, ax = plt.subplots() 
    ax.plot(ts, ball_contact[:])
    ax.axvline(ts[idx], linestyle='--', color = 'r' ,linewidth=1.5)
    ax.set_ylabel("contact")
    ax.set_xlabel("time [s]")
    ax.grid(True)
    plt.show()
    '''

    fig, axes = plt.subplots(3, 1)
    axes[0].plot(ts, us)
    axes[1].plot(ts, ys)
    axes[2].plot(ts ,ys_t)
    plt.show()

    #print ("ball velocity in z direction is :" , dxb0[idx , 2])

if __name__ == '__main__':
    main()
