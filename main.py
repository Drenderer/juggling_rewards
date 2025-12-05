"""
    Demonstrates some basic two ball juggling with a WAM arm.

    mail@kaiploeger.net
"""

from pathlib import Path

import mujoco as mj
import numpy as np
import time
from policies import CubicMP, ConstantMP, PiecewiseMP
from mujoco_environment import MjEnvironment, MjViewer, Arm, Ball
from rewards import survival_bonus, ball_distance_penalty, control_penalty

import matplotlib.pyplot as plt
from dynax import bandlimited_noise
from misc import generate_aprbs
import jax.random as jr
from diffrax import LinearInterpolation
import jax


DT = 0.002 #simulation time step
t_rest = int(0.1/DT)
XML_PATH = Path(__file__).parent / 'robot_description' / 'one_arm.xml'

#PD Controller Gains
Kp  = np.array([200.0, 300.0, 100.0, 100.0])
Kd  = np.array([  7.0,  15.0,   5.0,   2.5])
MAX_CTRL = np.array([150.0, 125.0,  40.0,  60.0]) #Torque limits (actuator saturation)


def get_policy():
    
    q_via_stroke = np.array([[-0.1,  1.12,  0.        ,  1.28],
                            [+0.08,  0.92,  0.,  1.00]          # changing the target policy can help to have different throw
                             ])
    
    
    
    dq_via_stroke = np.array([[0,  0,  0  ,  0],
                            [0,  0,  0       ,  0          ]
                            ])
    
    times_stroke = np.array([0.1])
    
    policy_wait = ConstantMP(pos=q_via_stroke[0], duration=t_rest*DT) #the original duration was 0.1
    
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


def get_ball_contact_force(model, data, ball_body_id):
    f1 = 0.0
    f2 = 0.0
    fn = 0.0
    """Return max normal contact force on the ball for the current step."""
    for i in range(data.ncon):
        contact = data.contact[i]
        b1 = model.geom_bodyid[contact.geom1]
        b2 = model.geom_bodyid[contact.geom2]

        if b1 == ball_body_id or b2 == ball_body_id:
            f = np.zeros(6, dtype=np.float64)
            mj.mj_contactForce(model, data, i, f)
            if ((f[0]**2 +f[1] **2 + f[2]**2) > (fn**2 +f1 **2 + f2**2)):
                if f[0] > 20 :
                    fn = 20
                else:
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



def main():
    policy = get_policy()

    model = mj.MjModel.from_xml_path(str(XML_PATH))
    data = mj.MjData(model)
    viewer = get_viwer(model, data)
    env = MjEnvironment(model, data, viewer)

    arm = Arm(model, data, 'wam')
    ball0 = Ball(model, data, 0)
   
    ball_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "balls/ball0")
    

    # reset env
    q, dq = policy(time=0)
    arm.q = q
    arm.dq = dq
    arm.tau = np.zeros(arm.num_dof)

    mj.mj_forward(model, data)
    ball0.x = arm.x + np.array([0.0, 0.0, 0.01])

    key = jr.key(33)  
    ts = np.linspace(0, 10, 5000)

    seed = int(time.time()) 
    key = jax.random.PRNGKey(seed)
    noise1 = bandlimited_noise(key = key , length=5000 , max_freq=10 , dt =DT)

    k = 0
    ts = []
    us = []
    ys = []
    ys_t = []
    ys_tt = []
    ball_force=[]
    ball_contact = []


    while env.time <= 5.0:
        q, dq = policy(k * DT)
        tau = pd_control(arm, q, dq)
        arm.tau = tau + 5* noise1[k]
        ball0.record_state()
        env.step()
        env.render()
    
        contact_forces = get_ball_contact_force(model , data , ball_body_id)
        contact = get_ball_contact(model , data , ball_body_id)
        k += 1
    
        ts.append(env.time)
        us.append(arm.tau)
        ys.append(arm.q)
        ys_t.append(arm.dq)
        ys_tt.append(arm.ddq)
        ball_force.append (contact_forces)
        ball_contact.append (contact)
        

    xb0, dxb0 = ball0.get_recording()
    ts = np.array(ts)
    us = np.array(us)
    ys = np.array(ys)
    ys_t = np.array(ys_t)
    ys_tt = np.array(ys_tt)
    xb0 = np.array(xb0)
    dxb0 = np.array(dxb0)
    ball_force = np.array(ball_force)
    ball_contact = np.array(ball_contact)
    


    idx_throw = np.where(ball_force[t_rest:] < 1e-04)[0]
    flag = False
    i=0
    while flag == False:
        if idx_throw[i+10] - idx_throw[i] ==10:
            idx_throw = int(t_rest + idx_throw[i])
            flag = True
        i=i+1
    print ("the time of throwing the ball is " , (idx_throw)* DT)




    idx_floor = np.where(xb0[:, 2] - 0.038 < 1e-4)[0]
    t_end = idx_floor[0]
    print ("the time of the ball touch floor " , (idx_floor[0])* DT)




    fig, axs = plt.subplots(3, 1, sharex=True)
    labels = ["x [m]", "y [m]", "z [m]"]
    for j, ax in enumerate(axs):
        ax.plot(ts[:t_end], xb0[:t_end, j] ,'-')
        ax.plot(ts[:t_end:50], xb0[:t_end:50, j], 'o')
        ax.axvline(ts[idx_throw], linestyle='--', color = 'r' ,linewidth=1.5)
        ax.set_ylabel(labels[j])
        ax.grid(True)

    axs[-1].set_xlabel("time [s]")
    plt.tight_layout()
    plt.show()


    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(xb0[:t_end, 0], xb0[:t_end, 1], xb0[:t_end, 2])
    ax.scatter(xb0[0, 0],      xb0[0, 1],      xb0[0, 2],      marker="o")  # start
    ax.scatter(xb0[t_end-1,0], xb0[t_end-1,1], xb0[t_end-1,2], marker="x")  # end
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Ball 0 trajectory")
    plt.tight_layout()
    plt.show()


    fig, ax = plt.subplots() 
    ax.plot(ts, ball_force[:,0])
    ax.axvline(ts[idx_throw], linestyle='--', color = 'r' ,linewidth=1.5)
    ax.set_ylabel("fn (normal force)")
    ax.set_xlabel("time [s]")
    ax.grid(True)
    plt.show()

    fig, ax = plt.subplots() 
    ax.plot(ts, ball_contact[:])
    ax.axvline(ts[idx_throw], linestyle='--', color = 'r' ,linewidth=1.5)
    ax.set_ylabel("contact")
    ax.set_xlabel("time [s]")
    ax.grid(True)
    plt.show()



    fig, axes = plt.subplots(3, 1)
    axes[0].plot(ts[:idx_throw], us[:idx_throw])
    axes[1].plot(ts[:idx_throw], ys[:idx_throw])
    axes[2].plot(ts[:idx_throw] ,ys_t[:idx_throw])
    plt.show()



if __name__ == '__main__':
    main()

