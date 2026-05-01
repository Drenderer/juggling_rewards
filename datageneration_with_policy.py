import jax
import time
import mujoco as mj
import numpy as np
from jax import numpy as jnp
from jax import random as jr
from dynax import bandlimited_noise

from main_with_policy import XML_PATH, get_policy , get_viwer , pd_control ,get_ball_contact_force, get_ball_contact
from mujoco_environment import Arm , Ball , MjEnvironment


DT = 0.002  # Time step for the simulation


def generate_trajectory (t_max,q1 , q2 , q4, t, render = False):

    ts, ys, ys_t, ys_tt, us = [], [], [], [], []
    ball_x, ball_xt, f, c = [], [], [], []

    policy = get_policy(q1 ,q2 ,q4,t)

    model = mj.MjModel.from_xml_path(str(XML_PATH))
    data = mj.MjData(model)
    viewer = get_viwer(model , data) if render else None
    env = MjEnvironment(model , data , viewer)

    arm = Arm(model , data , "wam")

    #reset env
    q, dq = policy (time = 0)
    arm.q = q
    arm.dq = dq
    arm.tau = np.zeros(arm.num_dof)

    mj.mj_forward(model , data)

    ball0 = Ball (model , data , 0)
    #ball0.x = arm.x + np.array([0.0 , 0.0 ,0.01])
    ball0.x = arm.x + np.array([0.0 , 0.0 ,0.0])
    ball_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "balls/ball0")
    i=0
    while env.time <=t_max:

        q , dq = policy(env.time)
        tau = pd_control(arm , q , dq)
        arm.tau = tau 
        i = i+1
        ball0.record_state()

        env.step()
        if render:
            env.render()
        
        contact_forces = get_ball_contact_force(model , data , ball_body_id)
        contact = get_ball_contact(model , data , ball_body_id)

        ts.append(env.time)
        us.append(arm.tau)
        ys.append(arm.q)
        ys_t.append(arm.dq)
        ys_tt.append(arm.ddq)
        f.append (contact_forces)
        c.append (contact)

    ball_x , ball_xt = ball0.get_recording()
    ts = np.array(ts)
    us = np.array(us)
    ys = np.array(ys)
    ys_t = np.array(ys_t)
    ys_tt = np.array(ys_tt)
    ball_x = np.array(ball_x)
    ball_xt = np.array(ball_xt)
    f = np.array(f)
    c = np.array(c)

    return ts , us , ys , ys_t , ys_tt, ball_x , ball_xt , f , c

def main():
    t_max = 3.0
    seed = int(time.time())
    ts, ys, ys_t, ys_tt, us = [], [], [], [], []
    ball_x, ball_xt, f, c = [], [], [], []
    
    number_trajectory = 20000

    for k in range (number_trajectory):
        seed = int(time.time())
        key = jr.PRNGKey(seed)
        k1, k2, k3 ,k4 = jr.split(key, 4)

        q1 = jr.uniform(k1 , shape=(4,) , minval=-0.25 , maxval=0.25)
        q2 = jr.uniform(k2 , shape=(4,) , minval= 0.65 , maxval=1.45)
        q4 = jr.uniform(k3 , shape=(4,) , minval=0.65 , maxval=1.45)
        t = jr.uniform(k4, shape=(3,) , minval=0.1 , maxval=0.4 )


        if k%100 == 0:
            print (k)
            print (q1)
        _ts , _us , _ys ,_ys_t , _ys_tt , _ball_x , _ball_xt , _f , _c = generate_trajectory (t_max ,q1 , q2 , q4,t, render=False)
        
        ts.append(_ts)
        us.append(_us)
        ys.append(_ys)
        ys_t.append(_ys_t)
        ys_tt.append(_ys_tt)
        ball_x.append(_ball_x)
        ball_xt.append(_ball_xt)
        f.append(_f)
        c.append(_c)
    
    assert all([np.array_equal(ts[0] ,t) for t in ts[1:]])

    np.savez('Data/Initial_data/third dataset/robot_throwing.npz' , ts =ts ,us = us, qs = ys , qs_t = ys_t , qs_tt = ys_tt)
    np.savez('Data/Initial_data/third dataset/ball_throwing.npz' , ts =ts ,ball_x = ball_x , ball_xt = ball_xt , f =f , c =c)
    print("Trajectory generation complete.")

if __name__ == '__main__' :
    main()