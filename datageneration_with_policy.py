import jax
import mujoco as mj
import numpy as np
from jax import numpy as jnp
from jax import random as jr
from dynax import bandlimited_noise

from main_with_policy import XML_PATH, get_policy , get_viwer , pd_control ,get_ball_contact_force, get_ball_contact
from mujoco_environment import Arm , Ball , MjEnvironment


DT = 0.002  # Time step for the simulation


def generate_trajectory (t_max, perturbation, q11 , q12 , q21 , q22 , q41, q42, render = False):

    ts, ys, ys_t, ys_tt, us = [], [], [], [], []
    ball_x, ball_xt, f, c = [], [], [], []

    policy = get_policy(q11 , q12 , q21 , q22 , q41, q42)

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
    ball0.x = arm.x + np.array([0.0 , 0.0 ,0.01])
    ball_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "balls/ball0")
    i=0
    while env.time <=t_max:

        q , dq = policy(env.time)
        tau = pd_control(arm , q , dq)
        arm.tau = tau #+5*perturbation[i]
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
    t_max = 2.0
    key = jr.PRNGKey(0)
    key1 ,key2 , key3 , key4 = jr.split(key,4)

    q11 = jnp.array([-0.5 , -0.3 , -0.1 , 0 , 0.1 , 0.3 , 0.5])
    w1 = jnp.array([-0.3 , -0.2 , -0.1 , 0.1 , 0.2 , 0.3])
    q21 = jnp.array([0.8 , 0.9 , 1 , 1.1 , 1.2 , 1.3])
    w2 = jnp.array([ 0 , 0.1 ,0.15 , 0.2 , 0.25 , 0.3])
    q41 = jnp.array([0.9 , 1 , 1.1 , 1.2 , 1.3 , 1.4])
    w4 = jnp.array ([0.1 ,0.2 , 0.3 , 0.4 ,0.5])

    Q11, W1, Q21, W2, Q41, W4 = jnp.meshgrid(q11, w1, q21, w2, q41, w4, indexing="ij")
    P = jnp.stack([Q11, W1, Q21, W2, Q41, W4], axis=-1).reshape(-1, 6)
    mask = (P[:,3] !=0) | (P[: ,5]>0.3)         #omiting all w2= 0 and w4=0.1 , 0.2 ,0.3
    P = P[mask]
    print (P.shape)

    key = jr.key(0)
    ts, ys, ys_t, ys_tt, us = [], [], [], [], []
    ball_x, ball_xt, f, c = [], [], [], []
    
    number_trajectory = len(P)
    k=0
    for noise_key in jr.split (key , number_trajectory):
        perturb = bandlimited_noise ( key = noise_key , length = int(t_max/DT) , max_freq=10 , dt =DT)
        if k%100 == 0:
            print (k)
        _ts , _us , _ys ,_ys_t , _ys_tt , _ball_x , _ball_xt , _f , _c = generate_trajectory (t_max , perturb , P[k,0] , P[k,0] + P[k,1] , 
                                                                                              P[k,2] , P[k,2] - P[k,3] , P[k,4] , P[k,4] - P[k,5], render=False)
        k=k+1
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

    np.savez('Data/robot_throwing.npz' , ts =ts ,us = us, qs = ys , qs_t = ys_t , qs_tt = ys_tt)
    np.savez('Data/ball_throwing.npz' , ts =ts ,ball_x = ball_x , ball_xt = ball_xt , f =f , c =c)
    print("Trajectory generation complete.")

if __name__ == '__main__' :
    main()