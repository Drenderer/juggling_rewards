"""
    Demonstrates some basic two ball juggling with a WAM arm.

    mail@kaiploeger.net
"""

from pathlib import Path

import mujoco as mj
import numpy as np

from policies import CubicMP, ConstantMP, PiecewiseMP
from mujoco_environment import MjEnvironment, MjViewer, Arm, Ball
from rewards import survival_bonus, ball_distance_penalty, control_penalty

import matplotlib.pyplot as plt
from misc import generate_aprbs
import jax.random as jr
from diffrax import LinearInterpolation
import jax


DT = 0.002
XML_PATH = Path(__file__).parent / 'robot_description' / 'one_arm.xml'

Kp  = np.array([200.0, 300.0, 100.0, 100.0])
Kd  = np.array([  7.0,  15.0,   5.0,   2.5])
MAX_CTRL = np.array([150.0, 125.0,  40.0,  60.0])


def get_policy():
    q_via_stroke = np.array([[-0.1,  1.12,  0.        ,  1.28],
                            [-0.1,  0.92,  0.        ,  1.0],
                            [ 0.08, 1.12,  0.        ,  1.18],
                            [ 0.115, 0.92,  0.        ,  1.0],
                            [-0.08, 1.12,  0.        ,  1.18]])
    dq_via_stroke = np.zeros_like(q_via_stroke)
    times_stroke = np.array([0.1, 0.5, 0.6, 1.0])

    q_via_cyclic = np.array([[-0.08,  1.12, 0., 1.18],
                            [-0.12,  0.92, 0., 1.0],
                            [ 0.08,  1.12, 0., 1.18],
                            [ 0.12,  0.92, 0., 1.0]])
    dq_via_cyclic = np.zeros_like(q_via_cyclic)
    times_cyclic = np.array([0.1, 0.5, 0.6, 1.0])

    policy_wait = ConstantMP(pos=q_via_stroke[0], duration=0.1)
    policy_stroke = CubicMP(q_via_stroke, dq_via_stroke, times_stroke, cyclic=False)
    policy_cyclic = CubicMP(q_via_cyclic, dq_via_cyclic, times_cyclic, cyclic=True)
    policy = PiecewiseMP([policy_wait, policy_stroke, policy_cyclic])
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


def reward_function(arm, ball0, ball1):
    reward =  1 * survival_bonus()
    reward += 0.05 * ball_distance_penalty(ball0.x, ball1.x)
    reward += 0.0002 * control_penalty(arm.tau)
    return reward


def main():
    policy = get_policy()

    model = mj.MjModel.from_xml_path(str(XML_PATH))
    data = mj.MjData(model)
    viewer = get_viwer(model, data)
    env = MjEnvironment(model, data, viewer)

    arm = Arm(model, data, 'wam')
    ball0 = Ball(model, data, 0)
    ball1 = Ball(model, data, 1)


    # reset env
    q, dq = policy(time=0)
    arm.q = q
    arm.dq = dq
    arm.tau = np.zeros(arm.num_dof)

    mj.mj_forward(model, data)
    ball0.x = arm.x + np.array([0.0, 0.0, 0.01])
    ball1.x = np.array([0.88, -0.1, 2.7])

    # Defint the APRBS tau noise
    key = jr.key(1)
    ts = np.linspace(0, 10, 5000)
    def get_noise(key):
        noise = generate_aprbs(key, ts.size, num_jumps=10, initial_value=0.5)
        noise = 0.5*(noise - 0.5)
        return noise
    noises = jax.vmap(get_noise)(jr.split(key, 4)).T
    noise_interp = LinearInterpolation(ts, noises)
    plt.plot(ts, noises)

    @jax.jit
    def interp_noise(t):
        noise = noise_interp.evaluate(t)
        return noise

    k = 0
    cum_reward = 0
    ts = []
    us = []
    ys = []
    ys_t = []
    ys_tt = []
    while env.time <= 10.0:
        q, dq = policy(k * DT)
        # q += interp_noise(env.time)
        tau = pd_control(arm, q, dq)

        reward = reward_function(arm, ball0, ball1)
        cum_reward += reward
        arm.tau = tau

        env.step()

        env.render()
        k += 1

        ts.append(env.time)
        us.append(arm.tau)
        ys.append(arm.q)
        ys_t.append(arm.dq)
        ys_tt.append(arm.ddq)

        # if k % 10 == 0:
            # print(f"{k * DT:.2f} sec, ~{np.floor(k*DT*2):.0f} catches, reward: {reward:.2f}, cum_reward: {cum_reward:.2f}")

    ts = np.array(ts)
    us = np.array(us)
    ys = np.array(ys)
    ys_t = np.array(ys_t)
    ys_tt = np.array(ys_tt)
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(ts, us)
    axes[1].plot(ts, ys)
    plt.show()

    # Save the data
    np.savez('juggle_data.npz', ts=ts, us=us, ys=ys, ys_t=ys_t, ys_tt=ys_tt)

if __name__ == '__main__':
    main()

