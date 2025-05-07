"""
    This file generates data of the robot dynamics by randomly perturbing the policy and 
    writing the motor moments and the joint angles (velocities and accelerations) to a file.
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
import jax.numpy as jnp

from typing import Callable


DT = 0.002  # Time step for the simulation
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


def generate_trajectory(noise: Callable, t_max: float, render=False):  

    ts = []
    us = []
    ys = []
    ys_t = []
    ys_tt = []

    policy = get_policy()

    model = mj.MjModel.from_xml_path(str(XML_PATH))
    data = mj.MjData(model)
    viewer = get_viwer(model, data) if render else None
    env = MjEnvironment(model, data, viewer)

    arm = Arm(model, data, 'wam')

    # reset env
    q, dq = policy(time=0)
    arm.q = q
    arm.dq = dq
    arm.tau = np.zeros(arm.num_dof)

    ts.append(env.time)
    us.append(arm.tau)
    ys.append(arm.q)
    ys_t.append(arm.dq)
    ys_tt.append(arm.ddq)

    mj.mj_forward(model, data)
    
    while env.time <= t_max:
        q, dq = policy(env.time)
        q += noise(env.time)
        tau = pd_control(arm, q, dq)

        arm.tau = tau

        env.step()

        if render:
            env.render()

        ts.append(env.time)
        us.append(arm.tau)
        ys.append(arm.q)
        ys_t.append(arm.dq)
        ys_tt.append(arm.ddq)


    ts = np.array(ts)
    us = np.array(us)
    ys = np.array(ys)
    ys_t = np.array(ys_t)
    ys_tt = np.array(ys_tt)
    
    return ts, us, ys, ys_t, ys_tt
    

def get_abrbs(key, max_value, t_max:float, num_jumps:int=10):

    # Define the APRBS tau noise
    ts = np.linspace(0, 10, int(t_max/DT))
    def get_noise(key):
        noise = generate_aprbs(key, ts.size, num_jumps=10, initial_value=0.5)
        noise = max_value*2*(noise - 0.5)
        return noise
    noises = jax.vmap(get_noise)(jr.split(key, 4)).T
    noise_interp = LinearInterpolation(ts, noises)

    @jax.jit
    def interp_noise(t):
        noise = noise_interp.evaluate(t)
        return noise 
    
    return interp_noise

def get_multisine(key, max_value:float, num_freqs:int=10, num_data=4):

    freqs = jnp.arange(num_freqs)
    sin_amplitudes = jr.normal(key, shape=(num_data, num_freqs)) / freqs.size
    # cos_amplitudes = jr.normal(key, shape=(num_data, num_freqs)) / freqs.size

    @jax.jit
    def noise(t):
        s = sin_amplitudes * jnp.sin(freqs * t)
        # c = cos_amplitudes * jnp.cos(freqs * t)
        return max_value * s.sum(axis=1) #+ c.sum(axis=1)

    return noise


def main():

    t_max = 10.0
    
    key = jr.key(0)

    ts = []
    us = []
    ys = []
    ys_t = []
    ys_tt = []
    for noise_key in jr.split(key, 1):
        # noise = get_noise(noise_key, max_value=0.25, t_max)
        noise = get_multisine(noise_key, max_value=0.25)
        _ts, _us, _ys, _ys_t, _ys_tt = generate_trajectory(noise, t_max, render=False)
        ts.append(_ts)
        us.append(_us)
        ys.append(_ys)
        ys_t.append(_ys_t)
        ys_tt.append(_ys_tt)

    ts = np.stack(ts, axis=0)
    us = np.stack(us, axis=0)
    ys = np.stack(ys, axis=0)
    ys_t = np.stack(ys_t, axis=0)
    ys_tt = np.stack(ys_tt, axis=0)

    fig, axes = plt.subplots(2, 1)
    n = 0
    axes[0].plot(ts[n], us[n])
    axes[1].plot(ts[n], ys[n])
    plt.show()

    # Save the data
    np.savez('juggle_data.npz', ts=ts, us=us, ys=ys, ys_t=ys_t, ys_tt=ys_tt)


if __name__ == '__main__':
    main()