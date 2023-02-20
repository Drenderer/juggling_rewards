"""
    Demonstrates some basic two ball juggling with a WAM arm.

    mail@kaiploeger.net
"""

from pathlib import Path

import mujoco as mj
import numpy as np

from policies import CubicMP, ConstantMP, PiecewiseMP
from mujoco_environment import MjEnvironment, MjViewer, Arm, Ball



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
    viewer._run_speed = 0.1
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

    k = 0
    while True:
        q, dq = policy(k * DT)
        tau = pd_control(arm, q, dq)
        arm.tau = tau
        env.step()
        env.render()
        k += 1
        if k % 500 == 0:
            print(f"{k * DT:.0f}sec, ~{k*DT*2:.0f} catches")



if __name__ == '__main__':
    main()

