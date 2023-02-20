"""
    Wrapping objects in mujoco environments into a simple interface.

    mail@kaiploeger.net
"""
from abc import ABC, abstractmethod
import sys
from threading import Lock
import time

import glfw
import imageio
import mujoco as mj
import numpy as np


class MjViewer:
    """Render a mujoco environment using glfw
    """
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self._gui_lock = Lock()
        self._button_left_pressed = False
        self._button_right_pressed = False
        self._left_double_click_pressed = False
        self._right_double_click_pressed = False
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self._paused = False
        self._transparent = False
        self._contacts = False
        self._joints = False
        self._wire_frame = False
        self._inertias = False
        self._com = False
        self._render_every_frame = False
        self._image_idx = 0
        self._image_path = "/tmp/frame_%07d.png"
        self._time_per_render = 1 / 60.0
        self._run_speed = 1.0
        self._loop_count = 0
        self._advance_by_one_step = False
        self._hide_menu = True

        # glfw init
        glfw.init()
        width, height = glfw.get_video_mode(glfw.get_primary_monitor()).size
        # self.window = glfw.create_window(width // 2, height // 2, "mujoco", None, None)
        self.window = glfw.create_window(width // 4 * 3, height // 4 * 3, "mujoco", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(self.window)
        window_width, _ = glfw.get_window_size(self.window)
        self._scale = framebuffer_width * 1.0 / window_width

        # set callbacks
        glfw.set_cursor_pos_callback(self.window, self._cursor_pos_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)
        glfw.set_key_callback(self.window, self._key_callback)
        self._last_left_click_time = None
        self._last_right_click_time = None

        # create options, camera, scene, context
        self.vopt = mj.MjvOption()
        self.cam = mj.MjvCamera()
        self.scn = mj.MjvScene(self.model, maxgeom=10000)
        self.pert = mj.MjvPerturb()
        self.ctx = mj.MjrContext(
            self.model, mj.mjtFontScale.mjFONTSCALE_150.value
        )

        # get viewport
        self.viewport = mj.MjrRect(0, 0, framebuffer_width, framebuffer_height)

        # overlay, markers
        self._overlay = {}
        self._markers = []

    def _key_callback(self, window, key, scancode, action, mods):
        if action != glfw.RELEASE:
            return
        # Switch cameras
        elif key == glfw.KEY_TAB:
            self.cam.fixedcamid += 1
            self.cam.type = mj.mjtCamera.mjCAMERA_FIXED
            if self.cam.fixedcamid >= self.model.ncam:
                self.cam.fixedcamid = -1
                self.cam.type = mj.mjtCamera.mjCAMERA_FREE
        # Pause simulation
        elif key == glfw.KEY_SPACE and self._paused is not None:
            self._paused = not self._paused
        # Advances simulation by one step.
        elif key == glfw.KEY_RIGHT and self._paused is not None:
            self._advance_by_one_step = True
            self._paused = True
        # Slows down simulation
        elif key == glfw.KEY_S:
            self._run_speed /= 2.0
        # Speeds up simulation
        elif key == glfw.KEY_F:
            self._run_speed *= 2.0
        # Turn off / turn on rendering every frame.
        elif key == glfw.KEY_D:
            self._render_every_frame = not self._render_every_frame
        # Capture screenshot
        elif key == glfw.KEY_T:
            img = np.zeros(
                (
                    glfw.get_framebuffer_size(self.window)[1],
                    glfw.get_framebuffer_size(self.window)[0],
                    3,
                ),
                dtype=np.uint8,
            )
            mj.mjr_readPixels(img, None, self.viewport, self.ctx)
            imageio.imwrite(self._image_path % self._image_idx, np.flipud(img))
            self._image_idx += 1
        # Display contact forces
        elif key == glfw.KEY_C:
            self._contacts = not self._contacts
            self.vopt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = self._contacts
            self.vopt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = self._contacts
        elif key == glfw.KEY_J:
            self._joints = not self._joints
            self.vopt.flags[mj.mjtVisFlag.mjVIS_JOINT] = self._joints
        # Display coordinate frames
        elif key == glfw.KEY_E:
            self.vopt.frame = 1 - self.vopt.frame
        # Hide overlay menu
        elif key == glfw.KEY_H:
            self._hide_menu = not self._hide_menu
        # Make transparent
        elif key == glfw.KEY_R:
            self._transparent = not self._transparent
            if self._transparent:
                self.model.geom_rgba[:, 3] /= 5.0
            else:
                self.model.geom_rgba[:, 3] *= 5.0
        # Display inertia
        elif key == glfw.KEY_I:
            self._inertias = not self._inertias
            self.vopt.flags[mj.mjtVisFlag.mjVIS_INERTIA] = self._inertias
        # Display center of mass
        elif key == glfw.KEY_M:
            self._com = not self._com
            self.vopt.flags[mj.mjtVisFlag.mjVIS_COM] = self._com
        # Wireframe Rendering
        elif key == glfw.KEY_W:
            self._wire_frame = not self._wire_frame
            self.scn.flags[mj.mjtRndFlag.mjRND_WIREFRAME] = self._wire_frame
        # Geom group visibility
        elif key in (glfw.KEY_0, glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4):
            self.vopt.geomgroup[key - glfw.KEY_0] ^= 1
        # Quit
        if key == glfw.KEY_ESCAPE or key==glfw.KEY_CAPS_LOCK:
            print("Pressed ESC!")
            print("Quitting.")
            glfw.set_window_should_close(self.window, True)
            # raise KeyboardInterrupt

    def _cursor_pos_callback(self, window, xpos, ypos):
        if not (self._button_left_pressed or self._button_right_pressed):
            return

        mod_shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if self._button_right_pressed:
            action = (
                mj.mjtMouse.mjMOUSE_MOVE_H
                if mod_shift
                else mj.mjtMouse.mjMOUSE_MOVE_V
            )
        elif self._button_left_pressed:
            action = (
                mj.mjtMouse.mjMOUSE_ROTATE_H
                if mod_shift
                else mj.mjtMouse.mjMOUSE_ROTATE_V
            )
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM

        dx = int(self._scale * xpos) - self._last_mouse_x
        dy = int(self._scale * ypos) - self._last_mouse_y
        width, height = glfw.get_framebuffer_size(window)

        with self._gui_lock:
            if self.pert.active:
                mj.mjv_movePerturb(
                    self.model,
                    self.data,
                    action,
                    dx / height,
                    dy / height,
                    self.scn,
                    self.pert,
                )
            else:
                mj.mjv_moveCamera(
                    self.model, action, dx / height, dy / height, self.scn, self.cam
                )

        self._last_mouse_x = int(self._scale * xpos)
        self._last_mouse_y = int(self._scale * ypos)

    def _mouse_button_callback(self, window, button, act, mods):
        self._button_left_pressed = (
            button == glfw.MOUSE_BUTTON_LEFT and act == glfw.PRESS
        )
        self._button_right_pressed = (
            button == glfw.MOUSE_BUTTON_RIGHT and act == glfw.PRESS
        )

        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = int(self._scale * x)
        self._last_mouse_y = int(self._scale * y)

        # detect a left- or right- doubleclick
        self._left_double_click_pressed = False
        self._right_double_click_pressed = False
        time_now = glfw.get_time()

        if self._button_left_pressed:
            if self._last_left_click_time is None:
                self._last_left_click_time = glfw.get_time()

            time_diff = time_now - self._last_left_click_time
            if time_diff > 0.01 and time_diff < 0.3:
                self._left_double_click_pressed = True
            self._last_left_click_time = time_now

        if self._button_right_pressed:
            if self._last_right_click_time is None:
                self._last_right_click_time = glfw.get_time()

            time_diff = time_now - self._last_right_click_time
            if time_diff > 0.01 and time_diff < 0.2:
                self._right_double_click_pressed = True
            self._last_right_click_time = time_now

        # set perturbation
        key = mods == glfw.MOD_CONTROL
        newperturb = 0
        if key and self.pert.select > 0:
            # right: translate, left: rotate
            if self._button_right_pressed:
                newperturb = mj.mjtPertBit.mjPERT_TRANSLATE
            if self._button_left_pressed:
                newperturb = mj.mjtPertBit.mjPERT_ROTATE

            # perturbation onste: reset reference
            if newperturb and not self.pert.active:
                mj.mjv_initPerturb(self.model, self.data, self.scn, self.pert)
        self.pert.active = newperturb

        # handle doubleclick
        if self._left_double_click_pressed or self._right_double_click_pressed:
            # determine selection mode
            selmode = 0
            if self._left_double_click_pressed:
                selmode = 1
            if self._right_double_click_pressed:
                selmode = 2
            if self._right_double_click_pressed and key:
                selmode = 3

            # find geom and 3D click point, get corresponding body
            width, height = self.viewport.width, self.viewport.height
            aspectratio = width / height
            relx = x / width
            rely = (self.viewport.height - y) / height
            selpnt = np.zeros((3, 1), dtype=np.float64)
            selgeom = np.zeros((1, 1), dtype=np.int32)
            selskin = np.zeros((1, 1), dtype=np.int32)
            selbody = mj.mjv_select(
                self.model,
                self.data,
                self.vopt,
                aspectratio,
                relx,
                rely,
                self.scn,
                selpnt,
                selgeom,
                selskin,
            )

            # set lookat point, start tracking is requested
            if selmode == 2 or selmode == 3:
                # set cam lookat
                if selbody >= 0:
                    self.cam.lookat = selpnt.flatten()
                # switch to tracking camera if dynamic body clicked
                if selmode == 3 and selbody > 0:
                    self.cam.type = mj.mjtCamera.mjCAMERA_TRACKING
                    self.cam.trackbodyid = selbody
                    self.cam.fixedcamid = -1
            # set body selection
            else:
                if selbody >= 0:
                    # record selection
                    self.pert.select = selbody
                    self.pert.skinselect = selskin
                    # compute localpos
                    vec = selpnt.flatten() - self.data.xpos[selbody]
                    mat = self.data.xmat[selbody].reshape(3, 3)
                    self.pert.localpos = self.data.xmat[selbody].reshape(3, 3).dot(vec)
                else:
                    self.pert.select = 0
                    self.pert.skinselect = -1
            # stop perturbation on select
            self.pert.active = 0

        # 3D release
        if act == glfw.RELEASE:
            self.pert.active = 0

    def _scroll_callback(self, window, x_offset, y_offset):
        with self._gui_lock:
            mj.mjv_moveCamera(
                self.model,
                mj.mjtMouse.mjMOUSE_ZOOM,
                0,
                -0.05 * y_offset,
                self.scn,
                self.cam,
            )

    def add_marker(self, **marker_params):
        self._markers.append(marker_params)

    def _add_marker_to_scene(self, marker):
        if self.scn.ngeom >= self.scn.maxgeom:
            raise RuntimeError("Ran out of geoms. maxgeom: %d" % self.scn.maxgeom)

        g = self.scn.geoms[self.scn.ngeom]
        # default values.
        g.dataid = -1
        g.objtype = mj.mjtObj.mjOBJ_UNKNOWN
        g.objid = -1
        g.category = mj.mjtCatBit.mjCAT_DECOR
        g.texid = -1
        g.texuniform = 0
        g.texrepeat[0] = 1
        g.texrepeat[1] = 1
        g.emission = 0
        g.specular = 0.5
        g.shininess = 0.5
        g.reflectance = 0
        g.type = mj.mjtGeom.mjGEOM_BOX
        g.size[:] = np.ones(3) * 0.1
        g.mat[:] = np.eye(3)
        g.rgba[:] = np.ones(4)

        for key, value in marker.items():
            if isinstance(value, (int, float, mj._enums.mjtGeom)):
                setattr(g, key, value)
            elif isinstance(value, (tuple, list, np.ndarray)):
                attr = getattr(g, key)
                attr[:] = np.asarray(value).reshape(attr.shape)
            elif isinstance(value, str):
                assert key == "label", "Only label is a string in mjtGeom."
                if value is None:
                    g.label[0] = 0
                else:
                    g.label = value
            elif hasattr(g, key):
                raise ValueError(
                    "mjtGeom has attr {} but type {} is invalid".format(
                        key, type(value)
                    )
                )
            else:
                raise ValueError("mjtGeom doesn't have field %s" % key)

        self.scn.ngeom += 1

        return

    def _create_overlay(self):
        topleft = mj.mjtGridPos.mjGRID_TOPLEFT
        topright = mj.mjtGridPos.mjGRID_TOPRIGHT
        bottomleft = mj.mjtGridPos.mjGRID_BOTTOMLEFT
        bottomright = mj.mjtGridPos.mjGRID_BOTTOMRIGHT

        def add_overlay(gridpos, text1, text2):
            if gridpos not in self._overlay:
                self._overlay[gridpos] = ["", ""]
            self._overlay[gridpos][0] += text1 + "\n"
            self._overlay[gridpos][1] += text2 + "\n"

        if self._render_every_frame:
            add_overlay(topleft, "", "")
        else:
            add_overlay(
                topleft,
                "Run speed = %.3f x real time" % self._run_speed,
                "[S]lower, [F]aster",
            )
        add_overlay(
            topleft, "Ren[d]er every frame", "On" if self._render_every_frame else "Off"
        )
        add_overlay(
            topleft,
            "Switch camera (#cams = %d)" % (self.model.ncam + 1),
            "[Tab] (camera ID = %d)" % self.cam.fixedcamid,
        )
        add_overlay(topleft, "[C]ontact forces", "On" if self._contacts else "Off")
        add_overlay(topleft, "[J]oints", "On" if self._joints else "Off")
        add_overlay(topleft, "[I]nertia", "On" if self._inertias else "Off")
        add_overlay(topleft, "Center of [M]ass", "On" if self._com else "Off")
        add_overlay(topleft, "T[r]ansparent", "On" if self._transparent else "Off")
        add_overlay(topleft, "[W]ireframe", "On" if self._wire_frame else "Off")
        if self._paused is not None:
            if not self._paused:
                add_overlay(topleft, "Stop", "[Space]")
            else:
                add_overlay(topleft, "Start", "[Space]")
                add_overlay(topleft, "Advance simulation by one step", "[right arrow]")
        add_overlay(
            topleft, "Referenc[e] frames", "On" if self.vopt.frame == 1 else "Off"
        )
        add_overlay(topleft, "[H]ide Menu", "")
        if self._image_idx > 0:
            fname = self._image_path % (self._image_idx - 1)
            add_overlay(topleft, "Cap[t]ure frame", "Saved as %s" % fname)
        else:
            add_overlay(topleft, "Cap[t]ure frame", "")
        add_overlay(topleft, "Toggle geomgroup visibility", "0-4")

        add_overlay(bottomleft, "FPS", "%d%s" % (1 / self._time_per_render, ""))
        add_overlay(bottomleft, "Solver iterations", str(self.data.solver_iter + 1))
        add_overlay(
            bottomleft, "Step", str(round(self.data.time / self.model.opt.timestep))
        )
        add_overlay(bottomleft, "timestep", "%.5f" % self.model.opt.timestep)

    def apply_perturbations(self):
        self.data.xfrc_applied = np.zeros_like(self.data.xfrc_applied)
        mj.mjv_applyPerturbPose(self.model, self.data, self.pert, 0)
        mj.mjv_applyPerturbForce(self.model, self.data, self.pert)

    def render(self):
        # mjv_updateScene, mjr_render, mjr_overlay
        def update():
            # fill overlay items
            self._create_overlay()

            render_start = time.time()
            if self.window is None:
                return
            elif glfw.window_should_close(self.window):
                glfw.terminate()
                sys.exit(0)
            self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(
                self.window
            )
            with self._gui_lock:
                # update scene
                mj.mjv_updateScene(
                    self.model,
                    self.data,
                    self.vopt,
                    self.pert,
                    self.cam,
                    mj.mjtCatBit.mjCAT_ALL.value,
                    self.scn,
                )
                # marker items
                for marker in self._markers:
                    self._add_marker_to_scene(marker)
                # render
                mj.mjr_render(self.viewport, self.scn, self.ctx)
                # overlay items
                if not self._hide_menu:
                    for gridpos, [t1, t2] in self._overlay.items():
                        mj.mjr_overlay(
                            mj.mjtFontScale.mjFONTSCALE_150,
                            gridpos,
                            self.viewport,
                            t1,
                            t2,
                            self.ctx,
                        )
                glfw.swap_buffers(self.window)
            glfw.poll_events()
            self._time_per_render = 0.9 * self._time_per_render + 0.1 * (
                time.time() - render_start
            )

            # clear overlay
            self._overlay.clear()

        if self._paused:
            while self._paused:
                update()
                if self._advance_by_one_step:
                    self._advance_by_one_step = False
                    break
        else:
            self._loop_count += self.model.opt.timestep / (
                self._time_per_render * self._run_speed
            )
            if self._render_every_frame:
                self._loop_count = 1
            while self._loop_count > 0:
                update()
                self._loop_count -= 1

        # clear markers
        self._markers[:] = []

        # apply perturbation (should this come before mj_step?)
        self.apply_perturbations()

    def close(self):
        glfw.terminate()


class Environment(ABC):
    """Abstract base class for all types environments
    """
    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self):
        pass


class MjEnvironment(Environment):
    """Abstracts interactions with mujoco such that the environment can be used as a black box
    """
    def __init__(self, m: mj.MjModel, d: mj.MjData, v: MjViewer|None=None):
        self.model = m
        self.data = d
        self.viewer = v

    def step(self, n_steps=1):
        for _ in range(n_steps):
            mj.mj_step(self.model, self.data)

    def reset(self):
        mj.mj_resetData(self.model, self.data)

    def update(self):
        mj.mj_forward(self.model, self.data)

    def render(self):
        if self.viewer is not None:
            self.viewer.render()


class Ball:
    def __init__(self, m: mj.MjModel, d: mj.MjData, ball_id: int):
        """Simple interface for all juggling balls"""
        self.model = m
        self.data = d
        self.ball_id = ball_id
        self.name = 'ball'+str(ball_id)
        self.free_joint = self.data.joint(self.name)
        self.mocap = self.model.body('balls_des/'+self.name)
        self._xs_rec = []
        self._dxs_rec = []

    @property
    def x(self) -> np.ndarray:
        return self.free_joint.qpos[:3]

    @x.setter
    def x(self, x: np.ndarray):
        self.free_joint.qpos[:3] = x

    @property
    def dx(self) -> np.ndarray:
        return self.free_joint.qvel[:3]

    @dx.setter
    def dx(self, dx):
        self.free_joint.qvel[:3] = dx

    def get_touch_down(self, h_td) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """predicts when and where the ball will land, given the height of a touch-down plane"""
        g = self.model.opt.gravity
        t_td = - self.dx[2]/g[2] + np.sqrt(self.dx[2]**2/g[2]**2 -2*(self.x[2]-h_td)/g[2])
        x_td = self.x + self.dx*t_td + 0.5*g*t_td**2
        dx_td = self.dx + g*t_td
        return t_td, x_td, dx_td

    def set_mocap(self, x):
        self.data.mocap_pos[self.mocap.mocapid] = x

    def record_state(self):
        self._xs_rec.append(self.x)
        self._dxs_rec.append(self.dx)

    def get_recording(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self._xs_rec), np.array(self._dxs_rec)

    def clear_recording(self):
        self._xs_rec = []
        self.dxs_vel = []


class Arm:
    num_dof = 4

    def __init__(self, m: mj.MjModel, d: mj.MjData, name: str):
        """Simple interface for a robot arm"""
        self.model = m
        self.data = d
        self.name = name

        self.joint_names = [f"{self.name}/joints/shoulder_yaw",
                            f"{self.name}/joints/shoulder_pitch",
                            f"{self.name}/joints/shoulder_roll",
                            f"{self.name}/joints/elbow"]
        self.joint_ids = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, joint_name) for joint_name in self.joint_names]
        for joint_id, joint_name in zip(self.joint_ids, self.joint_names):
            assert joint_id != -1, f"Joint '{joint_name}' not found in model. Available joints: {[m.joint(i).name for i in range(m.njnt)]}"

        self.motor_names = [f"{self.name}/motors/shoulder_yaw",
                            f"{self.name}/motors/shoulder_pitch",
                            f"{self.name}/motors/shoulder_roll",
                            f"{self.name}/motors/elbow"]
        self.motor_ids = [mj.mj_name2id(m, mj.mjtObj.mjOBJ_ACTUATOR, act_name) for act_name in self.motor_names]
        for motor_id, motor_name in zip(self.motor_ids, self.motor_names):
            assert motor_id != -1, f"Motor '{motor_name}' not found in model. Available motors: {[m.actuator(i).name for i in range(m.nu)]}"

        self._q_rec, self._dq_rec, self._ddq_rec = [], [], []
        self._x_rec, self._dx_rec= [], []
        self._tau_rec = []
        self._tool_site = self.model.site(self.name + "/sites/tool")

    @property
    def q(self) -> np.ndarray:
        return self.data.qpos[self.joint_ids].copy()

    @q.setter
    def q(self, q: np.ndarray):
        self.data.qpos[self.joint_ids] = q.copy()

    @property
    def dq(self) -> np.ndarray:
        return self.data.qvel[self.joint_ids].copy()

    @dq.setter
    def dq(self, dq):
        self.data.qvel[self.joint_ids] = dq.copy()

    @property
    def ddq(self) -> np.ndarray:
        return self.data.qacc[self.joint_ids].copy()

    @ddq.setter
    def ddq(self, ddq):
        self.data.qacc[self.joint_ids] = ddq.copy()

    @property
    def x(self) -> np.ndarray:
        return self.data.site_xpos[self._tool_site.id].copy()

    @property
    def dx(self) -> np.ndarray:  # TODO: why is result of mj_objectVelocity() rotated?
        return self.jac @ self.dq

    @property
    def tau(self):
        return self.data.ctrl[self.motor_ids].copy()

    @tau.setter
    def tau(self, tau: np.ndarray):
        self.data.ctrl[self.motor_ids] = tau.copy()

    @property
    def jac(self) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mj.mj_jacSite(self.model, self.data, jacp, jacr, self._tool_site.id)
        return jacp[:, self.joint_ids]

    def record_state(self):
        self._q_rec.append(self.q)
        self._dq_rec.append(self.dq)
        self._ddq_rec.append(self.ddq)
        self._x_rec.append(self.x)
        self._dx_rec.append(self.dx)
        self._tau_rec.append(self.tau)

    def get_recording(self):
        return {'q':   np.array(self._q_rec),
                'dq':  np.array(self._dq_rec),
                'ddq': np.array(self._ddq_rec),
                'x':   np.array(self._x_rec),
                'dx':  np.array(self._dx_rec),
                'tau': np.array(self._tau_rec)}

    def clear_recording(self):
        self._q_rec = []
        self._dq_rec = []
        self._ddq_rec = []
        self._x_rec = []
        self._dx_rec = []
        self._tau_rec = []


