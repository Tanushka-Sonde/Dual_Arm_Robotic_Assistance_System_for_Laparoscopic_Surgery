# =============================================================================
#  Franka Panda + TDCR (50-seg freeze-12) Combined Controller
#  
#  Panda arm : DLS velocity IK  (from dls_velocity_ctrl)
#  TDCR      : Keyboard teleoperation (pynput)
#
#  Keyboard layout:
#  ─────────────────────────────────────────────────────
#  Panda IK (target position):
#    Arrow UP/DOWN    : move target +/- Y
#    Arrow LEFT/RIGHT : move target +/- X  
#    W / S            : move target +/- Z
#    R                : reset target to initial EE position
#
#  TDCR bend (muscle actuators A_1, A_2):
#    K                : activate tendon t1 (bend one way)
#    L                : activate tendon t2 (bend other way)
#
#  TDCR gripper (muscle actuators A_3, A_4):
#    Z                : close gripper (increase activation)
#    C                : open gripper  (decrease activation)
#    X                : auto-grasp pulse (close then release)
#  ─────────────────────────────────────────────────────
# =============================================================================

import os
import sys
import time
import threading
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from pynput import keyboard

# ── Project root (adjust to your structure) ──────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.dls_velocity_control.dls_velocity_ctrl import DLSVelocityPlanner
from utils.quaternion_math.quaternion_math import quaternion_math

# ── Model path ────────────────────────────────────────────────────────────────
XML_PATH = str(
    Path(__file__).resolve().parent.parent
    / "description" / "franka_emika_panda" / "scene_1frank_tdcr50S.xml"
)

# ── Load model ────────────────────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)

# ── Reset to home keyframe ────────────────────────────────────────────────────
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
if key_id != -1:
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    print("[INFO] Reset to 'home' keyframe.")
else:
    print("[WARN] 'home' keyframe not found, using default state.")
    mujoco.mj_resetData(model, data)

mujoco.mj_forward(model, data)

# ── IK / Panda control settings ──────────────────────────────────────────────
CONTROL_START_TIME  = 0.5   # seconds before IK activates
POS_GAIN            = 2.0   # task-space proportional gain [1/s]
ORI_GAIN            = 1.0
MAX_CARTESIAN_SPEED = 0.010  # m/s
MAX_ANGULAR_SPEED   = 0.05   # rad/s
TARGET_STEP         = 0.01  # m per key press

# ── TDCR actuator indices (from XML actuator order) ──────────────────────────
# Index 0-6 : panda arm actuators (actuator1..7)
# Index 7   : A_1 (bend t1)
# Index 8   : A_2 (bend t2)
# Index 9   : A_3 (gripper 1)
# Index 10  : A_4 (gripper 2)
IDX_A1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "A_1")
IDX_A2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "A_2")
IDX_A3 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "A_3")
IDX_A4 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "A_4")
print(f"[INFO] TDCR actuator indices: A_1={IDX_A1}, A_2={IDX_A2}, A_3={IDX_A3}, A_4={IDX_A4}")

# ── DLS Velocity Planner for panda arm ───────────────────────────────────────
EE_SITE_NAME = "attachment_site"
ee_site_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME)

if ee_site_id == -1:
    print(f"[ERROR] Site '{EE_SITE_NAME}' not found. Available sites:")
    for i in range(model.nsite):
        print(f"  {i}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)}")
    sys.exit(1)

# gripper_cfg not needed for single panda (no gripper actuators in panda set)
planner = DLSVelocityPlanner(
    model=model,
    data=data,
    kd=5.0,
    site_name=EE_SITE_NAME,
    damping=1e-2,
    gripper_cfg=[],          # panda has no finger actuators in this model
    for_multi=False,
    actuator_mode="position",
)

# Initial EE position → becomes starting target
mujoco.mj_forward(model, data)
ee_pos0  = data.site_xpos[planner.site_id].copy()
ee_R0    = data.site_xmat[planner.site_id].reshape(3, 3)
print(f"[INFO] Initial EE pos: {ee_pos0}")
print(f"[INFO] Using site '{EE_SITE_NAME}' (id={ee_site_id})")

# Shared mutable target (updated by keyboard thread)
target_pos  = ee_pos0.copy()
target_quat = None          # None → align local Z to world -Z

# ── Shared keyboard state ─────────────────────────────────────────────────────
_lock       = threading.Lock()
key_state   = {
    'k': False, 'l': False,
    'z': False, 'c': False, 'x': False,
    'w': False, 's': False, 'r': False,
}
pressed_once = {'x': False, 'r': False}

# Arrow keys need special handling
arrow_state = {
    'up': False, 'down': False, 'left': False, 'right': False
}

def on_press(key):
    with _lock:
        try:
            ch = key.char.lower() if key.char else None
            if ch and ch in key_state:
                key_state[ch] = True
        except AttributeError:
            # Special keys
            if key == keyboard.Key.up:    arrow_state['up']    = True
            if key == keyboard.Key.down:  arrow_state['down']  = True
            if key == keyboard.Key.left:  arrow_state['left']  = True
            if key == keyboard.Key.right: arrow_state['right'] = True

def on_release(key):
    with _lock:
        try:
            ch = key.char.lower() if key.char else None
            if ch and ch in key_state:
                key_state[ch] = False
        except AttributeError:
            if key == keyboard.Key.up:    arrow_state['up']    = False
            if key == keyboard.Key.down:  arrow_state['down']  = False
            if key == keyboard.Key.left:  arrow_state['left']  = False
            if key == keyboard.Key.right: arrow_state['right'] = False
        if key == keyboard.Key.esc:
            return False  # stop listener

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

# ── TDCR state ────────────────────────────────────────────────────────────────
tdcr_act    = [0.0, 0.0, 0.0, 0.0]   # [A_1, A_2, A_3, A_4]
grasp_auto  = False
grasp_level = 0.0

BEND_RATE   = 0.05    # activation change per step for bend
GRIP_RATE   = 0.002   # activation change per step for gripper

# ── Main loop ─────────────────────────────────────────────────────────────────
v = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=True)
print("[INFO] Viewer launched. Press ESC in terminal to quit.")
print()
print("  PANDA IK controls:")
print("    Arrow UP/DOWN     target Y")
print("    Arrow LEFT/RIGHT  target X")
print("    W / S             target Z")
print("    R                 reset target to initial EE pos")
print()
print("  TDCR controls:")
print("    K  bend (t1)    L  bend (t2)")
print("    Z  close grip   C  open grip")
print("    X  auto-grasp pulse")

while v.is_running():
    with v.lock():

        # ── 1. Update target from keyboard ───────────────────────────────────
        with _lock:
            ks = dict(key_state)
            ar = dict(arrow_state)

        if ar['up']:    target_pos[1] += TARGET_STEP
        if ar['down']:  target_pos[1] -= TARGET_STEP
        if ar['left']:  target_pos[0] -= TARGET_STEP
        if ar['right']: target_pos[0] += TARGET_STEP
        if ks['w']:     target_pos[2] += TARGET_STEP
        if ks['s']:     target_pos[2] -= TARGET_STEP
        if ks['r']:
            target_pos[:] = ee_pos0
            print(f"[INFO] Target reset to {ee_pos0}")

        # ── 2. Panda DLS IK ──────────────────────────────────────────────────
        if data.time > CONTROL_START_TIME:
            ee_pos = data.site_xpos[planner.site_id].copy()
            ee_R   = data.site_xmat[planner.site_id].reshape(3, 3)

            # Position error → desired linear velocity
            pos_err = target_pos - ee_pos
            v_cmd   = POS_GAIN * pos_err

            # Orientation error → desired angular velocity
            if target_quat is None:
                cur_z   = ee_R[:, 2]
                ori_err = np.cross(cur_z, np.array([0.0, 0.0, -1.0]))
            else:
                q_t = np.array(target_quat, float)
                q_t /= np.linalg.norm(q_t)
                q_wxyz = np.zeros(4)
                mujoco.mju_mat2Quat(q_wxyz, data.site_xmat[planner.site_id])
                q_c = quaternion_math.wxyz_to_xyzw(q_wxyz)
                if np.dot(q_t, q_c) < 0.0:
                    q_t = -q_t
                ori_err = quaternion_math.quat_log_error(q_t, q_c)

            w_cmd = ORI_GAIN * ori_err

            # Speed caps
            lin_spd = np.linalg.norm(v_cmd)
            if lin_spd > MAX_CARTESIAN_SPEED and lin_spd > 1e-9:
                v_cmd *= MAX_CARTESIAN_SPEED / lin_spd
            ang_spd = np.linalg.norm(w_cmd)
            if ang_spd > MAX_ANGULAR_SPEED and ang_spd > 1e-9:
                w_cmd *= MAX_ANGULAR_SPEED / ang_spd

            # Send twist to DLS planner (updates ctrl[0:7])
            planner.track_twist(v_cmd, w_cart=w_cmd)

        # ── 3. TDCR teleoperation ────────────────────────────────────────────
        # Bend actuators (A_1 / A_2)
        if ks['k']:
            tdcr_act[0] = min(tdcr_act[0] + BEND_RATE, 1.0)
        else:
            tdcr_act[0] = max(tdcr_act[0] - BEND_RATE, 0.0)

        if ks['l']:
            tdcr_act[1] = min(tdcr_act[1] + BEND_RATE, 1.0)
        else:
            tdcr_act[1] = max(tdcr_act[1] - BEND_RATE, 0.0)

        # Auto-grasp pulse (X key)
        if ks['x'] and not grasp_auto:
            grasp_auto  = True
            grasp_level = 1.0
            with _lock:
                key_state['x'] = False  # consume

        if grasp_auto:
            grasp_level = max(grasp_level - 0.01, 0.0)
            tdcr_act[2] = grasp_level
            tdcr_act[3] = grasp_level
            if grasp_level == 0.0:
                grasp_auto = False
        else:
            # Manual grip
            if ks['z']:
                tdcr_act[2] = min(tdcr_act[2] + GRIP_RATE, 1.0)
                tdcr_act[3] = min(tdcr_act[3] + GRIP_RATE, 1.0)
            elif ks['c']:
                tdcr_act[2] = max(tdcr_act[2] - GRIP_RATE, 0.0)
                tdcr_act[3] = max(tdcr_act[3] - GRIP_RATE, 0.0)

        # Write TDCR activations to ctrl (indices IDX_A1..IDX_A4)
        data.ctrl[IDX_A1] = tdcr_act[0]
        data.ctrl[IDX_A2] = tdcr_act[1]
        data.ctrl[IDX_A3] = tdcr_act[2]
        data.ctrl[IDX_A4] = tdcr_act[3]

        # ── 4. Step simulation ───────────────────────────────────────────────
        mujoco.mj_step(model, data)

    v.sync()

    # Optional: print status at reduced rate
    # (comment out if it slows things down)
    # print(f"  EE: {data.site_xpos[planner.site_id]}  "
    #       f"Target: {target_pos}  TDCR: {tdcr_act}", end='\r')

listener.stop()
print("\n[INFO] Simulation ended.")