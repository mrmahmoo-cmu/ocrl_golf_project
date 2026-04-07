"""
visualize_swing.py

Opens a MuJoCo viewer window and runs the full simulation visually:
  - Controlled torque swing (not just gravity)
  - Air drag on the ball
  - Rolling deceleration on terrain
  - Hole completion (ball freezes at hole)

Usage:
    py -3.13 scripts/visualize_swing.py

Controls in the viewer window:
    Space     = pause/unpause
    Backspace = reset
    Mouse     = rotate camera (left-click drag), zoom (scroll), pan (right-click drag)
    Tab       = toggle left panel
    Esc       = quit
"""

import sys
sys.path.insert(0, ".")

import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

# Import our environment's physics (drag, terrain, hole logic)
from envs.golf_env import (
    DRAG_K, ROLLING_DECEL, HOLE_RADIUS, TERRAIN_FAIRWAY,
    TerrainMap, GolfSwingEnv,
)

# ── Configuration ──────────────────────────────────────────────────
HOLE_POS = np.array([65.0, 0.0])     # where to place the hole (adjust as needed)
PLAYBACK_SPEED = 0.5                   # 1.0 = real-time, 0.5 = half speed, 2.0 = double
MAX_TIME = 15.0                        # seconds of simulation


def swing_torque(t):
    """Open-loop torque profile for the downswing."""
    if t < 0.15:
        return [-120.0, 0.0]
    elif t < 0.25:
        return [-80.0, 0.0]
    else:
        return [0.0, 0.0]


def apply_ball_forces(model, data, ball_body_id, terrain, ball_in_hole):
    """Apply air drag and rolling deceleration to the ball."""
    if ball_in_hole:
        data.xfrc_applied[ball_body_id, :] = 0.0
        return

    ball_pos = data.xpos[ball_body_id]
    ball_vel = data.cvel[ball_body_id][3:]  # linear velocity
    speed = np.linalg.norm(ball_vel)

    if speed < 1e-6:
        return

    vel_dir = ball_vel / speed

    # Air drag
    drag_force = -DRAG_K * speed ** 2 * vel_dir
    data.xfrc_applied[ball_body_id, :3] = drag_force

    # Rolling deceleration (on ground)
    GROUND_Z = 0.05
    if ball_pos[2] < GROUND_Z:
        horiz_vel = ball_vel[:2].copy()
        horiz_speed = np.linalg.norm(horiz_vel)
        if horiz_speed > 0.01:
            horiz_dir = horiz_vel / horiz_speed
            decel = terrain.deceleration(ball_pos[0], ball_pos[1])
            ball_mass = model.body_mass[ball_body_id]
            brake = -decel * ball_mass * np.append(horiz_dir, 0.0)
            data.xfrc_applied[ball_body_id, :3] += brake[:3]


def main():
    # Load model
    model_path = Path(__file__).parent.parent / "models" / "arm_club_ball.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Move hole to desired position (update the body pos in the model)
    hole_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hole")
    model.body_pos[hole_body_id][:2] = HOLE_POS

    # Get IDs
    elbow_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "elbow")
    wrist_jnt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wrist")
    elbow_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "elbow_torque")
    wrist_act = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "wrist_torque")
    ball_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    club_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "club_head")
    ball_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_joint")
    ball_qvel_start = model.jnt_dofadr[ball_jnt_id]

    # Set initial backswing position
    data.qpos[elbow_jnt] = 3 * np.pi / 4   # arm up, backswing
    data.qpos[wrist_jnt] = 0.0
    mujoco.mj_forward(model, data)

    terrain = TerrainMap()
    sim_dt = model.opt.timestep

    # State tracking
    ball_in_hole = False
    contact_detected = False
    ball_launched = False
    printed_contact = False
    printed_launch = False
    printed_hole = False
    printed_stop = False

    print("=" * 60)
    print("VISUAL SIMULATION — watch the MuJoCo viewer window")
    print("=" * 60)
    print(f"Hole at:    ({HOLE_POS[0]}, {HOLE_POS[1]})")
    print(f"Playback:   {PLAYBACK_SPEED}x real-time")
    print(f"Max time:   {MAX_TIME}s")
    print()
    print("Opening viewer... (this may take a moment)")
    print()

    # Launch the viewer (opens a window)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set camera to a nice initial view
        viewer.cam.azimuth = 90       # looking along +X (down the fairway)
        viewer.cam.elevation = -20     # slightly above
        viewer.cam.distance = 5        # zoom level
        viewer.cam.lookat[:] = [0, 0, 0.5]  # look at the tee area

        print("Viewer opened. Simulation starting...\n")

        #wall_start = time.time()
        sim_start = data.time

        input("Press Enter in the terminal to start the simulation...")
        print("Starting!\n")
        wall_start = time.time()

        while viewer.is_running() and data.time < MAX_TIME:
            step_start = time.time()

            # Get current sim time for torque profile
            t = data.time

            # Apply swing torques
            torques = swing_torque(t)
            data.ctrl[elbow_act] = torques[0]
            data.ctrl[wrist_act] = torques[1]

            # Apply ball physics (drag + rolling decel)
            apply_ball_forces(model, data, ball_body_id, terrain, ball_in_hole)

            # Check hole
            if not ball_in_hole:
                ball_pos = data.xpos[ball_body_id]
                dist = np.linalg.norm(ball_pos[:2] - HOLE_POS)
                if dist < HOLE_RADIUS and ball_pos[2] < 0.05:
                    ball_in_hole = True
                    data.qvel[ball_qvel_start:ball_qvel_start+6] = 0.0
                    data.xfrc_applied[ball_body_id, :] = 0.0

            # Freeze ball if in hole
            if ball_in_hole:
                data.qvel[ball_qvel_start:ball_qvel_start+6] = 0.0

            # Step physics
            mujoco.mj_step(model, data)

            # Check contact
            if not contact_detected:
                for i in range(data.ncon):
                    c = data.contact[i]
                    if {c.geom1, c.geom2} == {ball_geom_id, club_geom_id}:
                        contact_detected = True
                        break

            # Track ball state
            ball_vel = data.cvel[ball_body_id][3:]
            ball_speed = np.linalg.norm(ball_vel)
            ball_pos = data.xpos[ball_body_id]

            if contact_detected and not ball_launched and ball_speed > 0.5:
                ball_launched = True

            # Print events
            if contact_detected and not printed_contact:
                printed_contact = True
                club_vel = data.cvel[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "club_head_body")][3:]
                print(f"  [CONTACT]  t={data.time:.3f}s  club_speed={np.linalg.norm(club_vel):.1f} m/s")

            if ball_launched and not printed_launch:
                printed_launch = True
                angle = np.degrees(np.arctan2(ball_vel[2], ball_vel[0]))
                print(f"  [LAUNCH]   t={data.time:.3f}s  speed={ball_speed:.1f} m/s  angle={angle:.1f} deg")

            if ball_in_hole and not printed_hole:
                printed_hole = True
                print(f"  [HOLED!]   t={data.time:.2f}s  pos=({ball_pos[0]:.2f}, {ball_pos[1]:.2f})")

            if ball_launched and not ball_in_hole and ball_speed < 0.01 and not printed_stop:
                printed_stop = True
                dist = np.linalg.norm(ball_pos[:2] - HOLE_POS)
                print(f"  [STOPPED]  t={data.time:.2f}s  pos=({ball_pos[0]:.2f}, {ball_pos[1]:.2f})  dist_to_hole={dist:.2f}m")

            # Sync viewer (updates the visual display)
            viewer.sync()

            # Real-time pacing: sleep to match desired playback speed
            sim_elapsed = data.time - sim_start
            wall_target = sim_elapsed / PLAYBACK_SPEED
            wall_elapsed = time.time() - wall_start
            sleep_time = wall_target - wall_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Stop if ball is done
            if ball_in_hole:
                # Keep viewer open for a few seconds so user can see
                break
            if ball_launched and ball_speed < 0.01:
                break

    # Keep viewer open until user closes the window
        print("\nSimulation done. Close the viewer window to exit.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)

    print(f"\n{'='*60}")
    print("Simulation complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
