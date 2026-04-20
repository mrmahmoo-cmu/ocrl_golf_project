"""
visualize_result.py

Loads a saved optimization result and replays the optimized swing
in the MuJoCo viewer using the same GolfSwingEnv class that the
optimizer uses, guaranteeing identical physics

Usage:
    py -3.13 scripts/visualize_result.py
    py -3.13 scripts/visualize_result.py courses/course2_result.npz
"""

import sys
sys.path.insert(0, ".")

import time
import numpy as np
import mujoco.viewer
from pathlib import Path

from envs.golf_env import GolfSwingEnv


PLAYBACK_SPEED = 0.5
MAX_TIME = 15.0


def get_torque_at_time(torque_profile, bin_dt, swing_duration, n_bins, t):
    if t >= swing_duration:
        return np.array([0.0, 0.0])
    bin_idx = min(int(t / bin_dt), n_bins - 1)
    return torque_profile[bin_idx]


def main():
    if len(sys.argv) > 1:
        result_path = sys.argv[1]
    else:
        result_path = "courses/course2_result.npz"

    print(f"Loading results from: {result_path}")
    data_file = np.load(result_path, allow_pickle=True)

    elbow_init = float(data_file["elbow_init"])
    torque_profile = data_file["torque_profile"]
    swing_duration = float(data_file["swing_duration"])
    n_bins = int(data_file["n_bins"])
    bin_dt = float(data_file["bin_dt"])
    hole_pos = data_file["hole_pos"]
    cost = float(data_file["cost"])

    has_aim = bool(data_file.get("has_aim", False))
    aim_angle = float(data_file["aim_angle"]) if has_aim else 0.0

    # Load wind
    if "wind" in data_file:
        wind = data_file["wind"].astype(np.float64)
    else:
        wind = None

    # Load model path
    if "model_path" in data_file:
        model_path = Path(str(data_file["model_path"]))
    else:
        model_path = None

    # Load terrain zones
    terrain_zones_raw = data_file["terrain_zones"]
    terrain_zones = []
    for row in terrain_zones_raw:
        terrain_zones.append((float(row[0]), float(row[1]),
                              float(row[2]), float(row[3]), str(row[4])))

    print(f"  Hole at:         ({hole_pos[0]}, {hole_pos[1]})")
    if has_aim:
        print(f"  Aim angle:       {np.degrees(aim_angle):.1f} deg")
    print(f"  Backswing:       {np.degrees(elbow_init):.1f} deg")
    print(f"  Best cost:       {cost:.4f}m")
    if wind is not None and np.linalg.norm(wind) > 0:
        print(f"  Wind:            ({wind[0]:.1f}, {wind[1]:.1f}, {wind[2]:.1f}) m/s")
    print(f"  Model:           {model_path}")
    print()

    # ── Create the SAME environment the optimizer used ──
    # use the GolfSwingEnv
    env = GolfSwingEnv(
        ctrl_dt=0.001,          # fine timestep for smooth visual
        max_time=MAX_TIME,
        hole_pos=hole_pos,
        terrain_zones=terrain_zones,
        enable_drag=True,
        enable_rolling_decel=True,
        model_path=model_path,
        wind=wind,
    )

    obs = env.reset(elbow_init=elbow_init, wrist_init=0.0,
                    aim_init=aim_angle if has_aim else None)

    print("Opening viewer...")
    print()

    # ── Launch viewer using the env's own model and data ──
    # The viewer references env.data directly, so when env.step()
    # updates the physics state, viewer.sync() shows the new state
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        mid_x = hole_pos[0] / 2
        mid_y = hole_pos[1] / 2
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -30
        viewer.cam.distance = max(float(hole_pos[0]), 20) * 1.2
        viewer.cam.lookat[:] = [mid_x, mid_y, 0.5]

        input("Press Enter to start the optimized swing...")
        print("Running!\n")

        wall_start = time.time()
        sim_start = env.data.time

        printed_contact = False
        printed_launch = False
        printed_hole = False
        printed_stop = False

        for step in range(env.max_steps):
            if not viewer.is_running():
                break

            t = step * env.ctrl_dt
            torque = get_torque_at_time(torque_profile, bin_dt, swing_duration, n_bins, t)
            obs, info = env.step(torque)

            # Print events
            if info["contact"] and not printed_contact:
                printed_contact = True
                club_speed = np.linalg.norm(obs["club_head_vel"])
                print(f"  [CONTACT]  t={info['time']:.3f}s  club_speed={club_speed:.1f} m/s")

            if info["ball_launched"] and not printed_launch:
                printed_launch = True
                lv = info["ball_launch_vel"]
                speed = np.linalg.norm(lv)
                vert = np.degrees(np.arctan2(lv[2], np.linalg.norm(lv[:2])))
                horiz = np.degrees(np.arctan2(lv[1], lv[0]))
                print(f"  [LAUNCH]   speed={speed:.1f} m/s  vert={vert:.1f}°  horiz={horiz:.1f}°")

            if info["in_hole"] and not printed_hole:
                printed_hole = True
                bp = obs["ball_pos"]
                print(f"  [HOLED!]   t={info['time']:.2f}s  pos=({bp[0]:.2f}, {bp[1]:.2f})")

            if info["ball_stopped"] and not printed_stop:
                printed_stop = True
                bp = obs["ball_pos"]
                print(f"  [STOPPED]  t={info['time']:.2f}s  pos=({bp[0]:.2f}, {bp[1]:.2f})  "
                      f"dist={info['dist_to_hole']:.3f}m  terrain={info['terrain']}")

            # Sync viewer
            viewer.sync()

            # Real-time pacing
            sim_elapsed = env.data.time - sim_start
            wall_target = sim_elapsed / PLAYBACK_SPEED
            wall_elapsed = time.time() - wall_start
            sleep_time = wall_target - wall_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            if info["done"]:
                break

        # Final summary
        bp = obs["ball_pos"]
        print(f"\n  Final: pos=({bp[0]:.2f}, {bp[1]:.2f}, {bp[2]:.2f})  "
              f"dist={info['dist_to_hole']:.3f}m  "
              f"in_hole={info['in_hole']}  terrain={info['terrain']}")

        # Keep viewer open
        print("\nDone. Close the viewer window to exit.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)

    print("\nViewer closed.")


if __name__ == "__main__":
    main()