"""
visualize_result.py

Replays a saved optimization result in the MuJoCo viewer.
Uses GolfEnv directly for identical physics.

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

from envs.golf_env import GolfEnv


PLAYBACK_SPEED = 0.5


def main():
    result_path = sys.argv[1] if len(sys.argv) > 1 else "courses/course1_result.npz"

    print(f"Loading results from: {result_path}")
    data = np.load(result_path, allow_pickle=True)

    speed = float(data["speed"])
    vert_angle = float(data["vert_angle"])
    horiz_angle = float(data["horiz_angle"])
    hole_pos = data["hole_pos"]
    cost = float(data["cost"])

    wind = data["wind"].astype(np.float64) if "wind" in data else None
    model_path = Path(str(data["model_path"])) if "model_path" in data else None

    terrain_zones = []
    for row in data["terrain_zones"]:
        terrain_zones.append((float(row[0]), float(row[1]),
                              float(row[2]), float(row[3]), str(row[4])))

    print(f"  Hole at:         ({hole_pos[0]}, {hole_pos[1]})")
    print(f"  Launch speed:    {speed:.2f} m/s")
    print(f"  Vertical angle:  {np.degrees(vert_angle):.2f} deg")
    print(f"  Horizontal angle:{np.degrees(horiz_angle):.2f} deg")
    print(f"  Best cost:       {cost:.4f}m")
    if wind is not None and np.linalg.norm(wind) > 0:
        print(f"  Wind:            ({wind[0]:.1f}, {wind[1]:.1f}, {wind[2]:.1f}) m/s")
    print()

    env = GolfEnv(
        ctrl_dt=0.001, max_time=15.0,
        hole_pos=hole_pos, terrain_zones=terrain_zones,
        model_path=model_path, wind=wind,
    )
    env.reset()

    print("Opening viewer...")
    print()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        mid_x = float(hole_pos[0]) / 2
        mid_y = float(hole_pos[1]) / 2
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -30
        viewer.cam.distance = max(float(hole_pos[0]), 20) * 1.2
        viewer.cam.lookat[:] = [mid_x, mid_y, 0.5]

        input("Press Enter to launch the ball...")
        print("Launched!\n")

        env.launch_ball(speed, vert_angle, horiz_angle)
        lv = env.ball_launch_vel
        lv_speed = np.linalg.norm(lv)
        lv_vert = np.degrees(np.arctan2(lv[2], np.linalg.norm(lv[:2])))
        lv_horiz = np.degrees(np.arctan2(lv[1], lv[0]))
        print(f"  [LAUNCH]   speed={lv_speed:.1f} m/s  vert={lv_vert:.1f}°  horiz={lv_horiz:.1f}°")

        wall_start = time.time()
        sim_start = env.data.time
        printed_hole = False
        printed_stop = False

        for _ in range(env.max_steps):
            if not viewer.is_running():
                break

            info = env.step()

            if info["in_hole"] and not printed_hole:
                printed_hole = True
                bp = info["ball_pos"]
                print(f"  [HOLED!]   t={info['time']:.2f}s  pos=({bp[0]:.2f}, {bp[1]:.2f})")

            if info["ball_stopped"] and not printed_stop:
                printed_stop = True
                bp = info["ball_pos"]
                print(f"  [STOPPED]  t={info['time']:.2f}s  pos=({bp[0]:.2f}, {bp[1]:.2f})  "
                      f"dist={info['dist_to_hole']:.3f}m  terrain={info['terrain']}")

            viewer.sync()

            sim_elapsed = env.data.time - sim_start
            wall_target = sim_elapsed / PLAYBACK_SPEED
            wall_elapsed = time.time() - wall_start
            if wall_target - wall_elapsed > 0:
                time.sleep(wall_target - wall_elapsed)

            if info["done"]:
                break

        bp = info["ball_pos"]
        print(f"\n  Final: pos=({bp[0]:.2f}, {bp[1]:.2f}, {bp[2]:.2f})  "
              f"dist={info['dist_to_hole']:.3f}m  "
              f"in_hole={info['in_hole']}  terrain={info['terrain']}")

        print("\nDone. Close the viewer window to exit.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)

    print("\nViewer closed.")


if __name__ == "__main__":
    main()