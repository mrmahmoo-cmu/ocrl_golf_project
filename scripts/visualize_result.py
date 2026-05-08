"""
visualize_result.py

Replays saved optimization results in the MuJoCo viewer.
Handles both single-shot (Courses 1 & 2) and multi-stroke (Course 3) formats.
Uses GolfEnv directly for identical physics.
"""

import sys
sys.path.insert(0, ".")

import time
import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path

from envs.golf_env import GolfEnv


PLAYBACK_SPEED = 0.5


def visualize_single_shot(env, speed, vert_angle, horiz_angle, hole_pos):
    """Visualize a single-shot result (Courses 1 & 2)."""
    env.reset()

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
        print(f"  [LAUNCH]   speed={np.linalg.norm(lv):.1f} m/s  "
              f"vert={np.degrees(np.arctan2(lv[2], np.linalg.norm(lv[:2]))):.1f}°  "
              f"horiz={np.degrees(np.arctan2(lv[1], lv[0])):.1f}°")

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


def visualize_multi_stroke(env, strokes, hole_pos):
    """Visualize a multi-stroke result (Course 3)."""
    env.reset()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        mid_x = float(hole_pos[0]) / 2
        mid_y = float(hole_pos[1]) / 2
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -30
        viewer.cam.distance = max(float(hole_pos[0]), 20) * 1.2
        viewer.cam.lookat[:] = [mid_x, mid_y, 0.5]

        for i, stroke in enumerate(strokes):
            stroke_num = int(stroke[0])
            speed = stroke[1]
            vert_angle = stroke[2]
            horiz_angle = stroke[3]
            start_x, start_y, start_z = stroke[4], stroke[5], stroke[6]

            # Position ball at stroke start
            qp = env.ball_qpos_start
            env.data.qpos[qp]     = start_x
            env.data.qpos[qp + 1] = start_y
            env.data.qpos[qp + 2] = start_z
            # Zero out velocity and rotation
            qv = env.ball_qvel_start
            env.data.qvel[qv:qv+6] = 0.0
            env.data.qpos[qp+3:qp+7] = [1, 0, 0, 0]  # reset quaternion
            mujoco.mj_forward(env.model, env.data)

            # Reset env state for new stroke
            env.ball_launched = False
            env.ball_launch_vel = None
            env.ball_in_hole = False
            env.ball_landed = False
            env.ball_landed_terrain = None
            env.step_count = 0

            input(f"\nPress Enter for stroke {stroke_num} "
                  f"({speed:.1f} m/s, {np.degrees(vert_angle):.1f}° up, "
                  f"{np.degrees(horiz_angle):.1f}° aim)...")

            env.launch_ball(speed, vert_angle, horiz_angle)
            lv = env.ball_launch_vel
            print(f"  [STROKE {stroke_num}] speed={np.linalg.norm(lv):.1f} m/s  "
                  f"from ({start_x:.1f}, {start_y:.1f})")

            wall_start = time.time()
            sim_start = env.data.time
            printed_hole = False
            printed_stop = False

            for _ in range(env.max_steps):
                if not viewer.is_running():
                    return
                info = env.step()

                if info["in_hole"] and not printed_hole:
                    printed_hole = True
                    bp = info["ball_pos"]
                    print(f"  [HOLED!]   t={info['time']:.2f}s  pos=({bp[0]:.2f}, {bp[1]:.2f})")

                if info["ball_stopped"] and not printed_stop:
                    printed_stop = True
                    bp = info["ball_pos"]
                    print(f"  [STOPPED]  pos=({bp[0]:.2f}, {bp[1]:.2f})  "
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

        print("\nAll strokes complete. Close the viewer window to exit.")
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)

    print("\nViewer closed.")


def main():
    if len(sys.argv) > 1:
        result_path = sys.argv[1]
    else:
        print("Which course to visualize?")
        print("  1 — Course 1: Straight Shot")
        print("  2 — Course 2: Gentle Dogleg")
        print("  3 — Course 3: White Dogwood")
        choice = input("\nEnter course number (1/2/3): ").strip()
        result_path = f"courses/course{choice}_result.npz"

    print(f"\nLoading results from: {result_path}")
    data = np.load(result_path, allow_pickle=True)

    hole_pos = data["hole_pos"]
    wind = data["wind"].astype(np.float64) if "wind" in data else None
    model_path = Path(str(data["model_path"])) if "model_path" in data else None

    terrain_zones = []
    for row in data["terrain_zones"]:
        terrain_zones.append((float(row[0]), float(row[1]),
                              float(row[2]), float(row[3]), str(row[4])))

    # Detect format: multi-stroke has "strokes" key, single-shot has "speed"
    is_multi_stroke = "strokes" in data

    if is_multi_stroke:
        strokes = data["strokes"]
        total = int(data["total_strokes"])
        holed = bool(data["holed"])
        print(f"  Course type:     Multi-stroke ({total} strokes, holed={holed})")
        print(f"  Hole at:         ({hole_pos[0]}, {hole_pos[1]})")
        if wind is not None and np.linalg.norm(wind) > 0:
            print(f"  Wind:            ({wind[0]:.1f}, {wind[1]:.1f}, {wind[2]:.1f}) m/s")
        print()
        for s in strokes:
            print(f"  Stroke {int(s[0])}: {s[1]:.1f} m/s, {np.degrees(s[2]):.1f}° up, "
                  f"{np.degrees(s[3]):.1f}° aim  "
                  f"from ({s[4]:.1f}, {s[5]:.1f})")
    else:
        speed = float(data["speed"])
        vert_angle = float(data["vert_angle"])
        horiz_angle = float(data["horiz_angle"])
        cost = float(data["cost"])
        print(f"  Hole at:         ({hole_pos[0]}, {hole_pos[1]})")
        print(f"  Launch speed:    {speed:.2f} m/s")
        print(f"  Vertical angle:  {np.degrees(vert_angle):.2f} deg")
        print(f"  Horizontal angle:{np.degrees(horiz_angle):.2f} deg")
        print(f"  Best cost:       {cost:.4f}m")
        if wind is not None and np.linalg.norm(wind) > 0:
            print(f"  Wind:            ({wind[0]:.1f}, {wind[1]:.1f}, {wind[2]:.1f}) m/s")

    print()

    env = GolfEnv(
        ctrl_dt=0.002, max_time=15.0,
        hole_pos=hole_pos, terrain_zones=terrain_zones,
        model_path=model_path, wind=wind,
    )

    if is_multi_stroke:
        visualize_multi_stroke(env, strokes, hole_pos)
    else:
        visualize_single_shot(env, speed, vert_angle, horiz_angle, hole_pos)


if __name__ == "__main__":
    main()