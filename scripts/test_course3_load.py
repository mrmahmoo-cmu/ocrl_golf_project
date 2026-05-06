"""
Quick Course 3 load/smoke test.

This checks:
1. Course 3 config imports correctly.
2. MuJoCo model loads through GolfEnv.
3. A basic direct-launch shot runs without crashing.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from courses.course3 import HOLE_POS, MODEL_PATH, TERRAIN_ZONES, WIND, AIM_ANGLE_TO_HOLE
from envs.golf_env import GolfEnv


def main():
    env = GolfEnv(
        model_path=MODEL_PATH,
        hole_pos=HOLE_POS,
        terrain_zones=TERRAIN_ZONES,
        wind=WIND,
        max_time=10.0,
    )

    env.reset()

    # Basic test shot, not optimized.
    speed = 35.0
    vert_angle = 0.25
    horiz_angle = AIM_ANGLE_TO_HOLE

    env.launch_ball(speed, vert_angle, horiz_angle)

    info = None
    while True:
        info = env.step()
        if info["done"]:
            break

    print("Course 3 smoke test completed.")
    print(f"Final ball position: {info['ball_pos']}")
    print(f"Distance to hole: {info['dist_to_hole']:.3f} m")
    print(f"Final terrain: {info['terrain']}")
    print(f"Landed terrain: {info['landed_terrain']}")
    print(f"In hole: {info['in_hole']}")


if __name__ == "__main__":
    main()