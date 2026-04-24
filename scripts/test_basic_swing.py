"""test_basic_swing.py — Verifies the ball launch pipeline (no arm)."""
import sys
sys.path.insert(0, ".")
import numpy as np
from envs.golf_env import GolfEnv

def run_test():
    env = GolfEnv(ctrl_dt=0.001, max_time=5.0,
                  enable_drag=False, enable_rolling_decel=False)
    env.reset()

    print("=" * 60)
    print("BASIC LAUNCH TEST")
    print("=" * 60)
    print(f"Hole at: {env.hole_pos}")
    print()

    # Launch: 20 m/s, 30 deg up, straight ahead
    speed = 20.0
    vert = np.radians(30)
    horiz = 0.0
    env.launch_ball(speed, vert, horiz)

    print(f"Launch: {speed} m/s at {np.degrees(vert):.0f} deg vert, {np.degrees(horiz):.0f} deg horiz")
    print(f"Expected vx={speed*np.cos(vert)*np.cos(horiz):.1f}, "
          f"vy={speed*np.cos(vert)*np.sin(horiz):.1f}, "
          f"vz={speed*np.sin(vert):.1f}")
    print()

    max_h = 0
    for _ in range(env.max_steps):
        info = env.step()
        max_h = max(max_h, info["ball_pos"][2])
        if info["done"]:
            break

    final_x = info["ball_pos"][0]
    final_z = info["ball_pos"][2]

    print(f"{'─'*60}")
    checks = {
        "Ball launched": env.ball_launched,
        "Has height": max_h > 1.0,
        "Moves forward": final_x > 10.0,
        "Toward +X": final_x > 0,
        "Returns to ground": final_z < 0.5,
    }
    ok = all(checks.values())
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    print(f"\n  Max height: {max_h:.1f}m")
    print(f"  Final X:    {final_x:.1f}m")

    print(f"\n{'='*60}")
    print(f"{'ALL SYSTEMS GO' if ok else 'ISSUES FOUND'}")
    print(f"{'='*60}")
    return ok

if __name__ == "__main__":
    sys.exit(0 if run_test() else 1)