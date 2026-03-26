"""
test_basic_swing.py — Verifies the swing pipeline with corrected geometry.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
from envs.golf_env import GolfSwingEnv

def run_simple_swing():
    env = GolfSwingEnv(ctrl_dt=0.001, max_time=5.0,
                       enable_drag=False, enable_rolling_decel=False)
    obs = env.reset()  # defaults: elbow=+3π/4, wrist=0

    print("=" * 60)
    print("GOLF SWING TEST (corrected geometry)")
    print("=" * 60)
    print(f"Hole at:       {env.hole_pos}")
    print(f"Initial elbow: {np.degrees(obs['elbow_pos']):+.1f} deg")
    print(f"Initial wrist: {np.degrees(obs['wrist_pos']):+.1f} deg")
    print(f"Club head:     {obs['club_head_pos']}")
    print(f"Ball:          {obs['ball_pos']}")
    print()

    contact_time = None
    launch_vel = None
    max_ball_height = 0
    trajectory = []

    for step in range(env.max_steps):
        t = step * env.ctrl_dt

        # NEGATIVE elbow torque = downswing toward +X
        if t < 0.15:
            action = [-120.0, 0.0]
        elif t < 0.25:
            action = [-80.0, 0.0]
        else:
            action = [0.0, 0.0]

        obs, info = env.step(action)

        if info["ball_launched"]:
            max_ball_height = max(max_ball_height, obs["ball_pos"][2])
            trajectory.append(obs["ball_pos"].copy())

        if info["contact"] and contact_time is None:
            contact_time = info["time"]
            club_vel = obs["club_head_vel"]
            print(f"[CONTACT] t={contact_time:.4f}s")
            print(f"  Club speed: {np.linalg.norm(club_vel):.1f} m/s")
            print(f"  Club vel_x: {club_vel[0]:+.1f} m/s {'(toward hole)' if club_vel[0] > 0 else '(WRONG DIR)'}")

        if info["ball_launched"] and launch_vel is None:
            launch_vel = info["ball_launch_vel"]
            speed = np.linalg.norm(launch_vel)
            angle = np.degrees(np.arctan2(launch_vel[2], launch_vel[0]))
            print(f"\n[LAUNCH]")
            print(f"  Speed: {speed:.1f} m/s, Angle: {angle:.1f} deg")
            print(f"  Direction: {'toward hole (+X)' if launch_vel[0] > 0 else 'AWAY from hole (-X)'}")

        if info["done"]:
            break

    print(f"\n{'─'*60}")
    checks = {
        "Model loads":          True,
        "Contact detected":     contact_time is not None,
        "Ball launched":        launch_vel is not None,
        "Ball toward hole":     launch_vel is not None and launch_vel[0] > 0,
        "Ball has height":      max_ball_height > 0.1,
        "Ball moves fwd":       obs["ball_pos"][0] > 1.0,
    }
    all_pass = True
    for name, ok in checks.items():
        status = "PASS" if ok else "FAIL"
        if not ok: all_pass = False
        print(f"  [{status}] {name}")

    print(f"\n{'='*60}")
    print("ALL SYSTEMS GO" if all_pass else "ISSUES DETECTED")
    print(f"{'='*60}")
    return all_pass

if __name__ == "__main__":
    success = run_simple_swing()
    sys.exit(0 if success else 1)
