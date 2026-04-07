"""test_basic_swing.py — Verifies the swing pipeline."""
import sys
sys.path.insert(0, ".")
import numpy as np
from envs.golf_env import GolfSwingEnv

def run_simple_swing():
    env = GolfSwingEnv(ctrl_dt=0.001, max_time=5.0,
                       enable_drag=False, enable_rolling_decel=False)
    obs = env.reset()
    print("=" * 60)
    print("GOLF SWING TEST")
    print("=" * 60)
    print(f"Hole at: {env.hole_pos}")
    print(f"Elbow: {np.degrees(obs['elbow_pos']):+.1f} deg, Wrist: {np.degrees(obs['wrist_pos']):+.1f} deg")
    print(f"Club: {obs['club_head_pos']}, Ball: {obs['ball_pos']}\n")

    contact_time = None; launch_vel = None; max_h = 0; traj = []
    for step in range(env.max_steps):
        t = step * env.ctrl_dt
        if t < 0.15: action = [-120.0, 0.0]
        elif t < 0.25: action = [-80.0, 0.0]
        else: action = [0.0, 0.0]
        obs, info = env.step(action)
        if info["ball_launched"]:
            max_h = max(max_h, obs["ball_pos"][2]); traj.append(obs["ball_pos"].copy())
        if info["contact"] and contact_time is None:
            contact_time = info["time"]
            cv = obs["club_head_vel"]
            print(f"[CONTACT] t={contact_time:.4f}s  speed={np.linalg.norm(cv):.1f} m/s  vel_x={cv[0]:+.1f}")
        if info["ball_launched"] and launch_vel is None:
            launch_vel = info["ball_launch_vel"]
            s = np.linalg.norm(launch_vel); a = np.degrees(np.arctan2(launch_vel[2], launch_vel[0]))
            print(f"[LAUNCH] {s:.1f} m/s at {a:.1f} deg {'toward hole' if launch_vel[0]>0 else 'WRONG'}")
        if info["done"]: break

    print(f"\n{'─'*60}")
    checks = {"Model loads": True, "Contact": contact_time is not None,
              "Launched": launch_vel is not None,
              "Toward hole": launch_vel is not None and launch_vel[0] > 0,
              "Has height": max_h > 0.1, "Moves fwd": obs["ball_pos"][0] > 1.0}
    ok = all(checks.values())
    for n, v in checks.items(): print(f"  [{'PASS' if v else 'FAIL'}] {n}")
    print(f"\n{'='*60}\n{'ALL SYSTEMS GO' if ok else 'ISSUES'}\n{'='*60}")
    return ok

if __name__ == "__main__": sys.exit(0 if run_simple_swing() else 1)