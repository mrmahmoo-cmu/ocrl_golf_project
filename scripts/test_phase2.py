"""
test_phase2.py — Verifies air drag, terrain, rolling deceleration.
"""
import sys
sys.path.insert(0, ".")
import numpy as np
from envs.golf_env import GolfSwingEnv, TerrainMap

def swing_torque(t):
    """Downswing: negative elbow torque."""
    if t < 0.15:
        return [-120.0, 0.0]
    elif t < 0.25:
        return [-80.0, 0.0]
    else:
        return [0.0, 0.0]

def run_episode(env, label=""):
    obs = env.reset()
    launch_vel = None
    max_height = 0
    terrain_log = set()
    trajectory = []

    for step in range(env.max_steps):
        t = step * env.ctrl_dt
        obs, info = env.step(swing_torque(t))

        if info["ball_launched"]:
            max_height = max(max_height, obs["ball_pos"][2])
            trajectory.append(obs["ball_pos"].copy())
            terrain_log.add(info["terrain"])
        if info["ball_launched"] and launch_vel is None:
            launch_vel = info["ball_launch_vel"]
        if info["done"]:
            break

    traj = np.array(trajectory) if trajectory else np.zeros((1, 3))
    return {
        "label": label,
        "launch_speed": np.linalg.norm(launch_vel) if launch_vel is not None else 0,
        "launch_angle": np.degrees(np.arctan2(launch_vel[2], launch_vel[0])) if launch_vel is not None else 0,
        "max_height": max_height,
        "final_pos": obs["ball_pos"].copy(),
        "final_speed": info["ball_speed"],
        "dist_to_hole": info["dist_to_hole"],
        "ball_stopped": info["ball_stopped"],
        "in_hole": info["in_hole"],
        "x_travel": traj[-1, 0] - traj[0, 0] if len(traj) > 1 else 0,
        "terrains_visited": terrain_log,
        "sim_time": info["time"],
        "ball_toward_hole": launch_vel is not None and launch_vel[0] > 0,
    }

def print_results(r):
    print(f"\n  [{r['label']}]")
    print(f"  Launch:       {r['launch_speed']:.1f} m/s at {r['launch_angle']:.1f} deg")
    print(f"  Max height:   {r['max_height']:.2f} m")
    print(f"  X travel:     {r['x_travel']:.2f} m")
    print(f"  Final pos:    ({r['final_pos'][0]:.2f}, {r['final_pos'][1]:.2f}, {r['final_pos'][2]:.3f})")
    print(f"  Final speed:  {r['final_speed']:.4f} m/s")
    print(f"  Dist to hole: {r['dist_to_hole']:.2f} m")
    print(f"  Stopped:      {r['ball_stopped']}")
    print(f"  Terrains:     {r['terrains_visited']}")

def main():
    print("=" * 60)
    print("PHASE 2 TEST — Corrected Geometry + Physics")
    print("=" * 60)

    print("\n── Test 1: Full Physics ──")
    r_full = run_episode(
        GolfSwingEnv(ctrl_dt=0.001, max_time=15.0, enable_drag=True, enable_rolling_decel=True),
        "Full Physics")
    print_results(r_full)

    print("\n── Test 2: No Physics (baseline) ──")
    r_none = run_episode(
        GolfSwingEnv(ctrl_dt=0.001, max_time=15.0, enable_drag=False, enable_rolling_decel=False),
        "No Physics")
    print_results(r_none)

    print("\n── Test 3: Terrain Classification ──")
    tm = TerrainMap()
    tests = [(0,0,"tee"),(10,0,"fairway"),(37.5,0,"sand"),(45,0,"green"),(10,7,"rough"),(60,0,"rough")]
    terrain_ok = True
    for x, y, exp in tests:
        act = tm.classify(x, y)
        ok = act == exp
        if not ok: terrain_ok = False
        print(f"  ({x:5.1f},{y:4.1f}) → {act:10s} expected {exp:10s} [{'OK' if ok else 'FAIL'}]")

    print(f"\n{'─'*60}")
    print("CHECKLIST:")
    checks = {
        "Contact + launch":       r_full["launch_speed"] > 5,
        "Ball toward hole (+X)":  r_full["ball_toward_hole"],
        "Drag reduces distance":  r_full["x_travel"] < r_none["x_travel"] * 0.8,
        "Ball stops (full)":      r_full["ball_stopped"],
        "Ball doesn't stop (no)": not r_none["ball_stopped"],
        "Terrain classification": terrain_ok,
        "Max height realistic":   0.1 < r_full["max_height"] < 50,
    }
    all_pass = True
    for name, ok in checks.items():
        if not ok: all_pass = False
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    print(f"\n{'='*60}")
    print("PHASE 2 " + ("ALL SYSTEMS GO" if all_pass else "ISSUES DETECTED"))
    print(f"{'='*60}")
    print(f"\n  {'Config':<20s} {'X travel':>10s} {'Stopped':>10s}")
    print(f"  {'─'*20} {'─'*10} {'─'*10}")
    for r in [r_none, r_full]:
        print(f"  {r['label']:<20s} {r['x_travel']:>9.1f}m {'Yes' if r['ball_stopped'] else 'No':>10s}")
    return all_pass

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
