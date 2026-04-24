"""test_phase2.py — Verifies drag, terrain, rolling decel, and hole completion."""
import sys
sys.path.insert(0, ".")
import numpy as np
from envs.golf_env import GolfEnv, TerrainMap

def run_sim(env, speed, vert_deg, horiz_deg=0):
    env.reset()
    env.launch_ball(speed, np.radians(vert_deg), np.radians(horiz_deg))
    max_h = 0
    terrains = set()
    for _ in range(env.max_steps):
        info = env.step()
        max_h = max(max_h, info["ball_pos"][2])
        terrains.add(info["terrain"])
        if info["done"]:
            break
    return {
        "final_x": info["ball_pos"][0],
        "max_h": max_h,
        "stopped": info["ball_stopped"],
        "in_hole": info["in_hole"],
        "speed": info["ball_speed"],
        "dist": info["dist_to_hole"],
        "terrains": terrains,
        "time": info["time"],
    }

def main():
    print("=" * 60)
    print("PHASE 2 TEST")
    print("=" * 60)

    print("\n── Full Physics (drag + rolling) ──")
    rf = run_sim(GolfEnv(ctrl_dt=0.001, max_time=15.0), 25, 30)
    print(f"  Travel: {rf['final_x']:.1f}m  Height: {rf['max_h']:.1f}m  Stopped: {rf['stopped']}  Terrains: {rf['terrains']}")

    print("\n── No Physics ──")
    rn = run_sim(GolfEnv(ctrl_dt=0.001, max_time=15.0, enable_drag=False, enable_rolling_decel=False), 25, 30)
    print(f"  Travel: {rn['final_x']:.1f}m  Stopped: {rn['stopped']}")

    print("\n── Hole Completion (hole at 65m) ──")
    rh = run_sim(GolfEnv(ctrl_dt=0.001, max_time=15.0, hole_pos=(65.0, 0.0)), 25, 30)
    print(f"  In hole: {rh['in_hole']}  Final X: {rh['final_x']:.2f}  Speed: {rh['speed']:.4f}")

    print("\n── Terrain Classification ──")
    tm = TerrainMap()
    tests = [(0,0,"tee"), (10,0,"fairway"), (37.5,0,"sand"), (45,0,"green"), (10,7,"rough"), (60,0,"rough")]
    tok = all(tm.classify(x,y)==e for x,y,e in tests)
    for x,y,e in tests:
        a = tm.classify(x,y)
        print(f"  ({x},{y})→{a} [{'OK' if a==e else 'FAIL'}]")

    print(f"\n{'─'*60}\nCHECKLIST:")
    checks = {
        "Ball launches": rf["final_x"] > 5,
        "Drag reduces distance": rf["final_x"] < rn["final_x"] * 0.8,
        "Ball stops (with physics)": rf["stopped"],
        "Ball doesn't stop (no physics)": not rn["stopped"],
        "Terrain classification": tok,
        "Hole completion": rh["in_hole"],
        "Ball frozen at hole": rh["speed"] < 0.001,
    }
    ok = all(checks.values())
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    print(f"\n{'='*60}")
    print(f"PHASE 2 {'ALL SYSTEMS GO' if ok else 'ISSUES FOUND'}")
    print(f"{'='*60}")
    return ok

if __name__ == "__main__":
    sys.exit(0 if main() else 1)