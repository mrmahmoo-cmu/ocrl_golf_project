"""test_phase2.py — Verifies drag, terrain, rolling decel, and hole completion."""
import sys
sys.path.insert(0, ".")
import numpy as np
from envs.golf_env import GolfSwingEnv, TerrainMap

def swing(t):
    if t < 0.15: return [-120.0, 0.0]
    elif t < 0.25: return [-80.0, 0.0]
    else: return [0.0, 0.0]

def run(env, label=""):
    obs = env.reset(); lv = None; mh = 0; tl = set(); tr = []
    for step in range(env.max_steps):
        obs, info = env.step(swing(step * env.ctrl_dt))
        if info["ball_launched"]:
            mh = max(mh, obs["ball_pos"][2]); tr.append(obs["ball_pos"].copy()); tl.add(info["terrain"])
        if info["ball_launched"] and lv is None: lv = info["ball_launch_vel"]
        if info["done"]: break
    t = np.array(tr) if tr else np.zeros((1,3))
    return {"label":label, "speed":np.linalg.norm(lv) if lv is not None else 0,
            "angle":np.degrees(np.arctan2(lv[2],lv[0])) if lv is not None else 0, "mh":mh,
            "fpos":obs["ball_pos"].copy(), "fspd":info["ball_speed"],
            "dist":info["dist_to_hole"], "stop":info["ball_stopped"],
            "hole":info["in_hole"], "xt":t[-1,0]-t[0,0] if len(t)>1 else 0,
            "terr":tl, "toward":lv is not None and lv[0]>0, "time":info["time"]}

def main():
    print("="*60); print("PHASE 2 TEST"); print("="*60)

    print("\n── Full Physics ──")
    rf = run(GolfSwingEnv(ctrl_dt=0.001, max_time=15.0), "Full")
    print(f"  Launch: {rf['speed']:.1f} m/s at {rf['angle']:.1f} deg")
    print(f"  Travel: {rf['xt']:.1f}m  Height: {rf['mh']:.1f}m  Stopped: {rf['stop']}  Terrains: {rf['terr']}")

    print("\n── No Physics ──")
    rn = run(GolfSwingEnv(ctrl_dt=0.001, max_time=15.0, enable_drag=False, enable_rolling_decel=False), "None")
    print(f"  Travel: {rn['xt']:.1f}m  Stopped: {rn['stop']}")

    print("\n── Hole Completion (hole at 65m) ──")
    rh = run(GolfSwingEnv(ctrl_dt=0.001, max_time=15.0, hole_pos=(65.0, 0.0)), "Hole")
    print(f"  In hole: {rh['hole']}  Final pos: ({rh['fpos'][0]:.2f}, {rh['fpos'][2]:.3f})  Speed: {rh['fspd']:.4f}")

    print("\n── Terrain ──")
    tm = TerrainMap()
    tests = [(0,0,"tee"),(10,0,"fairway"),(37.5,0,"sand"),(45,0,"green"),(10,7,"rough"),(60,0,"rough")]
    tok = all(tm.classify(x,y)==e for x,y,e in tests)
    for x,y,e in tests:
        a=tm.classify(x,y); print(f"  ({x},{y})→{a} [{'OK' if a==e else 'FAIL'}]")

    print(f"\n{'─'*60}\nCHECKLIST:")
    checks = {"Launch": rf["speed"]>5, "Toward hole": rf["toward"],
              "Drag works": rf["xt"]<rn["xt"]*0.8, "Ball stops": rf["stop"],
              "No-phys doesn't stop": not rn["stop"], "Terrain": tok,
              "Hole completion": rh["hole"], "Ball frozen at hole": rh["fspd"]<0.001}
    ok = all(checks.values())
    for n,v in checks.items(): print(f"  [{'PASS' if v else 'FAIL'}] {n}")
    print(f"\n{'='*60}\nPHASE 2 {'ALL SYSTEMS GO' if ok else 'ISSUES'}\n{'='*60}")
    return ok

if __name__ == "__main__": sys.exit(0 if main() else 1)