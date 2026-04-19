"""
optimize_course1.py

CMA-ES trajectory optimization for Course 1: Gentle Dogleg.
Uses restarts: if CMA-ES stalls, it restarts with smaller sigma
around the best solution found so far.

Usage:
    py -3.13 scripts/optimize_course1.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import time
import cma
from pathlib import Path

from envs.golf_env import GolfSwingEnv
from courses.course1 import (
    HOLE_POS, TERRAIN_ZONES, MODEL_PATH,
    STRAIGHT_LINE_DIST, AIM_ANGLE_TO_HOLE,
)

# ── Optimization configuration ─────────────────────────────────────
SWING_DURATION = 0.35
N_BINS = 7
BIN_DT = SWING_DURATION / N_BINS

CTRL_DT = 0.005
MAX_TIME = 10.0

BOUNDS_LOWER = [0.0, np.pi/4] + [-150.0, -60.0] * N_BINS
BOUNDS_UPPER = [np.radians(30), np.pi] + [150.0, 60.0] * N_BINS
N_VARS = 2 + N_BINS * 2

# CMA-ES settings
POP_SIZE = 32
MAX_EVALS_PER_RUN = 2000
MAX_RESTARTS = 3            # restart up to 3 times if not hole-in-one

# Penalties
NO_CONTACT_PENALTY = 1000.0
WRONG_DIRECTION_PENALTY = 500.0
TIMEOUT_PENALTY = 200.0
SAND_PENALTY = 10.0
ROUGH_PENALTY = 25.0


def decode_params(x):
    x_clamped = np.clip(x, BOUNDS_LOWER, BOUNDS_UPPER)
    return x_clamped[0], x_clamped[1], x_clamped[2:].reshape(N_BINS, 2)


def get_torque_at_time(torque_profile, t):
    if t >= SWING_DURATION:
        return np.array([0.0, 0.0])
    return torque_profile[min(int(t / BIN_DT), N_BINS - 1)]


def simulate(aim_angle, elbow_init, torque_profile):
    env = GolfSwingEnv(
        ctrl_dt=CTRL_DT, max_time=MAX_TIME, hole_pos=HOLE_POS,
        terrain_zones=TERRAIN_ZONES, model_path=MODEL_PATH,
    )
    obs = env.reset(elbow_init=elbow_init, wrist_init=0.0, aim_init=aim_angle)
    for step in range(env.max_steps):
        obs, info = env.step(get_torque_at_time(torque_profile, step * env.ctrl_dt))
        if info["done"]:
            break
    return info["dist_to_hole"], {
        "contact": env.contact_detected, "launched": env.ball_launched,
        "in_hole": info["in_hole"], "ball_stopped": info["ball_stopped"],
        "final_pos": obs["ball_pos"].copy(), "launch_vel": env.ball_launch_vel,
        "terrain": info["terrain"],
    }


def cost_function(x):
    aim, elbow, torques = decode_params(x)
    dist, info = simulate(aim, elbow, torques)
    if not info["contact"]:
        return NO_CONTACT_PENALTY
    if info["launch_vel"] is not None and info["launch_vel"][0] < 0:
        return WRONG_DIRECTION_PENALTY
    if not info["ball_stopped"] and not info["in_hole"]:
        return TIMEOUT_PENALTY + dist
    if info["in_hole"]:
        # Penalize if the ball passed through rough/sand on the way
        if info.get("landed_terrain") == "rough":
            return 50.0  # not zero — better than missing, worse than a clean shot
        if info.get("landed_terrain") == "sand":
            return 30.0
        return 0.0
    cost = dist
    if info["terrain"] == "sand": cost += SAND_PENALTY
    if info["terrain"] == "rough": cost += ROUGH_PENALTY
    return cost


def run_cmaes(x0, sigma0, seed, max_evals, label=""):
    """Run one CMA-ES optimization. Returns (best_x, best_cost, history, eval_count)."""
    best_cost = float("inf")
    best_x = None
    gen_count = 0
    eval_count = 0
    start_time = time.time()
    history = []

    opts = {
        'popsize': POP_SIZE,
        'maxfevals': max_evals,
        'bounds': [BOUNDS_LOWER, BOUNDS_UPPER],
        'tolfun': 1e-5,
        'verb_disp': 0,
        'verb_log': 0,
        'seed': seed,
    }

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    while not es.stop():
        candidates = es.ask()
        costs = []
        for c in candidates:
            cost = cost_function(c)
            costs.append(cost)
            eval_count += 1
            if cost < best_cost:
                best_cost = cost
                best_x = c.copy()

        es.tell(candidates, costs)
        gen_count += 1
        elapsed = time.time() - start_time
        history.append(best_cost)

        status = ""
        if best_cost == 0.0: status = " *** HOLE IN ONE ***"
        elif best_cost < 0.5: status = " (almost there!)"
        elif best_cost < 1.0: status = " (very close!)"
        elif best_cost < 5.0: status = " (approaching green)"

        print(f"  {label}Gen {gen_count:3d} | evals: {eval_count:5d} | "
              f"best: {best_cost:8.3f}m | elapsed: {elapsed:6.1f}s{status}")

        if best_cost == 0.0:
            break

    return best_x, best_cost, history, eval_count


def main():
    print("=" * 60)
    print("COURSE 1: Gentle Dogleg — CMA-ES with Restarts")
    print("=" * 60)
    print(f"Hole:            ({HOLE_POS[0]}, {HOLE_POS[1]})")
    print(f"Distance:        {STRAIGHT_LINE_DIST:.1f}m")
    print(f"Ideal aim:       {np.degrees(AIM_ANGLE_TO_HOLE):.1f} deg")
    print(f"Variables:        {N_VARS}")
    print(f"Population:       {POP_SIZE}")
    print(f"Max evals/run:    {MAX_EVALS_PER_RUN}")
    print(f"Max restarts:     {MAX_RESTARTS}")
    print()

    # Initial guess: aim toward hole, strong backswing, aggressive downswing
    x0 = np.zeros(N_VARS)
    x0[0] = AIM_ANGLE_TO_HOLE
    x0[1] = 3 * np.pi / 4
    for i in range(N_BINS):
        x0[2 + i*2] = -80.0
        x0[2 + i*2 + 1] = 0.0

    sigma0 = 0.3 * np.mean(np.array(BOUNDS_UPPER) - np.array(BOUNDS_LOWER))

    global_best_cost = float("inf")
    global_best_x = None
    total_evals = 0
    all_history = []
    t_total_start = time.time()

    for restart in range(MAX_RESTARTS + 1):
        if restart == 0:
            print("── Initial run ──\n")
            current_x0 = x0.copy()
            current_sigma = sigma0
            seed = 42
        else:
            print(f"\n── Restart {restart} (refining around best: {global_best_cost:.3f}m) ──\n")
            # Restart from best found so far with smaller sigma
            current_x0 = global_best_x.copy()
            current_sigma = sigma0 * (0.3 ** restart)  # shrink sigma each restart
            seed = 42 + restart * 17

        best_x, best_cost, history, evals = run_cmaes(
            current_x0, current_sigma, seed, MAX_EVALS_PER_RUN,
            label=f"[R{restart}] " if restart > 0 else "",
        )

        total_evals += evals
        all_history.extend(history)

        if best_cost < global_best_cost:
            global_best_cost = best_cost
            global_best_x = best_x.copy()

        if global_best_cost == 0.0:
            break

    t_total = time.time() - t_total_start

    # Decode best
    best_aim, best_elbow, best_torques = decode_params(global_best_x)

    # Final simulation at fine timestep
    print(f"\n{'─'*60}")
    print("Running final simulation...\n")

    env_final = GolfSwingEnv(
        ctrl_dt=0.001, max_time=15.0, hole_pos=HOLE_POS,
        terrain_zones=TERRAIN_ZONES, model_path=MODEL_PATH,
    )
    obs = env_final.reset(elbow_init=best_elbow, wrist_init=0.0, aim_init=best_aim)

    trajectory = []
    for step in range(env_final.max_steps):
        t = step * env_final.ctrl_dt
        obs, info = env_final.step(get_torque_at_time(best_torques, t))
        if info["ball_launched"]:
            trajectory.append(obs["ball_pos"].copy())
        if info["done"]:
            break

    # Report
    print("RESULT:")
    print(f"  Method:          CMA-ES with restarts")
    print(f"  Total time:      {t_total:.1f}s")
    print(f"  Total evals:     {total_evals}")
    print(f"  Best cost:       {global_best_cost:.4f}")
    print()
    print(f"  Aim angle:       {np.degrees(best_aim):.1f} deg")
    print(f"  Backswing angle: {np.degrees(best_elbow):.1f} deg")
    print(f"  Torque profile:")
    for i in range(N_BINS):
        ts, te = i * BIN_DT, (i + 1) * BIN_DT
        print(f"    [{ts:.3f}-{te:.3f}s]  "
              f"elbow={best_torques[i,0]:+7.1f} Nm  wrist={best_torques[i,1]:+6.1f} Nm")
    print()
    print(f"  Ball final:      ({obs['ball_pos'][0]:.2f}, {obs['ball_pos'][1]:.2f}, {obs['ball_pos'][2]:.2f})")
    print(f"  Dist to hole:    {info['dist_to_hole']:.4f} m")
    print(f"  In hole:         {info['in_hole']}")
    print(f"  Terrain:         {info['terrain']}")

    if env_final.ball_launch_vel is not None:
        lv = env_final.ball_launch_vel
        speed = np.linalg.norm(lv)
        vert = np.degrees(np.arctan2(lv[2], np.linalg.norm(lv[:2])))
        horiz = np.degrees(np.arctan2(lv[1], lv[0]))
        print(f"  Launch speed:    {speed:.1f} m/s")
        print(f"  Vertical angle:  {vert:.1f} deg")
        print(f"  Horizontal aim:  {horiz:.1f} deg")

    # Save
    save_path = Path("courses/course1_result.npz")
    np.savez(
        save_path,
        aim_angle=best_aim, elbow_init=best_elbow,
        torque_profile=best_torques,
        swing_duration=SWING_DURATION, n_bins=N_BINS, bin_dt=BIN_DT,
        hole_pos=HOLE_POS,
        terrain_zones=np.array(TERRAIN_ZONES, dtype=object),
        cost=global_best_cost,
        trajectory=np.array(trajectory) if trajectory else np.zeros((1, 3)),
        cost_history=np.array(all_history),
        model_path=str(MODEL_PATH), has_aim=True,
    )
    print(f"\n  Saved to: {save_path}")

    print(f"\n{'='*60}")
    if info["in_hole"]:
        print("HOLE IN ONE!")
    else:
        print(f"Best distance: {info['dist_to_hole']:.3f}m from hole")
        print(f"Landed on: {info['terrain']}")
    print(f"{'='*60}")
    print(f"\nTo visualize: py -3.13 scripts/visualize_result.py")


if __name__ == "__main__":
    main()