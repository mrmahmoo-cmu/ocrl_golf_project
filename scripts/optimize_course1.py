"""
optimize_course1.py

CMA-ES optimization for Course 1: Straight Shot.
3 decision variables: launch speed, vertical angle, horizontal angle.

Usage:
    py -3.13 scripts/optimize_course1.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import time
import cma
from pathlib import Path

from envs.golf_env import GolfEnv
from courses.course1 import (
    HOLE_POS, TERRAIN_ZONES, MODEL_PATH,
    STRAIGHT_LINE_DIST, AIM_ANGLE_TO_HOLE,
)

# ── Bounds ──────────────────────────────────────────────────────────
# [speed (m/s), vert_angle (rad), horiz_angle (rad)]
BOUNDS_LOWER = [5.0,  np.radians(5),  np.radians(-15)]
BOUNDS_UPPER = [30.0, np.radians(60), np.radians(15)]
N_VARS = 3

POP_SIZE = 16
MAX_EVALS_PER_RUN = 1000
MAX_RESTARTS = 2

NO_CONTACT_PENALTY = 1000.0
TIMEOUT_PENALTY = 200.0
SAND_PENALTY = 10.0
ROUGH_PENALTY = 25.0


def simulate(speed, vert_angle, horiz_angle):
    env = GolfEnv(
        ctrl_dt=0.002, max_time=15.0, hole_pos=HOLE_POS,
        terrain_zones=TERRAIN_ZONES, model_path=MODEL_PATH,
    )
    env.reset()
    env.launch_ball(speed, vert_angle, horiz_angle)

    for _ in range(env.max_steps):
        info = env.step()
        if info["done"]:
            break

    return info


def cost_function(x):
    x_clamped = np.clip(x, BOUNDS_LOWER, BOUNDS_UPPER)
    speed, vert, horiz = x_clamped

    info = simulate(speed, vert, horiz)

    if not info["ball_stopped"] and not info["in_hole"]:
        return TIMEOUT_PENALTY + info["dist_to_hole"]
    if info["in_hole"]:
        return 0.0

    cost = info["dist_to_hole"]
    if info["terrain"] == "sand": cost += SAND_PENALTY
    if info["terrain"] == "rough": cost += ROUGH_PENALTY
    return cost


def run_cmaes(x0, sigma0, seed, max_evals, label=""):
    best_cost = float("inf")
    best_x = None
    gen_count = 0
    eval_count = 0
    start_time = time.time()
    history = []

    opts = {
        'popsize': POP_SIZE, 'maxfevals': max_evals,
        'bounds': [BOUNDS_LOWER, BOUNDS_UPPER],
        'tolfun': 1e-6, 'verb_disp': 0, 'verb_log': 0, 'seed': seed,
    }
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    while not es.stop():
        candidates = es.ask()
        costs = [cost_function(c) for c in candidates]
        eval_count += len(costs)
        for c, cost in zip(candidates, costs):
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
        elif best_cost < 5.0: status = " (on the green)"

        print(f"  {label}Gen {gen_count:3d} | evals: {eval_count:4d} | "
              f"best: {best_cost:8.3f}m | elapsed: {elapsed:5.1f}s{status}")
        if best_cost == 0.0:
            break

    return best_x, best_cost, history, eval_count


def main():
    print("=" * 60)
    print("COURSE 1: Straight Shot — CMA-ES (3 variables)")
    print("=" * 60)
    print(f"Hole:            ({HOLE_POS[0]}, {HOLE_POS[1]})")
    print(f"Distance:        {STRAIGHT_LINE_DIST:.1f}m")
    print(f"Variables:        speed, vert_angle, horiz_angle")
    print(f"Population:       {POP_SIZE}")
    print(f"Max evals/run:    {MAX_EVALS_PER_RUN}")
    print()

    # Initial guess
    x0 = np.array([18.0, np.radians(25), AIM_ANGLE_TO_HOLE])
    sigma0 = 0.3 * np.mean(np.array(BOUNDS_UPPER) - np.array(BOUNDS_LOWER))

    global_best_cost = float("inf")
    global_best_x = None
    total_evals = 0
    all_history = []
    t_start = time.time()

    for restart in range(MAX_RESTARTS + 1):
        if restart == 0:
            print("── Initial run ──\n")
            current_x0 = x0.copy()
            current_sigma = sigma0
            seed = 42
        else:
            print(f"\n── Restart {restart} (best: {global_best_cost:.3f}m) ──\n")
            current_x0 = global_best_x.copy()
            current_sigma = sigma0 * (0.3 ** restart)
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

    t_total = time.time() - t_start
    best_x_clamped = np.clip(global_best_x, BOUNDS_LOWER, BOUNDS_UPPER)
    speed, vert, horiz = best_x_clamped

    # Final simulation
    print(f"\n{'─'*60}")
    print("Running final simulation...\n")
    info = simulate(speed, vert, horiz)

    print("RESULT:")
    print(f"  Total time:      {t_total:.1f}s")
    print(f"  Total evals:     {total_evals}")
    print(f"  Best cost:       {global_best_cost:.4f}")
    print(f"\n  Launch speed:    {speed:.2f} m/s")
    print(f"  Vertical angle:  {np.degrees(vert):.2f} deg")
    print(f"  Horizontal angle:{np.degrees(horiz):.2f} deg")
    print(f"\n  Ball final:      ({info['ball_pos'][0]:.2f}, {info['ball_pos'][1]:.2f}, {info['ball_pos'][2]:.2f})")
    print(f"  Dist to hole:    {info['dist_to_hole']:.4f} m")
    print(f"  In hole:         {info['in_hole']}")
    print(f"  Terrain:         {info['terrain']}")

    save_path = Path("courses/course1_result.npz")
    np.savez(
        save_path,
        speed=speed, vert_angle=vert, horiz_angle=horiz,
        hole_pos=HOLE_POS,
        terrain_zones=np.array(TERRAIN_ZONES, dtype=object),
        cost=global_best_cost,
        cost_history=np.array(all_history),
        model_path=str(MODEL_PATH),
    )
    print(f"\n  Saved to: {save_path}")

    print(f"\n{'='*60}")
    if info["in_hole"]:
        print("HOLE IN ONE!")
    else:
        print(f"Best distance: {info['dist_to_hole']:.3f}m from hole")
    print(f"{'='*60}")
    print(f"\nTo visualize: py -3.13 scripts/visualize_result.py")


if __name__ == "__main__":
    main()