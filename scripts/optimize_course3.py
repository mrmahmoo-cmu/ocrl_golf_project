"""
optimize_course3.py

Multi-stroke CMA-ES optimization for Course 3: White Dogwood.

The course is ~117m, too far for one shot. Strategy: optimize each
stroke greedily with early stopping — if the optimizer hasn't improved
by more than 1m in 8 generations, stop and take the next shot.

3 decision variables per stroke: speed, vertical angle, horizontal angle.
Includes wind, water penalty, and landed-terrain penalties.

Usage:
    py -3.13 scripts/optimize_course3.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import time
import cma
import mujoco

from envs.golf_env import GolfEnv
from courses.course3 import (
    HOLE_POS, TERRAIN_ZONES, MODEL_PATH, WIND,
    STRAIGHT_LINE_DIST, AIM_ANGLE_TO_HOLE,
)

# Per-stroke config
BOUNDS_LOWER = [5.0,  np.radians(5),  np.radians(-30)]
BOUNDS_UPPER = [30.0, np.radians(60), np.radians(60)]
N_VARS = 3

POP_SIZE = 16
MAX_EVALS_PER_RUN = 500
MAX_RESTARTS = 1
MAX_STROKES = 6

# Early stopping: if best hasn't improved by this much in N gens, move on
STAGNATION_THRESHOLD = 1.0   # meters
STAGNATION_GENS = 8          # generations without improvement

# Penalties
TIMEOUT_PENALTY = 500.0
SAND_PENALTY = 10.0
ROUGH_PENALTY = 15.0
WATER_PENALTY = 200.0
LANDED_WATER_PENALTY = 200.0
LANDED_SAND_PENALTY = 20.0
LANDED_ROUGH_PENALTY = 10.0


def simulate_shot(ball_start, speed, vert_angle, horiz_angle):
    env = GolfEnv(
        ctrl_dt=0.002, max_time=15.0, hole_pos=HOLE_POS,
        terrain_zones=TERRAIN_ZONES, model_path=MODEL_PATH,
        wind=WIND,
    )
    env.reset()

    # Move ball to starting position
    qp = env.ball_qpos_start
    env.data.qpos[qp]     = ball_start[0]
    env.data.qpos[qp + 1] = ball_start[1]
    env.data.qpos[qp + 2] = ball_start[2]
    mujoco.mj_forward(env.model, env.data)

    env.launch_ball(speed, vert_angle, horiz_angle)

    for _ in range(env.max_steps):
        info = env.step()
        if info["done"]:
            break

    return info


def cost_function(x, ball_start):
    x_clamped = np.clip(x, BOUNDS_LOWER, BOUNDS_UPPER)
    speed, vert, horiz = x_clamped

    info = simulate_shot(ball_start, speed, vert, horiz)

    if not info["ball_stopped"] and not info["in_hole"]:
        return TIMEOUT_PENALTY + info["dist_to_hole"]

    if info["in_hole"]:
        # Penalize if landed on bad terrain
        lt = info.get("landed_terrain")
        if lt == "water":
            return LANDED_WATER_PENALTY
        if lt == "sand":
            return LANDED_SAND_PENALTY
        if lt == "rough":
            return LANDED_ROUGH_PENALTY
        return 0.0

    cost = info["dist_to_hole"]

    terrain = info["terrain"]
    if terrain == "water":
        cost += WATER_PENALTY
    elif terrain == "sand":
        cost += SAND_PENALTY
    elif terrain == "rough":
        cost += ROUGH_PENALTY

    if info.get("landed_terrain") == "water":
        cost += LANDED_WATER_PENALTY

    return cost


def optimize_stroke(ball_start, stroke_num):
    dx = HOLE_POS[0] - ball_start[0]
    dy = HOLE_POS[1] - ball_start[1]
    aim_to_hole = np.arctan2(dy, dx)
    dist_to_hole = np.sqrt(dx**2 + dy**2)

    speed_guess = min(25.0, max(8.0, dist_to_hole * 0.4))
    vert_guess = np.radians(30) if dist_to_hole > 15 else np.radians(20)

    x0 = np.array([speed_guess, vert_guess, aim_to_hole])
    sigma0 = 0.3 * np.mean(np.array(BOUNDS_UPPER) - np.array(BOUNDS_LOWER))

    global_best_cost = float("inf")
    global_best_x = None

    for restart in range(MAX_RESTARTS + 1):
        if restart == 0:
            current_x0 = x0.copy()
            current_sigma = sigma0
            seed = 42 + stroke_num * 100
        else:
            current_x0 = global_best_x.copy()
            current_sigma = sigma0 * (0.3 ** restart)
            seed = 42 + stroke_num * 100 + restart * 17

        opts = {
            'popsize': POP_SIZE, 'maxfevals': MAX_EVALS_PER_RUN,
            'bounds': [BOUNDS_LOWER, BOUNDS_UPPER],
            'tolfun': 1e-6, 'verb_disp': 0, 'verb_log': 0, 'seed': seed,
        }
        es = cma.CMAEvolutionStrategy(current_x0, current_sigma, opts)

        gen = 0
        stagnation_count = 0
        prev_best = float("inf")

        while not es.stop():
            candidates = es.ask()
            costs = [cost_function(c, ball_start) for c in candidates]
            for c, cost in zip(candidates, costs):
                if cost < global_best_cost:
                    global_best_cost = cost
                    global_best_x = c.copy()
            es.tell(candidates, costs)
            gen += 1

            # Early stopping: check stagnation
            improvement = prev_best - global_best_cost
            if improvement < STAGNATION_THRESHOLD:
                stagnation_count += 1
            else:
                stagnation_count = 0
            prev_best = global_best_cost

            status = ""
            if global_best_cost == 0.0: status = " *** HOLED ***"
            elif global_best_cost < 1.0: status = " (very close!)"
            elif global_best_cost < 5.0: status = " (on the green)"
            elif global_best_cost < 20.0: status = " (approaching)"

            r_label = f"[R{restart}] " if restart > 0 else ""
            print(f"    {r_label}Gen {gen:3d} | best: {global_best_cost:8.3f}m{status}")

            if global_best_cost == 0.0:
                break

            if stagnation_count >= STAGNATION_GENS:
                print(f"    → Early stop: no significant improvement in {STAGNATION_GENS} generations")
                break

        if global_best_cost == 0.0:
            break

    # Final sim with best params
    best_clamped = np.clip(global_best_x, BOUNDS_LOWER, BOUNDS_UPPER)
    speed, vert, horiz = best_clamped
    info = simulate_shot(ball_start, speed, vert, horiz)

    return best_clamped, global_best_cost, info


def main():
    print("=" * 60)
    print("COURSE 3: White Dogwood — Multi-Stroke CMA-ES")
    print("=" * 60)
    print(f"Hole:            ({HOLE_POS[0]}, {HOLE_POS[1]})")
    print(f"Distance:        {STRAIGHT_LINE_DIST:.1f}m")
    if np.linalg.norm(WIND) > 0:
        print(f"Wind:            ({WIND[0]}, {WIND[1]}, {WIND[2]}) m/s")
    print(f"Max strokes:     {MAX_STROKES}")
    print(f"Early stop:      {STAGNATION_THRESHOLD}m improvement / {STAGNATION_GENS} gens")
    print(f"Variables/stroke: {N_VARS}")
    print()

    ball_pos = np.array([0.05, 0.0, 0.0214])
    all_strokes = []
    t_start = time.time()

    for stroke in range(1, MAX_STROKES + 1):
        dist_remaining = np.linalg.norm(HOLE_POS - ball_pos[:2])
        print(f"{'─'*60}")
        print(f"STROKE {stroke} | from ({ball_pos[0]:.1f}, {ball_pos[1]:.1f}) | "
              f"{dist_remaining:.1f}m to hole")
        print(f"{'─'*60}")

        params, cost, info = optimize_stroke(ball_pos, stroke)
        speed, vert, horiz = params

        stroke_data = {
            "stroke": stroke,
            "start_pos": ball_pos.copy(),
            "speed": speed,
            "vert_angle": vert,
            "horiz_angle": horiz,
            "end_pos": info["ball_pos"].copy(),
            "dist_to_hole": info["dist_to_hole"],
            "in_hole": info["in_hole"],
            "terrain": info["terrain"],
            "landed_terrain": info["landed_terrain"],
        }
        all_strokes.append(stroke_data)

        print(f"\n  Shot: {speed:.1f} m/s, {np.degrees(vert):.1f}° up, "
              f"{np.degrees(horiz):.1f}° aim")
        print(f"  Landed: ({info['ball_pos'][0]:.1f}, {info['ball_pos'][1]:.1f}) "
              f"on {info['terrain']}")
        print(f"  Distance to hole: {info['dist_to_hole']:.2f}m")

        if info["in_hole"]:
            print(f"\n  *** HOLED on stroke {stroke}! ***")
            break

        if info["terrain"] == "water":
            print(f"  *** BALL IN WATER — penalty drop ***")
            dx = ball_pos[0] - info["ball_pos"][0]
            dy = ball_pos[1] - info["ball_pos"][1]
            d = np.sqrt(dx**2 + dy**2)
            if d > 0:
                ball_pos = info["ball_pos"].copy()
                ball_pos[0] += 2.0 * dx / d
                ball_pos[1] += 2.0 * dy / d
                ball_pos[2] = 0.0214
            else:
                ball_pos = info["ball_pos"].copy()
                ball_pos[2] = 0.0214
            stroke_data["penalty"] = True
            print(f"  Dropped at ({ball_pos[0]:.1f}, {ball_pos[1]:.1f})")
        else:
            ball_pos = info["ball_pos"].copy()
            ball_pos[2] = 0.0214

        print()

    t_total = time.time() - t_start

    # Summary
    print(f"\n{'='*60}")
    print("RESULT SUMMARY")
    print(f"{'='*60}")
    total_strokes = len(all_strokes)
    holed = all_strokes[-1]["in_hole"] if all_strokes else False
    print(f"  Total strokes:  {total_strokes}")
    print(f"  Holed:          {holed}")
    print(f"  Total time:     {t_total:.1f}s")
    if np.linalg.norm(WIND) > 0:
        print(f"  Wind:           ({WIND[0]}, {WIND[1]}, {WIND[2]}) m/s")
    print()

    for s in all_strokes:
        penalty = " [WATER]" if s.get("penalty") else ""
        hole_str = " *** HOLED ***" if s["in_hole"] else ""
        print(f"  Stroke {s['stroke']}: "
              f"({s['start_pos'][0]:.1f}, {s['start_pos'][1]:.1f}) → "
              f"({s['end_pos'][0]:.1f}, {s['end_pos'][1]:.1f}) "
              f"[{s['speed']:.1f} m/s, {np.degrees(s['vert_angle']):.1f}°, "
              f"{np.degrees(s['horiz_angle']):.1f}°] "
              f"on {s['terrain']}{penalty}{hole_str}")

    # Save
    save_path = Path("courses/course3_result.npz")
    np.savez(
        save_path,
        strokes=np.array([(s["stroke"], s["speed"], s["vert_angle"], s["horiz_angle"],
                           s["start_pos"][0], s["start_pos"][1], s["start_pos"][2],
                           s["end_pos"][0], s["end_pos"][1], s["end_pos"][2],
                           s["dist_to_hole"], int(s["in_hole"]))
                          for s in all_strokes]),
        hole_pos=HOLE_POS,
        terrain_zones=np.array(TERRAIN_ZONES, dtype=object),
        model_path=str(MODEL_PATH),
        wind=WIND,
        total_strokes=total_strokes,
        holed=holed,
    )
    print(f"\n  Saved to: {save_path}")

    print(f"\n{'='*60}")
    if holed:
        print(f"COMPLETED IN {total_strokes} STROKES!")
    else:
        final_dist = all_strokes[-1]["dist_to_hole"]
        print(f"DID NOT HOLE — {final_dist:.2f}m remaining after {total_strokes} strokes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()