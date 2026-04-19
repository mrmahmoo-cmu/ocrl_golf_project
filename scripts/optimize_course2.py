"""
optimize_course2.py

CMA-ES trajectory optimization for Course 2: The Ridge.

Same approach as Course 1 but with wind and a longer course.
The optimizer must account for the crosswind when choosing aim direction.

Usage:
    py -3.13 scripts/optimize_course2.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import time
import cma
from pathlib import Path

from envs.golf_env import GolfSwingEnv
from courses.course2 import (
    HOLE_POS, TERRAIN_ZONES, MODEL_PATH, WIND,
    STRAIGHT_LINE_DIST, AIM_ANGLE_TO_HOLE,
)

# ── Optimization configuration ─────────────────────────────────────
SWING_DURATION = 0.35
N_BINS = 7
BIN_DT = SWING_DURATION / N_BINS

CTRL_DT = 0.005
MAX_TIME = 12.0

BOUNDS_LOWER = [0.0, np.pi/4] + [-150.0, -60.0] * N_BINS
BOUNDS_UPPER = [np.radians(45), np.pi] + [150.0, 60.0] * N_BINS

N_VARS = 2 + N_BINS * 2

SIGMA0 = 0.3
MAX_EVALS = 5000
POP_SIZE = 32

NO_CONTACT_PENALTY = 1000.0
WRONG_DIRECTION_PENALTY = 500.0
TIMEOUT_PENALTY = 200.0
SAND_PENALTY = 10.0
ROUGH_PENALTY = 25.0


def decode_params(x):
    x_clamped = np.clip(x, BOUNDS_LOWER, BOUNDS_UPPER)
    aim_angle = x_clamped[0]
    elbow_init = x_clamped[1]
    torque_profile = x_clamped[2:].reshape(N_BINS, 2)
    return aim_angle, elbow_init, torque_profile


def get_torque_at_time(torque_profile, t):
    if t >= SWING_DURATION:
        return np.array([0.0, 0.0])
    bin_idx = min(int(t / BIN_DT), N_BINS - 1)
    return torque_profile[bin_idx]


def simulate(aim_angle, elbow_init, torque_profile):
    env = GolfSwingEnv(
        ctrl_dt=CTRL_DT,
        max_time=MAX_TIME,
        hole_pos=HOLE_POS,
        terrain_zones=TERRAIN_ZONES,
        enable_drag=True,
        enable_rolling_decel=True,
        model_path=MODEL_PATH,
        wind=WIND,
    )

    obs = env.reset(elbow_init=elbow_init, wrist_init=0.0, aim_init=aim_angle)

    for step in range(env.max_steps):
        t = step * env.ctrl_dt
        torque = get_torque_at_time(torque_profile, t)
        obs, info = env.step(torque)
        if info["done"]:
            break

    return info["dist_to_hole"], {
        "time": info["time"],
        "contact": env.contact_detected,
        "launched": env.ball_launched,
        "in_hole": info["in_hole"],
        "ball_stopped": info["ball_stopped"],
        "final_pos": obs["ball_pos"].copy(),
        "final_speed": info["ball_speed"],
        "launch_vel": env.ball_launch_vel,
        "terrain": info["terrain"],
    }


def cost_function(x):
    aim_angle, elbow_init, torque_profile = decode_params(x)
    dist, info = simulate(aim_angle, elbow_init, torque_profile)

    if not info["contact"]:
        return NO_CONTACT_PENALTY
    if info["launch_vel"] is not None and info["launch_vel"][0] < 0:
        return WRONG_DIRECTION_PENALTY
    if not info["ball_stopped"] and not info["in_hole"]:
        return TIMEOUT_PENALTY + dist
    if info["in_hole"]:
        return 0.0

    cost = dist
    if info["terrain"] == "sand":
        cost += SAND_PENALTY
    if info["terrain"] == "rough":
        cost += ROUGH_PENALTY
    return cost


def main():
    print("=" * 60)
    print("COURSE 2: The Ridge — CMA-ES Optimization")
    print("=" * 60)
    print(f"Hole:            ({HOLE_POS[0]}, {HOLE_POS[1]})")
    print(f"Distance:        {STRAIGHT_LINE_DIST:.1f}m")
    print(f"Ideal aim:       {np.degrees(AIM_ANGLE_TO_HOLE):.1f} deg")
    print(f"Wind:            ({WIND[0]}, {WIND[1]}, {WIND[2]}) m/s")
    print(f"Variables:        {N_VARS}")
    print(f"Population:       {POP_SIZE} per generation")
    print(f"Max evaluations:  {MAX_EVALS}")
    print()

    # Initial guess
    x0 = np.zeros(N_VARS)
    x0[0] = AIM_ANGLE_TO_HOLE
    x0[1] = 3 * np.pi / 4
    for i in range(N_BINS):
        x0[2 + i*2] = -80.0
        x0[2 + i*2 + 1] = 0.0

    ranges = np.array(BOUNDS_UPPER) - np.array(BOUNDS_LOWER)
    sigma0_scaled = SIGMA0 * np.mean(ranges)

    best_cost = float("inf")
    best_x = None
    gen_count = 0
    eval_count = 0
    start_time = time.time()
    cost_history = []

    print("Starting CMA-ES optimization...\n")

    opts = {
        'popsize': POP_SIZE,
        'maxfevals': MAX_EVALS,
        'bounds': [BOUNDS_LOWER, BOUNDS_UPPER],
        'tolfun': 1e-4,
        'verb_disp': 0,
        'verb_log': 0,
        'seed': 42,
    }

    es = cma.CMAEvolutionStrategy(x0, sigma0_scaled, opts)

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
        cost_history.append(best_cost)

        status = ""
        if best_cost == 0.0:
            status = " *** HOLE IN ONE ***"
        elif best_cost < 1.0:
            status = " (very close!)"
        elif best_cost < 5.0:
            status = " (approaching green)"
        elif best_cost >= NO_CONTACT_PENALTY:
            status = " (no contact yet)"

        print(f"  Gen {gen_count:3d} | evals: {eval_count:5d} | "
              f"best: {best_cost:8.3f}m | elapsed: {elapsed:6.1f}s{status}")

        if best_cost == 0.0:
            break

    t_elapsed = time.time() - start_time
    best_aim, best_elbow_init, best_torques = decode_params(best_x)

    # Final simulation
    print(f"\n{'─'*60}")
    print("Running final simulation...\n")

    env_final = GolfSwingEnv(
        ctrl_dt=0.001,
        max_time=15.0,
        hole_pos=HOLE_POS,
        terrain_zones=TERRAIN_ZONES,
        model_path=MODEL_PATH,
        wind=WIND,
    )
    obs = env_final.reset(elbow_init=best_elbow_init, wrist_init=0.0, aim_init=best_aim)

    trajectory = []
    for step in range(env_final.max_steps):
        t = step * env_final.ctrl_dt
        torque = get_torque_at_time(best_torques, t)
        obs, info = env_final.step(torque)
        if info["ball_launched"]:
            trajectory.append(obs["ball_pos"].copy())
        if info["done"]:
            break

    # Report
    print("RESULT:")
    print(f"  Method:          CMA-ES")
    print(f"  Total time:      {t_elapsed:.1f}s")
    print(f"  Evaluations:     {eval_count}")
    print(f"  Generations:     {gen_count}")
    print(f"  Best cost:       {best_cost:.4f}")
    print()
    print(f"  Aim angle:       {np.degrees(best_aim):.1f} deg")
    print(f"  Backswing angle: {np.degrees(best_elbow_init):.1f} deg")
    print(f"  Wind:            ({WIND[0]}, {WIND[1]}, {WIND[2]}) m/s")
    print(f"  Torque profile:")
    for i in range(N_BINS):
        ts = i * BIN_DT
        te = (i + 1) * BIN_DT
        print(f"    [{ts:.3f}-{te:.3f}s]  "
              f"elbow={best_torques[i, 0]:+7.1f} Nm  wrist={best_torques[i, 1]:+6.1f} Nm")
    print()
    print(f"  Ball final pos:  ({obs['ball_pos'][0]:.2f}, {obs['ball_pos'][1]:.2f}, {obs['ball_pos'][2]:.2f})")
    print(f"  Dist to hole:    {info['dist_to_hole']:.4f} m")
    print(f"  In hole:         {info['in_hole']}")
    print(f"  Terrain:         {info['terrain']}")

    if env_final.ball_launch_vel is not None:
        lv = env_final.ball_launch_vel
        speed = np.linalg.norm(lv)
        angle_vert = np.degrees(np.arctan2(lv[2], np.linalg.norm(lv[:2])))
        angle_horiz = np.degrees(np.arctan2(lv[1], lv[0]))
        print(f"  Launch speed:    {speed:.1f} m/s")
        print(f"  Vertical angle:  {angle_vert:.1f} deg")
        print(f"  Horizontal aim:  {angle_horiz:.1f} deg")

    # Save results
    save_path = Path("courses/course2_result.npz")
    np.savez(
        save_path,
        aim_angle=best_aim,
        elbow_init=best_elbow_init,
        torque_profile=best_torques,
        swing_duration=SWING_DURATION,
        n_bins=N_BINS,
        bin_dt=BIN_DT,
        hole_pos=HOLE_POS,
        terrain_zones=np.array(TERRAIN_ZONES, dtype=object),
        cost=best_cost,
        trajectory=np.array(trajectory) if trajectory else np.zeros((1, 3)),
        cost_history=np.array(cost_history),
        model_path=str(MODEL_PATH),
        has_aim=True,
        wind=WIND,
    )
    print(f"\n  Results saved to: {save_path}")

    print(f"\n{'='*60}")
    if info["in_hole"]:
        print("HOLE IN ONE!")
    else:
        print(f"Best distance: {info['dist_to_hole']:.3f}m from hole")
        print(f"Landed on: {info['terrain']}")
    print(f"{'='*60}")
    print(f"\nTo visualize: py -3.13 scripts/visualize_result.py courses/course2_result.npz")


if __name__ == "__main__":
    main()
