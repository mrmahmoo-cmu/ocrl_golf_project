"""
optimize_course3_sequence.py

Full-horizon / sequence CMA-ES optimization for Course 3: White Dogwood.

WHY THIS FILE EXISTS
--------------------
The existing Course 3 optimizer uses a greedy strategy:
    1. Optimize stroke 1 from the current ball position.
    2. Execute that stroke.
    3. Move the ball to the new position.
    4. Optimize stroke 2 from there.
    5. Repeat until the ball is holed.

That works well and is fast, but each stroke is optimized locally.

This script implements a second strategy:
    Full-horizon / sequence optimization.

Instead of optimizing one stroke at a time, CMA-ES optimizes the entire
K-shot plan together.

Example for K = 3:
    Candidate vector x =
        [
            speed_1, vertical_angle_1, horizontal_angle_1,
            speed_2, vertical_angle_2, horizontal_angle_2,
            speed_3, vertical_angle_3, horizontal_angle_3
        ]

This means shot 1 can be optimized while considering how it affects
shot 2 and shot 3.

WHAT WE TEST
------------
We test different planning horizons:
    K = 1
    K = 2
    K = 3

Each shot has 3 decision variables:
    1. launch speed
    2. vertical launch angle
    3. horizontal launch angle

Therefore:
    K = 1 -> 3 variables
    K = 2 -> 6 variables
    K = 3 -> 9 variables

WATER / POND RULE
-----------------
If the ball lands in the pond:
    1. Add +1 penalty stroke.
    2. Move the ball to a simplified drop position outside the water.
    3. Continue the sequence if shots remain.

Final golf score:
    total_score = actual_shots + penalty_strokes

This makes the water behavior closer to real golf scoring.
"""


# ============================================================
# Imports and repo path setup
# ============================================================

import sys
from pathlib import Path
import time

# This file lives in scripts/.
# parents[1] moves one level up to the repo root:
#     scripts/optimize_course3_sequence.py -> repo_root/
#
# Adding the repo root to sys.path lets us import:
#     envs.golf_env
#     courses.course3
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import cma
import mujoco

from envs.golf_env import GolfEnv
from courses.course3 import (
    HOLE_POS,
    TERRAIN_ZONES,
    MODEL_PATH,
    WIND,
    STRAIGHT_LINE_DIST,
    AIM_ANGLE_TO_HOLE,
)


# ============================================================
# General configuration
# ============================================================

# Golf ball radius/height used in the MuJoCo XML.
# We use this when placing the ball at a new stroke start location.
BALL_Z = 0.0214

# Initial tee position.
# This should match the ball start position in course3.xml.
START_POS = np.array([0.05, 0.0, BALL_Z])

# We test sequence lengths K = 1, 2, 3.
# K means "number of physical shots in the planned sequence."
K_MIN = 2
K_MAX = 3

# Each shot is parameterized by:
#     [speed, vertical_angle, horizontal_angle]
VARS_PER_SHOT = 3

# Bounds for one shot:
#     speed in m/s
#     vertical angle in radians
#     horizontal angle in radians
#
# These bounds are repeated K times for K-shot optimization.
ONE_SHOT_LOWER = np.array([
    5.0,              # minimum launch speed
    np.radians(5),    # minimum vertical angle
    np.radians(-30),  # minimum horizontal aim angle
])

ONE_SHOT_UPPER = np.array([
    30.0,             # maximum launch speed
    np.radians(60),   # maximum vertical angle
    np.radians(60),   # maximum horizontal aim angle
])

# CMA-ES population size.
# Each generation tries POP_SIZE candidate sequences.
#
# Example:
#     K = 3 and POP_SIZE = 24
#     means 24 candidate 3-shot plans are simulated per generation.
POP_SIZE = 24

# Maximum number of candidate evaluations per CMA-ES run.
# Since each candidate may contain multiple shots, K = 3 is more expensive
# than K = 1.
MAX_EVALS_PER_RUN = 1200

# Number of restarts after the initial CMA-ES run.
# MAX_RESTARTS = 1 means:
#     initial run + 1 restart
MAX_RESTARTS = 1

# Print progress every N generations to avoid an extremely messy terminal.
# Set to 1 if you want every generation printed.
PRINT_EVERY_N_GENS = 5


# ============================================================
# Scoring / cost constants
# ============================================================

# If a sequence does not hole the ball, it receives this base penalty.
# Then final distance and hazard penalties are added.
FAILURE_PENALTY = 1000.0

# This makes the golf score very important when a sequence succeeds.
#
# Example:
#     total_score = 2 -> cost starts around 200
#     total_score = 3 -> cost starts around 300
#
# This encourages CMA-ES to find lower-score solutions.
STROKE_SCORE_WEIGHT = 100.0

# Extra penalty applied when a successful solution includes water penalties.
# This means:
#     clean score 3 is preferred over score 3 with water penalty,
#     if both have same total_score.
PENALTY_STROKE_EXTRA_WEIGHT = 25.0

# Penalty for timeouts / sequences that do not settle cleanly.
TIMEOUT_PENALTY = 250.0

# Hazard penalties for failed or partial sequences.
# These help CMA-ES rank bad candidates instead of treating them all equally.
WATER_HIT_COST = 200.0
SAND_FINAL_COST = 20.0
ROUGH_FINAL_COST = 10.0

# First attempted drop distance after water.
# If this drop is still inside water, the function tries larger distances.
DROP_BACK_DISTANCE = 2.0


# ============================================================
# Helper functions
# ============================================================

def bounds_for_k(k):
    """
    Create lower/upper CMA-ES bounds for a K-shot sequence.

    For one shot, we have:
        [speed_min, vert_min, horiz_min]
        [speed_max, vert_max, horiz_max]

    For K = 3, we need:
        lower = [
            speed_min, vert_min, horiz_min,
            speed_min, vert_min, horiz_min,
            speed_min, vert_min, horiz_min
        ]

    Same idea for upper.
    """
    lower = np.tile(ONE_SHOT_LOWER, k)
    upper = np.tile(ONE_SHOT_UPPER, k)
    return lower, upper


def classify_terrain_xy(x, y):
    """
    Classify the terrain type at XY position (x, y).

    This mirrors the logic in TerrainMap.classify() inside golf_env.py.

    Important detail:
        Terrain zones use "first match wins."

    That means if a point is inside multiple zones, the first zone in
    TERRAIN_ZONES is returned.

    If no terrain zone contains the point, we treat it as rough.
    """
    for x_min, x_max, y_min, y_max, terrain in TERRAIN_ZONES:
        if x_min <= x <= x_max and y_min <= y <= y_max:
            return terrain

    return "rough"


def make_initial_guess(k):
    """
    Build a reasonable initial guess for a K-shot plan.

    CMA-ES does not require a perfect initial guess, but a reasonable one
    helps it converge faster.

    Strategy:
        Split the straight-line distance roughly across K shots.

    Example:
        If the hole is 117 m away:
            K = 1 -> each shot roughly 117 m
            K = 2 -> each shot roughly 58.5 m
            K = 3 -> each shot roughly 39 m

    We convert that rough distance into a speed guess using a simple scaling.
    CMA-ES will then adjust speed, vertical angle, and direction.
    """
    distance_per_shot = STRAIGHT_LINE_DIST / k

    # Speed guess is capped between 8 and 25 m/s.
    # This prevents tiny speeds for short horizons and excessive speeds for long ones.
    speed_guess = min(25.0, max(8.0, distance_per_shot * 0.4))

    # Farther shots use a higher launch angle.
    # Shorter approach shots use a lower angle.
    vert_guess = np.radians(30) if distance_per_shot > 15.0 else np.radians(20)

    # Initial aim points directly at the hole.
    # The optimizer can later discover that aiming away from the hole is better.
    x0 = []
    for _ in range(k):
        x0.extend([speed_guess, vert_guess, AIM_ANGLE_TO_HOLE])

    return np.array(x0, dtype=float)


def unpack_shots(x, k):
    """
    Convert a flat CMA-ES decision vector into K shot parameter triples.

    Example for K = 2:
        x = [
            speed_1, vert_1, horiz_1,
            speed_2, vert_2, horiz_2
        ]

    returns:
        [
            (speed_1, vert_1, horiz_1),
            (speed_2, vert_2, horiz_2)
        ]

    We also clamp x to the allowed bounds before using it.
    """
    lower, upper = bounds_for_k(k)
    x_clamped = np.clip(x, lower, upper)

    shots = []

    for i in range(k):
        # Starting index of shot i in the flat vector.
        j = i * VARS_PER_SHOT

        speed = x_clamped[j]
        vert = x_clamped[j + 1]
        horiz = x_clamped[j + 2]

        shots.append((speed, vert, horiz))

    return shots, x_clamped


def compute_drop_position(previous_pos, water_pos):
    """
    Compute a simplified penalty drop position after a water hit.

    Real golf relief is more nuanced, but for the simulator we use:

        If ball enters water:
            move it backward from the water position toward the previous
            shot start position.

    This approximates dropping outside the pond, behind the entry direction.

    Inputs:
        previous_pos:
            position where the shot started

        water_pos:
            position where water was detected

    Output:
        drop position outside the water if possible
    """
    previous_xy = previous_pos[:2]
    water_xy = water_pos[:2]

    # Direction from water position back toward where the shot came from.
    direction = previous_xy - water_xy
    norm = np.linalg.norm(direction)

    if norm < 1e-9:
        # Fallback if previous_pos and water_pos are basically identical.
        # Move backward in x.
        direction = np.array([-1.0, 0.0])
    else:
        direction = direction / norm

    # Try increasingly larger drop distances until the drop is outside water.
    # This prevents dropping the ball back into the pond.
    for dist in [DROP_BACK_DISTANCE, 3.0, 4.0, 5.0, 7.0, 10.0]:
        drop_xy = water_xy + dist * direction
        terrain = classify_terrain_xy(drop_xy[0], drop_xy[1])

        if terrain != "water":
            return np.array([drop_xy[0], drop_xy[1], BALL_Z])

    # Last fallback: return a point 10 m back even if something weird happens.
    drop_xy = water_xy + 10.0 * direction
    return np.array([drop_xy[0], drop_xy[1], BALL_Z])


# ============================================================
# MuJoCo simulation
# ============================================================

def simulate_single_shot(ball_start, speed, vert_angle, horiz_angle):
    """
    Simulate one shot from an arbitrary starting position.

    This is similar to Course 2's simulate() function, but it supports
    starting from any position, not only the tee.

    That is required for multi-shot planning because shot 2 starts wherever
    shot 1 ended, and shot 3 starts wherever shot 2 ended.

    Returns:
        info dictionary from GolfEnv, plus:
            info["in_water"]  -> True/False
            info["water_pos"] -> position where water was detected, or None

    Water detection logic:
        We only count water if:
            1. the ball is touching/contacting the ground, and
            2. its terrain classification is "water".

        This avoids penalizing a ball that simply flies over the pond.
    """
    env = GolfEnv(
        ctrl_dt=0.002,
        max_time=15.0,
        hole_pos=HOLE_POS,
        terrain_zones=TERRAIN_ZONES,
        model_path=MODEL_PATH,
        wind=WIND,
    )

    # Reset MuJoCo simulation to the XML starting state.
    env.reset()

    # Move the ball to the desired start position.
    # ball_qpos_start is the starting index of the ball's free-joint qpos.
    qp = env.ball_qpos_start

    # ball_qvel_start is the starting index of the ball's free-joint qvel.
    qv = env.ball_qvel_start

    # Free joint position:
    #     qpos[qp + 0] = x
    #     qpos[qp + 1] = y
    #     qpos[qp + 2] = z
    env.data.qpos[qp] = ball_start[0]
    env.data.qpos[qp + 1] = ball_start[1]
    env.data.qpos[qp + 2] = ball_start[2]

    # Reset ball orientation quaternion.
    # Free joint orientation is stored after position as:
    #     [qw, qx, qy, qz]
    # Identity quaternion = [1, 0, 0, 0]
    env.data.qpos[qp + 3: qp + 7] = np.array([1.0, 0.0, 0.0, 0.0])

    # Reset translational and rotational velocities to zero.
    env.data.qvel[qv: qv + 6] = 0.0

    # Tell MuJoCo to recompute all derived quantities after manually changing qpos/qvel.
    mujoco.mj_forward(env.model, env.data)

    # Directly set initial ball velocity using the "magic hand" launch model.
    env.launch_ball(speed, vert_angle, horiz_angle)

    info = None

    # Step the simulation forward until:
    #     ball stops,
    #     ball goes in hole,
    #     water is detected,
    #     or max time is reached.
    for _ in range(env.max_steps):
        info = env.step()

        # Check whether the ball is currently touching the ground.
        # This uses the environment's internal MuJoCo contact check.
        on_ground_now, _ = env._get_ground_contact_info()

        # Count water only if the ball is on the ground/contacting while in water.
        # This is important because flying over water should not be penalized.
        if on_ground_now and info["terrain"] == "water":
            info = dict(info)
            info["in_water"] = True
            info["water_pos"] = info["ball_pos"].copy()
            info["done"] = True
            return info

        if info["done"]:
            break

    if info is None:
        raise RuntimeError("Simulation ended without producing info.")

    # Convert to normal dict so we can safely add fields.
    info = dict(info)

    # Extra safety:
    # If the final resting terrain is water, count it as water.
    if info["terrain"] == "water":
        info["in_water"] = True
        info["water_pos"] = info["ball_pos"].copy()
    else:
        info["in_water"] = False
        info["water_pos"] = None

    return info


def simulate_sequence(x, k):
    """
    Simulate a full K-shot sequence.

    This is the core difference between this method and the greedy method.

    Greedy:
        optimize one shot, execute it, then optimize next shot.

    Sequence:
        CMA-ES proposes all K shots at once.
        We simulate them in order and return one score for the whole plan.

    Inputs:
        x:
            flat vector of all shot parameters

        k:
            number of planned shots

    Output:
        dictionary containing full sequence result:
            strokes
            holed
            actual_shots
            penalty_strokes
            total_score
            final position
            final distance
            final terrain
    """
    shots, x_clamped = unpack_shots(x, k)

    # Start every candidate sequence from the tee.
    ball_pos = START_POS.copy()

    # Store details about each simulated shot for reporting and visualization.
    strokes = []

    # Count golf penalty strokes from water.
    penalty_strokes = 0

    # Becomes True if any shot holes the ball.
    holed = False

    for shot_idx, (speed, vert, horiz) in enumerate(shots, start=1):
        start_pos = ball_pos.copy()

        # Simulate this shot from the current ball position.
        info = simulate_single_shot(start_pos, speed, vert, horiz)

        # Record everything useful from this shot.
        stroke_data = {
            "shot": shot_idx,
            "start_pos": start_pos.copy(),
            "speed": speed,
            "vert_angle": vert,
            "horiz_angle": horiz,
            "end_pos": info["ball_pos"].copy(),
            "dist_to_hole": info["dist_to_hole"],
            "terrain": info["terrain"],
            "landed_terrain": info.get("landed_terrain"),
            "in_hole": bool(info["in_hole"]),
            "in_water": bool(info.get("in_water", False)),
            "drop_pos": None,
        }

        strokes.append(stroke_data)

        # If the ball is holed, the sequence ends immediately.
        # Any unused planned shots are ignored.
        if info["in_hole"]:
            holed = True
            ball_pos = info["ball_pos"].copy()
            break

        # Water / pond handling:
        #     add one penalty stroke,
        #     compute a simplified drop position,
        #     continue from the drop position if there are shots remaining.
        if info.get("in_water", False):
            penalty_strokes += 1

            water_pos = info["water_pos"] if info["water_pos"] is not None else info["ball_pos"]
            drop_pos = compute_drop_position(start_pos, water_pos)

            stroke_data["drop_pos"] = drop_pos.copy()
            ball_pos = drop_pos.copy()
            continue

        # Normal non-water shot:
        # continue from the ball's final resting position.
        ball_pos = info["ball_pos"].copy()

        # Keep the ball on/above the course surface for the next launch.
        ball_pos[2] = BALL_Z

    # After simulating all available shots, compute final summary.
    final_dist = np.linalg.norm(HOLE_POS - ball_pos[:2])
    final_terrain = classify_terrain_xy(ball_pos[0], ball_pos[1])

    # actual_shots is the number of physical simulated launches.
    actual_shots = len(strokes)

    # total_score follows golf-style scoring:
    #     physical shots + penalty strokes
    total_score = actual_shots + penalty_strokes

    # Count how many water hits occurred.
    water_hits = sum(1 for s in strokes if s["in_water"])

    return {
        "k": k,
        "x": x_clamped.copy(),
        "strokes": strokes,
        "holed": holed,
        "actual_shots": actual_shots,
        "penalty_strokes": penalty_strokes,
        "total_score": total_score,
        "water_hits": water_hits,
        "final_pos": ball_pos.copy(),
        "final_dist": final_dist,
        "final_terrain": final_terrain,
    }


# ============================================================
# Cost function
# ============================================================

def score_sequence_result(result):
    """
    Convert a full sequence result into one scalar cost for CMA-ES.

    CMA-ES only understands one number:
        lower cost = better candidate.

    Therefore, this function turns a complete K-shot simulation into one cost.

    If the sequence holes the ball:
        prioritize golf score:
            total_score = actual_shots + penalty_strokes

        add a small extra penalty for water so clean solutions are preferred.

    If the sequence does not hole the ball:
        use a large failure penalty, then add:
            final distance to hole,
            hazard penalties,
            water hit penalties.
    """
    if result["holed"]:
        # Main successful-sequence cost:
        # lower golf score is better.
        cost = STROKE_SCORE_WEIGHT * result["total_score"]

        # Extra penalty to prefer clean solutions when total_score is equal.
        cost += PENALTY_STROKE_EXTRA_WEIGHT * result["penalty_strokes"]

        return cost

    # Failed sequences are worse than successful ones.
    cost = FAILURE_PENALTY

    # Among failures, closer to the hole is better.
    cost += result["final_dist"]

    # Penalize bad final terrain.
    if result["final_terrain"] == "water":
        cost += WATER_HIT_COST
    elif result["final_terrain"] == "sand":
        cost += SAND_FINAL_COST
    elif result["final_terrain"] == "rough":
        cost += ROUGH_FINAL_COST

    # Penalize any water hits during the sequence.
    cost += WATER_HIT_COST * result["water_hits"]

    # Penalize penalty strokes in failed sequences too.
    cost += TIMEOUT_PENALTY * result["penalty_strokes"]

    return cost


def evaluate_candidate(x, k):
    """
    Evaluate one CMA-ES candidate.

    This is the function used inside the CMA-ES loop.

    Steps:
        1. Simulate the full K-shot sequence.
        2. Convert the sequence result into a scalar cost.
        3. Return both cost and detailed result.
    """
    result = simulate_sequence(x, k)
    cost = score_sequence_result(result)
    return cost, result


# ============================================================
# CMA-ES optimizer
# ============================================================

def run_cmaes_for_k(k):
    """
    Optimize a K-shot sequence using CMA-ES.

    CMA-ES is a derivative-free optimizer:
        It does not need gradients.
        It only needs candidate vectors and their costs.

    For this project:
        candidate vector = K-shot golf plan
        cost = score from score_sequence_result()

    Example:
        K = 3
        n_vars = 9
        CMA-ES searches for the best 9 values.
    """
    n_vars = k * VARS_PER_SHOT
    lower, upper = bounds_for_k(k)

    # Initial guess around a simple "split distance evenly" strategy.
    x0 = make_initial_guess(k)

    # Initial search spread.
    # Larger sigma means wider exploration.
    # Smaller sigma means more local refinement.
    sigma0 = 0.3 * np.mean(upper - lower)

    # Track best result across initial run and restarts.
    global_best_cost = float("inf")
    global_best_x = None
    global_best_result = None

    # Store best-cost history for possible plotting later.
    all_history = []

    # Count how many candidate sequences were evaluated.
    total_evals = 0

    print(f"\n{'=' * 70}")
    print(f"FULL-HORIZON CMA-ES | K = {k} shots | variables = {n_vars}")
    print(f"{'=' * 70}")

    for restart in range(MAX_RESTARTS + 1):
        if restart == 0:
            # First run starts from the initial guess.
            current_x0 = x0.copy()
            current_sigma = sigma0
            seed = 123 + 1000 * k
            label = ""
        else:
            # Restart from the best solution found so far,
            # but with a smaller search radius.
            current_x0 = global_best_x.copy()
            current_sigma = sigma0 * (0.3 ** restart)
            seed = 123 + 1000 * k + 17 * restart
            label = f"[R{restart}] "

        # CMA-ES options.
        opts = {
            "popsize": POP_SIZE,
            "maxfevals": MAX_EVALS_PER_RUN,
            "bounds": [lower.tolist(), upper.tolist()],
            "tolfun": 1e-6,

            # Turn off cma package's default printing/logging.
            # We print our own cleaner progress messages.
            "verb_disp": 0,
            "verb_log": 0,

            # Fixed seed for repeatability.
            "seed": seed,
        }

        # Create CMA-ES optimizer object.
        es = cma.CMAEvolutionStrategy(current_x0, current_sigma, opts)

        gen = 0
        t0 = time.time()

        # Main CMA-ES loop.
        while not es.stop():
            # ask():
            #     CMA-ES proposes a population of candidate K-shot sequences.
            candidates = es.ask()

            costs = []
            results = []

            # Evaluate every candidate sequence.
            for candidate in candidates:
                cost, result = evaluate_candidate(candidate, k)

                costs.append(cost)
                results.append(result)

                # Manually track the best candidate across all generations.
                if cost < global_best_cost:
                    global_best_cost = cost
                    global_best_x = np.array(candidate, dtype=float).copy()
                    global_best_result = result

            # tell():
            #     Give CMA-ES the candidates and their costs.
            #     CMA-ES then updates its internal search distribution.
            es.tell(candidates, costs)

            gen += 1
            total_evals += len(candidates)
            all_history.append(global_best_cost)

            elapsed = time.time() - t0

            # Build a short status message for terminal output.
            if global_best_result is not None and global_best_result["holed"]:
                status = (
                    f"HOLED | score={global_best_result['total_score']} "
                    f"shots={global_best_result['actual_shots']} "
                    f"pen={global_best_result['penalty_strokes']}"
                )
            else:
                status = (
                    f"not holed | dist={global_best_result['final_dist']:.2f}m "
                    f"terrain={global_best_result['final_terrain']}"
                )

            # Print every few generations to keep the terminal readable.
            if gen == 1 or gen % PRINT_EVERY_N_GENS == 0:
                print(
                    f"  {label}Gen {gen:3d} | evals {total_evals:4d} | "
                    f"best cost {global_best_cost:8.3f} | {status} | "
                    f"{elapsed:5.1f}s"
                )

            # Absolute best possible outcome:
            #     hole in 1 physical shot with no penalties.
            #
            # If this happens, no need to continue.
            if (
                global_best_result is not None
                and global_best_result["holed"]
                and global_best_result["total_score"] <= 1
            ):
                break

        # If we already found a hole-in-one, do not restart.
        if (
            global_best_result is not None
            and global_best_result["holed"]
            and global_best_result["total_score"] <= 1
        ):
            break

    return {
        "k": k,
        "best_x": global_best_x,
        "best_cost": global_best_cost,
        "best_result": global_best_result,
        "history": np.array(all_history),
        "total_evals": total_evals,
    }


# ============================================================
# Reporting and saving
# ============================================================

def print_sequence_summary(result):
    """
    Print a compact summary of one sequence result.

    This is called after each K finishes and again for the selected best result.

    It prints:
        each shot's start/end position,
        shot parameters,
        terrain,
        water penalties,
        total score.
    """
    print()
    print("Sequence:")

    for s in result["strokes"]:
        water = " [WATER +1]" if s["in_water"] else ""
        holed = " *** HOLED ***" if s["in_hole"] else ""

        print(
            f"  Shot {s['shot']}: "
            f"({s['start_pos'][0]:.1f}, {s['start_pos'][1]:.1f}) -> "
            f"({s['end_pos'][0]:.1f}, {s['end_pos'][1]:.1f}) "
            f"[{s['speed']:.1f} m/s, "
            f"{np.degrees(s['vert_angle']):.1f} deg up, "
            f"{np.degrees(s['horiz_angle']):.1f} deg aim] "
            f"terrain={s['terrain']}{water}{holed}"
        )

        if s["drop_pos"] is not None:
            print(
                f"           penalty drop -> "
                f"({s['drop_pos'][0]:.1f}, {s['drop_pos'][1]:.1f})"
            )

    print()
    print(f"  Holed:           {result['holed']}")
    print(f"  Actual shots:    {result['actual_shots']}")
    print(f"  Penalty strokes: {result['penalty_strokes']}")
    print(f"  Golf score:      {result['total_score']}")
    print(f"  Final distance:  {result['final_dist']:.3f} m")
    print(f"  Final terrain:   {result['final_terrain']}")


def strokes_to_array(strokes):
    """
    Convert stroke dictionaries into a NumPy object array for saving.

    The first columns are intentionally compatible with visualize_result.py:

        0: shot/stroke number
        1: speed
        2: vertical angle
        3: horizontal angle
        4: start_x
        5: start_y
        6: start_z

    Extra columns store additional information for analysis/reporting.
    """
    rows = []

    for s in strokes:
        drop = s["drop_pos"]

        # If there was no drop, store NaN values.
        if drop is None:
            drop = np.array([np.nan, np.nan, np.nan])

        rows.append([
            s["shot"],
            s["speed"],
            s["vert_angle"],
            s["horiz_angle"],

            s["start_pos"][0],
            s["start_pos"][1],
            s["start_pos"][2],

            s["end_pos"][0],
            s["end_pos"][1],
            s["end_pos"][2],

            s["dist_to_hole"],
            s["terrain"],
            s["landed_terrain"],

            int(s["in_hole"]),
            int(s["in_water"]),

            drop[0],
            drop[1],
            drop[2],
        ])

    return np.array(rows, dtype=object)


def save_best_result(best_package, all_packages):
    """
    Save the selected best result to an .npz file.

    This file can later be loaded for:
        report tables,
        plotting,
        visualization using visualize_result.py.

    Important compatibility note:
        visualize_result.py expects multi-stroke files to contain:
            strokes
            hole_pos
            terrain_zones
            model_path
            wind
            total_strokes
            holed
    """
    best_result = best_package["best_result"]

    save_path = Path("courses/course3_sequence_result.npz")

    # Store summary of every tested horizon K.
    # This is useful for a comparison table in the report.
    horizon_summary = []

    for package in all_packages:
        r = package["best_result"]

        horizon_summary.append([
            package["k"],
            package["best_cost"],
            int(r["holed"]),
            r["actual_shots"],
            r["penalty_strokes"],
            r["total_score"],
            r["final_dist"],
            r["final_terrain"],
            package["total_evals"],
        ])

    np.savez(
        save_path,

        # Metadata about the method.
        method="full_horizon_sequence_cmaes",

        # Best selected horizon.
        best_k=best_package["k"],

        # Best raw decision vector found by CMA-ES.
        best_x=best_package["best_x"],

        # Best scalar cost.
        best_cost=best_package["best_cost"],

        # Stroke-by-stroke result.
        strokes=strokes_to_array(best_result["strokes"]),

        # Course/environment data needed for replay.
        hole_pos=HOLE_POS,
        terrain_zones=np.array(TERRAIN_ZONES, dtype=object),
        model_path=str(MODEL_PATH),
        wind=WIND,

        # Result summary.
        actual_shots=best_result["actual_shots"],

        # Compatibility key for visualize_result.py.
        # The visualizer expects this name for multi-stroke files.
        total_strokes=best_result["actual_shots"],

        penalty_strokes=best_result["penalty_strokes"],
        total_score=best_result["total_score"],
        holed=best_result["holed"],
        final_dist=best_result["final_dist"],
        final_pos=best_result["final_pos"],

        # Summary over all tested K values.
        horizon_summary=np.array(horizon_summary, dtype=object),
    )

    print(f"\nSaved best sequence result to: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    """
    Main experiment driver.

    Steps:
        1. Print setup.
        2. Run full-horizon CMA-ES for K = 1, 2, 3.
        3. Print the best result for each K.
        4. Compare all successful horizons.
        5. Select the best result by golf score.
        6. Save the selected result.
    """
    print("=" * 70)
    print("COURSE 3: White Dogwood — Full-Horizon Sequence CMA-ES")
    print("=" * 70)
    print(f"Hole:             ({HOLE_POS[0]}, {HOLE_POS[1]})")
    print(f"Distance:         {STRAIGHT_LINE_DIST:.1f} m")
    print(f"K tested:          {K_MIN} to {K_MAX}")
    print(f"Variables:         K * 3")
    print(f"Population:        {POP_SIZE}")
    print(f"Max evals/run:     {MAX_EVALS_PER_RUN}")
    print(f"Restarts:          {MAX_RESTARTS}")
    print(f"Water rule:        +1 penalty stroke + drop")

    if np.linalg.norm(WIND) > 0:
        print(f"Wind:              ({WIND[0]}, {WIND[1]}, {WIND[2]}) m/s")

    print()

    t_start = time.time()
    all_packages = []

    # Run sequence optimization for each planning horizon.
    for k in range(K_MIN, K_MAX + 1):
        package = run_cmaes_for_k(k)
        all_packages.append(package)

        print(f"\nBest result for K={k}:")
        print_sequence_summary(package["best_result"])

    # Keep only horizons that successfully holed the ball.
    successful = [p for p in all_packages if p["best_result"]["holed"]]

    print(f"\n{'=' * 70}")
    print("FINAL COMPARISON")
    print(f"{'=' * 70}")

    # Print all K results side by side.
    for package in all_packages:
        r = package["best_result"]

        print(
            f"K={package['k']}: "
            f"holed={r['holed']} | "
            f"actual shots={r['actual_shots']} | "
            f"penalties={r['penalty_strokes']} | "
            f"golf score={r['total_score']} | "
            f"final dist={r['final_dist']:.2f} m | "
            f"best cost={package['best_cost']:.3f}"
        )

    if successful:
        # The smallest K that produced a successful physical sequence.
        shortest_successful_k = min(p["k"] for p in successful)

        # Select the best result by:
        #     1. lowest golf score
        #     2. fewer penalty strokes
        #     3. fewer physical shots
        #     4. smaller K
        #
        # This means a clean 3-shot result can be preferred over a
        # 2-shot result with water penalties if the golf score is better.
        best_package = min(
            successful,
            key=lambda p: (
                p["best_result"]["total_score"],
                p["best_result"]["penalty_strokes"],
                p["best_result"]["actual_shots"],
                p["k"],
            ),
        )

        best = best_package["best_result"]

        print()
        print(f"Shortest successful physical horizon: K = {shortest_successful_k}")
        print(f"Best selected horizon:                K = {best_package['k']}")
        print(f"Best golf score:                      {best['total_score']}")
        print(f"Actual shots:                         {best['actual_shots']}")
        print(f"Penalty strokes:                      {best['penalty_strokes']}")

        print_sequence_summary(best)

        save_best_result(best_package, all_packages)

    else:
        # If none of the horizons holed the ball, choose the closest final result.
        best_package = min(
            all_packages,
            key=lambda p: p["best_result"]["final_dist"],
        )

        print()
        print("No horizon holed the ball.")
        print(f"Closest result came from K={best_package['k']}")
        print_sequence_summary(best_package["best_result"])

        save_best_result(best_package, all_packages)

    total_time = time.time() - t_start
    print(f"\nTotal optimization time: {total_time:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()