# Golf Trajectory Optimization — MuJoCo + CMA-ES

A golf ball is launched with optimized speed and angles across progressively harder courses. MuJoCo handles the physics (contact, gravity, collisions, bouncing), and CMA-ES optimizes the launch parameters to achieve a hole-in-one.

## Project Structure

```
golf_project/
├── models/
│   ├── course1.xml                # Course 1: Straight Shot
│   └── course2.xml                # Course 2: Gentle Dogleg
├── envs/
│   ├── __init__.py
│   └── golf_env.py                # Physics environment (drag, rolling, wind, terrain)
├── courses/
│   ├── __init__.py
│   ├── course1.py                 # Course 1 config (hole pos, terrain zones)
│   └── course2.py                 # Course 2 config (+ wind, landed-terrain penalties)
├── scripts/
│   ├── test_basic_swing.py        # Sanity check: does the ball launch correctly?
│   ├── test_phase2.py             # Full physics test: drag, terrain, hole completion
│   ├── visualize_result.py        # Replay optimized results in MuJoCo viewer
│   ├── optimize_course1.py        # CMA-ES optimizer for Course 1
│   └── optimize_course2.py        # CMA-ES optimizer for Course 2
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
pip install mujoco gymnasium numpy scipy cma
```

Requires Python 3.10+.

## How to Run

**Run the optimizer for a course:**
```bash
cd golf_project
py -3.13 scripts/optimize_course1.py
py -3.13 scripts/optimize_course2.py
```
The optimizer prints progress each generation (best distance to hole, elapsed time) and saves the result to `courses/courseN_result.npz` when finished.

**Visualize the optimized result:**
```bash
py -3.13 scripts/visualize_result.py                               # Course 1 (default)
py -3.13 scripts/visualize_result.py courses/course2_result.npz    # Course 2
```
Opens a MuJoCo viewer. Press Enter in the terminal to launch the ball. The viewer shows ball flight, bouncing, and rolling. The terminal prints launch and landing events. Close the viewer window to exit.

Viewer controls: left-click drag to rotate, scroll to zoom, right-click drag to pan.

**Run the test suite (no viewer, terminal output only):**
```bash
py -3.13 scripts/test_basic_swing.py
py -3.13 scripts/test_phase2.py
```

## Courses

### Course 1: "Straight Shot"
A simple 35m straight course with one sand bunker near the hole. No wind, no dogleg. The optimizer converges to a hole-in-one within a few generations. Hole at (35, 0).

### Course 2: "Gentle Dogleg"
A 39m course that bends right toward the hole at (38, 8). Features a hill on the fairway, two sand bunkers, a 1.5 m/s crosswind, and landed-terrain penalties that discourage the ball from landing in rough or sand. The optimizer must account for wind drift when choosing the launch angle.

## What Each File Does

### `models/course1.xml` / `models/course2.xml`
Course-specific MuJoCo models. Each defines the terrain layout (fairway, green, sand, rough), decorative trees (cylinder trunks + sphere canopies), a flag pole at the hole, a tee marker, and the golf ball with a free joint. All decorative elements use `contype="0" conaffinity="0"` so the ball passes through them. There is no robot arm — the ball's initial velocity is set directly by the optimizer.

### `envs/golf_env.py`
The Python environment that wraps MuJoCo. It loads an XML model, launches the ball with a specified speed and angle, and applies physics that can't be defined in XML alone:

- **Aerodynamic drag** — Quadratic drag using the ball's velocity relative to the wind. A tailwind reduces drag, a headwind increases it, and a crosswind pushes the ball sideways.
- **Slope-aware rolling deceleration** — Uses MuJoCo's contact normals to project braking force along the surface plane, scaled by cos(slope angle). Different terrain types apply different deceleration rates (green: 0.8 m/s², fairway: 2.0, rough: 5.0, sand: 8.0).
- **Persistent ground detection** — A `ball_landed` flag activates on first ground contact after launch and stays on permanently, preventing contact flickering from adding noise to the cost function.
- **Hole completion** — The ball freezes when its XY distance to the hole is within `HOLE_RADIUS`, it has landed at least once, and it is currently in contact with the ground.
- **Wind** — 3D wind vector passed at construction. Defaults to zero.

The environment provides `reset()`, `launch_ball(speed, vert_angle, horiz_angle)`, and `step()` methods used by both the optimizer and visualizer.

### `courses/course1.py` / `courses/course2.py`
Course configuration files. Each defines the hole position, terrain zone boundaries (axis-aligned rectangles, first match wins), and model path. Course 2 additionally defines a wind vector.

### `scripts/optimize_course1.py` / `scripts/optimize_course2.py`
CMA-ES optimizers. Each parameterizes the shot as **3 decision variables**: launch speed (m/s), vertical angle (radians), and horizontal angle (radians). The cost function runs a full MuJoCo simulation and returns distance to the hole, with penalties for timeout and landing on unfavorable terrain.

CMA-ES with restarts is used: the initial run explores broadly, and subsequent restarts refine around the best solution found with progressively smaller search radii. Course 2 additionally passes wind to the simulation and penalizes hole-in-ones that land in rough or sand.

### `scripts/visualize_result.py`
Replays saved optimization results in the MuJoCo viewer. Creates a `GolfEnv` instance with the same parameters used during optimization (model, terrain zones, wind), guaranteeing identical physics. Accepts an optional command-line argument for the result file path; defaults to Course 1.

### `scripts/test_basic_swing.py`
Basic test script. Launches the ball at 20 m/s, 30° up, straight ahead with no drag. Verifies: ball launches, gains height, moves forward in +X, and returns to ground.

### `scripts/test_phase2.py`
Full physics test. Compares runs with and without drag/rolling to verify each system works. Checks: drag reduces travel distance, ball stops on its own, terrain zones classify correctly, ball freezes at the hole.

## Optimization: CMA-ES

CMA-ES is used instead of gradient-based methods because the cost function is non-differentiable — contact bouncing, terrain transitions, and hole completion create discontinuities.

With only 3 variables, the optimizer converges in seconds rather than minutes. A population of 16 candidates per generation works well.

## Physics Summary

| Feature | Implementation |
|---|---|
| Contact & collisions | MuJoCo solver (solref/solimp parameters) |
| Air drag | Quadratic, Cd=0.25, wind-relative velocity |
| Rolling friction | Terrain-dependent braking force, slope-aware |
| Ball bounce | MuJoCo contact elasticity (natural, not scripted) |
| Wind | Modifies effective drag via relative velocity model |
| Ground detection | MuJoCo contact list, persistent `ball_landed` flag |
| Hole completion | XY distance < radius + currently on ground → freeze ball |

## Current Status

- Course 1: Straight Shot — hole-in-one achieved (~4 generations, ~20 seconds)
- Course 2: Gentle Dogleg with wind — hole-in-one achieved (~4 generations, ~22 seconds)
- Course 3: Multi-stroke course