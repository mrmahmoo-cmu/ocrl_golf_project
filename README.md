# Golf Swing Optimization — MuJoCo + CMA-ES

A 2-DOF robot arm swings a golf club to hit a ball across progressively harder courses. MuJoCo handles the physics (contact, gravity, collisions), and CMA-ES optimizes the swing trajectory to achieve a hole-in-one.

## Project Structure

```
golf_project/
├── models/
│   ├── arm_club_ball.xml          # Base model (no aim joint, used by tests)
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
│   ├── test_basic_swing.py        # Sanity check: does the arm hit the ball?
│   ├── test_phase2.py             # Full physics test: drag, terrain, hole completion
│   ├── visualize_swing.py         # Open-loop swing visualization
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
py -3.13 scripts/visualize_result.py                          # Course 1 (default)
py -3.13 scripts/visualize_result.py courses/course2_result.npz   # Course 2
```
Opens a MuJoCo viewer. Press Enter in the terminal to start the swing. The viewer shows the arm swing, ball flight, bouncing, and rolling. The terminal prints contact, launch, and landing events. Close the viewer window to exit.

Viewer controls: left-click drag to rotate, scroll to zoom, right-click drag to pan.

**Run the test suite (no viewer, terminal output only):**
```bash
py -3.13 scripts/test_basic_swing.py
py -3.13 scripts/test_phase2.py
```

## Courses

### Course 1: "Straight Shot"
A simple 35m straight course with one sand bunker near the hole. No wind, no dogleg. The optimizer converges to a hole-in-one relatively quickly, demonstrating basic CMA-ES behavior. Hole at (35, 0).

### Course 2: "Gentle Dogleg"
A 39m course that bends right toward the hole at (38, 8). Features a hill on the fairway, two sand bunkers, a 1.5 m/s crosswind, and landed-terrain penalties that discourage the ball from landing in rough or sand. Significantly harder to optimize — typically requires multiple CMA-ES restarts.

## What Each File Does

### `models/arm_club_ball.xml`
The base MuJoCo model. Defines the 2-DOF robot arm (elbow + wrist hinge joints), club shaft and head, golf ball with a free joint, and flat terrain. Used by the test scripts. Does not include an aim joint.

### `models/course1.xml` / `models/course2.xml`
Course-specific MuJoCo models. Each adds an aim joint (turntable hinge for aiming direction), course-specific terrain layout (fairway, green, sand, rough), decorative trees (cylinder trunks + sphere canopies), a flag pole at the hole, and a tee marker. All decorative elements use `contype="0" conaffinity="0"` so the ball passes through them.

### `envs/golf_env.py`
The Python environment that wraps MuJoCo. It loads an XML model and adds physics that can't be defined in XML alone:

- **Aerodynamic drag**
- **Slope-aware rolling deceleration**
- **Persistent ground detection**
- **Hole completion**
- **Wind**

The environment provides `reset()`, `step(action)`, and `get_obs()` methods used by both the optimizer and visualizer.

### `courses/course1.py` / `courses/course2.py`
Course configuration files. Each defines the hole position, terrain zone boundaries (axis-aligned rectangles, first match wins), model path, and derived quantities (straight-line distance, ideal aim angle). Course 2 additionally defines a wind vector and is used with landed-terrain penalties.

### `scripts/optimize_course1.py` / `scripts/optimize_course2.py`
CMA-ES trajectory optimizers. Each parameterizes the swing as 16 decision variables: aim angle, backswing angle, and 7 torque bins × 2 joints (elbow + wrist). The cost function runs a full simulation and returns distance to the hole, with penalties for no contact, wrong direction, timeout, and landing on unfavorable terrain.

CMA-ES with restarts is used: the initial run explores broadly, and subsequent restarts refine around the best solution found with progressively smaller search radii (sigma). Course 2 additionally passes wind to the simulation and penalizes hole-in-ones that land in rough or sand.

### `scripts/visualize_result.py`
Replays saved optimization results in the MuJoCo viewer. Creates an actual `GolfSwingEnv` instance with the same parameters used during optimization (model, terrain zones, wind), guaranteeing identical physics. Accepts an optional command-line argument for the result file path; defaults to Course 1.

### `scripts/test_basic_swing.py`
Sanity check using the base model. Runs an open-loop swing and verifies: model loads, club contacts ball, ball launches toward the hole, ball gains height, ball moves forward. All six checks must pass.

### `scripts/test_phase2.py`
Full physics test using the base model. Compares runs with and without drag/rolling to verify each system works. Checks: drag reduces travel distance, ball stops on its own, terrain zones classify correctly, ball freezes at the hole.

## Optimization Method: CMA-ES

CMA-ES is used instead of gradient-based methods because the cost function is non-differentiable: contact events, terrain transitions, and hole completion create discontinuities that prevent gradient computation.

## Physics Summary

| Feature | Implementation |
|---|---|
| Contact & collisions | MuJoCo solver (solref/solimp parameters) |
| Air drag | Quadratic, Cd=0.25, wind-relative velocity |
| Rolling friction | Terrain-dependent braking force, slope-aware |
| Ball bounce | MuJoCo contact elasticity (natural, not scripted) |
| Wind | Modifies effective drag via relative velocity model |
| Ground detection | MuJoCo contact list, persistent `ball_landed` flag |
| Hole completion | XY distance < radius + ball landed → freeze ball |

## Current Status

- **Phase 1 (done):** 2-DOF arm swings and hits the ball
- **Phase 2 (done):** Air drag, terrain friction, rolling deceleration, hole completion
- **Phase 3 (in progress):** Course design, CMA-ES optimization, wind, slope-aware rolling
  - Course 1: Straight Shot — hole-in-one achieved
  - Course 2: Gentle Dogleg with wind — hole-in-one achieved
  - Course 3: Multi-stroke course
