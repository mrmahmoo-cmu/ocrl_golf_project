# Golf Trajectory Optimization — MuJoCo + CMA-ES

Optimizes golf shot parameters across three progressively harder courses using CMA-ES. MuJoCo simulates ball physics (drag, wind, terrain friction, bouncing, rolling), and CMA-ES finds the launch speed and angles that get the ball into the hole.

## Project Structure

```
golf_project/
├── models/
│   ├── course1.xml                # Course 1: Straight Shot
│   ├── course2.xml                # Course 2: Gentle Dogleg
│   └── course3.xml                # Course 3: White Dogwood
├── envs/
│   ├── __init__.py
│   └── golf_env.py                # Physics environment
├── courses/
│   ├── __init__.py
│   ├── course1.py                 # Course 1 config
│   ├── course2.py                 # Course 2 config
│   └── course3.py                 # Course 3 config
├── scripts/
│   ├── optimize_course1.py        # CMA-ES optimizer for Course 1
│   ├── optimize_course2.py        # CMA-ES optimizer for Course 2
│   ├── optimize_course3.py        # Greedy multi-stroke optimizer for Course 3
│   ├── optimize_course3_sequence.py  # Full-horizon sequence optimizer for Course 3
│   ├── visualize_result.py        # Replay results in MuJoCo viewer
│   ├── test_course3_load.py       # Smoke test for Course 3
│   └── view_course3.py            # View Course 3 layout (no optimization)
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

**Optimize a course:**
```bash
cd golf_project
py -3.13 scripts/optimize_course1.py
py -3.13 scripts/optimize_course2.py
py -3.13 scripts/optimize_course3.py
py -3.13 scripts/optimize_course3_sequence.py
```
Each optimizer prints generation-by-generation progress and saves results to `courses/courseN_result.npz`.

**Visualize results:**
```bash
py -3.13 scripts/visualize_result.py
```
Prompts you to select a course (1, 2, or 3), then opens the MuJoCo viewer and replays the optimized shot(s). For Course 3, each stroke is shown sequentially with a prompt between strokes. Viewer controls: left-click drag to rotate, scroll to zoom, right-click drag to pan.

## Courses

**Course 1: "Straight Shot"** — 35m straight course with one sand bunker. No wind. Serves as the baseline to verify the optimization method works. Goal: hole-in-one.

**Course 2: "Gentle Dogleg"** — 39m course that bends right with a hill, two sand bunkers, and a crosswind. Adds wind physics and landed-terrain penalties. The optimizer must compensate for wind drift. Goal: hole-in-one.

**Course 3: "White Dogwood"** — 117m course inspired by Augusta National Hole 11. Features a winding fairway, a water hazard near the green, bunkers, mounds, wind, and decorative trees. Too long for a single shot, so the optimizer plans multiple strokes, avoiding the pond and minimizing total strokes. Goal: complete in as few strokes as possible.

## File Descriptions

**`envs/golf_env.py`** — Core physics environment. Wraps MuJoCo and adds aerodynamic drag (wind-relative), slope-aware rolling deceleration by terrain type, persistent ground detection, and hole completion logic. Exposes `reset()`, `launch_ball()`, and `step()` methods. No robot arm — the ball's initial velocity is set directly.

**`courses/course1.py`, `course2.py`, `course3.py`** — Configuration for each course: hole position, terrain zone boundaries, model path, and wind vector. Terrain zones are axis-aligned rectangles checked in priority order (first match wins).

**`models/course1.xml`, `course2.xml`, `course3.xml`** — MuJoCo model files defining terrain geometry, decorative elements (trees, flag poles, tee markers), lighting, and the golf ball. Decorative elements have collisions disabled so the ball passes through them.

**`scripts/optimize_course1.py`, `optimize_course2.py`** — Single-shot CMA-ES optimizers. Three decision variables per shot: launch speed, vertical angle, horizontal angle. Use restarts with shrinking search radius if the initial run stalls.

**`scripts/optimize_course3.py`** — Greedy multi-stroke CMA-ES optimizer. Optimizes one stroke at a time from the ball's current position, with early stopping per stroke when improvement stagnates. Includes water hazard penalties and penalty drops.

**`scripts/optimize_course3_sequence.py`** — Full-horizon sequence CMA-ES optimizer. Optimizes all K strokes jointly rather than one at a time, allowing stroke 1 to be planned with awareness of how it affects subsequent strokes. Tests planning horizons of K=2 and K=3, then selects the best result by golf score. Includes water penalty drop logic and golf-style scoring (physical shots + penalty strokes).

**`scripts/visualize_result.py`** — Loads saved results and replays them in the MuJoCo viewer using the same `GolfEnv` class as the optimizer, guaranteeing identical physics. Handles both single-shot and multi-stroke formats.

**`scripts/test_course3_load.py`** — Smoke test that verifies Course 3's model loads and a basic shot runs without errors.

**`scripts/view_course3.py`** — Opens Course 3 in the MuJoCo viewer for visual inspection without running any optimization.

## Optimization Methods

### Greedy Per-Stroke (Courses 1, 2, 3)
CMA-ES optimizes each shot independently. For single-shot courses (1 and 2), this means finding the one best launch. For Course 3, each stroke is optimized from wherever the ball currently sits, then the ball is moved to its resting position and the next stroke is optimized. Three decision variables per stroke: launch speed, vertical angle, horizontal angle.

### Full-Horizon Sequence (Course 3)
CMA-ES optimizes all K strokes simultaneously as a single decision vector (K×3 variables). This allows earlier strokes to be planned with awareness of how they affect later strokes. For example, stroke 1 might intentionally land shorter to give stroke 2 a cleaner line to the hole. The full-horizon approach found a 2-stroke solution on Course 3, compared to 3 strokes from the greedy approach.

Both methods use CMA-ES as the underlying optimizer, with restarts and shrinking search radii to refine solutions.

## Physics

All ball physics are applied in Python on top of MuJoCo's contact solver:

- **Drag**: Quadratic, computed from velocity relative to wind
- **Rolling**: Terrain-dependent braking force projected along the surface plane using contact normals
- **Bounce**: Handled naturally by MuJoCo's contact elasticity
- **Wind**: Shifts the reference frame for drag computation
- **Ground detection**: Persistent flag prevents contact flickering noise
- **Hole detection**: Requires ball to be within hole radius AND currently touching the ground
- **Water hazard**: Ball landing in water incurs a penalty stroke and drop outside the hazard