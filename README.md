# Golf Swing Optimization — MuJoCo Environment

## Project Structure

```
golf_project/
├── models/
│   └── arm_club_ball.xml        # MuJoCo model: defines all physical objects
├── envs/
│   ├── __init__.py
│   └── golf_env.py              # Python environment: drives the simulation
├── scripts/
│   ├── test_basic_swing.py      # Quick test: does the arm hit the ball?
│   ├── test_phase2.py           # Full test: drag, terrain, hole completion
│   └── visualize_swing.py       # Visual simulation in MuJoCo viewer
├── courses/                     # Course definitions (Phase 3)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
pip install mujoco gymnasium numpy
```

Requires Python 3.10+.

## How to Run

**Watch the simulation visually:**
```bash
cd golf_project
py -3.13 scripts/visualize_swing.py
```
This opens a MuJoCo viewer window where you can see the arm swing, the ball launch, and the ball roll down the course. The terminal will prompt you to press Enter before starting. After the simulation finishes, the viewer stays open so you can inspect the scene. Close the window to exit.

Viewer controls: left-click drag to rotate, scroll to zoom, right-click drag to pan.

You can edit the top of `visualize_swing.py` to change the hole position or playback speed.

**Run the test suite (no viewer, just terminal output):**
```bash
py -3.13 scripts/test_basic_swing.py
py -3.13 scripts/test_phase2.py
```

## What Each File Does

### `models/arm_club_ball.xml`
The MuJoCo model file. Written in XML, it defines every physical object in the simulation: the 2-DOF robot arm (elbow + wrist joints), the club shaft and head, the golf ball, the terrain surfaces (tee, fairway, sand, green, rough), and the hole marker. It also defines material properties like friction and mass. This is what MuJoCo's physics engine uses to compute collisions, gravity, and contact forces.

### `envs/golf_env.py`
The Python environment that drives the simulation. It loads the XML model and adds the physics that can't be defined in XML alone: air drag on the ball, surface-dependent rolling deceleration, and hole completion logic (the ball freezes when it reaches the hole). It also provides a clean API (`reset()`, `step()`, `get_obs()`) that the optimization and RL code in Phase 3 will use.

### `scripts/test_basic_swing.py`
A quick sanity check. Runs an open-loop swing with no drag or terrain effects and verifies that the arm makes contact with the ball and launches it in the correct direction (+X, toward the hole). If all checks pass, the core pipeline is working.

### `scripts/test_phase2.py`
A more thorough test. Verifies that air drag reduces travel distance, that terrain zones are classified correctly, that the ball comes to rest on its own, and that the ball freezes when it reaches the hole. Compares runs with and without physics to confirm each system is doing its job.

### `scripts/visualize_swing.py`
The visual simulation. Runs the same physics as the test scripts but opens a MuJoCo viewer window so you can watch everything happen. This is the main way to see your progress — any changes to the model or environment will show up here.

## How the XML and Python Work Together

The XML model defines *what exists* — the arm, ball, terrain, and their physical properties. MuJoCo uses this to simulate physics (gravity, collisions, joint motion).

The Python code defines *what happens* — it applies the swing torques to the arm joints, adds air drag and ground friction forces to the ball, detects when the ball enters the hole, and controls the simulation timing. Without the Python code, the model just sits there under gravity.

The visual simulation (`visualize_swing.py`) combines both: it loads the XML model, runs the Python physics, and displays the result in a viewer window.

## Current Status

- **Phase 1 (done):** 2-DOF arm swings and hits the ball toward the hole
- **Phase 2 (done):** Air drag, terrain friction zones, rolling deceleration, hole completion
- **Phase 3 (next):** Course design and swing optimization (trajectory optimization + RL)